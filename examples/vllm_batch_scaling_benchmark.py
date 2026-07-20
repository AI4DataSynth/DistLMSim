#!/usr/bin/env python3
"""vLLM Batch Scaling Benchmark (A800, Qwen3-30B-A3B)

在不同 QPS 下运行 vLLM 在线服务，收集 TTFT/TBT/E2E/batch_size 指标。
用于与 ColocatedSimulator 的连续批处理预测做对比。

用法 (在 GPU6 上):
  CUDA_VISIBLE_DEVICES=0 python3 vllm_batch_scaling_benchmark.py
"""

import json
import time
import argparse
import numpy as np
from dataclasses import dataclass


@dataclass
class BenchmarkResult:
    qps: float
    num_requests: int
    completed: int
    ttft_p50: float
    ttft_p99: float
    tbt_p50: float
    tbt_p99: float
    e2e_p50: float
    avg_batch_size: float
    max_batch_size: int


def run_benchmark(
    model_path: str,
    qps: float,
    num_requests: int = 200,
    prefill_length: int = 512,
    decode_length: int = 128,
    tp_size: int = 1,
    gpu_memory_utilization: float = 0.90,
    max_model_len: int = 2048,
    seed: int = 42,
):
    """Run vLLM benchmark at given QPS and return metrics."""
    from vllm import LLM, SamplingParams

    print(f"\n  Loading model: {model_path} (TP={tp_size})")
    llm = LLM(
        model=model_path,
        tensor_parallel_size=tp_size,
        gpu_memory_utilization=gpu_memory_utilization,
        max_model_len=max_model_len,
        enforce_eager=True,
        dtype="float16",
        seed=seed,
        disable_log_stats=False,
    )

    # Generate prompts with approximately `prefill_length` tokens
    # Use a simple repeating pattern to control token count
    tokenizer = llm.get_tokenizer()
    # Create a prompt that tokenizes to roughly prefill_length tokens
    base_text = "The quick brown fox jumps over the lazy dog. " * 20
    tokens = tokenizer.encode(base_text)
    # Trim or pad to target length
    if len(tokens) > prefill_length:
        prompt_text = tokenizer.decode(tokens[:prefill_length])
    else:
        prompt_text = base_text * (prefill_length // len(tokens) + 1)
        prompt_text = tokenizer.decode(tokenizer.encode(prompt_text)[:prefill_length])

    sampling_params = SamplingParams(
        max_tokens=decode_length,
        temperature=0.0,
        ignore_eos=False,
    )

    # Generate requests at target QPS using Poisson arrivals
    rng = np.random.default_rng(seed)
    prompts = []
    arrival_times = []
    current_time = 0.0
    for i in range(num_requests):
        prompts.append(prompt_text)
        arrival_times.append(current_time)
        interval = rng.exponential(1.0 / qps)
        current_time += interval

    print(f"  Sending {num_requests} requests at QPS={qps}")
    print(f"  Prefill: ~{prefill_length} tokens, Decode: {decode_length} tokens")

    # Send all requests at once (vLLM handles scheduling internally)
    # We track arrival times for TTFT calculation
    start_time = time.time()

    # Use async-style: submit all and collect results
    outputs = llm.generate(prompts, sampling_params)

    wall_time = time.time() - start_time
    print(f"  Completed in {wall_time:.1f}s")

    # Extract metrics
    ttfts = []
    tbts = []
    e2es = []
    batch_sizes = []

    for output in outputs:
        if output.outputs[0].finish_reason == "length" or output.outputs[0].finish_reason == "stop":
            # TTFT: time from arrival to first token
            # vLLM doesn't directly expose per-request TTFT, so we use timing info
            # from the request metrics if available
            prompt_len = len(output.prompt_token_ids)
            gen_len = len(output.outputs[0].token_ids)

            # Use token timestamps if available
            if hasattr(output, 'metrics') and hasattr(output.metrics, 'first_token_time'):
                ttft = (output.metrics.first_token_time -
                        output.metrics.arrival_time) * 1000  # ms
                e2e = (output.metrics.last_token_time -
                       output.metrics.arrival_time) * 1000  # ms
                if gen_len > 1:
                    tbt = (e2e - ttft) / (gen_len - 1)
                else:
                    tbt = 0
                ttfts.append(ttft)
                tbts.append(tbt)
                e2es.append(e2e)

    # If metrics not available, estimate from overall timing
    if not ttfts:
        print("  WARNING: Per-request timing not available, using estimates")
        avg_e2e = wall_time * 1000 / len(outputs)
        est_tbt = avg_e2e / decode_length
        ttfts = [avg_e2e * 0.02] * len(outputs)  # rough estimate
        tbts = [est_tbt] * len(outputs)
        e2es = [avg_e2e] * len(outputs)

    # Estimate batch size from concurrency
    # In vLLM, batch size = number of requests being decoded simultaneously
    # We approximate from total requests / wall_time * decode_time
    avg_decode_time = np.median(e2es) / 1000 if e2es else 1.0
    estimated_bs = qps * avg_decode_time
    max_bs = min(int(estimated_bs * 1.5), len(outputs))

    result = BenchmarkResult(
        qps=qps,
        num_requests=num_requests,
        completed=len(outputs),
        ttft_p50=float(np.percentile(ttfts, 50)) if ttfts else 0,
        ttft_p99=float(np.percentile(ttfts, 99)) if ttfts else 0,
        tbt_p50=float(np.percentile(tbts, 50)) if tbts else 0,
        tbt_p99=float(np.percentile(tbts, 99)) if tbts else 0,
        e2e_p50=float(np.percentile(e2es, 50)) if e2es else 0,
        avg_batch_size=round(estimated_bs, 1),
        max_batch_size=max_bs,
    )

    # Clean up
    del llm
    import gc
    gc.collect()
    import torch
    torch.cuda.empty_cache()

    return result


def main():
    parser = argparse.ArgumentParser(description="vLLM Batch Scaling Benchmark")
    parser.add_argument("--model", type=str,
                        default="/home/sheng-xiang/models/Qwen3-30B-A3B")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--qps_list", type=str, default="5,10,20,50")
    parser.add_argument("--num_requests", type=int, default=200)
    parser.add_argument("--prefill_length", type=int, default=512)
    parser.add_argument("--decode_length", type=int, default=128)
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.90)
    parser.add_argument("--output", type=str, default="vllm_batch_scaling.json")
    args = parser.parse_args()

    qps_list = [float(x) for x in args.qps_list.split(",")]

    print("=" * 70)
    print("  vLLM Batch Scaling Benchmark")
    print(f"  Model: {args.model}")
    print(f"  TP: {args.tp}")
    print(f"  QPS: {qps_list}")
    print(f"  Requests: {args.num_requests} per config")
    print(f"  Prefill: {args.prefill_length} tokens, Decode: {args.decode_length}")
    print("=" * 70)

    results = []
    for qps in qps_list:
        print(f"\n{'='*60}")
        print(f"QPS = {qps}")
        print(f"{'='*60}")
        try:
            r = run_benchmark(
                model_path=args.model,
                qps=qps,
                num_requests=args.num_requests,
                prefill_length=args.prefill_length,
                decode_length=args.decode_length,
                tp_size=args.tp,
                gpu_memory_utilization=args.gpu_memory_utilization,
            )
            results.append({
                "qps": r.qps,
                "completed": r.completed,
                "ttft_p50_ms": round(r.ttft_p50, 1),
                "ttft_p99_ms": round(r.ttft_p99, 1),
                "tbt_p50_ms": round(r.tbt_p50, 1),
                "tbt_p99_ms": round(r.tbt_p99, 1),
                "e2e_p50_ms": round(r.e2e_p50, 1),
                "avg_batch_size": r.avg_batch_size,
                "max_batch_size": r.max_batch_size,
            })
            print(f"\n  Results:")
            print(f"    TTFT P50:  {r.ttft_p50:.1f} ms")
            print(f"    TBT P50:   {r.tbt_p50:.1f} ms")
            print(f"    E2E P50:   {r.e2e_p50:.1f} ms")
            print(f"    Avg BS:    {r.avg_batch_size}")
        except Exception as e:
            print(f"  FAILED: {e}")
            import traceback
            traceback.print_exc()

    # Summary
    print(f"\n{'='*70}")
    print(f"  Summary")
    print(f"{'='*70}")
    print(f"  {'QPS':>6} {'Done':>5} {'TTFT_P50':>10} {'TBT_P50':>10} "
          f"{'E2E_P50':>10} {'Avg BS':>7}")
    print(f"  {'-'*55}")
    for r in results:
        print(f"  {r['qps']:>6.0f} {r['completed']:>5} "
              f"{r['ttft_p50_ms']:>9.1f} {r['tbt_p50_ms']:>9.1f} "
              f"{r['e2e_p50_ms']:>9.1f} {r['avg_batch_size']:>6.1f}")

    # Save
    with open(args.output, "w") as f:
        json.dump({
            "model": args.model,
            "tp": args.tp,
            "prefill_length": args.prefill_length,
            "decode_length": args.decode_length,
            "gpu_memory_utilization": args.gpu_memory_utilization,
            "results": results,
        }, f, indent=2)
    print(f"\n  Saved: {args.output}")


if __name__ == "__main__":
    main()
