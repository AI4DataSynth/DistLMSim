"""Tests for distlmsim.config module."""

import sys
import unittest
from unittest.mock import patch

sys.path.insert(0, '.')

from distlmsim.config import (
    DeviceSKUConfig,
    NodeSKUConfig,
    NVLinkConfig,
    RDMAConfig,
    NetworkTopologyConfig,
    ModelConfig,
    ReplicaConfig,
    ClusterConfig,
    DisaggregatedConfig,
    SchedulingConfig,
    RequestGeneratorConfig,
    MetricsConfig,
    SimulationConfig,
)
from distlmsim.types import (
    DeviceSKUType,
    NodeSKUType,
    InterconnectType,
    RDMAProtocolType,
    NetworkModelMode,
    DeploymentMode,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    KVCacheTransferStrategy,
)


class TestDeviceSKUConfig(unittest.TestCase):
    """Test DeviceSKUConfig defaults."""

    def test_default_values(self):
        """Verify DeviceSKUConfig has correct default values."""
        config = DeviceSKUConfig()
        self.assertEqual(config.device_type, DeviceSKUType.A800)
        self.assertEqual(config.fp16_tflops, 25.0)
        self.assertEqual(config.memory_gb, 80.0)
        self.assertEqual(config.memory_bandwidth_gbps, 2039.0)


class TestNodeSKUConfig(unittest.TestCase):
    """Test NodeSKUConfig defaults."""

    def test_default_values(self):
        """Verify NodeSKUConfig has correct default values."""
        config = NodeSKUConfig()
        self.assertEqual(config.node_type, NodeSKUType.A800_DGX)
        self.assertEqual(config.num_gpus, 8)
        self.assertIsInstance(config.device_sku, DeviceSKUConfig)

    def test_nested_device_sku(self):
        """Verify nested DeviceSKUConfig is properly initialized."""
        config = NodeSKUConfig()
        self.assertEqual(config.device_sku.device_type, DeviceSKUType.A800)
        self.assertEqual(config.device_sku.fp16_tflops, 25.0)


class TestNVLinkConfig(unittest.TestCase):
    """Test NVLinkConfig defaults."""

    def test_default_values(self):
        """Verify NVLinkConfig has correct default values."""
        config = NVLinkConfig()
        self.assertEqual(config.interconnect_type, InterconnectType.NVLINK_SWITCH)
        self.assertEqual(config.bandwidth_gbps, 600.0)
        self.assertEqual(config.num_links_per_gpu, 12)
        self.assertEqual(config.latency_us, 1.5)
        self.assertEqual(config.nvswitch_bandwidth_gbps, 900.0)


class TestRDMAConfig(unittest.TestCase):
    """Test RDMAConfig defaults."""

    def test_default_values(self):
        """Verify RDMAConfig has correct default values."""
        config = RDMAConfig()
        self.assertEqual(config.protocol, RDMAProtocolType.ROCE_V2)
        self.assertEqual(config.bandwidth_gbps, 200.0)
        self.assertEqual(config.latency_us, 2.0)
        self.assertEqual(config.congestion_control, "DCQCN")
        self.assertTrue(config.ecn_enabled)
        self.assertTrue(config.pfc_enabled)
        self.assertTrue(config.ib_subnet_manager)
        self.assertEqual(config.ib_service_level, 0)


class TestNetworkTopologyConfig(unittest.TestCase):
    """Test NetworkTopologyConfig defaults."""

    def test_default_values(self):
        """Verify NetworkTopologyConfig has correct default values."""
        config = NetworkTopologyConfig()
        self.assertIsInstance(config.nvlink, NVLinkConfig)
        self.assertIsInstance(config.rdma, RDMAConfig)
        self.assertEqual(config.model_mode, NetworkModelMode.HYBRID)
        self.assertEqual(config.topology_type, "fat_tree")
        self.assertEqual(config.num_switch_layers, 2)
        self.assertEqual(config.oversubscription_ratio, 1.0)
        self.assertEqual(config.nccl_cpu_launch_overhead_ms, 0.02)
        self.assertEqual(config.nccl_cpu_skew_overhead_per_device_ms, 0.0)


class TestModelConfig(unittest.TestCase):
    """Test ModelConfig defaults."""

    def test_default_values(self):
        """Verify ModelConfig has correct default values."""
        config = ModelConfig()
        self.assertEqual(config.model_name, "Qwen3-30B-A3B")
        self.assertEqual(config.num_layers, 48)
        self.assertEqual(config.num_q_heads, 32)
        self.assertEqual(config.num_kv_heads, 4)
        self.assertEqual(config.embedding_dim, 2048)
        self.assertEqual(config.mlp_hidden_dim, 0)
        self.assertEqual(config.num_experts, 128)
        self.assertEqual(config.top_k_experts, 8)
        self.assertEqual(config.vocab_size, 151936)


class TestReplicaConfig(unittest.TestCase):
    """Test ReplicaConfig defaults."""

    def test_default_values(self):
        """Verify ReplicaConfig has correct default values."""
        config = ReplicaConfig()
        self.assertIsInstance(config.model, ModelConfig)
        self.assertIsInstance(config.device_sku, DeviceSKUConfig)
        self.assertEqual(config.tensor_parallel_size, 1)
        self.assertEqual(config.num_pipeline_stages, 1)
        self.assertEqual(config.expert_parallel_size, 1)
        self.assertFalse(config.enable_expert_parallel)
        self.assertEqual(config.scheduler_type, ReplicaSchedulerType.SARATHI)
        self.assertEqual(config.max_batch_size, 256)
        self.assertEqual(config.max_num_tokens, 16384)


class TestClusterConfig(unittest.TestCase):
    """Test ClusterConfig defaults."""

    def test_default_values(self):
        """Verify ClusterConfig has correct default values."""
        config = ClusterConfig()
        self.assertEqual(config.num_nodes, 2)
        self.assertIsInstance(config.node_sku, NodeSKUConfig)
        self.assertIsInstance(config.network, NetworkTopologyConfig)
        self.assertEqual(config.num_replicas, 1)
        self.assertIsInstance(config.replica, ReplicaConfig)
        self.assertEqual(config.deployment_mode, DeploymentMode.COLOCATED)


class TestDisaggregatedConfig(unittest.TestCase):
    """Test DisaggregatedConfig defaults."""

    def test_default_values(self):
        """Verify DisaggregatedConfig has correct default values."""
        config = DisaggregatedConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.num_prefill_nodes, 1)
        self.assertEqual(config.num_decode_nodes, 1)
        self.assertEqual(config.kv_cache_transfer_strategy, KVCacheTransferStrategy.DIRECT)
        self.assertFalse(config.kv_cache_compression)
        self.assertEqual(config.kv_cache_compression_ratio, 2.0)
        self.assertEqual(config.prefill_batch_size, 32)
        self.assertEqual(config.decode_batch_size, 256)
        self.assertTrue(config.enable_chunked_prefill)
        self.assertEqual(config.prefill_chunk_size, 4096)


class TestSchedulingConfig(unittest.TestCase):
    """Test SchedulingConfig defaults."""

    def test_default_values(self):
        """Verify SchedulingConfig has correct default values."""
        config = SchedulingConfig()
        self.assertEqual(config.global_scheduler_type, GlobalSchedulerType.ROUND_ROBIN)
        self.assertEqual(config.replica_scheduler_type, ReplicaSchedulerType.SARATHI)
        self.assertFalse(config.enable_request_migration)
        self.assertEqual(config.migration_interval_ms, 1000.0)
        self.assertEqual(config.load_balancing_interval_ms, 500.0)


class TestRequestGeneratorConfig(unittest.TestCase):
    """Test RequestGeneratorConfig defaults."""

    def test_default_values(self):
        """Verify RequestGeneratorConfig has correct default values."""
        config = RequestGeneratorConfig()
        self.assertEqual(config.generator_type, "synthetic")
        self.assertEqual(config.qps, 10.0)
        self.assertEqual(config.prefill_length, 2048)
        self.assertEqual(config.decode_length, 512)
        self.assertIsNone(config.trace_file)
        self.assertEqual(config.length_distribution, "normal")
        self.assertEqual(config.length_cv, 0.3)


class TestMetricsConfig(unittest.TestCase):
    """Test MetricsConfig defaults."""

    def test_default_values(self):
        """Verify MetricsConfig has correct default values."""
        config = MetricsConfig()
        self.assertTrue(config.enable_detailed_logging)
        self.assertEqual(config.output_dir, "results")
        self.assertEqual(config.percentiles, [50, 90, 95, 99])
        self.assertFalse(config.enable_plots)


class TestSimulationConfig(unittest.TestCase):
    """Test SimulationConfig defaults and from_cli method."""

    def test_default_values(self):
        """Verify SimulationConfig has correct default values."""
        config = SimulationConfig()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.time_limit_s, 60.0)
        self.assertIsInstance(config.cluster, ClusterConfig)
        self.assertIsInstance(config.disaggregated, DisaggregatedConfig)
        self.assertIsInstance(config.scheduling, SchedulingConfig)
        self.assertIsInstance(config.request, RequestGeneratorConfig)
        self.assertIsInstance(config.metrics, MetricsConfig)

    def test_nested_config_defaults(self):
        """Verify nested configs are properly initialized with defaults."""
        config = SimulationConfig()
        # Check cluster nested config
        self.assertEqual(config.cluster.num_nodes, 2)
        self.assertEqual(config.cluster.node_sku.num_gpus, 8)
        # Check network nested config
        self.assertEqual(config.cluster.network.nvlink.bandwidth_gbps, 600.0)
        self.assertEqual(config.cluster.network.rdma.bandwidth_gbps, 200.0)
        # Check model nested config
        self.assertEqual(config.cluster.replica.model.model_name, "Qwen3-30B-A3B")
        self.assertEqual(config.cluster.replica.model.num_layers, 48)

    @patch('sys.argv', ['main.py'])
    def test_from_cli_defaults(self):
        """Test from_cli with default arguments."""
        config = SimulationConfig.from_cli()
        self.assertEqual(config.seed, 42)
        self.assertEqual(config.log_level, "INFO")
        self.assertEqual(config.time_limit_s, 60.0)
        self.assertEqual(config.cluster.num_nodes, 2)
        self.assertEqual(config.cluster.node_sku.num_gpus, 8)
        self.assertEqual(config.cluster.num_replicas, 1)

    @patch('sys.argv', ['main.py', '--seed', '123', '--num_nodes', '4', '--qps', '20.0'])
    def test_from_cli_custom_args(self):
        """Test from_cli with custom arguments."""
        config = SimulationConfig.from_cli()
        self.assertEqual(config.seed, 123)
        self.assertEqual(config.cluster.num_nodes, 4)
        self.assertEqual(config.request.qps, 20.0)

    @patch('sys.argv', ['main.py', '--rdma_protocol', 'INFINIBAND', '--rdma_bandwidth_gbps', '400.0'])
    def test_from_cli_network_args(self):
        """Test from_cli with network arguments."""
        config = SimulationConfig.from_cli()
        self.assertEqual(config.cluster.network.rdma.protocol, RDMAProtocolType.INFINIBAND)
        self.assertEqual(config.cluster.network.rdma.bandwidth_gbps, 400.0)

    @patch('sys.argv', ['main.py', '--tensor_parallel_size', '4', '--num_pipeline_stages', '2'])
    def test_from_cli_parallelism_args(self):
        """Test from_cli with parallelism arguments."""
        config = SimulationConfig.from_cli()
        self.assertEqual(config.cluster.replica.tensor_parallel_size, 4)
        self.assertEqual(config.cluster.replica.num_pipeline_stages, 2)

    @patch('sys.argv', ['main.py', '--disaggregated', '--num_prefill_nodes', '2'])
    def test_from_cli_disaggregated_args(self):
        """Test from_cli with disaggregated arguments."""
        config = SimulationConfig.from_cli()
        self.assertTrue(config.disaggregated.enabled)
        self.assertEqual(config.disaggregated.num_prefill_nodes, 2)


if __name__ == '__main__':
    unittest.main()
