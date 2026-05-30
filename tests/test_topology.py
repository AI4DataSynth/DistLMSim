"""Tests for distlmsim.topology module."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.config import NVLinkConfig, RDMAConfig, NetworkTopologyConfig
from distlmsim.types import RDMAProtocolType
from distlmsim.topology.nvlink_model import NVLinkModel
from distlmsim.topology.rdma_model import RDMAModel
from distlmsim.topology.communication_cost import CommunicationCostCalculator
from distlmsim.topology.network_topology import NetworkTopology


class TestNVLinkModelAllReduce(unittest.TestCase):
    """Test NVLinkModel AllReduce."""

    def setUp(self):
        self.config = NVLinkConfig()
        self.model = NVLinkModel(self.config, num_gpus_per_node=8)

    def test_single_gpu_returns_zero(self):
        """Verify allreduce time is 0 for single GPU."""
        time_ms = self.model.get_allreduce_time(num_gpus=1, data_size_bytes=1024)
        self.assertEqual(time_ms, 0.0)

    def test_multi_gpu_positive(self):
        """Verify allreduce time is positive for multi-GPU."""
        time_ms = self.model.get_allreduce_time(num_gpus=4, data_size_bytes=1024 * 1024)
        self.assertGreater(time_ms, 0.0)

    def test_increases_with_data_size(self):
        """Verify allreduce time increases with data size."""
        time_small = self.model.get_allreduce_time(num_gpus=4, data_size_bytes=1024)
        time_large = self.model.get_allreduce_time(num_gpus=4, data_size_bytes=1024 * 1024)
        self.assertGreater(time_large, time_small)

    def test_increases_with_gpu_count(self):
        """Verify allreduce time increases with GPU count (ring factor)."""
        time_2 = self.model.get_allreduce_time(num_gpus=2, data_size_bytes=1024 * 1024)
        time_8 = self.model.get_allreduce_time(num_gpus=8, data_size_bytes=1024 * 1024)
        self.assertGreater(time_8, time_2)


class TestNVLinkModelSendRecv(unittest.TestCase):
    """Test NVLinkModel Send/Recv."""

    def setUp(self):
        self.config = NVLinkConfig()
        self.model = NVLinkModel(self.config, num_gpus_per_node=8)

    def test_positive_time(self):
        """Verify send_recv time is positive."""
        time_ms = self.model.get_send_recv_time(data_size_bytes=1024 * 1024)
        self.assertGreater(time_ms, 0.0)

    def test_increases_with_data_size(self):
        """Verify send_recv time increases with data size."""
        time_small = self.model.get_send_recv_time(data_size_bytes=1024)
        time_large = self.model.get_send_recv_time(data_size_bytes=1024 * 1024)
        self.assertGreater(time_large, time_small)


class TestNVLinkModelAllToAll(unittest.TestCase):
    """Test NVLinkModel All-to-All."""

    def setUp(self):
        self.config = NVLinkConfig()
        self.model = NVLinkModel(self.config, num_gpus_per_node=8)

    def test_single_gpu_returns_zero(self):
        """Verify alltoall time is 0 for single GPU."""
        time_ms = self.model.get_alltoall_time(num_gpus=1, data_size_per_gpu_bytes=1024)
        self.assertEqual(time_ms, 0.0)

    def test_multi_gpu_positive(self):
        """Verify alltoall time is positive for multi-GPU."""
        time_ms = self.model.get_alltoall_time(num_gpus=4, data_size_per_gpu_bytes=1024 * 1024)
        self.assertGreater(time_ms, 0.0)


class TestRDMAModelTransfer(unittest.TestCase):
    """Test RDMAModel transfer time."""

    def test_roce_v2_transfer(self):
        """Verify RoCEv2 transfer time is positive."""
        config = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2)
        model = RDMAModel(config)
        time_ms = model.get_transfer_time(data_size_bytes=1024 * 1024)
        self.assertGreater(time_ms, 0.0)

    def test_protocol_overhead_ordering(self):
        """Verify protocol overhead: TCP/IP > RoCEv2 > InfiniBand > 0."""
        configs = {
            'roce': RDMAConfig(protocol=RDMAProtocolType.ROCE_V2),
            'ib': RDMAConfig(protocol=RDMAProtocolType.INFINIBAND),
            'tcp': RDMAConfig(protocol=RDMAProtocolType.TCP_IP),
        }
        models = {k: RDMAModel(c) for k, c in configs.items()}
        data_size = 10 * 1024 * 1024  # 10 MB

        time_roce = models['roce'].get_transfer_time(data_size)
        time_ib = models['ib'].get_transfer_time(data_size)
        time_tcp = models['tcp'].get_transfer_time(data_size)

        # RoCEv2 has more overhead than IB, TCP has most
        self.assertGreater(time_roce, time_ib)
        self.assertGreater(time_tcp, time_roce)
        self.assertGreater(time_ib, 0.0)


class TestRDMAModelAllReduce(unittest.TestCase):
    """Test RDMAModel AllReduce."""

    def setUp(self):
        self.config = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2)
        self.model = RDMAModel(self.config)

    def test_single_node_returns_zero(self):
        """Verify allreduce time is 0 for single node."""
        time_ms = self.model.get_allreduce_time(num_nodes=1, data_size_bytes=1024)
        self.assertEqual(time_ms, 0.0)

    def test_multi_node_positive(self):
        """Verify allreduce time is positive for multi-node."""
        time_ms = self.model.get_allreduce_time(num_nodes=2, data_size_bytes=1024 * 1024)
        self.assertGreater(time_ms, 0.0)


class TestRDMAModelAllToAll(unittest.TestCase):
    """Test RDMAModel All-to-All."""

    def setUp(self):
        self.config = RDMAConfig(protocol=RDMAProtocolType.ROCE_V2)
        self.model = RDMAModel(self.config)

    def test_single_node_returns_zero(self):
        """Verify alltoall time is 0 for single node."""
        time_ms = self.model.get_alltoall_time(num_nodes=1, data_size_per_node_bytes=1024)
        self.assertEqual(time_ms, 0.0)

    def test_multi_node_positive(self):
        """Verify alltoall time is positive for multi-node."""
        time_ms = self.model.get_alltoall_time(num_nodes=2, data_size_per_node_bytes=1024 * 1024)
        self.assertGreater(time_ms, 0.0)


class TestCommunicationCostCalculator(unittest.TestCase):
    """Test CommunicationCostCalculator."""

    def setUp(self):
        self.network_config = NetworkTopologyConfig()
        self.topology = NetworkTopology.from_config(self.network_config, num_nodes=2)
        self.calc = CommunicationCostCalculator(
            self.network_config, self.topology, num_gpus_per_node=8
        )

    def test_tp_allreduce_single_gpu(self):
        """Verify TP allreduce is 0 for tp_size=1."""
        time_ms = self.calc.tensor_parallel_allreduce(tp_size=1, data_size_bytes=1024)
        self.assertEqual(time_ms, 0.0)

    def test_tp_allreduce_multi_gpu(self):
        """Verify TP allreduce is positive for tp_size>1."""
        time_ms = self.calc.tensor_parallel_allreduce(tp_size=4, data_size_bytes=1024 * 1024)
        self.assertGreater(time_ms, 0.0)

    def test_pp_send_recv_same_node(self):
        """Verify PP send/recv on same node uses NVLink."""
        time_ms = self.calc.pipeline_parallel_send_recv(
            data_size_bytes=1024 * 1024,
            src_stage_node_id=0,
            dst_stage_node_id=0,
        )
        self.assertGreater(time_ms, 0.0)

    def test_pp_send_recv_cross_node(self):
        """Verify PP send/recv across nodes uses RDMA."""
        time_ms = self.calc.pipeline_parallel_send_recv(
            data_size_bytes=1024 * 1024,
            src_stage_node_id=0,
            dst_stage_node_id=1,
        )
        self.assertGreater(time_ms, 0.0)

    def test_ep_alltoall_single_gpu(self):
        """Verify EP alltoall is 0 for ep_size=1."""
        time_ms = self.calc.expert_parallel_alltoall(
            ep_size=1, data_size_per_gpu_bytes=1024, node_ids=[0]
        )
        self.assertEqual(time_ms, 0.0)

    def test_ep_alltoall_multi_gpu_same_node(self):
        """Verify EP alltoall on same node uses NVLink."""
        time_ms = self.calc.expert_parallel_alltoall(
            ep_size=4, data_size_per_gpu_bytes=1024 * 1024, node_ids=[0, 0, 0, 0]
        )
        self.assertGreater(time_ms, 0.0)

    def test_kv_cache_transfer(self):
        """Verify KV cache transfer time is positive."""
        time_ms = self.calc.kv_cache_transfer(
            kv_cache_size_bytes=10 * 1024 * 1024,
            src_node_id=0,
            dst_node_id=1,
        )
        self.assertGreater(time_ms, 0.0)

    def test_kv_cache_transfer_compression(self):
        """Verify compression reduces transfer time."""
        time_no_comp = self.calc.kv_cache_transfer(
            kv_cache_size_bytes=10 * 1024 * 1024,
            src_node_id=0,
            dst_node_id=1,
            compression_ratio=1.0,
        )
        time_comp = self.calc.kv_cache_transfer(
            kv_cache_size_bytes=10 * 1024 * 1024,
            src_node_id=0,
            dst_node_id=1,
            compression_ratio=2.0,
        )
        self.assertGreater(time_no_comp, time_comp)


class TestNetworkTopology(unittest.TestCase):
    """Test NetworkTopology."""

    def setUp(self):
        self.config = NetworkTopologyConfig()
        self.topo = NetworkTopology.from_config(self.config, num_nodes=4)

    def test_build_topology(self):
        """Verify topology is built with correct number of links."""
        links = self.topo.get_all_links()
        # 4 nodes -> 4*3/2 = 6 bidirectional pairs -> 12 directional links
        self.assertEqual(len(links), 12)

    def test_get_path_same_node(self):
        """Verify same-node path is empty."""
        path = self.topo.get_path(src_node=0, dst_node=0)
        self.assertEqual(len(path), 0)

    def test_get_path_different_nodes(self):
        """Verify cross-node path has at least one link."""
        path = self.topo.get_path(src_node=0, dst_node=1)
        self.assertGreater(len(path), 0)

    def test_get_effective_bandwidth_same_node(self):
        """Verify same-node bandwidth uses NVLink."""
        bw = self.topo.get_effective_bandwidth(src_node=0, dst_node=0)
        self.assertEqual(bw, self.config.nvlink.nvswitch_bandwidth_gbps)

    def test_get_effective_bandwidth_cross_node(self):
        """Verify cross-node bandwidth uses RDMA."""
        bw = self.topo.get_effective_bandwidth(src_node=0, dst_node=1)
        self.assertEqual(bw, self.config.rdma.bandwidth_gbps)

    def test_get_latency_same_node(self):
        """Verify same-node latency uses NVLink."""
        lat = self.topo.get_latency(src_node=0, dst_node=0)
        self.assertEqual(lat, self.config.nvlink.latency_us)

    def test_get_latency_cross_node(self):
        """Verify cross-node latency uses RDMA."""
        lat = self.topo.get_latency(src_node=0, dst_node=1)
        self.assertEqual(lat, self.config.rdma.latency_us)

    def test_is_same_node(self):
        """Verify is_same_node correctly identifies GPUs on same node."""
        # GPUs 0-7 on node 0, GPUs 8-15 on node 1
        self.assertTrue(self.topo.is_same_node(0, 7, gpus_per_node=8))
        self.assertFalse(self.topo.is_same_node(0, 8, gpus_per_node=8))


if __name__ == '__main__':
    unittest.main()
