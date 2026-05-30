"""Tests for distlmsim.types module."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.types import (
    DeviceSKUType,
    NodeSKUType,
    InterconnectType,
    RDMAProtocolType,
    NetworkModelMode,
    ParallelismType,
    DeploymentMode,
    NodeRole,
    GlobalSchedulerType,
    ReplicaSchedulerType,
    KVCacheTransferStrategy,
    EventType,
)


class TestDeviceSKUType(unittest.TestCase):
    """Test DeviceSKUType enum."""

    def test_all_values_exist(self):
        """Verify all expected GPU device types exist."""
        expected = ['A800', 'A100', 'H100', 'H200']
        for name in expected:
            self.assertTrue(hasattr(DeviceSKUType, name))

    def test_enum_comparison(self):
        """Verify enum comparison works correctly."""
        self.assertEqual(DeviceSKUType.A800, DeviceSKUType.A800)
        self.assertNotEqual(DeviceSKUType.A800, DeviceSKUType.H100)

    def test_enum_is_instance(self):
        """Verify enum members are proper instances."""
        self.assertIsInstance(DeviceSKUType.A800, DeviceSKUType)


class TestNodeSKUType(unittest.TestCase):
    """Test NodeSKUType enum."""

    def test_all_values_exist(self):
        """Verify all expected node types exist."""
        expected = ['A800_DGX', 'H100_DGX']
        for name in expected:
            self.assertTrue(hasattr(NodeSKUType, name))

    def test_enum_comparison(self):
        """Verify enum comparison works correctly."""
        self.assertEqual(NodeSKUType.A800_DGX, NodeSKUType.A800_DGX)
        self.assertNotEqual(NodeSKUType.A800_DGX, NodeSKUType.H100_DGX)


class TestInterconnectType(unittest.TestCase):
    """Test InterconnectType enum."""

    def test_all_values_exist(self):
        """Verify all expected interconnect types exist."""
        expected = ['NVLINK_SWITCH', 'NVLINK_MESH']
        for name in expected:
            self.assertTrue(hasattr(InterconnectType, name))


class TestRDMAProtocolType(unittest.TestCase):
    """Test RDMAProtocolType enum."""

    def test_all_values_exist(self):
        """Verify all expected RDMA protocols exist."""
        expected = ['ROCE_V2', 'INFINIBAND', 'TCP_IP']
        for name in expected:
            self.assertTrue(hasattr(RDMAProtocolType, name))

    def test_enum_comparison(self):
        """Verify enum comparison works correctly."""
        self.assertEqual(RDMAProtocolType.ROCE_V2, RDMAProtocolType.ROCE_V2)
        self.assertNotEqual(RDMAProtocolType.ROCE_V2, RDMAProtocolType.INFINIBAND)


class TestNetworkModelMode(unittest.TestCase):
    """Test NetworkModelMode enum."""

    def test_all_values_exist(self):
        """Verify all expected network model modes exist."""
        expected = ['ANALYTICAL', 'PROFILING', 'HYBRID']
        for name in expected:
            self.assertTrue(hasattr(NetworkModelMode, name))


class TestParallelismType(unittest.TestCase):
    """Test ParallelismType enum."""

    def test_all_values_exist(self):
        """Verify all expected parallelism types exist."""
        expected = ['TENSOR', 'PIPELINE', 'DATA', 'EXPERT']
        for name in expected:
            self.assertTrue(hasattr(ParallelismType, name))


class TestDeploymentMode(unittest.TestCase):
    """Test DeploymentMode enum."""

    def test_all_values_exist(self):
        """Verify all expected deployment modes exist."""
        expected = ['COLOCATED', 'DISAGGREGATED']
        for name in expected:
            self.assertTrue(hasattr(DeploymentMode, name))


class TestNodeRole(unittest.TestCase):
    """Test NodeRole enum."""

    def test_all_values_exist(self):
        """Verify all expected node roles exist."""
        expected = ['MIXED', 'PREFILL', 'DECODE']
        for name in expected:
            self.assertTrue(hasattr(NodeRole, name))


class TestGlobalSchedulerType(unittest.TestCase):
    """Test GlobalSchedulerType enum."""

    def test_all_values_exist(self):
        """Verify all expected global scheduler types exist."""
        expected = ['ROUND_ROBIN', 'RANDOM', 'LEAST_OUTSTANDING', 'TOPOLOGY_AWARE']
        for name in expected:
            self.assertTrue(hasattr(GlobalSchedulerType, name))


class TestReplicaSchedulerType(unittest.TestCase):
    """Test ReplicaSchedulerType enum."""

    def test_all_values_exist(self):
        """Verify all expected replica scheduler types exist."""
        expected = ['SARATHI', 'VLLM', 'ORCA', 'FCFS']
        for name in expected:
            self.assertTrue(hasattr(ReplicaSchedulerType, name))


class TestKVCacheTransferStrategy(unittest.TestCase):
    """Test KVCacheTransferStrategy enum."""

    def test_all_values_exist(self):
        """Verify all expected KV cache transfer strategies exist."""
        expected = ['DIRECT', 'PIPELINED', 'STORE_FORWARD']
        for name in expected:
            self.assertTrue(hasattr(KVCacheTransferStrategy, name))


class TestEventType(unittest.TestCase):
    """Test EventType enum."""

    def test_all_values_exist(self):
        """Verify all expected event types exist."""
        expected = [
            'REQUEST_ARRIVAL', 'GLOBAL_SCHEDULE', 'REPLICA_SCHEDULE',
            'BATCH_STAGE_ARRIVAL', 'BATCH_STAGE_END', 'BATCH_END',
            'PREFILL_COMPLETE', 'KV_CACHE_TRANSFER_START',
            'KV_CACHE_TRANSFER_END', 'DECODE_START',
            'EXPERT_ASSIGNMENT', 'EXPERT_COMM_START', 'EXPERT_COMM_END',
        ]
        for name in expected:
            self.assertTrue(hasattr(EventType, name))


if __name__ == '__main__':
    unittest.main()
