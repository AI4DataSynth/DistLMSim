"""Tests for distlmsim.cluster module."""

import sys
import unittest

sys.path.insert(0, '.')

from distlmsim.config import (
    ClusterConfig,
    NodeSKUConfig,
    DeviceSKUConfig,
    NetworkTopologyConfig,
    ReplicaConfig,
    ModelConfig,
)
from distlmsim.cluster.node import PhysicalNode, GPUDevice
from distlmsim.cluster.cluster import Cluster
from distlmsim.cluster.resource_manager import ResourceManager
from distlmsim.topology.network_topology import NetworkTopology
from distlmsim.types import NodeRole


class TestPhysicalNode(unittest.TestCase):
    """Test PhysicalNode."""

    def setUp(self):
        self.sku = NodeSKUConfig()
        self.node = PhysicalNode(node_id=0, sku=self.sku, global_gpu_id_offset=0)

    def test_creation(self):
        """Verify basic PhysicalNode creation."""
        self.assertEqual(self.node.id, 0)
        self.assertEqual(self.node.role, NodeRole.MIXED)
        self.assertEqual(self.node.num_gpus, 8)

    def test_gpu_count(self):
        """Verify correct number of GPUs."""
        self.assertEqual(len(self.node.gpus), 8)
        self.assertEqual(len(self.node.global_gpu_ids), 8)

    def test_global_gpu_ids(self):
        """Verify global GPU IDs are correctly offset."""
        node2 = PhysicalNode(node_id=1, sku=self.sku, global_gpu_id_offset=8)
        self.assertEqual(node2.global_gpu_ids[0], 8)
        self.assertEqual(node2.global_gpu_ids[-1], 15)

    def test_memory_allocation_success(self):
        """Verify successful memory allocation."""
        success = self.node.allocate_gpu_memory(local_gpu_id=0, size_gb=10.0)
        self.assertTrue(success)
        self.assertAlmostEqual(self.node.gpus[0].memory_used_gb, 10.0)
        self.assertAlmostEqual(self.node.gpus[0].memory_free_gb, 70.0)

    def test_memory_allocation_failure(self):
        """Verify memory allocation fails when insufficient memory."""
        success = self.node.allocate_gpu_memory(local_gpu_id=0, size_gb=90.0)
        self.assertFalse(success)

    def test_memory_free(self):
        """Verify memory free works correctly."""
        self.node.allocate_gpu_memory(local_gpu_id=0, size_gb=30.0)
        self.assertAlmostEqual(self.node.gpus[0].memory_used_gb, 30.0)
        self.node.free_gpu_memory(local_gpu_id=0, size_gb=10.0)
        self.assertAlmostEqual(self.node.gpus[0].memory_used_gb, 20.0)

    def test_memory_free_floor_at_zero(self):
        """Verify memory free doesn't go below zero."""
        self.node.allocate_gpu_memory(local_gpu_id=0, size_gb=5.0)
        self.node.free_gpu_memory(local_gpu_id=0, size_gb=10.0)
        self.assertAlmostEqual(self.node.gpus[0].memory_used_gb, 0.0)

    def test_total_memory_free(self):
        """Verify total memory free calculation."""
        total_free = self.node.get_total_memory_free_gb()
        self.assertAlmostEqual(total_free, 8 * 80.0)  # 8 GPUs * 80 GB each

    def test_available_gpus(self):
        """Verify get_available_gpus returns all GPUs initially."""
        available = self.node.get_available_gpus()
        self.assertEqual(len(available), 8)

    def test_add_remove_replica(self):
        """Verify adding and removing replica IDs."""
        self.node.add_replica(0)
        self.assertIn(0, self.node.replica_ids)
        self.node.add_replica(1)
        self.assertIn(1, self.node.replica_ids)
        self.node.remove_replica(0)
        self.assertNotIn(0, self.node.replica_ids)
        self.assertIn(1, self.node.replica_ids)

    def test_role_setter(self):
        """Verify role can be set."""
        self.node.role = NodeRole.PREFILL
        self.assertEqual(self.node.role, NodeRole.PREFILL)


class TestGPUDevice(unittest.TestCase):
    """Test GPUDevice."""

    def test_creation(self):
        """Verify GPUDevice creation."""
        gpu = GPUDevice(global_id=0, local_id=0, node_id=0)
        self.assertEqual(gpu.global_id, 0)
        self.assertEqual(gpu.local_id, 0)
        self.assertEqual(gpu.node_id, 0)
        self.assertAlmostEqual(gpu.memory_used_gb, 0.0)
        self.assertAlmostEqual(gpu.memory_total_gb, 80.0)
        self.assertTrue(gpu.is_available)

    def test_memory_free_property(self):
        """Verify memory_free_gb property."""
        gpu = GPUDevice(global_id=0, local_id=0, node_id=0, memory_used_gb=30.0)
        self.assertAlmostEqual(gpu.memory_free_gb, 50.0)


class TestClusterFromConfig(unittest.TestCase):
    """Test Cluster.from_config."""

    def test_creates_correct_number_of_nodes(self):
        """Verify cluster creates the correct number of nodes."""
        config = ClusterConfig(num_nodes=4)
        cluster = Cluster.from_config(config)
        self.assertEqual(cluster.num_nodes, 4)

    def test_total_gpus(self):
        """Verify total GPU count."""
        config = ClusterConfig(num_nodes=2)
        cluster = Cluster.from_config(config)
        self.assertEqual(cluster.total_gpus, 16)  # 2 nodes * 8 GPUs

    def test_creates_correct_number_of_replicas(self):
        """Verify cluster creates the correct number of replicas."""
        config = ClusterConfig(num_nodes=2, num_replicas=2)
        cluster = Cluster.from_config(config)
        self.assertEqual(len(cluster.replicas), 2)

    def test_single_replica(self):
        """Verify single replica is properly created."""
        config = ClusterConfig(num_nodes=1, num_replicas=1)
        cluster = Cluster.from_config(config)
        self.assertEqual(len(cluster.replicas), 1)
        replica = cluster.get_replica(0)
        self.assertEqual(replica.model_name, config.replica.model.model_name)

    def test_replica_gpu_assignment(self):
        """Verify replica gets the correct number of GPUs."""
        config = ClusterConfig(
            num_nodes=1,
            num_replicas=1,
            replica=ReplicaConfig(
                tensor_parallel_size=4,
                num_pipeline_stages=1,
            ),
        )
        cluster = Cluster.from_config(config)
        replica = cluster.get_replica(0)
        self.assertEqual(len(replica.gpu_ids), 4)

    def test_replica_with_pp(self):
        """Verify replica with pipeline parallelism."""
        config = ClusterConfig(
            num_nodes=2,
            num_replicas=1,
            replica=ReplicaConfig(
                tensor_parallel_size=4,
                num_pipeline_stages=2,
            ),
        )
        cluster = Cluster.from_config(config)
        replica = cluster.get_replica(0)
        # TP=4, PP=2 -> 8 GPUs total
        self.assertEqual(len(replica.gpu_ids), 8)

    def test_get_node(self):
        """Verify get_node returns the correct node."""
        config = ClusterConfig(num_nodes=2)
        cluster = Cluster.from_config(config)
        node = cluster.get_node(0)
        self.assertIsInstance(node, PhysicalNode)
        self.assertEqual(node.id, 0)

    def test_are_gpus_on_same_node(self):
        """Verify GPU co-location check."""
        config = ClusterConfig(num_nodes=2)
        cluster = Cluster.from_config(config)
        self.assertTrue(cluster.are_gpus_on_same_node(0, 7))   # Both on node 0
        self.assertFalse(cluster.are_gpus_on_same_node(0, 8))  # Node 0 vs node 1


class TestResourceManager(unittest.TestCase):
    """Test ResourceManager."""

    def setUp(self):
        sku = NodeSKUConfig()
        self.nodes = [
            PhysicalNode(node_id=0, sku=sku, global_gpu_id_offset=0),
            PhysicalNode(node_id=1, sku=sku, global_gpu_id_offset=8),
        ]
        network_config = NetworkTopologyConfig()
        self.topology = NetworkTopology.from_config(network_config, num_nodes=2)
        self.rm = ResourceManager(self.nodes, self.topology)

    def test_allocate_replica_tp_affinity(self):
        """Verify TP group is allocated on same node."""
        gpu_ids = self.rm.allocate_replica(replica_id=0, num_gpus=4, tp_size=4)
        self.assertEqual(len(gpu_ids), 4)
        # All GPUs should be on the same node (first 4 available)
        node_ids = set(gid // 8 for gid in gpu_ids)
        self.assertEqual(len(node_ids), 1)

    def test_allocate_replica_cross_node(self):
        """Verify PP groups can span nodes."""
        gpu_ids = self.rm.allocate_replica(replica_id=0, num_gpus=8, tp_size=4)
        self.assertEqual(len(gpu_ids), 8)

    def test_release_replica(self):
        """Verify release_replica frees GPU resources."""
        self.rm.allocate_replica(replica_id=0, num_gpus=4, tp_size=4)
        self.assertEqual(self.rm.get_available_gpu_count(), 16 - 4)
        self.rm.release_replica(replica_id=0)
        self.assertEqual(self.rm.get_available_gpu_count(), 16)

    def test_get_gpu_allocation(self):
        """Verify get_gpu_allocation returns correct mapping."""
        self.rm.allocate_replica(replica_id=0, num_gpus=4, tp_size=4)
        allocation = self.rm.get_gpu_allocation()
        self.assertEqual(len(allocation), 4)
        for gpu_id, replica_id in allocation.items():
            self.assertEqual(replica_id, 0)

    def test_assign_node_roles(self):
        """Verify assign_node_roles sets node roles correctly."""
        self.rm.assign_node_roles(
            prefill_node_ids=[0],
            decode_node_ids=[1],
        )
        self.assertEqual(self.nodes[0].role, NodeRole.PREFILL)
        self.assertEqual(self.nodes[1].role, NodeRole.DECODE)

    def test_allocate_insufficient_resources(self):
        """Verify allocation fails gracefully when resources insufficient."""
        # Allocate 16 GPUs (all available)
        self.rm.allocate_replica(replica_id=0, num_gpus=8, tp_size=8)
        self.rm.allocate_replica(replica_id=1, num_gpus=8, tp_size=8)
        # Now no GPUs left
        with self.assertRaises(RuntimeError):
            self.rm.allocate_replica(replica_id=2, num_gpus=4, tp_size=4)


if __name__ == '__main__':
    unittest.main()
