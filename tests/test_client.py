import unittest
from gdb_client.topology import ClusterTopology, NodeInfo

try:
    import polars as pl
    HAS_POLARS = True
except ImportError:
    HAS_POLARS = False


class TestTopologyAndClient(unittest.TestCase):
    def test_topology_routing(self):
        topology = ClusterTopology.__new__(ClusterTopology)
        topology.nodes = [
            NodeInfo(node_id=1, http_url="http://127.0.0.1:8847", flight_port=8848, client_flight_port=8860),
            NodeInfo(node_id=2, http_url="http://127.0.0.1:8846", flight_port=8849, client_flight_port=8861),
            NodeInfo(node_id=3, http_url="http://127.0.0.1:8845", flight_port=8850, client_flight_port=8862),
        ]
        topology.replication_factor = 3
        topology.partitions = 8

        self.assertEqual(topology.total_nodes, 3)

        # Key modulo routing tests
        target_0 = topology.target_node_for_key(0)
        self.assertEqual(target_0.node_id, 1)
        self.assertEqual(target_0.client_flight_port, 8860)

        target_1 = topology.target_node_for_key(1)
        self.assertEqual(target_1.node_id, 2)
        self.assertEqual(target_1.client_flight_port, 8861)

        target_2 = topology.target_node_for_key(2)
        self.assertEqual(target_2.node_id, 3)
        self.assertEqual(target_2.client_flight_port, 8862)

        target_3 = topology.target_node_for_key(3)
        self.assertEqual(target_3.node_id, 1)

    @unittest.skipUnless(HAS_POLARS, "polars is required for this test")
    def test_polars_partitioning_logic(self):
        df = pl.DataFrame({
            "id": [1, 2, 3, 4, 5, 6],
            "name": ["A", "B", "C", "D", "E", "F"],
        })
        total_nodes = 3
        df_part = df.with_columns((pl.col("id") % total_nodes).cast(pl.UInt32).alias("_partition"))
        partitions = df_part.partition_by("_partition", as_dict=True)

        self.assertEqual(len(partitions), 3)
        self.assertEqual(sum(len(sub) for sub in partitions.values()), 6)


if __name__ == "__main__":
    unittest.main()
