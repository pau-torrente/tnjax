import unittest
import jax
import jax.numpy as jnp
import networkx as nx
import matplotlib
from tnjax.tensor_networks.tensor_network import TensorNetwork

matplotlib.use("Agg")  # use non-GUI backend for test environments

TENSORS = {
    "A1": jnp.array(jax.random.uniform(jax.random.PRNGKey(0), (1, 2, 3))),
    "A2": jnp.array(jax.random.uniform(jax.random.PRNGKey(0), (3, 2, 3))),
    "A3": jnp.array(jax.random.uniform(jax.random.PRNGKey(0), (3, 2, 1))),
}

EDGES = [(("A1", 2), ("A2", 0)), (("A2", 2), ("A3", 0))]


class TestTensorNetwork(unittest.TestCase):

    def setUp(self):
        # Example tensors: simple 3-site MPS chain
        self.tensors = TENSORS
        self.edges = EDGES

    def test_build_graph_true(self):
        """Check that a graph is built when build_nx_graph=True."""
        tn = TensorNetwork(self.tensors, self.edges, build_nx_graph=True)
        self.assertIsInstance(tn.graph, nx.Graph)
        self.assertEqual(set(tn.graph.nodes), set(self.tensors.keys()))

    def test_build_graph_false(self):
        """Graph should be None when build_nx_graph=False."""
        tn = TensorNetwork(self.tensors, self.edges, build_nx_graph=False)
        self.assertIsNone(tn.graph)

    def test_graph_edges_metadata(self):
        """Edges should have correct metadata."""
        tn = TensorNetwork(self.tensors, self.edges, build_nx_graph=True)
        for a, b, data in tn.graph.edges(data=True):
            if {a, b} == {"A1", "A2"}:
                self.assertEqual(data["a_leg"], 2)
                self.assertEqual(data["b_leg"], 0)
            elif {a, b} == {"A2", "A3"}:
                self.assertEqual(data["a_leg"], 2)
                self.assertEqual(data["b_leg"], 0)

    def test_plot_executes(self):
        """Ensure the plot method runs without exceptions."""
        tn = TensorNetwork(self.tensors, self.edges, build_nx_graph=True)
        try:
            tn.plot(show_open_legs=True)
        except Exception as e:
            self.fail(f"plot() raised an exception: {e}")

    def test_plot_open_leg_count(self):
        """Check that open-leg dummy nodes correspond to unconnected legs."""
        tn = TensorNetwork(self.tensors, self.edges, build_nx_graph=True)
        G = tn.graph.copy()

        # count open legs before plotting
        expected_open_legs = sum(
            tensor.ndim - G.degree(name) for name, tensor in self.tensors.items()
        )

        tn.plot(show_open_legs=True)

        # Rebuild dummy count directly from plot-generated graph logic
        dummy_count = sum(
            1 for name, data in tn.graph.nodes(data=True) if data.get("dummy", False)
        )

        self.assertEqual(expected_open_legs, dummy_count)

    def test_invalid_edge_format(self):
        """Ensure an edge with a wrong tuple structure raises TypeError."""
        invalid_edges = [(("A1", 0),)]  # incomplete edge
        with self.assertRaises(TypeError):
            TensorNetwork(self.tensors, invalid_edges, build_nx_graph=True)
