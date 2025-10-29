import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt


class TensorNetwork:
    def __init__(
        self,
        tensors: dict[int | str, jnp.ndarray],
        edges: list[tuple[tuple[int, int]]],
        build_nx_graph: bool = False,
    ):
        self.tensors = tensors
        self.edges = edges

        self._graph = self._build_graph(tensors, edges) if build_nx_graph else None

    def _build_graph(
        self,
        tensors: dict[int | str, jnp.ndarray],
        edges: list[tuple[tuple[int | str, int], tuple[int | str, int]]],
    ):
        G = nx.Graph()
        for name, tensor in tensors.items():
            G.add_node(name, ndim=tensor.ndim)

        for (a, ai), (b, bi) in edges:
            G.add_edge(a, b, a_leg=ai, b_leg=bi)

        return G

    @property
    def graph(self):
        if self._graph is not None:
            return self._graph
        else:
            print("Graph was not initialized")
            return self._graph

    def plot(self, show_open_legs: bool = True, **kwargs):
        """
        Plot the tensor network.

        Args:
            show_open_legs: whether to display dangling legs
        """
        G = self._graph.copy()
        dummy_nodes = []
        open_count = 0

        # Add open-leg dummy nodes if requested
        if show_open_legs:
            for name, node in self._graph.nodes(data=True):
                ndim = node["ndim"]
                degree = G.degree(name)

                # Add dummy nodes for unconnected legs
                for i in range(ndim - degree):
                    dummy_name = rf"$i_{{{open_count}}}$"
                    dummy_nodes.append(dummy_name)
                    G.add_node(dummy_name, dummy=True)
                    G.add_edge(name, dummy_name, a_leg=i, b_leg=None)
                    open_count += 1

        pos = nx.spring_layout(G)

        # Separate main and dummy nodes
        main_nodes = [n for n in G.nodes if not G.nodes[n].get("dummy", False)]
        dummy_nodes = [n for n in G.nodes if G.nodes[n].get("dummy", False)]

        # Plot main tensors
        nx.draw_networkx_nodes(
            G, pos, nodelist=main_nodes, node_color="lightblue", node_size=500, **kwargs
        )
        nx.draw_networkx_labels(G, pos, labels={n: n for n in main_nodes})

        if show_open_legs:
            nx.draw_networkx_labels(G, pos, labels={n: n for n in dummy_nodes})

        # Plot edges
        nx.draw_networkx_edges(G, pos, alpha=0.5)

        plt.axis("off")
        plt.show()
