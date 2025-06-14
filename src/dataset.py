from collections.abc import Iterator

import equinox as eqx
import jax.experimental.sparse as jsparse
import jax.numpy as jnp
import jax.random as jr
import networkx as nx
import numpy as np
from jaxtyping import Array, Bool, Float, Int, PRNGKeyArray
from tqdm import tqdm


class GraphData(eqx.Module):
    adjacency: Int[jsparse.BCOO, "n_nodes n_nodes"]
    edges: Int[Array, "n_edges 2"]
    scores: Float[Array, " n_nodes"]
    mask: Bool[Array, " n_nodes"]

    @property
    def n_nodes(self) -> int:
        return self.scores.shape[-1]

    @classmethod
    def from_networkx(cls, graph: nx.DiGraph, max_nodes: int, max_edges: int) -> "GraphData":
        n_nodes, n_edges = graph.number_of_nodes(), graph.number_of_edges()
        assert n_nodes < max_nodes, "A virtual node is necessary for virtual edges"
        assert n_edges <= max_edges

        edges = np.full((max_edges, 2), fill_value=max_nodes - 1, dtype=int)
        edges[:n_edges] = np.array(nx.edges(graph))
        edges = jnp.array(edges)

        adjacency = jsparse.BCOO(
            (jnp.ones(max_edges, dtype=int), jnp.flip(edges, axis=1)),
            shape=(max_nodes, max_nodes),
        )

        scores = np.zeros((max_nodes,), float)
        scores[:n_nodes] = np.array([graph.nodes[i]["scores"] for i in range(len(graph))])
        scores = jnp.array(scores)

        mask = np.zeros((max_nodes,), bool)
        mask[:n_nodes] = True
        mask = jnp.array(mask)
        return cls(adjacency, edges, scores, mask)

    @classmethod
    def stack(cls, graphs: list["GraphData"]) -> "GraphData":
        return cls(
            adjacency=jsparse.sparsify(jnp.stack)([g.adjacency for g in graphs]),
            edges=jnp.stack([g.edges for g in graphs]),
            scores=jnp.stack([g.scores for g in graphs]),
            mask=jnp.stack([g.mask for g in graphs]),
        )


class Dataset:
    graphs: list[GraphData]

    def __init__(self, graphs: list[GraphData]):
        self.graphs = graphs

    def iter(self, batch_size: int, total_iters: int, key: PRNGKeyArray) -> Iterator[GraphData]:
        for sk in jr.split(key, total_iters):
            batch_ids = jr.choice(sk, len(self), (batch_size,))
            samples = [self[i] for i in batch_ids]
            yield GraphData.stack(samples)

    def __len__(self) -> int:
        return len(self.graphs)

    def __getitem__(self, index: int) -> GraphData:
        return self.graphs[index]

    @classmethod
    def split(
        cls, dataset: "Dataset", split: float, *, key: PRNGKeyArray
    ) -> tuple["Dataset", "Dataset"]:
        """Randomly split the dataset into two non-overlapping subsets."""
        assert 0.0 <= split <= 1.0

        training_size = int(len(dataset) * split)
        perm = jr.permutation(key, len(dataset))
        training_graphs = [dataset.graphs[i] for i in perm[:training_size]]
        test_graphs = [dataset.graphs[i] for i in perm[training_size:]]
        return cls(training_graphs), cls(test_graphs)

    @classmethod
    def generate(cls, n_nodes: int, n_graphs: int, key: PRNGKeyArray) -> "Dataset":
        """Generate a random dataset to learn from.

        The graphs are generated following the Erdős-Rényi model. The score to predict is the nodes'
        betweenness score.
        """
        graphs = []
        seeds = [int(jr.key_data(sk)[1]) for sk in jr.split(key, n_graphs)]
        for seed in tqdm(seeds, desc="Generating graphs", leave=False):
            graph = nx.erdos_renyi_graph(n_nodes, seed=seed, p=0.05, directed=False)
            graph = nx.to_directed(graph)

            # Keep the largest connected component.
            nodes = max(nx.weakly_connected_components(graph), key=len)
            graph = graph.subgraph(nodes).copy()

            # Relabel nodes from 0 to N.
            graph = nx.relabel_nodes(
                graph, {old_id: new_id for new_id, old_id in enumerate(graph.nodes)}
            )

            scores = nx.betweenness_centrality(graph)
            nx.set_node_attributes(graph, scores, "scores")

            graphs.append(graph)

        max_nodes = max(g.number_of_nodes() for g in graphs) + 1
        max_edges = max(g.number_of_edges() for g in graphs)
        graphs = [
            GraphData.from_networkx(graph, max_nodes, max_edges)
            for graph in tqdm(graphs, desc="To jax array", total=len(graphs), leave=False)
        ]
        return cls(graphs)
