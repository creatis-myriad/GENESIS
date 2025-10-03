from pathlib import Path

import pytest

from genesis.data.persevere import PersevereDataset

from .conftest import EDGES_KEY, NODES_KEY


@pytest.mark.parametrize("node_attrs_filter", [None, ["node_feat1"]])
@pytest.mark.parametrize("edge_attrs_filter", [None, ["edge_feat1"]])
@pytest.mark.parametrize("line_graph", [False, True])
def test_persevere_dataset(
    node_attrs_filter: list[str] | None,
    edge_attrs_filter: list[str] | None,
    line_graph: bool,
    dataset_root: Path,
    nodes_count: int,
    edges_count: int,
    node_features_count: int,
    edge_features_count: int,
) -> None:
    """Test PersevereDataset with and without attribute filters and in line graph mode."""
    dataset = PersevereDataset(
        root=str(dataset_root),
        force_reload=True,
        line_graph=line_graph,
        target_attr="target",
        node_attrs_filter=node_attrs_filter,
        edge_attrs_filter=edge_attrs_filter,
        json_to_nx_kwargs={
            "nodes": NODES_KEY,
            "edges": EDGES_KEY,
        },
    )

    node_features_count = 1 if node_attrs_filter else node_features_count
    edge_features_count = 1 if edge_attrs_filter else edge_features_count
    if line_graph:
        # Swap nodes and edges + decrease edges count to account for lost border edges
        node_features_count, edge_features_count = edge_features_count, node_features_count
        nodes_count, edges_count = edges_count, nodes_count
        edges_count -= 2

    for data in dataset:
        # Nodes tests
        if data.x is not None:
            assert data.x.shape == (nodes_count, node_features_count)

        # Edges tests
        if data.edge_attr is not None:
            assert data.edge_attr.shape == (edges_count, edge_features_count)

        # PyG Data attributes tests
        for attr in data.keys():  # noqa: SIM118
            assert attr in ["x", "y", "edge_index", "edge_attr", "graph_attr", "pos", "time", "num_nodes"]
