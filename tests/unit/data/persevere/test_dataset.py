from contextlib import nullcontext
from pathlib import Path

import pytest

from genesis.data.persevere import PersevereDataset

from .conftest import EDGES_KEY, NODES_KEY


@pytest.mark.parametrize("node_attrs_filter", [None, ["node_feat1"]])
@pytest.mark.parametrize("edge_attrs_filter", [None, ["edge_feat1"]])
@pytest.mark.parametrize("graph_attrs_filter", [None, ["graph_feat1"]])
@pytest.mark.parametrize("line_graph", [False, True])
def test_persevere_dataset(
    node_attrs_filter: list[str] | None,
    edge_attrs_filter: list[str] | None,
    graph_attrs_filter: list[str] | None,
    line_graph: bool,
    dataset_root: Path,
    nodes_count: int,
    edges_count: int,
    node_features_count: int,
    edge_features_count: int,
    graph_features_count: int,
) -> None:
    """Test PersevereDataset with and without attribute filters and in line graph mode."""
    exc_msg = None

    if (node_attrs_filter and "node" not in dataset_root.name) or (
        edge_attrs_filter and "edge" not in dataset_root.name
    ):
        # If filtering on node or edge attributes but the dataset does not contain any, expect KeyError
        dataset_init_expectation = pytest.raises(KeyError)
    elif line_graph and node_attrs_filter:
        # If filtering on node attributes in line graph mode, expect KeyError because node features are not converted
        # to edge features
        dataset_init_expectation = pytest.raises(KeyError)
    elif graph_attrs_filter and "graph" not in dataset_root.name:
        # If filtering on graph attributes but the dataset does not contain any, expect AttributeError
        # The error is different from node/edge attributes because graph attributes are customs and not built-in to PyG
        dataset_init_expectation = pytest.raises(AttributeError)
        exc_msg = "'GlobalStorage' object has no attribute"
    else:
        # Otherwise, no exceptions are expected to be raised, which can be checked using nullcontext. See pytest doc:
        # https://docs.pytest.org/en/stable/example/parametrize.html#parametrizing-conditional-raising
        dataset_init_expectation = nullcontext()

    with dataset_init_expectation as e:
        dataset = PersevereDataset(
            root=str(dataset_root),
            line_graph=line_graph,
            target_attr="target",
            node_attrs_filter=node_attrs_filter,
            edge_attrs_filter=edge_attrs_filter,
            graph_attrs_filter=graph_attrs_filter,
            json_to_nx_kwargs={
                "nodes": NODES_KEY,
                "edges": EDGES_KEY,
            },
        )

    # If dataset initialization should fail, check the expected exception was raised
    if not isinstance(dataset_init_expectation, nullcontext):
        assert e.type in dataset_init_expectation.expected_exceptions
        if exc_msg:
            assert exc_msg in str(e.value)
        return  # Stop test here if dataset initialization failed as expected

    if node_features_count and node_attrs_filter is not None:
        node_features_count = len(node_attrs_filter)
    if edge_features_count and edge_attrs_filter is not None:
        edge_features_count = len(edge_attrs_filter)
    if graph_features_count and graph_attrs_filter is not None:
        graph_features_count = len(graph_attrs_filter)

    if line_graph:
        # Swap nodes and edges + decrease edges count to account for lost border edges
        node_features_count, edge_features_count = edge_features_count, node_features_count
        nodes_count, edges_count = edges_count, nodes_count
        edges_count -= 2

    for data in dataset:
        # Nodes tests
        if nodes_count and node_features_count:
            assert data.x.shape == (nodes_count, node_features_count)

        # Edges tests
        if edges_count and edge_features_count and not line_graph:  # Line graph does not support edge features
            assert data.edge_attr.shape == (edges_count, edge_features_count)

        # Graph attributes tests
        if graph_features_count:
            assert data.graph_attr.shape == (graph_features_count,)

        # PyG Data attributes tests
        for attr in data.keys():  # noqa: SIM118
            assert attr in ["x", "y", "edge_index", "edge_attr", "graph_attr", "num_nodes"]
