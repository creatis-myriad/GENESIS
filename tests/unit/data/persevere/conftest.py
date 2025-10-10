import json
from pathlib import Path

import pytest
from _pytest.fixtures import FixtureRequest

# Keys used in the JSON files
NODES_KEY = "nodes"
EDGES_KEY = "links"
GRAPH_KEY = "graph"
ID_KEY = "id"
SOURCE_KEY = "source"
TARGET_KEY = "target"


@pytest.fixture(
    params=["node_edge_feats_dataset", "node_feats_dataset", "edge_feats_dataset", "graph_node_feats_dataset"]
)
def dataset_root(request: FixtureRequest, shared_datadir: Path) -> Path:
    """Returns the root data resources directory for the specified dataset to test."""
    return shared_datadir / request.param


@pytest.fixture
def json_graph(dataset_root: Path) -> dict:
    """Load the first JSON graph file found in the specified directory."""
    raw_dir = dataset_root / "raw"
    any_json_path = next(raw_dir.glob("*.json"))
    with open(any_json_path) as file:
        return json.load(file)


@pytest.fixture
def nodes_count(json_graph: dict) -> int:
    """Count the number of nodes in the JSON graph."""
    return len(json_graph[NODES_KEY])


@pytest.fixture
def edges_count(json_graph: dict) -> int:
    """Count the number of edges in the JSON graph."""
    return len(json_graph[EDGES_KEY])


@pytest.fixture
def node_features_count(json_graph: dict) -> int:
    """Count the number of node features."""
    return sum(1 for key in json_graph[NODES_KEY][0] if key != ID_KEY)


@pytest.fixture
def edge_features_count(json_graph: dict) -> int:
    """Count the number of edge features."""
    return sum(1 for key in json_graph[EDGES_KEY][0] if key not in [SOURCE_KEY, TARGET_KEY])


@pytest.fixture
def graph_features_count(json_graph: dict) -> int:
    """Count the number of graph features."""
    return sum(1 for key in json_graph[GRAPH_KEY] if key != TARGET_KEY)
