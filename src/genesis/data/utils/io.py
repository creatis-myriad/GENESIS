import json
from pathlib import Path
from typing import Any

import networkx as nx
import numpy as np
import pandas as pd
import torch
from monai.data import NibabelReader
from torch_geometric.data import Data

from genesis.data.utils import networkx_line_graph, networkx_to_pyg
from genesis.utils import RankedLogger

log = RankedLogger(__name__, rank_zero_only=True)


def load_nifti(filepath: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
    """Loads a NIFTI image and returns the image and its metadata.

    Args:
        filepath: Path to the NIFTI image.

    Returns:
        - Image array.
        - Image metadata.
    """
    nib_reader = NibabelReader()
    img_obj = nib_reader.read(filepath)
    return nib_reader.get_data(img_obj)


def json_to_pyg(
    json_path: Path,
    target_attr: str,
    target_dtype: torch.dtype,
    line_graph: bool = False,
    json_to_nx_kwargs: dict[str, Any] | None = None,
    nx_to_pyg_kwargs: dict[str, Any] | None = None,
) -> Data:
    """Parse a JSON file into a PyG `Data`, representing a graph.

    Args:
        json_path: File path to read as PyG graph.
        target_attr: Key of the graph attribute to use as target.
        target_dtype: Data type of the target attribute.
        line_graph: Whether to convert the parsed graph to its line graph.
        json_to_nx_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`.
        nx_to_pyg_kwargs: Node and edge features filters to pass to `pyg.utils.from_networkx`.

    Returns:
        PyG `Data` object loaded from the JSON file.
    """
    if json_to_nx_kwargs is None:
        json_to_nx_kwargs = {}
    if nx_to_pyg_kwargs is None:
        nx_to_pyg_kwargs = {}
    nx_graph = json_to_networkx(json_path, line_graph, **json_to_nx_kwargs)
    return networkx_to_pyg(nx_graph, target_attr, target_dtype, **nx_to_pyg_kwargs)


def json_to_networkx(
    json_path: Path, line_graph: bool = False, directed: bool = False, **node_link_graph_kwargs
) -> nx.Graph:
    """Parses a JSON file as the node-link data describing a NetworkX `Graph`.

    Args:
        json_path: File path to read as NetworkX graph.
        line_graph: Whether to convert the parsed graph to its line graph.
        directed: Whether the graph is directed.
        **node_link_graph_kwargs: Keys for serialized attribute names to pass to `nx.node_link_graph`.

    Returns:
        NetworkX `Graph` loaded from the JSON file.
    """
    with open(json_path) as file:
        json_graph = json.load(file)

    # If no keys for serialized attribute names are provided,
    # use default keys + set edges key to avoid warning
    graph = nx.node_link_graph(json_graph, directed=directed, **node_link_graph_kwargs or {"edges": "edges"})
    if line_graph:
        graph = networkx_line_graph(graph)
    return graph


def networkx_to_json(graph: nx.Graph, json_path: Path, **node_link_data_kwargs) -> None:
    """Saves a NetworkX `Graph` to a JSON file in node-link format.

    Args:
        graph: NetworkX graph to save.
        json_path: File path where to save the graph in JSON format.
        **node_link_data_kwargs: Keys for serialized attribute names to pass to `nx.node_link_data`.
    """
    node_link_data = nx.node_link_data(graph, **node_link_data_kwargs)
    with open(json_path, "w") as file:
        json.dump(node_link_data, file, indent=2, cls=NumpyEncoder)


class NumpyEncoder(json.JSONEncoder):
    """Custom JSON encoder for handling NumPy data types."""

    def default(self, obj: object) -> object:
        """Convert NumPy data types to native Python types for JSON serialization."""
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(self).default(obj)


def find_file(
    hint: Path | str,
    *,
    search_dirs: list[Path] | None = None,
    pattern: str = "*{id}*",
) -> Path:
    """Find a unique file based on the path or pattern.

    Resolve either:
        - a direct file path (if input_file.exists()), or
        - a patient ID (zero-padded to 4 digits) to any file matching `pattern`.

    Args:
        hint: either a Path to an existing file or a patient ID.
        search_dirs: list of directories to search under.
        pattern: a glob pattern containing '{id}' which will be replaced by the zero-padded ID.
            e.g. "*{id}*.json"

    Returns:
        The unique matching file path.

    Raises:
        FileNotFoundError: if no match, or if more than one unique match is found.
    """
    # 1) If they've passed a real file, just use it
    if Path(hint).is_file():
        return Path(hint).resolve()
    if search_dirs is None:
        raise ValueError("If `hint` is not a path to an existing file, `search_dirs` must be provided.")

    # 2) Otherwise interpret the stem as an ID
    patient_id = hint.zfill(4)
    pattern = pattern.format(id=patient_id)

    found = []
    for d in search_dirs:
        if d.is_dir():
            found.extend(d.rglob(pattern))

    unique = {p.resolve() for p in found}
    if not unique:
        raise FileNotFoundError(f"No file found for ID='{patient_id}' (pattern='{pattern}').")
    if len(unique) > 1:
        raise FileNotFoundError(f"Multiple matches for ID='{patient_id}' (pattern='{pattern}'): {list(unique)}")
    return unique.pop()


def load_and_clean_tabular_features(csv_filepath: Path, drop_na_subset: list[str] | None = None) -> pd.DataFrame:
    """Load and clean tabular patient features from a CSV file.

    Args:
        csv_filepath: Path to the tabular patient features CSV file.
        drop_na_subset: Critical columns where rows with NA values should be dropped.

    Returns:
        Cleaned tabular features as a pandas DataFrame.
    """
    df = pd.read_csv(csv_filepath, na_values=["nr", "NR"])
    # Standardize patient IDs to be zero-padded strings of length 4, and set as index
    df["patient_id"] = df["patient_id"].astype(str).str.zfill(4)
    df = df.set_index("patient_id")
    # Drop rows with missing values in critical columns
    if drop_na_subset:
        if isinstance(drop_na_subset, tuple):
            drop_na_subset = list(drop_na_subset)  # Convert tuple to list to correctly slice columns
        na_ids = df.index[df[drop_na_subset].isna().any(axis=1)].tolist()
        log.warning(
            f"Dropping clinical data from patients with at least one missing value in {drop_na_subset}: {na_ids}"
        )
        df.dropna(subset=drop_na_subset, inplace=True)
    # Clean troponin values: convert trace amount ("< 3") to 3 to allow casting to int
    df["troponin"] = df["troponin"].astype(str).str.replace("< 3", "3").astype(int)
    return df
