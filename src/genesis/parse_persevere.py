import json
from pathlib import Path
from typing import Any

import hydra
import pandas as pd
from omegaconf import DictConfig

from genesis.utils import RankedLogger, pre_hydra_routine

log = RankedLogger(__name__, rank_zero_only=True)


def add_graph_attributes(json_graph: dict[str, Any], graph_attributes: dict[str, Any]) -> dict[str, Any]:
    """Add attributes to the graph dictionary.

    Args:
        json_graph: JSON NetworkX graph dictionary.
        graph_attributes: Dictionary of attributes to add to the graph.

    Returns:
        JSON NetworkX graph with added attributes.
    """
    for key, value in graph_attributes.items():
        json_graph["graph"][key] = value
    return json_graph


def remove_nodes_links_attributes(
    json_graph: dict[str, Any], node_attribute_keys: list[str], link_attribute_keys: list[str]
) -> dict[str, Any]:
    """Remove unwanted attributes from nodes and links dictionaries.

    Args:
        json_graph: JSON NetworkX graph dictionary.
        node_attribute_keys: List of node attribute keys to remove.
        link_attribute_keys: List of link attribute keys to remove.

    Returns:
        JSON NetworkX graph with unwanted attributes removed.
    """
    for node in json_graph["nodes"]:
        for key in node_attribute_keys:
            node.pop(key, None)

    for link in json_graph["links"]:
        for key in link_attribute_keys:
            link.pop(key, None)
    return json_graph


def extract_patient_global_attributes(
    csv_path: Path, id_col: str, attr_cols: dict[str, str]
) -> dict[str, dict[str, Any]]:
    """Extract patient global attributes from a CSV file.

    Args:
        csv_path: Path to the CSV file.
        id_col: Name of the column containing patient IDs.
        attr_cols: Dictionary mapping attribute names to their respective column names.

    Returns:
        Dictionary mapping patient IDs to dictionaries of global attributes.
    """
    log.info(f"Extracting global attributes from {csv_path}")
    df = pd.read_csv(csv_path, index_col=id_col, usecols=[id_col, *attr_cols.values()])

    df.index = df.index.str[:4]  # Truncate patient IDs to the first 4 characters
    # Rename index and columns to the configured attribute names
    df.index.rename("patient_id", inplace=True)
    df.rename(columns={col: attr for attr, col in attr_cols.items()}, inplace=True)

    log.info(f"Extracted attributes for {len(df)} patients")
    return df.to_dict(orient="index")


@hydra.main(config_path="configs", config_name="parse_persevere", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """Parse raw JSON graphs into PyG-ready raw JSON graphs."""
    source_dir = Path(cfg.source_dir)
    pyg_raw_dir = Path(cfg.pyg_raw_dir)
    pyg_raw_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Parsing JSON graphs from '{source_dir}' to '{pyg_raw_dir}'")
    log.info(f"Global attributes to add: {list(cfg.graph_attrs.attrs_to_extract.keys())}")
    log.info(f"Node attributes to remove: {cfg.attrs_to_remove.node}")
    log.info(f"Link attributes to remove: {cfg.attrs_to_remove.link}")

    patient_attrs = extract_patient_global_attributes(
        csv_path=source_dir / cfg.graph_attrs.csv_file,
        id_col=cfg.graph_attrs.patient_id_column,
        attr_cols=cfg.graph_attrs.attrs_to_extract,
    )

    json_files = list(source_dir.glob("*.json"))
    log.info(f"Found {len(json_files)} JSON files to parse.")

    processed_count = 0
    skipped_count = 0

    for json_path in json_files:
        patient_prefix = json_path.stem[:4]

        if patient_prefix in patient_attrs:
            log.debug(f"Parsing file '{json_path.name}' for patient ID '{patient_prefix}'")
            with open(json_path) as f:
                json_graph = json.load(f)

            json_graph = add_graph_attributes(json_graph, patient_attrs[patient_prefix])
            json_graph = remove_nodes_links_attributes(json_graph, cfg.attrs_to_remove.node, cfg.attrs_to_remove.link)

            output_filename = f"{json_path.stem}_parsed.json"
            output_path = pyg_raw_dir / output_filename
            with open(output_path, "w") as file:
                json.dump(json_graph, file, indent=2)

            processed_count += 1
        else:
            log.warning(f"No global attributes found for patient ID '{patient_prefix}', skipping '{json_path.name}'")
            skipped_count += 1

    log.info(f"Parsing completed: {processed_count} files processed, {skipped_count} files skipped.")


def main() -> None:
    """Main entry point for training, before Hydra is called.

    This is a workaround for issues with Python packaging tools requiring a function to target for script entrypoints.
    It provides a target for entrypoints that comes before Hydra is called, allowing for pre-Hydra routines to be run
    (e.g. setting up environment variables, registering custom OmegaConf resolvers etc.)
    """
    pre_hydra_routine()
    hydra_main()


if __name__ == "__main__":
    main()
