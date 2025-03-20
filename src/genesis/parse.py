import json
import warnings
from pathlib import Path

import hydra
import pandas as pd
from omegaconf import DictConfig

from genesis.utils import RankedLogger

warnings.filterwarnings("ignore", category=UserWarning, module="openpyxl")

log = RankedLogger(__name__, rank_zero_only=True)


def add_graph_attributes(json_graph: dict, graph_attributes: dict) -> dict:
    """Add attributes to the graph dictionary."""
    for key, value in graph_attributes.items():
        json_graph["graph"][key] = value
    return json_graph


def remove_nodes_links_attributes(json_graph: dict, node_attribute_keys: list, link_attribute_keys: list) -> dict:
    """Remove unwanted attributes from nodes and links dictionaries."""
    for node in json_graph["nodes"]:
        for key in node_attribute_keys:
            node.pop(key, None)

    for link in json_graph["links"]:
        for key in link_attribute_keys:
            link.pop(key, None)
    return json_graph


def extract_patient_global_attributes(xlsx_path: Path, sheet_name: str, id_col: int, attr_cols: dict) -> dict:
    """Extract patient global attributes from an XLSX database."""
    log.info(f"Extracting global attributes from {xlsx_path} (sheet: {sheet_name})")
    df = pd.read_excel(xlsx_path, sheet_name=sheet_name, skiprows=1, header=None, dtype=str)
    df = df.dropna(subset=[id_col])
    df["patient_id"] = df.iloc[:, id_col].astype(str).str[:4]

    attrs_df = pd.DataFrame()
    for attr_name, col_idx in attr_cols.items():
        attrs_df[attr_name] = pd.to_numeric(df.iloc[:, col_idx], errors="coerce")

    attrs_df.index = df["patient_id"]
    attrs_df = attrs_df[~attrs_df.index.duplicated(keep="first")]
    log.info(f"Extracted attributes for {len(attrs_df)} patients")
    return attrs_df.to_dict(orient="index")


@hydra.main(config_path="configs", config_name="parse", version_base=None)
def main(cfg: DictConfig) -> None:
    """Parse raw JSON graphs into PyG-ready raw JSON graphs."""
    source_dir = Path(cfg.source_dir)
    pyg_raw_dir = Path(cfg.pyg_raw_dir)
    pyg_raw_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Parsing JSON graphs from '{source_dir}' to '{pyg_raw_dir}'")
    log.info(f"Global attributes to add: {list(cfg.global_attr.global_attr_columns.keys())}")
    log.info(f"Node attributes to remove: {cfg.attr_to_remove.node}")
    log.info(f"Link attributes to remove: {cfg.attr_to_remove.link}")

    patient_attrs = extract_patient_global_attributes(
        xlsx_path=source_dir / cfg.global_attr.db_filename,
        sheet_name=cfg.global_attr.patient_sheet,
        id_col=cfg.global_attr.patient_id_column,
        attr_cols=cfg.global_attr.global_attr_columns,
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
            json_graph = remove_nodes_links_attributes(json_graph, cfg.attr_to_remove.node, cfg.attr_to_remove.link)

            output_filename = f"{json_path.stem}_parsed.json"
            output_path = pyg_raw_dir / output_filename
            with open(output_path, "w") as file:
                json.dump(json_graph, file, indent=2)

            processed_count += 1
        else:
            log.warning(f"No global attributes found for patient ID '{patient_prefix}', skipping '{json_path.name}'")
            skipped_count += 1

    log.info(f"Parsing completed: {processed_count} files processed, {skipped_count} files skipped.")


if __name__ == "__main__":
    main()
