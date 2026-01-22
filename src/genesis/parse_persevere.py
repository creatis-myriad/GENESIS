import json
from pathlib import Path

import hydra
import networkx as nx
from omegaconf import DictConfig
from pandas.core.dtypes.common import is_integer_dtype

from genesis.data.utils import (
    NumpyEncoder,
    networkx_add_attrs,
    networkx_aggregate_attrs,
    networkx_remove_attrs,
    networkx_setdefault_attrs,
)
from genesis.utils import RankedLogger, pre_hydra_routine

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="configs", config_name="parse_persevere", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """Parse raw JSON graphs into PyG-ready raw JSON graphs."""
    source_dir = Path(cfg.source_dir)
    pyg_raw_dir = Path(cfg.pyg_raw_dir)
    pyg_raw_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Parsing JSON graphs from '{source_dir}' to '{pyg_raw_dir}'")
    json_files = list(source_dir.glob("*.json"))
    log.info(f"Found {len(json_files)} JSON files to parse")

    for key, attrs_to_remove in cfg.attrs_to_remove.items():
        log.info(f"{key.title()} attributes to remove: {attrs_to_remove}")
    log.info(f"Key to use for nodes data: '{cfg.node_link_data_nodes_key}'")
    log.info(f"Key to use for edges data: '{cfg.node_link_data_edges_key}'")

    agg_cfg = cfg.attrs_to_aggregate
    log.info(
        f"List-valued attributes to aggregate: \n"
        f"  nodes={agg_cfg.get('nodes', {})}, \n"
        f"  links={agg_cfg.get('links', {})}, "
    )

    tabular_dataset = hydra.utils.instantiate(cfg.global_features)
    tabular_data = tabular_dataset.data  # Extract the raw DataFrame from the dataset
    log.info(f"Extracted tabular features for {len(tabular_data)} patients")

    # If usecols is not specified, use all columns
    global_attrs = cfg.global_features.get("usecols", tabular_data.columns.tolist())
    # Do not include the index column in the global attributes if it is specified
    if (index_col := cfg.global_features.get("index_col")) and (index_col in global_attrs):
        global_attrs.remove(index_col)
    log.info(f"Tabular features to add: {global_attrs}")

    skipped_patient_ids = []

    for json_path in json_files:
        patient_id = json_path.stem[:4]
        if is_integer_dtype(tabular_data.index):
            patient_id = int(patient_id)

        if patient_id in tabular_data.index:
            log.debug(f"Parsing file '{json_path.name}' for patient ID '{patient_id}'")
            with open(json_path) as f:
                node_link_data = json.load(f)
                edges_key = "edges" if "edges" in node_link_data else "links"
                graph = nx.node_link_graph(node_link_data, edges=edges_key)

            # Add patient attributes as graph attributes
            patient_attrs = tabular_data.loc[patient_id, global_attrs].to_dict()
            graph = networkx_add_attrs(graph, "graph", patient_attrs)

            # Aggregate list attributes to scalar values
            graph = networkx_aggregate_attrs(graph, agg_cfg.get("nodes", {}), "nodes")
            graph = networkx_aggregate_attrs(graph, agg_cfg.get(edges_key, {}), edges_key)

            # Remove requested attributes
            for key, attrs_to_remove in cfg.attrs_to_remove.items():
                graph = networkx_remove_attrs(graph, key, attrs_to_remove)
                if key == "graph":
                    continue  # Skip setting default attributes for the graph
                graph = networkx_setdefault_attrs(graph, key, 0)

            # Rename 'nodes' and 'edges' keys in the node-link data, and save the modified graph
            node_link_data = nx.node_link_data(
                graph,
                nodes=cfg.node_link_data_nodes_key,
                edges=cfg.node_link_data_edges_key,
            )
            with open(pyg_raw_dir / json_path.name, "w") as file:
                json.dump(node_link_data, file, indent=2, cls=NumpyEncoder)

        else:
            skipped_patient_ids.append(patient_id)

    skipped_count = len(skipped_patient_ids)
    parsed_count = len(json_files) - skipped_count
    log.info(f"Parsed {parsed_count} JSON files")
    if skipped_patient_ids:
        log.warning(f"{skipped_count} patient(s) skipped because absent from tabular features: {skipped_patient_ids}")


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
