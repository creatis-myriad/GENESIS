import json
from pathlib import Path

import hydra
from omegaconf import DictConfig

from genesis.data.utils.networkx import node_link_data_add_attrs, node_link_data_remove_attrs
from genesis.utils import RankedLogger, pre_hydra_routine

log = RankedLogger(__name__, rank_zero_only=True)


@hydra.main(config_path="configs", config_name="parse_persevere", version_base=None)
def hydra_main(cfg: DictConfig) -> None:
    """Parse raw JSON graphs into PyG-ready raw JSON graphs."""
    source_dir = Path(cfg.source_dir)
    pyg_raw_dir = Path(cfg.pyg_raw_dir)
    pyg_raw_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Parsing JSON graphs from '{source_dir}' to '{pyg_raw_dir}'")
    global_attrs = list(cfg.clinical_data.usecols)
    # Do not include the index column in the global attributes if it is specified
    if index_col := cfg.clinical_data.get("index_col"):
        global_attrs.remove(index_col)
    log.info(f"Clinical attributes to add: {global_attrs}")
    for key, attrs_to_remove in cfg.attrs_to_remove.items():
        log.info(f"Remove {key} attributes: {attrs_to_remove}")

    clinical_data = hydra.utils.instantiate(cfg.clinical_data)
    log.info(f"Extracted attributes for {len(clinical_data)} patients")

    json_files = list(source_dir.glob("*.json"))
    log.info(f"Found {len(json_files)} JSON files to parse.")

    processed_count = 0
    skipped_count = 0

    for json_path in json_files:
        patient_id = json_path.stem[:4]

        if patient_id in clinical_data.indices():
            log.debug(f"Parsing file '{json_path.name}' for patient ID '{patient_id}'")
            with open(json_path) as f:
                node_link_data = json.load(f)

            patient_attrs = dict(zip(global_attrs, clinical_data.loc(patient_id), strict=False))
            node_link_data = node_link_data_add_attrs(node_link_data, "graph", patient_attrs)
            for key, attrs_to_remove in cfg.attrs_to_remove.items():
                node_link_data = node_link_data_remove_attrs(node_link_data, key, attrs_to_remove)

            output_filename = f"{json_path.stem}_parsed.json"
            output_path = pyg_raw_dir / output_filename
            with open(output_path, "w") as file:
                json.dump(node_link_data, file, indent=2)

            processed_count += 1
        else:
            log.warning(f"No global attributes found for patient ID '{patient_id}', skipping '{json_path.name}'")
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
