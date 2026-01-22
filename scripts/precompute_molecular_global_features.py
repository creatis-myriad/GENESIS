import shutil
from pathlib import Path

import pandas as pd
import rootutils
import torch
import yaml
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Dataset
from torch_geometric.datasets import ZINC

from genesis.data.data import GraphAttrData
from genesis.data.transforms.molecules import AddMolecularGlobalFeatures

DATASETS_ROOT = {
    "ZINC": "ZINC",
    "MolHIV": "ogbg_molhiv",
}
DATASETS_TARGETS = {
    "ZINC": "logP",
    "MolHIV": "inhibits_HIV",
}

data_dir = rootutils.find_root(indicator="pyproject.toml") / "data"
global_features = [
    "wiener",
    "circuit_rank",
    "independence",
    "hosoya",
    "max_matching",
    "second_eigenv",
    "spectral_radius",
    "zagreb_M1",
    "zagreb_M2",
]


def prepare_zinc(split: str = "train", reload: bool = False) -> ZINC:
    """Ensures fresh processing of the ZINC dataset, and adds molecular global features to the dataset."""
    # Add custom data class to safe globals to avoid user warning when loading the preprocessed dataset
    torch.serialization.add_safe_globals([GraphAttrData])

    return ZINC(
        root=str(data_dir / "ZINC"),
        subset=True,  # Use the subset proposed for benchmarking
        force_reload=reload,  # Ensure fresh processing of the dataset
        split=split,  # Only load the requested split
        pre_transform=AddMolecularGlobalFeatures(features=global_features),
    )


def prepare_molhiv(split: str = "train", reload: bool = False) -> PygGraphPropPredDataset:
    """Ensures fresh processing of the MolHIV dataset, and adds molecular global features to the dataset."""
    # The following imports and safe globals additions are a workaround to load OGB datasets in torch>=2.6
    # until this issue is resolved: https://github.com/snap-stanford/ogb/issues/497
    # TODO: Remove this workaround once the issue linked above is resolved and the fix is released
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr  # noqa: PLC0415
    from torch_geometric.data.storage import GlobalStorage  # noqa: PLC0415

    torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr])
    # Add custom data class to safe globals to avoid user warning when loading the preprocessed dataset
    torch.serialization.add_safe_globals([GraphAttrData])

    # Rename 'val' split to 'valid' to match OGB naming convention
    split = "valid" if split == "val" else split

    # Cleanup existing processed data to ensure fresh processing
    if reload:
        shutil.rmtree(data_dir / "ogbg_molhiv" / "processed", ignore_errors=True)

    molhiv = PygGraphPropPredDataset(
        root=str(data_dir),
        name="ogbg-molhiv",
        pre_transform=AddMolecularGlobalFeatures(features=global_features),
    )
    molhiv_split_idx = molhiv.get_idx_split()[split]  # Extract the requested split
    return molhiv[molhiv_split_idx]


def save_global_features_to_csv(dataset: Dataset, target_name: str, csv_file: Path) -> None:
    """Saves the global features of a dataset to a CSV file.

    Args:
        dataset: The dataset with global features to save.
        target_name: Name under which to identify the target (`y` attribute of the PyG dataset) in the CSV file.
        csv_file: Path to the CSV file to save the global features
    """
    features_df = pd.DataFrame(dataset.graph_attr.cpu().numpy(), columns=global_features)
    features_df[target_name] = dataset.y.cpu().numpy()
    csv_file.parent.mkdir(parents=True, exist_ok=True)
    features_df.to_csv(csv_file, index=False)


def compute_global_feature_stats(dataset: Dataset, feature: str) -> dict[str, float]:
    """Computes statistics for a given feature across a dataset.

    Args:
        dataset: The dataset with global features to compute statistics on.
        feature: The feature to compute statistics for.

    Returns:
        A dictionary of statistics and their values for the given feature.
    """
    feat_idx = global_features.index(feature)  # Determine order of feature in global features
    feature = dataset.graph_attr[:, feat_idx]  # Extract the feature across all graphs
    return {
        "mean": feature.mean().item(),
        "std": feature.std().item(),
        "min": feature.min().item(),
        "max": feature.max().item(),
    }


for dataset_name, init_func in [
    ("ZINC", prepare_zinc),
    ("MolHIV", prepare_molhiv),
]:
    # We force the dataset to be reloaded to make sure that the pre-transform computing global features is run.
    # However, the datasets' implementation process all splits at once (e.g. even if only one split is requested).
    # Use a flag to mark the first split request for the dataset, to disable force reloading if not and avoid reloading
    # the full dataset on each split.
    dataset = {}
    dataset_dir = data_dir / DATASETS_ROOT[dataset_name]
    is_first_split = True
    for split in ["train", "val", "test"]:
        # Initialize dataset split, pre-computing global features
        print(f"Preparing {split} split of {dataset_name} dataset...")
        dataset[split] = init_func(split=split, reload=is_first_split)
        is_first_split = False

        # Save the global features to CSV format
        global_features_csv = dataset_dir / "global_features" / f"{split}.csv"
        print(f"Saving global features for {split} split of {dataset_name} dataset to {global_features_csv}...")
        save_global_features_to_csv(dataset[split], DATASETS_TARGETS[dataset_name], global_features_csv)

    # Pre-compute statistics (e.g. to normalize data) on the training set only, to avoid val/test leakage
    global_features_records = {feat: compute_global_feature_stats(dataset["train"], feat) for feat in global_features}
    global_features_stats_yaml = dataset_dir / "global_features" / "train_stats.yaml"
    print(f"Saving train global features statistics of {dataset_name} dataset to {global_features_stats_yaml}")
    with open(global_features_stats_yaml, "w") as f:
        yaml.dump(global_features_records, f, sort_keys=False)

    print()  # Line break in prints between datasets
