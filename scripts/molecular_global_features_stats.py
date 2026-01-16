import shutil

import rootutils
from ogb.graphproppred import PygGraphPropPredDataset
from torch_geometric.data import Dataset
from torch_geometric.datasets import ZINC

from genesis.data.data import GraphAttrData
from genesis.data.transforms.molecules import AddMolecularGlobalFeatures

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


def prepare_zinc() -> ZINC:
    """Ensures fresh processing of the ZINC dataset, and adds molecular global features to the dataset."""
    return ZINC(
        root=str(data_dir / "ZINC"),
        subset=True,  # Use the subset proposed for benchmarking
        force_reload=True,  # Ensure fresh processing of the dataset
        split="train",  # Pre-compute statistics on the training set only, to avoid val/test leakage
        pre_transform=AddMolecularGlobalFeatures(features=global_features),
    )


def prepare_molhiv() -> PygGraphPropPredDataset:
    """Ensures fresh processing of the MolHIV dataset, and adds molecular global features to the dataset."""
    # The following imports and safe globals additions are a workaround to load OGB datasets in torch>=2.6
    # until this issue is resolved: https://github.com/snap-stanford/ogb/issues/497
    # TODO: Remove this workaround once the issue linked above is resolved and the fix is released
    import torch  # noqa: PLC0415
    from torch_geometric.data.data import DataEdgeAttr, DataTensorAttr  # noqa: PLC0415
    from torch_geometric.data.storage import GlobalStorage  # noqa: PLC0415

    torch.serialization.add_safe_globals([GlobalStorage, DataEdgeAttr, DataTensorAttr, GraphAttrData])

    # Cleanup existing processed data to ensure fresh processing
    shutil.rmtree(data_dir / "ogbg_molhiv" / "processed", ignore_errors=True)

    molhiv = PygGraphPropPredDataset(
        root=str(data_dir),
        name="ogbg-molhiv",
        pre_transform=AddMolecularGlobalFeatures(features=global_features),
    )
    # Pre-compute statistics on the training set only, to avoid val/test leakage
    molhiv_train_idx = molhiv.get_idx_split()["train"]
    return molhiv[molhiv_train_idx]


def compute_feature_stats(dataset: Dataset, feature: str) -> dict[str, float]:
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
    print(f"Preparing dataset {dataset_name}...")
    dataset = init_func()
    global_features_records = {feat: compute_feature_stats(dataset, feat) for feat in global_features}
    print(f"Global features statistics for dataset {dataset_name}:")
    print(global_features_records)
    print()
