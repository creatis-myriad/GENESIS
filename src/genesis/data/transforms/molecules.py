from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx

from genesis.data.data import GraphAttrData
from genesis.data.utils import features as graph_properties

GlobalFeature = Literal[
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


class AddMolecularGlobalFeatures(BaseTransform):
    """Add global features useful for characterizing molecules to the graph data object."""

    def __init__(
        self,
        features: list[GlobalFeature],
        attr_name: str = "graph_attr",
        normalize: Literal["min-max", "z-score"] | None = None,
        features_stats: dict[GlobalFeature, dict[Literal["mean", "std", "min", "max"], float]] | None = None,
    ) -> None:
        """Initializes an `AddMolecularGlobalFeatures` instance.

        Args:
            features: The list of molecular global features to compute and add to the graph data object.
            attr_name: The name of the attribute where to store the features in the data object.
            normalize: The normalization method to apply to the features.
            features_stats: Pre-computed statistics of the features for normalization. Required if `normalize` is not,
                ignored otherwise.
        """
        if normalize is not None and features_stats is None:
            raise ValueError("`features_stats` must be provided when `normalize` is not None.")

        self.attr_name = attr_name
        self._feat_funcs = {}
        self._norm_funcs = {}
        for feature in features:
            match feature:
                case "wiener":
                    self._feat_funcs[feature] = graph_properties.wiener_index
                case "circuit_rank":
                    self._feat_funcs[feature] = graph_properties.circuit_rank
                case "independence":
                    self._feat_funcs[feature] = graph_properties.independence_number
                case "hosoya":
                    self._feat_funcs[feature] = graph_properties.total_matchings
                case "max_matching":
                    self._feat_funcs[feature] = graph_properties.maximum_matching
                case "second_eigenv":
                    # Since the function returns a float64, i.e. double, cast to float so tensors created with this data
                    # will be float32, i.e. float, rather than float64, i.e. double.
                    self._feat_funcs[feature] = lambda x: float(graph_properties.second_eigenval(x))
                case "spectral_radius":
                    # Since the function can return a complex number, make sure to convert to float
                    self._feat_funcs[feature] = lambda x: float(graph_properties.spectral_radius(x))
                case "zagreb_M1":
                    self._feat_funcs[feature] = graph_properties.zagreb_index1
                case "zagreb_M2":
                    self._feat_funcs[feature] = graph_properties.zagreb_index2
                case _:
                    raise ValueError(f"Unknown molecular global feature '{feature}'.")

            if normalize is not None:
                if not (feat_stats := features_stats.get(feature, None)):
                    raise ValueError(f"Statistics for feature '{feature}' not found in `features_stats`.")
                match normalize:
                    case "min-max":
                        if not (feat_min := feat_stats.get("min", None)) or not (
                            feat_max := feat_stats.get("max", None)
                        ):
                            raise ValueError(
                                f"`min` and `max` must be provided in `features_stats` for feature '{feature}' "
                                f"when using 'min-max' normalization."
                            )
                        self._norm_funcs[feature] = (
                            lambda x, shift=feat_min, scale=feat_max - feat_min: (x - shift) / scale
                        )
                    case "z-score":
                        if not (mean := feat_stats.get("mean", None)) or not (std := feat_stats.get("std", None)):
                            raise ValueError(
                                f"`mean` and `std` must be provided in `features_stats` for feature '{feature}' "
                                f"when using 'z-score' normalization."
                            )
                        self._norm_funcs[feature] = lambda x, shift=mean, scale=std: (x - shift) / scale
                    case _:
                        raise ValueError(f"Unknown normalization method '{normalize}'.")

    def forward(self, data: Data) -> GraphAttrData:
        """Computes and adds molecular global features to the provided graph data object.

        Args:
            data: The input graph data object.

        Returns:
            Graph data object with added molecular global features.
        """
        nx_graph = to_networkx(data)
        global_features = {feat: func(nx_graph) for feat, func in self._feat_funcs.items()}
        if self._norm_funcs:
            global_features.update(
                {feat: norm_func(global_features[feat]) for feat, norm_func in self._norm_funcs.items()}
            )
        data[self.attr_name] = torch.tensor(list(global_features.values()), device=data.x.device)
        # Convert `Data` to `GraphAttrData` so that graph features are properly handled when batching
        return GraphAttrData.from_dict(data.to_dict())
