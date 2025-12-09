from typing import Literal

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import to_networkx

from genesis.data.data import GraphAttrData
from genesis.data.utils import features as graph_properties


class AddMolecularGlobalFeatures(BaseTransform):
    """Add global features useful for characterizing molecules to the graph data object."""

    def __init__(
        self,
        features: list[
            Literal[
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
        ],
        attr_name: str = "graph_attr",
    ) -> None:
        """Initializes an `AddMolecularGlobalFeatures` instance.

        Args:
            features: The list of molecular global features to compute and add to the graph data object.
            attr_name: The name of the attribute where to store the features in the data object.
        """
        self._feat_funcs = []
        for feature in features:
            match feature:
                case "wiener":
                    self._feat_funcs.append(graph_properties.wiener_index)
                case "circuit_rank":
                    self._feat_funcs.append(graph_properties.circuit_rank)
                case "independence":
                    self._feat_funcs.append(graph_properties.independence_number)
                case "hosoya":
                    self._feat_funcs.append(graph_properties.total_matchings)
                case "max_matching":
                    self._feat_funcs.append(graph_properties.maximum_matching)
                case "second_eigenv":
                    # Since the function returns a float64, i.e. double, cast to float so tensors created with this data
                    # will be float32, i.e. float, rather than float64, i.e. double.
                    self._feat_funcs.append(lambda x: float(graph_properties.second_eigenval(x)))
                case "spectral_radius":
                    # Since the function can return a complex number, make sure to convert to float
                    self._feat_funcs.append(lambda x: float(graph_properties.spectral_radius(x)))
                case "zagreb_M1":
                    self._feat_funcs.append(graph_properties.zagreb_index1)
                case "zagreb_M2":
                    self._feat_funcs.append(graph_properties.zagreb_index2)
                case _:
                    raise ValueError(f"Unknown molecular global feature '{feature}'.")

        self.attr_name = attr_name

    def forward(self, data: Data) -> GraphAttrData:
        """Computes and adds molecular global features to the provided graph data object.

        Args:
            data: The input graph data object.

        Returns:
            Graph data object with added molecular global features.
        """
        nx_graph = to_networkx(data)
        global_features = [feat_func(nx_graph) for feat_func in self._feat_funcs]
        data[self.attr_name] = torch.tensor(global_features, device=data.x.device)
        # Convert `Data` to `GraphAttrData` so that graph features are properly handled when batching
        return GraphAttrData.from_dict(data.to_dict())
