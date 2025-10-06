from abc import ABC

from torch import nn
from torch_geometric.transforms import BaseTransform


class LearnableTransform(BaseTransform, nn.Module, ABC):
    """A base class for transformations with learnable parameters."""
