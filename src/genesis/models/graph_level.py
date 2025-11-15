import collections
from typing import Literal

import torch
from torch import nn
from torch_geometric.data import Batch
from torch_geometric.nn import aggr

from genesis.models import GraphLitModule
from genesis.models.gnn.transform import LearnableTransform


class GraphLevelLitModule(GraphLitModule):
    """A LightningModule for GNNs aimed at graph-level tasks."""

    task_level = "graph"

    def __init__(
        self,
        task: Literal["binary", "multiclass", "multilabel", "regression"],
        encoder: nn.Module,
        readout: aggr.Aggregation,
        head: nn.Module,
        transforms: dict[str, LearnableTransform] | None = None,
        *args,
        **kwargs,
    ) -> None:
        """Initializes a `GraphLevelLitModule`.

        Args:
            task: Prediction task for which to configure the GNN model.
            encoder: The GNN model used to encode the graph.
            readout: The readout operation to use to aggregate node features into a single graph-level representation.
            head: The prediction head used to make predictions based on the graph-level representation.
            transforms: Transformations with learnable parameters (e.g. embedding) to apply to the input graphs before
                passing them to the encoder.
            *args: Additional positional arguments to pass to the superclass.
            **kwargs: Additional keyword arguments to pass to the superclass.
        """
        super().__init__(*args, **kwargs)

        # this line allows to access init params with 'self.hparams' attribute
        # also ensures init params will be stored in ckpt
        self.save_hyperparameters(ignore=["encoder", "readout", "head", "transforms"])

        if transforms:
            transforms = nn.Sequential(collections.OrderedDict(transforms))
        self.transforms = transforms
        self.encoder = encoder
        self.readout = readout
        self.head = head

    def forward(self, data: Batch) -> torch.Tensor:
        """Perform a forward pass through the model on a batch of graphs.

        Args:
            data: A batch of graphs, represented as one big (disconnected) graph.

        Returns:
            The predicted logits for the input graphs in the batch.
        """
        data = self.transforms(data) if self.transforms is not None else data
        x = self._encoder_step(data, data.x.float())  # Ensure graph node features are float
        x = self._readout_step(data, x)
        return self._head_step(data, x)

    def _encoder_step(self, data: Batch, x: torch.Tensor) -> torch.Tensor:
        # Extract different inputs depending on the types of features supported by the encoder,
        # casting features to float as needed
        encoder_forward_kwargs = {}
        if getattr(self.encoder, "supports_edge_attr", False):
            encoder_forward_kwargs["edge_attr"] = data.edge_attr.float() if data.edge_attr is not None else None
        if getattr(self.encoder, "supports_edge_weight", False):
            encoder_forward_kwargs["edge_weight"] = data.edge_weight.float() if data.edge_weight is not None else None
        if getattr(self.encoder, "supports_norm_batch", False):
            encoder_forward_kwargs["batch"] = data.batch
            encoder_forward_kwargs["batch_size"] = data.batch_size
        if getattr(self.encoder, "supports_batch", False):
            encoder_forward_kwargs["batch"] = data.batch
        if getattr(self.encoder, "supports_pe", False):
            encoder_forward_kwargs["pe"] = getattr(data, self.hparams.pe_attr)
        if getattr(self.encoder, "supports_graph_attr", False):
            encoder_forward_kwargs["graph_attr"] = (
                data.graph_attr.float() if data.get("graph_attr") is not None else None
            )

        return self.encoder(x, data.edge_index, **encoder_forward_kwargs)

    def _readout_step(self, data: Batch, x: torch.Tensor) -> torch.Tensor:
        # Pass the batch size to readout operation to avoid CPU communication/graph breaks
        return self.readout(x, ptr=data.ptr, dim_size=data.batch_size)

    def _head_step(self, data: Batch, x: torch.Tensor) -> torch.Tensor:
        # After the readout step, each graph as been reduced to one vector representation,
        # i.e. each element in the batch comes from a different graph, so we have to update the batch vector
        batch = torch.arange(data.batch_size, device=x.device)
        x = self.head(x, batch=batch, batch_size=data.batch_size)
        if self.hparams.task == "binary":
            x = x.squeeze(-1)  # Flatten the last dim when only one value is predicted
        return x


class LateFusionGraphLevelLitModule(GraphLevelLitModule):
    """A LightningModule that performs late fusion between GNN encoding and graph-level features."""

    def _head_step(self, data: Batch, x: torch.Tensor) -> torch.Tensor:
        x = torch.hstack((x, data.graph_attr))  # Concatenate graph-level features w/ graph encoding
        return super()._head_step(data, x)
