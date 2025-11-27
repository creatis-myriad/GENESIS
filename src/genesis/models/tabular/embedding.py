import torch
from rtdl_revisiting_models import CategoricalEmbeddings, LinearEmbeddings
from torch import Tensor, nn


class FeatureTokenizer(nn.Module):
    """Combines `LinearEmbeddings` and `CategoricalEmbeddings`.

    The "Feature Tokenizer" module from "Revisiting Deep Learning Models for Tabular Data" by Gorishniy et al. (2021).
    The module transforms continuous and categorical features to tokens (embeddings).

    Notes:
        - This is a port of the `FeatureTokenizer` class from v0.0.13 of the `rtdl` package using the updated underlying
          `CategoricalEmbeddings` and `LinearEmbeddings` from the `rtdl_revisiting_models` package, instead of the
          original `CategoricalFeatureTokenizer` and `NumericalFeatureTokenizer` from the `rtdl` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L260-L377

    Examples:
        >>> n_objects = 4
        >>> n_num_features = 3
        >>> n_cat_features = 2
        >>> d_token = 7
        >>> x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
        >>> x_num = torch.randn(n_objects, n_num_features)
        >>> tokenizer = FeatureTokenizer(n_num_features, {0: 2, 1: 3}, d_token)   # [2, 3] reflects cardinalities
        >>> tokens = tokenizer(torch.hstack([x_cat, x_num]))
        >>> tokens.shape  # (n_objects, n_num_features + n_cat_features, d_token)
        torch.Size([4, 5, 7])
    """

    def __init__(
        self,
        n_num_features: int,
        cat_cardinalities: dict[int, int],
        d_token: int,
    ) -> None:
        """Initializes `FeatureTokenizer` instance.

        Args:
            n_num_features: Number of continuous features. Set to 0 if there are no numerical features.
            cat_cardinalities: Indexes of categorical features in the input tensor to embed, along with the number of
                unique values for each feature. Pass an empty dict if there are no categorical features.
            d_token: Dimensionality of each token.
        """
        super().__init__()
        assert n_num_features >= 0, "n_num_features must be non-negative"
        assert n_num_features or cat_cardinalities, (
            "at least one of n_num_features or cat_cardinalities must be positive/non-empty"
        )
        self.n_num_features = n_num_features
        self.cat_cardinalities = cat_cardinalities
        self.num_tokenizer = LinearEmbeddings(n_num_features, d_token) if n_num_features else None
        self.cat_tokenizer = (
            CategoricalEmbeddings(list(cat_cardinalities.values()), d_token) if cat_cardinalities else None
        )

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.cat_tokenizer.d_token if self.num_tokenizer is None else self.num_tokenizer.d_token  # type: ignore

    def reset_parameters(self) -> None:
        """Resets all learnable parameters of the module."""
        if self.num_tokenizer is not None:
            self.num_tokenizer.reset_parameters()
        if self.cat_tokenizer is not None:
            self.cat_tokenizer.reset_parameters()

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass.

        Args:
            x: ([batch_size], n_num_features + n_cat_features), Input mix of numerical and categorical features.

        Returns:
            ([batch_size], n_num_features + n_cat_features, d_token), Tokenized features.

        Raises:
            AssertionError: if the described requirements for the inputs are not met.
        """
        if x.shape[-1] != self.n_num_features + len(self.cat_cardinalities):
            raise ValueError(
                f"Input has incorrect number of features. "
                f"Expected {self.n_num_features} (n_num_features) + {len(self.cat_cardinalities)} (n_cat_features), "
                f"got {x.shape[-1]}."
            )

        # Compute indices of categorical and numerical features
        full_indices = torch.arange(x.shape[-1], device=x.device)
        cat_indices = torch.tensor(list(self.cat_cardinalities), device=x.device)
        num_indices = full_indices[~torch.isin(full_indices, cat_indices)]

        xs = []
        if self.cat_tokenizer is not None:
            # Extract categorical features from input based on their indices
            x_cat = x.index_select(-1, cat_indices).to(torch.long)
            # Embed categorical features
            xs.append(self.cat_tokenizer(x_cat))
        if self.num_tokenizer is not None:
            # Extract numerical features from input based on their indices
            x_num = x.index_select(-1, num_indices)
            # Embed numerical features
            xs.append(self.num_tokenizer(x_num))

        # Concatenate indices by types and use argsort to find indices that would restore original order
        # (i.e. the inverse permutation applied to slice input into categorical and numerical features)
        tokenized_index_order = torch.cat((cat_indices, num_indices))
        slice_inverse_indices = torch.argsort(tokenized_index_order)

        # Concatenate numerical and categorical tokens, and reorder to original feature order
        x = torch.cat(xs, dim=-2)
        return x.index_select(-2, slice_inverse_indices)


class CLSToken(nn.Module):
    """CLS token module that appends a learnable token **to the end** of each sequence in the batch.

    Notes:
        - This is a port of the `CLSToken` class from v0.0.13 of the `rtdl` package. It mixes the original
          implementation with the simpler code of `_CLSEmbedding` from v0.0.2 of the `rtdl_revisiting_models` package.

    References:
        - Original implementation is here: https://github.com/yandex-research/rtdl/blob/f395a2db37bac74f3a209e90511e2cb84e218973/rtdl/modules.py#L380-L446

    Examples:
        >>> n_tokens = 3
        >>> d_token = 5
        >>> cls_token = CLSToken(d_token)
        >>> x = torch.randn(n_tokens, d_token)
        >>> x = cls_token(x)
        >>> x.shape # (n_tokens + 1, d_token)
        torch.Size([4, 5])
        >>> batch_size = 2
        >>> batch_x = torch.randn(batch_size, n_tokens, d_token)
        >>> batch_x = cls_token(batch_x)
        >>> batch_x.shape # (batch_size, n_tokens + 1, d_token)
        torch.Size([2, 4, 5])
    """

    def __init__(self, d_token: int) -> None:
        """Initializes `CLSToken` instance.

        Args:
            d_token: Dimensionality of each token.
        """
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_token))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initializes the weights using a uniform distribution."""
        d_rsqrt = self.weight.shape[-1] ** -0.5
        nn.init.uniform_(self.weight, -d_rsqrt, d_rsqrt)

    def expand(self, *leading_dimensions: int) -> Tensor:
        """Expand (repeat) the underlying CLS token to a tensor with the given leading dimensions.

        A possible use case is building a batch of CLS tokens.

        Note:
            Under the hood, the `torch.Tensor.expand` method is applied to the underlying `weight` parameter, so
            gradients will be propagated as expected.

        Args:
            leading_dimensions: Additional new dimensions.

        Returns:
            Tensor of shape `(*leading_dimensions, len(self.weight))`.
        """
        if not leading_dimensions:
            return self.weight
        new_dims = (1,) * (len(leading_dimensions) - 1)
        return self.weight.view(*new_dims, -1).expand(*leading_dimensions, -1)

    def forward(self, x: Tensor) -> Tensor:
        """Append self **to the end** of each item in the batch.

        Args:
            x: ([N], S, `d_model`), Input tensor.

        Returns:
            ([N], S + 1, `d_model`), Tensor with appended CLS token.
        """
        expanded_cls = self.expand(*x.shape[:-2], 1)
        return torch.cat([x, expanded_cls], dim=-2)
