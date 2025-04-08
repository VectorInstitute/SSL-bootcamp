"""Feature Tokenizer Transformer (FTTransformer) module."""

import time
import warnings
from typing import Callable, Dict, List, Optional, Tuple, Union, cast

import torch
from torch import Tensor, nn

from model.rtdl_modules import (
    _INTERNAL_ERROR_MESSAGE,
    MultiheadAttention,
    _all_or_none,
    _is_glu_activation,
    _make_nn_module,
    _TokenInitialization,
)


ModuleType = Union[str, Callable[..., nn.Module]]


# The following code is copied and modified from https://github.com/Yura52/rtdl
# Original code: https://github.com/Yura52/rtdl/blob/main/rtdl/modules.py
# Modified by: somaonishi
class NumericalFeatureTokenizer(nn.Module):
    """Transform continuous features to tokens (embeddings).

    This function transforms each continuous feature through a two-step process:

    1. Multiply the feature by a trainable vector.
    2. Add another trainable vector to the result.

    Each feature uses its own separate pair of trainable vectors; vectors are not
    shared between features.

    See Also
    --------
    FeatureTokenizer : Provides an illustration of this transformation.

    Notes
    -----
    Each feature has its separate pair of trainable vectors, i.e.,
    the vectors are not shared between features.
    """

    def __init__(
        self,
        continuous_columns: List[str],
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        continuous_columns : int
            The number of continuous (scalar) features.
        d_token : int
            The size of one token.
        bias : bool, optional
            If False, the transformation will include only multiplication.
            **Warning**: Setting `bias=False` leads to significantly worse results
            for Transformer-like (token-based) architectures. Default is True.
        initialization : {'uniform', 'normal'}, optional
            Initialization policy for parameters. Let `s = d ** -0.5`. Then, the
            corresponding distributions are `Uniform(-s, s)` and `Normal(0, s)`.
            In [gorishniy2021revisiting], the 'uniform' initialization was used.
            Default is 'uniform'.

        References
        ----------
        .. [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev, Valentin Khrulkov,
        Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021
        """
        super().__init__()
        self.initialization_ = _TokenInitialization.from_str(initialization)
        self.__d_token = d_token

        self.weight, self.bias = self.get_dict_parameter(continuous_columns, bias)

    def get_dict_parameter(
        self,
        continuous_columns: List[str],
        use_bias: bool,
    ) -> Tuple[nn.ParameterDict, Optional[nn.ParameterDict]]:
        """Get the dictionary of parameters."""
        weight = {}
        bias = {} if use_bias else None  # type: ignore
        for key in continuous_columns:
            weight[key] = nn.Parameter(Tensor(1, self.__d_token))
            if use_bias:
                bias[key] = nn.Parameter(Tensor(1, self.__d_token))  # type: ignore

        for parameter_dict in [weight, bias]:
            if parameter_dict is not None:
                for parameter in parameter_dict.values():
                    self.initialization_.apply(parameter, self.__d_token)

        return nn.ParameterDict(weight), nn.ParameterDict(bias) if use_bias else None

    def add_parameters(
        self,
        continuous_columns: List[str],
    ) -> None:
        """Add parameters to the module."""
        weight_add, bias_add = self.get_dict_parameter(continuous_columns, self.bias is not None)
        self.weight.update(weight_add)
        if self.bias is not None:
            self.bias.update(bias_add)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.weight)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.__d_token

    def forward(
        self,
        x_dict: Dict[str, Tensor],
    ) -> Tensor:
        """Perform the forward pass."""
        x_out = []
        for key, x in x_dict.items():
            x_ = self.weight[key][None] * x[..., None, None]
            if self.bias is not None:
                x_ = x_ + self.bias[key][None]
            x_out.append(x_)
        return torch.cat(x_out, dim=1)


# The following code is copied and modified from https://github.com/Yura52/rtdl
# Original code: https://github.com/Yura52/rtdl/blob/main/rtdl/modules.py
# Modified by: somaonishi
class CategoricalFeatureTokenizer(nn.Module):
    """Transforms categorical features to tokens (embeddings).

    The module efficiently implements a collection of `torch.nn.Embedding`
    (with optional biases).

    See Also
    --------
    `FeatureTokenizer` : For an illustration of feature tokenization.

    Examples
    --------
    .. testcode::

        # The input must contain integers. For example,
        # if the first feature can
        # take 3 distinct values,
        # then its cardinality is 3 and the first column
        # must contain values from the range `[0, 1, 2]`.
        cardinalities = [3, 10]
        x = torch.tensor([
            [0, 5],
            [1, 7],
            [0, 2],
            [2, 4]
        ])
        n_objects, n_features = x.shape
        d_token = 3
        tokenizer = CategoricalFeatureTokenizer(
            cardinalities, d_token, True, 'uniform'
        )
        tokens = tokenizer(x)
        assert tokens.shape == (n_objects, n_features, d_token)
    """

    def __init__(
        self,
        cardinality_dict: Dict[str, int],
        d_token: int,
        bias: bool,
        initialization: str,
    ) -> None:
        r"""Initialize the module.

        Parameters
        ----------
        cardinality_dict : Dict[str, int]
            The number of distinct values for each feature.
            For example, `cardinalities=[3, 4]`
            describes two features: the first can take values in
            the range `[0, 1, 2]` and the
            second can take values in the range `[0, 1, 2, 3]`.
        d_token : int
            The size of one token.
        bias : bool
            If `True`, a trainable vector is added to the embedding f
            or each feature,
            regardless of the feature value. The bias vectors are not
            shared between features.
        initialization : {'uniform', 'normal'}
            The initialization policy for parameters.
            Let \( s = d^{-0.5} \). The corresponding
            distributions are `Uniform(-s, s)` and `Normal(0, s)`.
            In the referenced paper, the 'uniform' initialization was used.

        References
        ----------
        .. [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev,
        Valentin Khrulkov, Artem Babenko,
        "Revisiting Deep Learning Models for Tabular Data", 2021.
        """
        super().__init__()
        assert cardinality_dict, "cardinalities must be non-empty"
        assert d_token > 0, "d_token must be positive"
        self.initialization_ = _TokenInitialization.from_str(initialization)
        self.__d_token = d_token

        self.embeddings = nn.ModuleDict(
            {key: nn.Embedding(cardinality, d_token) for key, cardinality in cardinality_dict.items()}
        )
        if bias:
            bias = {}  # type: ignore
            for key in cardinality_dict:
                bias[key] = nn.Parameter(Tensor(1, d_token))  # type: ignore
                self.initialization_.apply(bias[key], d_token)  # type: ignore
            self.bias = nn.ParameterDict(bias)

        for parameter in self.embeddings.values():
            self.initialization_.apply(parameter.weight, d_token)

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return len(self.category_offsets)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return self.__d_token

    def add_parameters(self, cardinality_dict: Dict[str, int]) -> None:
        """Add parameters to the module."""
        embeddings_add = nn.ModuleDict(
            {key: nn.Embedding(cardinality, self.__d_token) for key, cardinality in cardinality_dict.items()}
        )
        for parameter in embeddings_add.values():
            self.initialization_.apply(parameter.weight, self.__d_token)
        self.embeddings.update(embeddings_add)

        if self.bias is not None:
            bias = {}
            for key in cardinality_dict:
                bias[key] = nn.Parameter(Tensor(1, self.__d_token))
                self.initialization_.apply(bias[key], self.__d_token)
        self.bias.update(nn.ParameterDict(bias))

    def forward(self, x_dict: Dict[str, Tensor]) -> Tensor:
        """Perform the forward pass."""
        x_out = []

        for key, x in x_dict.items():
            if key not in self.embeddings:
                print(f"Key {key} not found in embeddings!")
            x_ = self.embeddings[key](x).unsqueeze(1)
            if self.bias is not None:
                x_ = x_ + self.bias[key][None]
            x_out.append(x_)
        return torch.cat(x_out, dim=1)


# The following code is copied and modified from https://github.com/Yura52/rtdl
# Original code: https://github.com/Yura52/rtdl/blob/main/rtdl/modules.py
# Modified by: somaonishi
class FeatureTokenizer(nn.Module):
    """Combines `NumericalFeatureTokenizer` and `CategoricalFeatureTokenizer`.

    The "Feature Tokenizer" module as described in [gorishniy2021revisiting].
    This module transforms continuous and categorical features into
    tokens (embeddings).

    In the following illustration, the red module in the upper brackets
    represents `NumericalFeatureTokenizer`, and the green module in the
    lower brackets represents `CategoricalFeatureTokenizer`.

    .. image:: ../images/feature_tokenizer.png
        :scale: 33%
        :alt: Feature Tokenizer

    Examples
    --------
    .. testcode::

        n_objects = 4
        n_num_features = 3
        n_cat_features = 2
        d_token = 7
        x_num = torch.randn(n_objects, n_num_features)
        x_cat = torch.tensor([[0, 1], [1, 0], [0, 2], [1, 1]])
        # [2, 3] reflects cardinalities
        tokenizer = FeatureTokenizer(n_num_features, [2, 3], d_token)
        tokens = tokenizer(x_num, x_cat)
        assert tokens.shape == (
            n_objects, n_num_features + n_cat_features, d_token
        )

    References
    ----------
    .. [gorishniy2021revisiting] Yury Gorishniy, Ivan Rubachev,
    Valentin Khrulkov, Artem Babenko,
    "Revisiting Deep Learning Models for Tabular Data", 2021.
    """

    def __init__(
        self,
        continuous_columns: Optional[List[str]],
        cat_cardinality_dict: Optional[Dict[str, int]],
        d_token: int,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        continuous_columns : List[str]
            The number of continuous features. Pass 0 if there are no
            numerical features.
        cat_cardinality_dict : Optional[Dict[str, int]]
            The number of unique values for each categorical feature.
            See `CategoricalFeatureTokenizer` for more details.
            Pass an empty list if there are no categorical features.
        d_token : int
            The size of a single token.
        """
        super().__init__()
        # assert len(continuous_columns) >= 0, "n_num_features must be non-negative"
        assert continuous_columns or cat_cardinality_dict, (
            "at least one of n_num_features or cat_cardinalities must be \
            positive/non-empty"
        )
        self.initialization = "uniform"
        self.num_tokenizer = (
            NumericalFeatureTokenizer(
                continuous_columns=continuous_columns,
                d_token=d_token,
                bias=True,
                initialization=self.initialization,
            )
            if continuous_columns
            else None
        )
        self.cat_tokenizer = (
            CategoricalFeatureTokenizer(cat_cardinality_dict, d_token, True, self.initialization)
            if cat_cardinality_dict
            else None
        )

    @property
    def n_tokens(self) -> int:
        """The number of tokens."""
        return sum(x.n_tokens for x in [self.num_tokenizer, self.cat_tokenizer] if x is not None)

    @property
    def d_token(self) -> int:
        """The size of one token."""
        return (  # type: ignore
            self.cat_tokenizer.d_token  # type: ignore
            if self.num_tokenizer is None
            else self.num_tokenizer.d_token  # type: ignore
        )

    def add_attribute(
        self,
        continuous_columns: Optional[List[str]] = None,
        cat_cardinality_dict: Optional[Dict[str, int]] = None,
    ) -> None:
        """Add attributes to the module."""
        assert continuous_columns is not None or cat_cardinality_dict is not None, (
            "At least one of n_num and cardinalities must be presented."
        )
        if continuous_columns is not None:
            if self.num_tokenizer is not None:
                self.num_tokenizer.add_parameters(continuous_columns)
            else:
                self.num_tokenizer = NumericalFeatureTokenizer(
                    continuous_columns,
                    self.d_token,
                    True,
                    self.initialization,
                )
        if cat_cardinality_dict is not None:
            if self.cat_tokenizer is not None:
                self.cat_tokenizer.add_parameters(cat_cardinality_dict)
            else:
                self.cat_tokenizer = CategoricalFeatureTokenizer(
                    cat_cardinality_dict,
                    self.d_token,
                    True,
                    self.initialization,
                )

    def forward(self, x_num: Optional[Dict[str, Tensor]], x_cat: Optional[Dict[str, Tensor]]) -> Tensor:
        """Perform the forward pass.

        Parameters
        ----------
        x_num : torch.Tensor
            Continuous features. Must be present if `n_num_features > 0` was passed
            to the constructor.
        x_cat : torch.Tensor
            Categorical features. See `CategoricalFeatureTokenizer.forward`
            for details.
            Must be present if a non-empty `cat_cardinalities` list was passed to
            the constructor.

        Returns
        -------
        torch.Tensor
            Tokens.

        Raises
        ------
        AssertionError
            If the described requirements for the inputs are not met.
        """
        assert x_num is not None or x_cat is not None, "At least one of x_num and x_cat must be presented"
        # assert _all_or_none(
        #     [self.num_tokenizer, x_num]
        # ), "If self.num_tokenizer is (not) None, then x_num must (not) be None"
        # assert _all_or_none(
        #     [self.cat_tokenizer, x_cat]
        # ), "If self.cat_tokenizer is (not) None, then x_cat must (not) be None"
        x = []
        if x_num is not None:
            x.append(self.num_tokenizer(x_num))  # type: ignore
        if x_cat is not None:
            x.append(self.cat_tokenizer(x_cat))  # type: ignore
        return x[0] if len(x) == 1 else torch.cat(x, dim=1)


# The following code is copied and modified from https://github.com/Yura52/rtdl
# Original code: https://github.com/Yura52/rtdl/blob/main/rtdl/modules.py
# Modified by: somaonishi
class Transformer(nn.Module):
    """Transformer with extra features.

    This module is the backbone of `FTTransformer`.
    """

    WARNINGS = {"first_prenormalization": True, "prenormalization": True}

    class FFN(nn.Module):
        """The Feed-Forward Network module used in every `Transformer` block."""

        def __init__(
            self,
            *,
            d_token: int,
            d_hidden: int,
            bias_first: bool,
            bias_second: bool,
            dropout: float,
            activation: ModuleType,
        ):
            """Initialize the module."""
            super().__init__()
            self.linear_first = nn.Linear(
                d_token,
                d_hidden * (2 if _is_glu_activation(activation) else 1),
                bias_first,
            )
            self.activation = _make_nn_module(activation)
            self.dropout = nn.Dropout(dropout)
            self.linear_second = nn.Linear(d_hidden, d_token, bias_second)

        def forward(self, x: Tensor) -> Tensor:
            """Perform the forward pass."""
            x = self.linear_first(x)
            x = self.activation(x)
            x = self.dropout(x)
            x = self.linear_second(x)
            return x

    def __init__(  # noqa: C901
        self,
        *,
        d_token: int,
        n_blocks: int,
        attention_n_heads: int,
        attention_dropout: float,
        attention_initialization: str,
        attention_normalization: str,
        ffn_d_hidden: int,
        ffn_dropout: float,
        ffn_activation: str,
        ffn_normalization: str,
        residual_dropout: float,
        prenormalization: bool,
        first_prenormalization: bool,
        last_layer_query_idx: Union[None, List[int], slice],
        n_tokens: Optional[int],
        kv_compression_ratio: Optional[float],
        kv_compression_sharing: Optional[str],
    ) -> None:
        """Initialize the module."""
        super().__init__()
        if isinstance(last_layer_query_idx, int):
            raise ValueError(
                "last_layer_query_idx must be None, list[int] or slice. "
                f"Do you mean last_layer_query_idx=[{last_layer_query_idx}] ?"
            )
        if not prenormalization:
            assert not first_prenormalization, (
                "If `prenormalization` is False, then `first_prenormalization` \
                must be False"
            )
        assert _all_or_none([n_tokens, kv_compression_ratio, kv_compression_sharing]), (
            "If any of the following arguments is (not) None, \
                then all of them must"
            "n_tokens, kv_compression_ratio, kv_compression_sharing"
        )
        assert kv_compression_sharing in [None, "headwise", "key-value", "layerwise"]
        if not prenormalization:
            if self.WARNINGS["prenormalization"]:
                warnings.warn(
                    "prenormalization is set to False. Are you sure about this? "
                    "The training can become less stable. "
                    "You can turn off this warning by tweaking the "
                    "rtdl.Transformer.WARNINGS dictionary.",
                    UserWarning,
                    stacklevel=2,
                )
            assert not first_prenormalization, (
                "If prenormalization is False, then first_prenormalization is \
                ignored and must be set to False"
            )
        if prenormalization and first_prenormalization and self.WARNINGS["first_prenormalization"]:
            warnings.warn(
                "first_prenormalization is set to True. Are you sure about this? "
                "For example, the vanilla FTTransformer with "
                "first_prenormalization=True performs SIGNIFICANTLY worse. "
                "You can turn off this warning by tweaking the "
                "rtdl.Transformer.WARNINGS dictionary.",
                UserWarning,
                stacklevel=2,
            )
            time.sleep(3)

        def make_kv_compression() -> ModuleType:
            """Make the key and value compression layers."""
            assert n_tokens and kv_compression_ratio, _INTERNAL_ERROR_MESSAGE  # for mypy
            # https://github.com/pytorch/fairseq/blob/1bba712622b8ae4efb3eb793a8a40da386fe11d0/examples/linformer/linformer_src/modules/multihead_linear_attention.py#L83
            return nn.Linear(n_tokens, int(n_tokens * kv_compression_ratio), bias=False)

        self.shared_kv_compression = (
            make_kv_compression() if kv_compression_ratio and kv_compression_sharing == "layerwise" else None
        )

        self.prenormalization = prenormalization
        self.last_layer_query_idx = last_layer_query_idx

        self.blocks = nn.ModuleList([])
        for layer_idx in range(n_blocks):
            layer = nn.ModuleDict(
                {
                    "attention": MultiheadAttention(
                        d_token=d_token,
                        n_heads=attention_n_heads,
                        dropout=attention_dropout,
                        bias=True,
                        initialization=attention_initialization,
                    ),
                    "ffn": Transformer.FFN(
                        d_token=d_token,
                        d_hidden=ffn_d_hidden,
                        bias_first=True,
                        bias_second=True,
                        dropout=ffn_dropout,
                        activation=ffn_activation,
                    ),
                    "attention_residual_dropout": nn.Dropout(residual_dropout),
                    "ffn_residual_dropout": nn.Dropout(residual_dropout),
                    "output": nn.Identity(),  # for hooks-based introspection
                }
            )
            if layer_idx or not prenormalization or first_prenormalization:
                layer["attention_normalization"] = _make_nn_module(attention_normalization, d_token)
            layer["ffn_normalization"] = _make_nn_module(ffn_normalization, d_token)
            if kv_compression_ratio and self.shared_kv_compression is None:
                layer["key_compression"] = make_kv_compression()
                if kv_compression_sharing == "headwise":
                    layer["value_compression"] = make_kv_compression()
                else:
                    assert kv_compression_sharing == "key-value", _INTERNAL_ERROR_MESSAGE
            self.blocks.append(layer)

    def _get_kv_compressions(self, layer: nn.ModuleDict) -> Tuple[Optional[nn.Module], Optional[nn.Module]]:
        return (
            (self.shared_kv_compression, self.shared_kv_compression)
            if self.shared_kv_compression is not None
            else (layer["key_compression"], layer["value_compression"])
            if "key_compression" in layer and "value_compression" in layer
            else (layer["key_compression"], layer["key_compression"])
            if "key_compression" in layer
            else (None, None)
        )

    def _start_residual(
        self,
        layer: nn.ModuleDict,
        stage: str,
        x: Tensor,
    ) -> Tensor:
        assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = x
        if self.prenormalization:
            norm_key = f"{stage}_normalization"
            if norm_key in layer:
                x_residual = layer[norm_key](x_residual)
        return x_residual

    def _end_residual(self, layer: nn.ModuleDict, stage: str, x: Tensor, x_residual: Tensor) -> Tensor:
        assert stage in ["attention", "ffn"], _INTERNAL_ERROR_MESSAGE
        x_residual = layer[f"{stage}_residual_dropout"](x_residual)
        x = x + x_residual
        if not self.prenormalization:
            x = layer[f"{stage}_normalization"](x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Perform the forward pass."""
        assert x.ndim == 3, "The input must have 3 dimensions: (n_objects, n_tokens, d_token)"
        for layer_idx, layer in enumerate(self.blocks):
            layer = cast(nn.ModuleDict, layer)

            query_idx = self.last_layer_query_idx if layer_idx + 1 == len(self.blocks) else None
            x_residual = self._start_residual(layer, "attention", x)
            x_residual, _ = layer["attention"](
                x_residual if query_idx is None else x_residual[:, query_idx],
                x_residual,
                *self._get_kv_compressions(layer),
            )
            if query_idx is not None:
                x = x[:, query_idx]
            x = self._end_residual(layer, "attention", x, x_residual)

            x_residual = self._start_residual(layer, "ffn", x)
            x_residual = layer["ffn"](x_residual)
            x = self._end_residual(layer, "ffn", x, x_residual)
            x = layer["output"](x)

        return x

    @torch.no_grad()
    def get_attention(self, x: Tensor, layer_idx: Union[int, str] = 0) -> Tensor:
        """Get the attention matrix."""
        assert x.ndim == 3, "The input must have 3 dimensions: (n_objects, n_tokens, d_token)"

        N, L, _ = x.shape

        def layer_attention(index: int) -> Tensor:
            layer = self.blocks[index]
            layer = cast(nn.ModuleDict, layer)

            x_residual = self._start_residual(layer, "attention", x)
            _, attention = layer["attention"](
                x_residual,
                x_residual,
                *self._get_kv_compressions(layer),
            )
            attention = torch.softmax(attention["attention_logits"], dim=-1)
            attention = attention.reshape(N, self.blocks[0]["attention"].n_heads, L, L)
            attention = attention.mean(1)
            attention = attention + torch.eye(L, device=x.device).unsqueeze(0)
            attention = attention / 2
            return attention

        if isinstance(layer_idx, str):
            assert layer_idx == "all"
            attention = layer_attention(0)
            for idx in range(1, len(self.blocks)):
                attention = torch.matmul(layer_attention(idx), attention)
            return attention
        return layer_attention(layer_idx)
