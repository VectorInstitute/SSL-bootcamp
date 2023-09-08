"""RTDL modules."""
import enum
import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


ModuleType = Union[str, Callable[..., nn.Module]]
_INTERNAL_ERROR_MESSAGE = "Internal error. Please, open an issue."


# The following code is copied from
# https://github.com/Yura52/rtdl/blob/main/rtdl/modules.py (MIT License)
def reglu(x: Tensor) -> Tensor:
    """Compute ReGLU activation function."""
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.relu(b)


def geglu(x: Tensor) -> Tensor:
    """Compute GEGLU activation function."""
    assert x.shape[-1] % 2 == 0
    a, b = x.chunk(2, dim=-1)
    return a * F.gelu(b)


def _is_glu_activation(activation: ModuleType) -> bool:
    return (
        isinstance(activation, str)
        and activation.endswith("GLU")
        or activation in [ReGLU, GEGLU]
    )


def _all_or_none(values: List[nn.Module]) -> bool:
    return all(x is None for x in values) or all(x is not None for x in values)


class ReGLU(nn.Module):
    """The ReGLU activation function from [shazeer2020glu]."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return reglu(x)


class GEGLU(nn.Module):
    """The GEGLU activation function from [shazeer2020glu]."""

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return geglu(x)


class _TokenInitialization(enum.Enum):
    UNIFORM = "uniform"
    NORMAL = "normal"

    @classmethod
    def from_str(cls, initialization: str) -> "_TokenInitialization":
        try:
            return cls(initialization)
        except ValueError as err:
            valid_values = [x.value for x in _TokenInitialization]
            raise ValueError(f"initialization must be one of {valid_values}") from err

    def apply(self, x: Tensor, d: int) -> None:
        d_sqrt_inv = 1 / math.sqrt(d)
        if self == _TokenInitialization.UNIFORM:
            # used in the paper "Revisiting Deep Learning Models for Tabular Data";
            # is equivalent to `nn.init.kaiming_uniform_(x, a=math.sqrt(5))` (which is
            # used by torch to initialize nn.Linear.weight, for example)
            nn.init.uniform_(x, a=-d_sqrt_inv, b=d_sqrt_inv)
        elif self == _TokenInitialization.NORMAL:
            nn.init.normal_(x, std=d_sqrt_inv)


def _make_nn_module(module_type: ModuleType, *args: Any) -> nn.Module:
    if isinstance(module_type, str):
        if module_type == "ReGLU":
            return ReGLU()
        elif module_type == "GEGLU":
            return GEGLU()
        else:
            try:
                cls = getattr(nn, module_type)
            except AttributeError as err:
                raise ValueError(
                    f"Failed to construct the module {module_type} \
                        with the arguments {args}"
                ) from err
            return cls(*args)
    else:
        return module_type(*args)


class MultiheadAttention(nn.Module):
    """Multihead Attention (self-/cross-) with optional 'linear' attention.

    To learn more about Multihead Attention, see [devlin2018bert].
    See the implementation of `Transformer` and the examples below to
    learn how to use the compression technique from [wang2020linformer] to speed
    up the module when the number of tokens is large.
    """

    def __init__(
        self,
        *,
        d_token: int,
        n_heads: int,
        dropout: float,
        bias: bool,
        initialization: str,
    ) -> None:
        """Initialize the module.

        Parameters
        ----------
        d_token : int
            The token size. Must be a multiple of `n_heads`.
        n_heads : int
            The number of heads. If greater than 1, then the module will have
            an additional output layer (the so-called "mixing" layer).
        dropout : float
            Dropout rate for the attention map. The dropout is applied to
            probabilities and does not affect logits.
        bias : bool
            If True, then input (and output, if present) layers also have bias.
            True is a reasonable default choice.
        initialization : {'kaiming', 'xavier'}
            Initialization for input projection layers. Must be one of
            ['kaiming', 'xavier']. 'kaiming' is a reasonable default choice.

        Raises
        ------
        AssertionError
            If requirements for the inputs are not met.
        """
        super().__init__()
        if n_heads > 1:
            assert d_token % n_heads == 0, "d_token must be a multiple of n_heads"
        assert initialization in ["kaiming", "xavier"]

        self.W_q = nn.Linear(d_token, d_token, bias)
        self.W_k = nn.Linear(d_token, d_token, bias)
        self.W_v = nn.Linear(d_token, d_token, bias)
        self.W_out = nn.Linear(d_token, d_token, bias) if n_heads > 1 else None
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout) if dropout else None

        for m in [self.W_q, self.W_k, self.W_v]:
            # the "xavier" branch tries to follow torch.nn.MultiheadAttention;
            # the second condition checks if W_v plays the role of W_out; the latter one
            # is initialized with Kaiming in torch
            if initialization == "xavier" and (
                m is not self.W_v or self.W_out is not None
            ):
                # gain is needed since W_qkv is represented with 3 separate layers (it
                # implies different fan_out)
                nn.init.xavier_uniform_(m.weight, gain=1 / math.sqrt(2))
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        if self.W_out is not None:
            nn.init.zeros_(self.W_out.bias)

    def _reshape(self, x: Tensor) -> Tensor:
        batch_size, n_tokens, d = x.shape
        d_head = d // self.n_heads
        return (
            x.reshape(batch_size, n_tokens, self.n_heads, d_head)
            .transpose(1, 2)
            .reshape(batch_size * self.n_heads, n_tokens, d_head)
        )

    def forward(
        self,
        x_q: Tensor,
        x_kv: Tensor,
        key_compression: Optional[nn.Linear],
        value_compression: Optional[nn.Linear],
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Perform the forward pass.

        Parameters
        ----------
        x_q : array_like
            Query tokens.
        x_kv : array_like
            Key-value tokens.
        key_compression : bool
            Linformer-style compression for keys.
        value_compression : bool
            Linformer-style compression for values.

        Returns
        -------
        tuple
            The first element is the transformed tokens, and the second element \
                is the attention statistics.
        """
        assert _all_or_none(
            [key_compression, value_compression]
        ), "If key_compression is (not) None, then value_compression must (not) be None"
        q, k, v = self.W_q(x_q), self.W_k(x_kv), self.W_v(x_kv)
        for tensor in [q, k, v]:
            assert tensor.shape[-1] % self.n_heads == 0, _INTERNAL_ERROR_MESSAGE
        if key_compression is not None:
            k = key_compression(k.transpose(1, 2)).transpose(1, 2)
            v = value_compression(v.transpose(1, 2)).transpose(1, 2)  # type: ignore

        batch_size = len(q)
        d_head_key = k.shape[-1] // self.n_heads
        d_head_value = v.shape[-1] // self.n_heads
        n_q_tokens = q.shape[1]

        q = self._reshape(q)
        k = self._reshape(k)
        attention_logits = q @ k.transpose(1, 2) / math.sqrt(d_head_key)
        attention_probs = F.softmax(attention_logits, dim=-1)
        if self.dropout is not None:
            attention_probs = self.dropout(attention_probs)
        x = attention_probs @ self._reshape(v)
        x = (
            x.reshape(batch_size, self.n_heads, n_q_tokens, d_head_value)
            .transpose(1, 2)
            .reshape(batch_size, n_q_tokens, self.n_heads * d_head_value)
        )
        if self.W_out is not None:
            x = self.W_out(x)
        return x, {
            "attention_logits": attention_logits,
            "attention_probs": attention_probs,
        }
