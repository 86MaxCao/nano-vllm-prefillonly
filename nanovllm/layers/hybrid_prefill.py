"""Hybrid Prefilling: chunked MLP execution to reduce peak activation memory.

MLP layers are token-independent (no sequence-dimension dependency), so they can
be computed in chunks without affecting correctness. This bounds peak memory from
`seq_len * intermediate_size` to `chunk_size * intermediate_size`, enabling much
longer input sequences on a single GPU.

Attention layers still see the full sequence — only MLP is chunked.
"""

import torch
from torch import nn

_HYBRID_PREFILL_ENABLED = False
_HYBRID_PREFILL_CHUNK_SIZE = 4096


def set_hybrid_prefill_config(enabled: bool, chunk_size: int = 4096):
    global _HYBRID_PREFILL_ENABLED, _HYBRID_PREFILL_CHUNK_SIZE
    _HYBRID_PREFILL_ENABLED = enabled
    _HYBRID_PREFILL_CHUNK_SIZE = chunk_size


def is_hybrid_prefill_enabled() -> bool:
    return _HYBRID_PREFILL_ENABLED


def get_chunk_size() -> int:
    return _HYBRID_PREFILL_CHUNK_SIZE


def chunked_mlp_forward(
    hidden_states: torch.Tensor,
    mlp: nn.Module,
    chunk_size: int | None = None,
) -> torch.Tensor:
    """Execute MLP in chunks along the token (first) dimension.

    Args:
        hidden_states: [total_tokens, hidden_size] — the flattened varlen tensor.
        mlp: The MLP module (gate_up_proj -> act_fn -> down_proj).
        chunk_size: Number of tokens per chunk. Uses global config if None.

    Returns:
        Output tensor with same shape as input.
    """
    if chunk_size is None:
        chunk_size = _HYBRID_PREFILL_CHUNK_SIZE

    seq_len = hidden_states.shape[0]

    if seq_len <= chunk_size:
        return mlp(hidden_states)

    outputs = []
    for start in range(0, seq_len, chunk_size):
        end = min(start + chunk_size, seq_len)
        chunk_out = mlp(hidden_states[start:end])
        outputs.append(chunk_out)

    return torch.cat(outputs, dim=0)
