from functools import lru_cache

import torch
from torch import nn


def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _compute_inv_freq(
    rotary_dim: int, base: float, rope_scaling: dict | None
) -> tuple[torch.Tensor, float]:
    """Return ``(inv_freq, position_scale)`` for the requested scaling scheme.

    ``position_scale`` divides positions before the angles are computed, which is
    how linear ("SuPE"-style) interpolation stretches the context window.
    """
    inv_freq = 1.0 / (
        base ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
    )
    if rope_scaling is None:
        return inv_freq, 1.0

    scaling = dict(rope_scaling)
    # HF configs use either `rope_type` or the legacy `type` key. Some Qwen
    # configs also stash `rope_theta` and mrope metadata in here.
    rope_type = scaling.get("rope_type") or scaling.get("type") or "default"
    rope_type = str(rope_type).lower()
    factor = float(scaling.get("factor", 1.0))

    if rope_type in ("default", "mrope", "interleaved_mrope"):
        # mRoPE reuses the default 1D frequencies; the multimodal section
        # dimensions are applied by the model when it builds position ids.
        return inv_freq, 1.0
    if rope_type == "linear":
        return inv_freq, factor
    if rope_type == "dynamic":
        # Dynamic NTK with the config-time sequence length; positions beyond it
        # would need per-step recomputation, which the static cache cannot do.
        adjusted = base * factor ** (rotary_dim / (rotary_dim - 2))
        inv_freq = 1.0 / (
            adjusted ** (torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim)
        )
        return inv_freq, 1.0
    if rope_type in ("llama3", "yarn"):
        low_freq_factor = float(scaling.get("low_freq_factor", 1.0))
        high_freq_factor = float(scaling.get("high_freq_factor", 4.0))
        original_max = float(
            scaling.get("original_max_position_embeddings", 8192)
        )
        wavelen = 2 * torch.pi / inv_freq
        low_wavelen = original_max / low_freq_factor
        high_wavelen = original_max / high_freq_factor
        smooth = (original_max / wavelen - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smoothed = (1 - smooth) * inv_freq / factor + smooth * inv_freq
        inv_freq = torch.where(wavelen > low_wavelen, inv_freq / factor, smoothed)
        inv_freq = torch.where(wavelen < high_wavelen, inv_freq, smoothed)
        return inv_freq, 1.0

    raise ValueError(
        f"Unsupported rope_scaling type {rope_type!r}. Supported: default, "
        f"linear, dynamic, llama3, yarn, mrope."
    )


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
        rope_scaling: dict | None = None,
    ) -> None:
        super().__init__()
        if rotary_dim > head_size:
            raise ValueError(
                f"rotary_dim ({rotary_dim}) cannot exceed head_size ({head_size})"
            )
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        inv_freq, position_scale = _compute_inv_freq(rotary_dim, base, rope_scaling)
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        if position_scale != 1.0:
            t = t / position_scale
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cache = torch.cat((freqs.cos(), freqs.sin()), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions]
        cos, sin = cos_sin.chunk(2, dim=-1)
        if self.rotary_dim == self.head_size:
            return (
                apply_rotary_emb(query, cos, sin),
                apply_rotary_emb(key, cos, sin),
            )
        # Partial rotary: only the leading rotary_dim channels are rotated.
        d = self.rotary_dim
        query = torch.cat(
            (apply_rotary_emb(query[..., :d], cos, sin), query[..., d:]), dim=-1
        )
        key = torch.cat(
            (apply_rotary_emb(key[..., :d], cos, sin), key[..., d:]), dim=-1
        )
        return query, key


@lru_cache(maxsize=None)
def _get_rope_cached(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling_items: tuple | None,
):
    rope_scaling = dict(rope_scaling_items) if rope_scaling_items is not None else None
    return RotaryEmbedding(
        head_size, rotary_dim, max_position, base, rope_scaling
    )


def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    """Return a shared RotaryEmbedding for the given geometry.

    Instances are cached per distinct argument set; a single-entry cache used to
    hand later-built layers a mismatched cos/sin table.
    """
    key = None
    if rope_scaling is not None:
        key = tuple(sorted((k, _hashable(v)) for k, v in rope_scaling.items()))
    return _get_rope_cached(head_size, rotary_dim, max_position, base, key)


def _hashable(value):
    if isinstance(value, list):
        return tuple(_hashable(v) for v in value)
    if isinstance(value, dict):
        return tuple(sorted((k, _hashable(v)) for k, v in value.items()))
    return value


def rope_params_from_config(config, default_theta: float = 10000.0):
    """Extract ``(rope_theta, rope_scaling)`` from an HF config.

    Some Qwen configs nest ``rope_theta`` inside ``rope_scaling``; that value
    wins, and the nested key is removed so it is not mistaken for a scaling
    parameter.
    """
    rope_theta = getattr(config, "rope_theta", None)
    if rope_theta is None:
        rope_theta = default_theta
    rope_scaling = getattr(config, "rope_scaling", None)

    if isinstance(rope_scaling, dict):
        rope_scaling = dict(rope_scaling)
        nested_theta = rope_scaling.pop("rope_theta", None)
        if nested_theta is not None:
            rope_theta = nested_theta
        if not rope_scaling:
            rope_scaling = None
    elif rope_scaling is not None:
        rope_scaling = None

    return float(rope_theta), rope_scaling
