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

        # Multimodal RoPE: map each frequency slot (rotary_dim // 2 of them)
        # to the position component that supplies it: 0=temporal, 1=height,
        # 2=width. Qwen3-VL interleaves the components across slots
        # (THWTHW...TT), Qwen2-VL/Qwen2.5-VL chunks them (TTT...HHH...WWW).
        self.mrope_section: list[int] | None = None
        comp = None
        if rope_scaling is not None and rope_scaling.get("mrope_section") is not None:
            section = [int(x) for x in rope_scaling["mrope_section"]]
            comp = torch.zeros(rotary_dim // 2, dtype=torch.long)
            if rope_scaling.get("mrope_interleaved"):
                for dim in (1, 2):
                    length = section[dim] * 3
                    comp[dim:length:3] = dim
            else:
                comp[section[0] : section[0] + section[1]] = 1
                comp[section[0] + section[1] :] = 2
            self.mrope_section = section
        self.register_buffer(
            "mrope_component", comp if comp is not None else torch.zeros(0, dtype=torch.long),
            persistent=False,
        )

    @torch.compile
    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if positions.dim() == 2:
            # Multimodal RoPE: positions is [3, num_tokens] (t, h, w).
            comp = self.mrope_component
            if comp.numel() != self.rotary_dim // 2:
                raise ValueError(
                    "mrope positions [3, N] require a rope with mrope_section configured"
                )
            cos_sin = self.cos_sin_cache[positions]  # [3, N, rotary_dim]
            cos, sin = cos_sin.chunk(2, dim=-1)  # each [3, N, rotary_dim // 2]
            # Slot j takes cos/sin of component comp[j].
            cos = torch.where(
                comp == 0, cos[0], torch.where(comp == 1, cos[1], cos[2])
            )
            sin = torch.where(
                comp == 0, sin[0], torch.where(comp == 1, sin[1], sin[2])
            )
        else:
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


def build_mrope_positions(
    sequence_lengths: list[int],
    seq_vision_placeholders: list[list[tuple[int, int]]] | None,
    image_grid_thw: torch.Tensor | None,
    spatial_merge_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Build ``[3, total_tokens]`` mrope position ids (t, h, w).

    Mirrors HF ``get_rope_index`` for image-only inputs, which is the scheme
    shared by Qwen2-VL/Qwen2.5-VL (chunked sections) and Qwen3-VL
    (interleaved sections): text tokens get identical sequential ids in all
    three components; image tokens get spatial ids and the running position
    advances by ``max(h', w')`` per image. Placeholder offsets are relative to
    the start of each sequence.
    """
    total = int(sum(sequence_lengths))
    pos = torch.zeros(3, total, dtype=torch.long, device=device)
    grids = image_grid_thw.tolist() if image_grid_thw is not None else []
    image_idx = 0
    base = 0
    for seq_len, placeholders in zip(
        sequence_lengths, seq_vision_placeholders or [[]] * len(sequence_lengths)
    ):
        cursor = 0
        current_pos = 0
        for offset, n_tokens in placeholders or []:
            text_len = offset - cursor
            if text_len > 0:
                idx = torch.arange(
                    current_pos, current_pos + text_len, device=device
                )
                pos[:, base + cursor : base + offset] = idx.unsqueeze(0)
                current_pos += text_len
            if image_idx >= len(grids):
                raise ValueError("image_grid_thw has fewer images than placeholders")
            t, h, w = (int(v) for v in grids[image_idx])
            image_idx += 1
            gh = h // spatial_merge_size
            gw = w // spatial_merge_size
            if n_tokens != t * gh * gw:
                raise ValueError(
                    f"placeholder length {n_tokens} does not match grid "
                    f"tokens {t}x{gh}x{gw}"
                )
            seg = slice(base + offset, base + offset + n_tokens)
            # Images have a single temporal frame: t is constant.
            pos[0, seg] = current_pos
            pos[1, seg] = (
                torch.arange(gh, device=device) + current_pos
            ).repeat_interleave(gw)
            pos[2, seg] = (torch.arange(gw, device=device) + current_pos).repeat(gh)
            current_pos += max(gh, gw)
            cursor = offset + n_tokens
        if cursor < seq_len:
            idx = torch.arange(
                current_pos, current_pos + (seq_len - cursor), device=device
            )
            pos[:, base + cursor : base + seq_len] = idx.unsqueeze(0)
        base += seq_len
    return pos


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
