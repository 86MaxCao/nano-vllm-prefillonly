"""Prefix KV / GDN state store for prefix-reuse prefill (SemIf serial/shared).

A PrefixEntry holds everything a suffix forward needs to attend over an
already-computed prefix: per-attention-layer post-RoPE K/V in varlen layout
(single sequence), plus per-GDN-layer conv/recurrent states (P3). Entries are
immutable once created; ``fork`` physically deep-copies so branches can never
pollute each other. Memory is accounted against a fixed quota; overflow raises
instead of evicting, matching SemIf's create -> fork -> score -> release usage.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

import torch


@dataclass
class PrefixEntry:
    prefix_len: int
    # Per attention layer: (k, v) of shape [prefix_len, num_kv_heads, head_dim].
    kv: list[tuple[torch.Tensor, torch.Tensor]]
    # Per GDN layer: (conv_state, recurrent_state). Empty for pure-attention
    # models; populated by the P3 GDN capture path.
    gdn: list[tuple[torch.Tensor, torch.Tensor]]
    # P4: M-RoPE end positions etc. for multimodal prefixes.
    multimodal_meta: dict | None = None

    @property
    def nbytes(self) -> int:
        total = 0
        for key, value in self.kv:
            total += key.nbytes + value.nbytes
        for conv_state, rec_state in self.gdn:
            total += conv_state.nbytes + rec_state.nbytes
        return total


class PrefixCacheStore:
    """Handle-addressed store with byte-quota accounting."""

    def __init__(self, max_bytes: int):
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.max_bytes = max_bytes
        self._entries: dict[int, PrefixEntry] = {}
        self._ids = itertools.count(1)
        self._used = 0

    @property
    def used_bytes(self) -> int:
        return self._used

    def __len__(self) -> int:
        return len(self._entries)

    def _check_quota(self, incoming: int) -> None:
        if self._used + incoming > self.max_bytes:
            raise RuntimeError(
                f"prefix cache quota exceeded: used {self._used} + incoming "
                f"{incoming} > max {self.max_bytes} bytes; release handles first"
            )

    def create(self, entry: PrefixEntry) -> int:
        if entry.prefix_len < 1:
            raise ValueError("prefix_len must be positive")
        self._check_quota(entry.nbytes)
        handle = next(self._ids)
        self._entries[handle] = entry
        self._used += entry.nbytes
        return handle

    def get(self, handle: int) -> PrefixEntry:
        try:
            return self._entries[handle]
        except KeyError:
            raise ValueError(f"unknown prefix handle {handle}") from None

    def fork(self, handle: int, n: int) -> list[int]:
        """Deep-copy an entry n times; returns the new handles.

        Quota is checked up front so a failed fork leaves the store untouched.
        """
        if n < 1:
            raise ValueError(f"fork count must be positive, got {n}")
        src = self.get(handle)
        self._check_quota(src.nbytes * n)
        handles = []
        for _ in range(n):
            entry = PrefixEntry(
                prefix_len=src.prefix_len,
                kv=[(k.clone(), v.clone()) for k, v in src.kv],
                gdn=[(c.clone(), r.clone()) for c, r in src.gdn],
                multimodal_meta=src.multimodal_meta,
            )
            fork_handle = next(self._ids)
            self._entries[fork_handle] = entry
            self._used += entry.nbytes
            handles.append(fork_handle)
        return handles

    def release(self, *handles: int) -> None:
        for handle in handles:
            entry = self._entries.pop(handle, None)
            if entry is None:
                raise ValueError(f"unknown prefix handle {handle}")
            self._used -= entry.nbytes
