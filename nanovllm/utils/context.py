from dataclasses import dataclass
import torch


@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    # Prefix-cache capture/replay (feat/prefill-kv-cache). Mutually exclusive:
    # capture runs during prefix creation, replay during suffix forward.
    kv_capture: list | None = None        # attention layers append (k, v) here
    prefix_kv: list | None = None         # per layer: list per sequence of (k, v)
    prefix_kv_bounds: list | None = None  # cumulative suffix boundaries (CPU ints)
    attn_call_index: int = 0              # replay: which layer is calling
    # P3 GDN fork placeholders (same call-order discipline as attention).
    gdn_capture: list | None = None
    gdn_initial: list | None = None
    gdn_call_index: int = 0

_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0, slot_mapping=None, context_lens=None, block_tables=None, kv_capture=None, prefix_kv=None, prefix_kv_bounds=None, gdn_capture=None, gdn_initial=None):
    global _CONTEXT
    assert kv_capture is None or prefix_kv is None, "capture and replay are mutually exclusive"
    assert gdn_capture is None or gdn_initial is None, "capture and replay are mutually exclusive"
    _CONTEXT = Context(is_prefill, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, context_lens, block_tables, kv_capture, prefix_kv, prefix_kv_bounds, 0, gdn_capture, gdn_initial, 0)

def reset_context():
    global _CONTEXT
    _CONTEXT = Context()
