import os
from dataclasses import dataclass, field

from transformers import AutoConfig


def _resolve_model_path(model: str, trust_remote_code: bool, revision: str | None = None) -> str:
    """Return a local directory for `model`, downloading from the Hub if needed.

    Accepting Hub ids keeps the documented ``LLM("Qwen/Qwen3-0.6B")`` usage
    working instead of failing on a local-directory assertion. Pass
    ``revision`` to pin a Hub commit; ``None`` keeps the legacy
    latest-available behaviour.
    """
    if os.path.isdir(model):
        return model
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise ValueError(
            f"Model path {model!r} is not a local directory and huggingface_hub "
            f"is not installed, so it cannot be downloaded. Install "
            f"huggingface_hub or pass a local path."
        ) from exc
    return snapshot_download(model, revision=revision)


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    is_multimodal: bool | None = None  # None = auto-detect from the HF config
    multimodal_model_type: str | None = None  # None = auto-detect
    is_reranker: bool = False  # Enable reranker support
    reranker_type: str | None = None  # "qwen3", "qwen3_vl", "gemma", "jina_v3", "jina_m0"
    is_original_qwen3_reranker: bool | None = None  # Auto-detected per reranker type; set True/False to override
    classifier_from_token: list[str] | None = None  # e.g., ["no", "yes"] for Qwen3-Reranker, ["Yes"] for Gemma-Reranker
    projector_dim: int = 512  # For jina-reranker-v3
    use_flex_attention: bool = True  # Retained for compatibility; jina_v3 runs on standard causal attention
    is_embedding: bool = False  # Enable embedding model support
    embedding_type: str | None = None  # "gemma2", "qwen3", "qwen3_vl", "jina_v4", "llavanext", "qwen2_vl_gme"
    pooling_type: str = "LAST"  # Pooling type: "LAST", "MEAN", "CLS"
    normalize_embeddings: bool = True  # Whether to normalize embeddings
    # Prefill-only optimizations
    prefill_only_mode: bool | None = None  # None = auto (on for embedding/reranker)
    max_tokens_hint: int | None = None  # Longest generation you intend to request
    max_prefill_batch_size: int = 1024  # Max batch size for prefill-only mode
    max_prefix_cache_bytes: int = 4 << 30  # Quota for the prefix KV/GDN state store
    trust_remote_code: bool = False  # Trust remote code for custom models
    revision: str | None = None  # Pin a Hub commit; local paths record it as a manifest label
    hf_config: AutoConfig | None = None
    eos: int | list[int] = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256 != 0:
            raise ValueError(
                "kvcache_block_size must be a positive multiple of 256, got "
                f"{self.kvcache_block_size}"
            )
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError(
                "tensor_parallel_size must be between 1 and 8, got "
                f"{self.tensor_parallel_size}"
            )
        self._original_model = self.model
        self.model = _resolve_model_path(
            self.model, self.trust_remote_code, self.revision
        )
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=self.trust_remote_code
        )
        self.model_revision = self._resolve_model_revision()

        # Multimodal models (e.g. Qwen3-VL) store the text settings in
        # hf_config.text_config.
        text_config = getattr(self.hf_config, "text_config", self.hf_config)

        max_position_embeddings = getattr(text_config, "max_position_embeddings", None)
        if max_position_embeddings is not None:
            self.max_model_len = min(self.max_model_len, max_position_embeddings)

        # eos may be defined within the text config, and may be a list of ids.
        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is None:
            eos_token_id = getattr(self.hf_config, "eos_token_id", None)
        if eos_token_id is not None:
            self.eos = eos_token_id

        if self.max_num_batched_tokens < self.max_model_len:
            raise ValueError(
                f"max_num_batched_tokens ({self.max_num_batched_tokens}) must "
                f"be >= max_model_len ({self.max_model_len}); otherwise a "
                "max-length prompt could never be scheduled."
            )

        self._resolve_multimodal()

        # Embedding and reranker models produce a pooled vector or a score from
        # a single forward pass, so decoding never happens.
        if self.is_embedding or self.is_reranker:
            self.prefill_only_mode = True
        elif self.prefill_only_mode is None:
            # A caller that only ever asks for one token also never decodes, so
            # the KV cache can be skipped entirely.
            self.prefill_only_mode = self.max_tokens_hint == 1

    def _resolve_model_revision(self) -> str | None:
        """Audit label for the exact weights in use.

        Local directories keep the caller-supplied revision string (a
        manifest label). Hub downloads resolve the cached commit hash so a
        floating ``revision=None`` still records what actually loaded.
        """
        if os.path.isdir(self._original_model):
            return self.revision
        try:
            from huggingface_hub import snapshot_download
            path = snapshot_download(
                self._original_model, revision=self.revision
            )
            commit = os.path.basename(path)
            if len(commit) == 40 and all(c in "0123456789abcdef" for c in commit):
                return commit
        except Exception:
            pass
        return self.revision

    def _resolve_multimodal(self):
        """Auto-detect multimodal support unless the caller was explicit.

        A vision config that goes unnoticed used to route VLMs to the text-only
        loader, which silently produced a model with no usable weights.
        """
        detected_type = self._detect_multimodal_type()

        if self.is_multimodal is None:
            self.is_multimodal = detected_type is not None

        if self.is_multimodal and self.multimodal_model_type is None:
            if detected_type is None:
                raise ValueError(
                    "is_multimodal=True but multimodal_model_type could not be "
                    "inferred from the config. Pass it explicitly (one of: "
                    "qwen3_vl, qwen2_5_vl, qwen2_vl, qwen3_5, llavanext)."
                )
            self.multimodal_model_type = detected_type

    def _detect_multimodal_type(self) -> str | None:
        """Infer the multimodal family from the HF config, then the path."""
        if not hasattr(self.hf_config, "vision_config"):
            return None

        model_type = (getattr(self.hf_config, "model_type", None) or "").lower()
        by_model_type = (
            ("qwen3_5", "qwen3_5"),
            ("qwen3_vl", "qwen3_vl"),
            ("qwen2_5_vl", "qwen2_5_vl"),
            ("qwen2_vl", "qwen2_vl"),
            ("llava", "llavanext"),
        )
        for needle, family in by_model_type:
            if needle in model_type:
                return family

        path = self.model.lower()
        by_path = (
            (("qwen3_5", "qwen3.5"), "qwen3_5"),
            (("qwen3-vl", "qwen3_vl"), "qwen3_vl"),
            (("qwen2.5-vl", "qwen2_5_vl"), "qwen2_5_vl"),
            (("qwen2-vl", "qwen2_vl"), "qwen2_vl"),
            (("llava",), "llavanext"),
        )
        for needles, family in by_path:
            if any(n in path for n in needles):
                return family
        return None

    @property
    def eos_token_ids(self) -> set[int]:
        """Normalise `eos` to a set, since configs may expose a list of ids."""
        if isinstance(self.eos, (list, tuple, set)):
            return {int(t) for t in self.eos}
        return {int(self.eos)}
