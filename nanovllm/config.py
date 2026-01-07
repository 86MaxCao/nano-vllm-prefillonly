import os
from dataclasses import dataclass
from transformers import AutoConfig


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    is_multimodal: bool = False  # Enable multimodal support
    multimodal_model_type: str = "qwen3_vl"  # "qwen3_vl", "qwen2_vl", "qwen2_5_vl", "llavanext"
    is_reranker: bool = False  # Enable reranker support
    reranker_type: str | None = None  # "qwen3", "gemma", "jina_v3", etc.
    is_original_qwen3_reranker: bool = False  # For original Qwen3-Reranker or Gemma-Reranker
    classifier_from_token: list[str] | None = None  # e.g., ["no", "yes"] for Qwen3-Reranker, ["Yes"] for Gemma-Reranker
    projector_dim: int = 512  # For jina-reranker-v3
    use_flex_attention: bool = True  # Use FlexAttention for listwise rerankers (jina_v3)
    is_embedding: bool = False  # Enable embedding model support
    embedding_type: str | None = None  # "gemma2", "qwen3", "jina_v3", etc.
    pooling_type: str = "LAST"  # Pooling type: "LAST", "MEAN", "CLS"
    normalize_embeddings: bool = True  # Whether to normalize embeddings
    # Prefill-only optimizations
    prefill_only_mode: bool = False  # Enable prefill-only mode (skip decode phase)
    max_prefill_batch_size: int = 1024  # Max batch size for prefill-only mode
    single_token_mode: bool = False  # Optimize for single token generation
    trust_remote_code: bool = False  # Trust remote code for custom models
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1

    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        trust_remote_code = getattr(self, "trust_remote_code", False)
        self.hf_config = AutoConfig.from_pretrained(
            self.model, trust_remote_code=trust_remote_code
        )

        # Multimodal models (e.g. Qwen3-VL) store the text settings in
        # hf_config.text_config.
        text_config = getattr(self.hf_config, "text_config", self.hf_config)

        max_position_embeddings = getattr(
            text_config,
            "max_position_embeddings",
            None,
        )
        if max_position_embeddings is not None:
            self.max_model_len = min(
                self.max_model_len,
                max_position_embeddings,
            )

        # eos may be defined within the text config
        eos_token_id = getattr(text_config, "eos_token_id", None)
        if eos_token_id is not None:
            self.eos = eos_token_id

        assert self.max_num_batched_tokens >= self.max_model_len
        
        # Auto-enable prefill_only_mode for embedding/reranker models
        if self.is_embedding or self.is_reranker:
            self.prefill_only_mode = True
        
        # Auto-enable single_token_mode if max_tokens would be 1
        # (This will be checked per-request in SamplingParams)
