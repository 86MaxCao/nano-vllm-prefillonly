"""Model loader for different model types (embedding, reranker, multimodal, text-only)."""
import importlib
import logging
import sys

import torch
from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.utils.loader import load_model

logger = logging.getLogger(__name__)


# Optional model implementations, imported on demand. Mapping each public symbol
# to its module lets an unrelated import failure (syntax error, missing
# third-party package) surface with its original traceback instead of being
# reported as "model not available".
_OPTIONAL_SYMBOLS = {
    "load_qwen3_vl_model": "nanovllm.models.qwen3_vl",
    "load_qwen2_vl_model": "nanovllm.models.qwen2_vl",
    "load_qwen2_5_vl_model": "nanovllm.models.qwen2_5_vl",
    "load_llavanext_model": "nanovllm.models.llavanext",
    "Qwen3Reranker": "nanovllm.models.qwen3_reranker",
    "GemmaReranker": "nanovllm.models.gemma_reranker",
    "GemmaForCausalLM": "nanovllm.models.gemma",
    "JinaRerankerV3": "nanovllm.models.jina_reranker_v3",
    "Gemma2Embedding": "nanovllm.models.gemma2_embedding",
    "Qwen3Embedding": "nanovllm.models.qwen3_embedding",
    "LLaVANextEmbedding": "nanovllm.models.llavanext_embedding",
    "Qwen2VLGmeEmbedding": "nanovllm.models.qwen2_vl_gme_embedding",
    "GmeQwen2VLConfig": "nanovllm.models.qwen2_vl_gme_embedding",
    "JinaEmbeddingsV4": "nanovllm.models.jina_v4_embedding",
    "JinaRerankerM0": "nanovllm.models.jina_m0_reranker",
    "Qwen3VLEmbedding": "nanovllm.models.qwen3_vl_embedding",
    "Qwen3VLReranker": "nanovllm.models.qwen3_vl_reranker",
    "load_qwen3_5_model": "nanovllm.models.qwen3_5",
    "Qwen3_5TextForCausalLM": "nanovllm.models.qwen3_5",
    "Qwen3NextForCausalLM": "nanovllm.models.qwen3_next",
}

# Each availability flag maps to a symbol whose import decides it.
_AVAILABILITY_FLAGS = {
    "QWEN3_VL_AVAILABLE": "load_qwen3_vl_model",
    "QWEN2_VL_AVAILABLE": "load_qwen2_vl_model",
    "QWEN2_5_VL_AVAILABLE": "load_qwen2_5_vl_model",
    "LLAVANEXT_AVAILABLE": "load_llavanext_model",
    "QWEN3_RERANKER_AVAILABLE": "Qwen3Reranker",
    "GEMMA_AVAILABLE": "GemmaReranker",
    "JINA_V3_AVAILABLE": "JinaRerankerV3",
    "GEMMA2_AVAILABLE": "Gemma2Embedding",
    "QWEN3_EMBEDDING_AVAILABLE": "Qwen3Embedding",
    "LLAVANEXT_EMBEDDING_AVAILABLE": "LLaVANextEmbedding",
    "QWEN2VL_GME_AVAILABLE": "Qwen2VLGmeEmbedding",
    "JINA_V4_AVAILABLE": "JinaEmbeddingsV4",
    "JINA_M0_AVAILABLE": "JinaRerankerM0",
    "QWEN3_VL_EMBEDDING_AVAILABLE": "Qwen3VLEmbedding",
    "QWEN3_VL_RERANKER_AVAILABLE": "Qwen3VLReranker",
    "QWEN3_5_AVAILABLE": "load_qwen3_5_model",
    "QWEN3_NEXT_AVAILABLE": "Qwen3NextForCausalLM",
}

_import_cache: dict[str, object] = {}
_import_errors: dict[str, Exception] = {}


def _try_import(symbol: str):
    """Import an optional symbol, remembering the failure reason."""
    if symbol in _import_cache:
        return _import_cache[symbol]
    if symbol in _import_errors:
        return None
    module_name = _OPTIONAL_SYMBOLS[symbol]
    try:
        obj = getattr(importlib.import_module(module_name), symbol)
    except Exception as exc:  # noqa: BLE001 - reported verbatim below
        _import_errors[symbol] = exc
        logger.warning(
            "Optional model %s is unavailable: %s: %s",
            symbol,
            type(exc).__name__,
            exc,
        )
        return None
    _import_cache[symbol] = obj
    return obj


def __getattr__(name: str):
    if name == "MULTIMODAL_AVAILABLE":
        return any(
            _try_import(_AVAILABILITY_FLAGS[flag]) is not None
            for flag in (
                "QWEN3_VL_AVAILABLE",
                "QWEN2_VL_AVAILABLE",
                "QWEN2_5_VL_AVAILABLE",
                "LLAVANEXT_AVAILABLE",
                "QWEN3_5_AVAILABLE",
            )
        )
    if name in _AVAILABILITY_FLAGS:
        return _try_import(_AVAILABILITY_FLAGS[name]) is not None
    if name in _OPTIONAL_SYMBOLS:
        obj = _try_import(name)
        if obj is None:
            raise ImportError(
                f"{name} could not be imported from "
                f"{_OPTIONAL_SYMBOLS[name]}"
            ) from _import_errors[name]
        return obj
    raise AttributeError(name)


def import_error_for(symbol: str) -> Exception | None:
    """Return the exception that made `symbol` unavailable, if any."""
    _try_import(symbol)
    return _import_errors.get(symbol)


from nanovllm.models.qwen3 import Qwen3ForCausalLM

try:
    from transformers import Gemma2Config
except ImportError:  # transformers < 4.42 exposed only GemmaConfig
    from transformers import GemmaConfig as Gemma2Config


def get_torch_dtype(hf_config) -> torch.dtype:
    """Extract torch dtype from HuggingFace config."""
    torch_dtype = getattr(hf_config, "torch_dtype", None)
    if torch_dtype is None and hasattr(hf_config, "text_config"):
        torch_dtype = getattr(hf_config.text_config, "torch_dtype", None)
    if isinstance(torch_dtype, str):
        resolved_dtype = getattr(torch, torch_dtype, None)
        if resolved_dtype is None:
            alias_map = {
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
                "float16": torch.float16,
            }
            resolved_dtype = alias_map.get(torch_dtype.lower())
        torch_dtype = resolved_dtype
    return torch_dtype if torch_dtype is not None else torch.float16


def get_target_dtype_for_embedding_reranker(hf_config) -> torch.dtype:
    """Determine target dtype for embedding/reranker models."""
    torch_dtype = get_torch_dtype(hf_config)
    if isinstance(torch_dtype, str):
        if torch_dtype == "float16":
            torch_dtype = torch.float16
        elif torch_dtype == "bfloat16":
            torch_dtype = torch.bfloat16
        elif torch_dtype == "float32":
            torch_dtype = torch.float32
        else:
            torch_dtype = None
    if torch_dtype == torch.float32 or torch_dtype is None:
        return torch.float16
    elif torch_dtype in (torch.float16, torch.bfloat16):
        return torch_dtype
    else:
        return torch.float16


def infer_embedding_type(config: Config, hf_config) -> str | None:
    """Auto-detect embedding_type from model path or config."""
    model_path_lower = config.model.lower()
    if "qwen3" in model_path_lower and ("vl" in model_path_lower or "vision" in model_path_lower):
        return "qwen3_vl"
    elif "qwen3" in model_path_lower:
        return "qwen3"
    elif "gemma2" in model_path_lower:
        return "gemma2"
    elif "jina" in model_path_lower and "v4" in model_path_lower:
        return "jina_v4"
    elif "jina" in model_path_lower and "v3" in model_path_lower:
        return "jina_v3"
    elif "llavanext" in model_path_lower:
        return "llavanext"
    elif "qwen2" in model_path_lower and "vl" in model_path_lower and "gme" in model_path_lower:
        return "qwen2_vl_gme"
    else:
        model_type = getattr(hf_config, "model_type", "").lower()
        if "qwen3" in model_type and ("vl" in model_type or hasattr(hf_config, "vision_config")):
            return "qwen3_vl"
        elif "qwen3" in model_type:
            return "qwen3"
        elif "gemma2" in model_type:
            return "gemma2"
    return None


def infer_reranker_type(config: Config, hf_config) -> str | None:
    """Auto-detect reranker_type from model path or config."""
    model_path_lower = config.model.lower()
    if "qwen3" in model_path_lower and ("vl" in model_path_lower or "vision" in model_path_lower):
        return "qwen3_vl"
    elif "qwen3" in model_path_lower:
        return "qwen3"
    elif "gemma" in model_path_lower and "rerank" in model_path_lower:
        return "gemma"
    elif "jina" in model_path_lower and "m0" in model_path_lower:
        return "jina_m0"
    elif "jina" in model_path_lower and "v3" in model_path_lower:
        return "jina_v3"
    else:
        model_type = getattr(hf_config, "model_type", "").lower()
        if "qwen3" in model_type and ("vl" in model_type or hasattr(hf_config, "vision_config")):
            return "qwen3_vl"
        elif "qwen3" in model_type:
            return "qwen3"
    return None


def infer_multimodal_model_type(config: Config, hf_config) -> str | None:
    """Auto-detect multimodal_model_type from model path or config."""
    model_path_lower = config.model.lower()
    model_type = getattr(hf_config, "model_type", "").lower()

    if "qwen3_5" in model_path_lower or "qwen3.5" in model_path_lower:
        return "qwen3_5"
    elif "qwen3" in model_path_lower and "vl" in model_path_lower:
        return "qwen3_vl"
    elif "qwen2_5" in model_path_lower and "vl" in model_path_lower:
        return "qwen2_5_vl"
    elif "qwen2" in model_path_lower and "vl" in model_path_lower:
        return "qwen2_vl"
    elif "llava" in model_path_lower:
        return "llavanext"
    elif "qwen3_5" in model_type:
        return "qwen3_5"
    elif "qwen3_vl" in model_type:
        return "qwen3_vl"
    elif "qwen2_5_vl" in model_type:
        return "qwen2_5_vl"
    elif "qwen2_vl" in model_type:
        return "qwen2_vl"
    elif "llava" in model_type:
        return "llavanext"
    return None


def create_qwen3_vl_name_mapping():
    """Create name mapping function for Qwen3VL models.

    This mapping is used for Qwen3VLEmbedding and Qwen3VLReranker,
    which both inherit from Qwen3VLForConditionalGeneration.

    Key mapping rules:
    - model.language_model.{layers,embed_tokens,norm,rotary_emb}.* 
      → language_model.model.* (text model sub-params need .model. level)
    - model.language_model.model.* → language_model.model.* (already has model level)
    - model.language_model.lm_head.* → language_model.lm_head.*
    - model.visual.* → visual.vision.*
    """
    def name_mapping(weight_name: str) -> str | None:
        if weight_name.startswith("model.language_model."):
            sub_name = weight_name[len("model.language_model."):]
            # Text model sub-params that live under language_model.model.*
            text_model_prefixes = (
                "model.",
                "embed_tokens.",
                "layers.",
                "norm.",
                "rotary_emb.",
            )
            if sub_name.startswith(text_model_prefixes):
                if sub_name.startswith("model."):
                    # Already has model level: model.xxx → language_model.xxx
                    sub_name = sub_name[len("model."):]
                # These params live under language_model.model.*, so add .model.
                sub_name = "language_model.model." + sub_name
            elif sub_name.startswith("lm_head."):
                sub_name = "language_model.lm_head." + sub_name[len("lm_head."):]
            else:
                sub_name = "language_model." + sub_name
            return sub_name
        if weight_name.startswith("model.visual."):
            sub_name = weight_name[len("model.visual."):]
            return "visual.vision." + sub_name
        return None
    return name_mapping


def create_legacy_qwenvl_name_mapping(
    skip_lm_head: bool = True,
    extra_prefixes: tuple[str, ...] = (),
):
    """Map legacy flat Qwen-VL checkpoint names onto current module paths.

    Older Qwen-VL derived checkpoints store text weights at the top level
    (``model.layers.*``, ``visual.*``) while current transformers nests them
    under ``model.language_model.*`` and ``model.visual.*``.

    Args:
        skip_lm_head: Drop ``lm_head.*`` tensors, for heads replaced downstream.
        extra_prefixes: Top-level prefixes to pass through unchanged, e.g.
            task-specific projection or scoring heads.
    """
    text_prefixes = ("model.embed_tokens.", "model.layers.", "model.norm.")

    def name_mapping(weight_name: str) -> str | None:
        if skip_lm_head and weight_name.startswith("lm_head."):
            return None
        if weight_name.startswith(text_prefixes):
            return "model.language_model." + weight_name[len("model."):]
        if weight_name.startswith("visual."):
            return "model." + weight_name
        if weight_name.startswith(("model.language_model.", "model.visual.")):
            # Already in the nested layout.
            return weight_name
        if extra_prefixes and weight_name.startswith(extra_prefixes):
            return weight_name
        return None

    return name_mapping


def create_jina_m0_name_mapping():
    """Name mapping for Jina Reranker M0 (legacy Qwen2-VL layout + score head)."""
    return create_legacy_qwenvl_name_mapping(
        skip_lm_head=True, extra_prefixes=("score.",)
    )


def create_jina_v4_name_mapping():
    """Name mapping for Jina Embeddings V4 (legacy layout + projector head)."""
    return create_legacy_qwenvl_name_mapping(
        skip_lm_head=True, extra_prefixes=("multi_vector_projector.",)
    )


# Optional model symbols are reached through the module object so that each
# attribute lookup goes through __getattr__ and triggers the lazy import.
_lazy = sys.modules[__name__]


class ModelLoader:
    """Handles loading of different model types."""

    @staticmethod
    def load_embedding_model(config: Config, hf_config, embedding_type: str,
                           pooling_type: str, normalize_embeddings: bool,
                           target_dtype: torch.dtype | None) -> torch.nn.Module:
        """Load an embedding model."""
        if embedding_type == "gemma2" and _lazy.GEMMA2_AVAILABLE:
            try:
                gemma2_config = Gemma2Config.from_pretrained(config.model)
            except Exception:
                gemma2_config = hf_config
            model = _lazy.Gemma2Embedding(
                gemma2_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "qwen3" and _lazy.QWEN3_EMBEDDING_AVAILABLE:
            text_config = getattr(hf_config, "text_config", hf_config)
            from transformers import Qwen3Config
            qwen3_config = Qwen3Config.from_dict(text_config.to_dict())
            model = _lazy.Qwen3Embedding(
                qwen3_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "llavanext" and _lazy.LLAVANEXT_EMBEDDING_AVAILABLE:
            model = _lazy.LLaVANextEmbedding(
                hf_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "qwen2_vl_gme" and _lazy.QWEN2VL_GME_AVAILABLE:
            gme_config = _lazy.GmeQwen2VLConfig.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
            model = _lazy.Qwen2VLGmeEmbedding(
                gme_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "jina_v4" and _lazy.JINA_V4_AVAILABLE:
            jina_v4_config = AutoConfig.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
            model = _lazy.JinaEmbeddingsV4(
                jina_v4_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(
                model, config.model, name_mapping=create_jina_v4_name_mapping()
            )
            return model

        elif embedding_type == "qwen3_vl" and _lazy.QWEN3_VL_EMBEDDING_AVAILABLE:
            qwen3_vl_config = AutoConfig.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
            embedding_model = _lazy.Qwen3VLEmbedding(
                qwen3_vl_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                embedding_model = embedding_model.to(target_dtype)
            name_mapping = create_qwen3_vl_name_mapping()
            load_model(embedding_model, config.model, name_mapping=name_mapping)
            return embedding_model

        else:
            raise ValueError(f"Unsupported embedding type: {embedding_type}")

    @staticmethod
    def load_reranker_model(config: Config, hf_config, reranker_type: str,
                          target_dtype: torch.dtype | None) -> torch.nn.Module:
        """Load a reranker model."""
        text_config = getattr(hf_config, "text_config", hf_config)
        # Per-type defaults for is_original and classifier_from_token.
        # Config values still override these when explicitly set.
        is_original = getattr(config, "is_original_qwen3_reranker", None)
        classifier_tokens = getattr(config, "classifier_from_token", None)

        if reranker_type == "qwen3":
            # qwen3 text rerankers always use yes/no token logits
            if is_original is None:
                is_original = True
            if classifier_tokens is None:
                classifier_tokens = ["no", "yes"]
            model = _lazy.Qwen3Reranker(
                text_config,
                is_original_reranker=is_original,
                classifier_from_token=classifier_tokens,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)

            if is_original:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.model)
                model.convert_from_original_reranker(tokenizer)
            return model

        elif reranker_type == "gemma" and _lazy.GEMMA_AVAILABLE:
            # gemma rerankers always use "Yes" token logit
            if is_original is None:
                is_original = True
            if classifier_tokens is None:
                classifier_tokens = ["Yes"]
            model = _lazy.GemmaReranker(
                text_config,
                is_original_reranker=is_original,
                classifier_from_token=classifier_tokens,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)

            if is_original:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.model)
                model.convert_from_original_reranker(tokenizer)
            return model

        elif reranker_type == "jina_v3" and _lazy.JINA_V3_AVAILABLE:
            projector_dim = getattr(config, "projector_dim", 512)
            use_flex_attention = getattr(config, "use_flex_attention", True)
            model = _lazy.JinaRerankerV3(
                text_config,
                projector_dim=projector_dim,
                use_flex_attention=use_flex_attention,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif reranker_type == "jina_m0" and _lazy.JINA_M0_AVAILABLE:
            jina_m0_config = AutoConfig.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
            model = _lazy.JinaRerankerM0(jina_m0_config)
            if target_dtype is not None:
                model = model.to(target_dtype)
            name_mapping = create_jina_m0_name_mapping()
            load_model(model, config.model, name_mapping=name_mapping)
            return model

        elif reranker_type == "qwen3_vl" and _lazy.QWEN3_VL_RERANKER_AVAILABLE:
            is_original = True  # VL rerankers always use yes/no token logits
            classifier_tokens = getattr(config, "classifier_from_token", ["no", "yes"])
            qwen3_vl_config = AutoConfig.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
            model = _lazy.Qwen3VLReranker(
                qwen3_vl_config,
                is_original_reranker=is_original,
                classifier_from_token=classifier_tokens,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            name_mapping = create_qwen3_vl_name_mapping()
            load_model(model, config.model, name_mapping=name_mapping)

            if is_original:
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=config.trust_remote_code)
                model.convert_from_original_reranker(tokenizer)
            return model

        else:
            raise ValueError(f"Unsupported reranker type: {reranker_type}")

    @staticmethod
    def load_multimodal_model(config: Config, multimodal_model_type: str) -> torch.nn.Module:
        """Load a multimodal model."""
        if multimodal_model_type == "qwen3_5" and _lazy.QWEN3_5_AVAILABLE:
            return _lazy.load_qwen3_5_model(config.model, config)
        elif multimodal_model_type == "qwen3_vl" and _lazy.QWEN3_VL_AVAILABLE:
            return _lazy.load_qwen3_vl_model(config.model, config)
        elif multimodal_model_type == "qwen2_vl" and _lazy.QWEN2_VL_AVAILABLE:
            return _lazy.load_qwen2_vl_model(config.model, config)
        elif multimodal_model_type == "qwen2_5_vl" and _lazy.QWEN2_5_VL_AVAILABLE:
            return _lazy.load_qwen2_5_vl_model(config.model, config)
        elif multimodal_model_type == "llavanext" and _lazy.LLAVANEXT_AVAILABLE:
            return _lazy.load_llavanext_model(config.model, config)
        else:
            raise ValueError(
                f"Unsupported multimodal_model_type: {multimodal_model_type} "
                f"or model not available"
            )

    @staticmethod
    def load_text_model(config: Config, hf_config) -> torch.nn.Module:
        """Load a text-only model."""
        if hasattr(hf_config, 'model_type') and hf_config.model_type == 'gemma' and _lazy.GEMMA_AVAILABLE:
            model = _lazy.GemmaForCausalLM(hf_config)
            load_model(model, config.model)
            return model
        else:
            text_config = getattr(hf_config, "text_config", hf_config)
            # Propagate tie_word_embeddings from parent config if missing on text_config
            if not hasattr(text_config, "tie_word_embeddings") and hasattr(hf_config, "tie_word_embeddings"):
                text_config.tie_word_embeddings = hf_config.tie_word_embeddings
            text_model_type = getattr(text_config, "model_type", None)
            # Qwen3.5 text models use Qwen3Next or _lazy.Qwen3_5TextForCausalLM
            if text_model_type in ("qwen3_next", "qwen3_5", "qwen3_5_moe"):
                if _lazy.QWEN3_NEXT_AVAILABLE:
                    model = _lazy.Qwen3NextForCausalLM(text_config)
                    load_model(model, config.model)
                    return model
                elif _lazy.QWEN3_5_AVAILABLE:
                    model = _lazy.Qwen3_5TextForCausalLM(text_config)
                    load_model(model, config.model)
                    return model
            # Default: Qwen3
            model = Qwen3ForCausalLM(text_config)
            load_model(model, config.model)
            return model
