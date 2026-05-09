"""Model loader for different model types (embedding, reranker, multimodal, text-only)."""
import logging
import torch
from transformers import AutoConfig

from nanovllm.config import Config
from nanovllm.utils.loader import load_model

logger = logging.getLogger(__name__)

# Import model classes with availability checks
try:
    from nanovllm.models.qwen3_vl import load_qwen3_vl_model
    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False

try:
    from nanovllm.models.qwen2_vl import load_qwen2_vl_model
    QWEN2_VL_AVAILABLE = True
except ImportError:
    QWEN2_VL_AVAILABLE = False

try:
    from nanovllm.models.qwen2_5_vl import load_qwen2_5_vl_model
    QWEN2_5_VL_AVAILABLE = True
except ImportError:
    QWEN2_5_VL_AVAILABLE = False

try:
    from nanovllm.models.llavanext import load_llavanext_model
    LLAVANEXT_AVAILABLE = True
except ImportError:
    LLAVANEXT_AVAILABLE = False

try:
    from nanovllm.models.qwen3_reranker import Qwen3Reranker
    QWEN3_RERANKER_AVAILABLE = True
except ImportError:
    QWEN3_RERANKER_AVAILABLE = False

try:
    from nanovllm.models.gemma_reranker import GemmaReranker
    from nanovllm.models.gemma import GemmaForCausalLM
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False

try:
    from nanovllm.models.jina_reranker_v3 import JinaRerankerV3
    JINA_V3_AVAILABLE = True
except ImportError:
    JINA_V3_AVAILABLE = False

try:
    from nanovllm.models.gemma2_embedding import Gemma2Embedding
    try:
        from transformers import Gemma2Config
    except ImportError:
        from transformers import GemmaConfig as Gemma2Config
    GEMMA2_AVAILABLE = True
except ImportError as e:
    GEMMA2_AVAILABLE = False

try:
    from nanovllm.models.qwen3_embedding import Qwen3Embedding
    QWEN3_EMBEDDING_AVAILABLE = True
except ImportError:
    QWEN3_EMBEDDING_AVAILABLE = False

try:
    from nanovllm.models.llavanext_embedding import LLaVANextEmbedding
    LLAVANEXT_EMBEDDING_AVAILABLE = True
except ImportError:
    LLAVANEXT_EMBEDDING_AVAILABLE = False

try:
    from nanovllm.models.qwen2_vl_gme_embedding import Qwen2VLGmeEmbedding, GmeQwen2VLConfig
    QWEN2VL_GME_AVAILABLE = True
except ImportError as e:
    QWEN2VL_GME_AVAILABLE = False

try:
    from nanovllm.models.jina_v4_embedding import JinaEmbeddingsV4
    JINA_V4_AVAILABLE = True
except ImportError as e:
    JINA_V4_AVAILABLE = False

try:
    from nanovllm.models.jina_m0_reranker import JinaRerankerM0
    JINA_M0_AVAILABLE = True
except ImportError as e:
    JINA_M0_AVAILABLE = False

try:
    from nanovllm.models.qwen3_vl_embedding import Qwen3VLEmbedding
    QWEN3_VL_EMBEDDING_AVAILABLE = True
except ImportError as e:
    QWEN3_VL_EMBEDDING_AVAILABLE = False

try:
    from nanovllm.models.qwen3_vl_reranker import Qwen3VLReranker
    QWEN3_VL_RERANKER_AVAILABLE = True
except ImportError as e:
    QWEN3_VL_RERANKER_AVAILABLE = False

from nanovllm.models.qwen3 import Qwen3ForCausalLM

MULTIMODAL_AVAILABLE = (
    QWEN3_VL_AVAILABLE or QWEN2_VL_AVAILABLE or
    QWEN2_5_VL_AVAILABLE or LLAVANEXT_AVAILABLE
)


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

    if "qwen3" in model_path_lower and "vl" in model_path_lower:
        return "qwen3_vl"
    elif "qwen2_5" in model_path_lower and "vl" in model_path_lower:
        return "qwen2_5_vl"
    elif "qwen2" in model_path_lower and "vl" in model_path_lower:
        return "qwen2_vl"
    elif "llava" in model_path_lower:
        return "llavanext"
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


def create_jina_m0_name_mapping():
    """Create name mapping function for Jina Reranker M0 model.

    The safetensors file uses flat names inherited from Qwen2VL's
    checkpoint format, while the transformers Qwen2VLForConditionalGeneration
    model prefixes text model params with model.language_model.*.

    Key mapping rules:
    - model.embed_tokens.* → model.language_model.embed_tokens.*
    - model.layers.* → model.language_model.layers.*
    - model.norm.* → model.language_model.norm.*
    - lm_head.* → skip (replaced by nn.Identity in JinaRerankerM0)
    - visual.* → model.visual.* (direct, just add model. prefix)
    - score.* → score.* (direct match, no transform)
    """
    def name_mapping(weight_name: str) -> str | None:
        # Skip lm_head — replaced by Identity in JinaRerankerM0
        if weight_name.startswith("lm_head."):
            return None
        # Text model params: model.{embed_tokens,layers,norm}.* 
        # → model.language_model.{embed_tokens,layers,norm}.*
        if weight_name.startswith("model.embed_tokens."):
            return "model.language_model." + weight_name[len("model."):]
        if weight_name.startswith("model.layers."):
            return "model.language_model." + weight_name[len("model."):]
        if weight_name.startswith("model.norm."):
            return "model.language_model." + weight_name[len("model."):]
        # Visual params: visual.* → model.visual.*
        if weight_name.startswith("visual."):
            return "model." + weight_name
        # Score MLP: score.* → score.* (direct match)
        if weight_name.startswith("score."):
            return weight_name
        return None
    return name_mapping


class ModelLoader:
    """Handles loading of different model types."""

    @staticmethod
    def load_embedding_model(config: Config, hf_config, embedding_type: str,
                           pooling_type: str, normalize_embeddings: bool,
                           target_dtype: torch.dtype | None) -> torch.nn.Module:
        """Load an embedding model."""
        if embedding_type == "gemma2" and GEMMA2_AVAILABLE:
            try:
                gemma2_config = Gemma2Config.from_pretrained(config.model)
            except Exception:
                gemma2_config = hf_config
            model = Gemma2Embedding(
                gemma2_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "qwen3" and QWEN3_EMBEDDING_AVAILABLE:
            text_config = getattr(hf_config, "text_config", hf_config)
            from transformers import Qwen3Config
            qwen3_config = Qwen3Config.from_dict(text_config.to_dict())
            model = Qwen3Embedding(
                qwen3_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "llavanext" and LLAVANEXT_EMBEDDING_AVAILABLE:
            model = LLaVANextEmbedding(
                hf_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "qwen2_vl_gme" and QWEN2VL_GME_AVAILABLE:
            gme_config = GmeQwen2VLConfig.from_pretrained(config.model, trust_remote_code=True)
            model = Qwen2VLGmeEmbedding(
                gme_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "jina_v4" and JINA_V4_AVAILABLE:
            jina_v4_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
            model = JinaEmbeddingsV4(
                jina_v4_config,
                pooling_type=pooling_type,
                normalize=normalize_embeddings,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif embedding_type == "qwen3_vl" and QWEN3_VL_EMBEDDING_AVAILABLE:
            qwen3_vl_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
            embedding_model = Qwen3VLEmbedding(
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
            model = Qwen3Reranker(
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

        elif reranker_type == "gemma" and GEMMA_AVAILABLE:
            # gemma rerankers always use "Yes" token logit
            if is_original is None:
                is_original = True
            if classifier_tokens is None:
                classifier_tokens = ["Yes"]
            model = GemmaReranker(
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

        elif reranker_type == "jina_v3" and JINA_V3_AVAILABLE:
            projector_dim = getattr(config, "projector_dim", 512)
            use_flex_attention = getattr(config, "use_flex_attention", True)
            model = JinaRerankerV3(
                text_config,
                projector_dim=projector_dim,
                use_flex_attention=use_flex_attention,
            )
            if target_dtype is not None:
                model = model.to(target_dtype)
            load_model(model, config.model)
            return model

        elif reranker_type == "jina_m0" and JINA_M0_AVAILABLE:
            jina_m0_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
            model = JinaRerankerM0(jina_m0_config)
            if target_dtype is not None:
                model = model.to(target_dtype)
            name_mapping = create_jina_m0_name_mapping()
            load_model(model, config.model, name_mapping=name_mapping)
            return model

        elif reranker_type == "qwen3_vl" and QWEN3_VL_RERANKER_AVAILABLE:
            is_original = True  # VL rerankers always use yes/no token logits
            classifier_tokens = getattr(config, "classifier_from_token", ["no", "yes"])
            qwen3_vl_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
            model = Qwen3VLReranker(
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
                tokenizer = AutoTokenizer.from_pretrained(config.model, trust_remote_code=True)
                model.convert_from_original_reranker(tokenizer)
            return model

        else:
            raise ValueError(f"Unsupported reranker type: {reranker_type}")

    @staticmethod
    def load_multimodal_model(config: Config, multimodal_model_type: str) -> torch.nn.Module:
        """Load a multimodal model."""
        if multimodal_model_type == "qwen3_vl" and QWEN3_VL_AVAILABLE:
            return load_qwen3_vl_model(config.model, config)
        elif multimodal_model_type == "qwen2_vl" and QWEN2_VL_AVAILABLE:
            return load_qwen2_vl_model(config.model, config)
        elif multimodal_model_type == "qwen2_5_vl" and QWEN2_5_VL_AVAILABLE:
            return load_qwen2_5_vl_model(config.model, config)
        elif multimodal_model_type == "llavanext" and LLAVANEXT_AVAILABLE:
            return load_llavanext_model(config.model, config)
        else:
            raise ValueError(
                f"Unsupported multimodal_model_type: {multimodal_model_type} "
                f"or model not available"
            )

    @staticmethod
    def load_text_model(config: Config, hf_config) -> torch.nn.Module:
        """Load a text-only model."""
        if hasattr(hf_config, 'model_type') and hf_config.model_type == 'gemma' and GEMMA_AVAILABLE:
            model = GemmaForCausalLM(hf_config)
            load_model(model, config.model)
            return model
        else:
            text_config = getattr(hf_config, "text_config", hf_config)
            model = Qwen3ForCausalLM(text_config)
            load_model(model, config.model)
            return model
