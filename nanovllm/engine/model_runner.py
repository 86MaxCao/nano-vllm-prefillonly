import pickle
import random
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context
from nanovllm.utils.loader import load_model

try:
    from nanovllm.models.qwen3_vl import load_qwen3_vl_model
    QWEN3_VL_AVAILABLE = True
except ImportError:
    QWEN3_VL_AVAILABLE = False
    print("[ModelRunner] qwen3_vl module unavailable")

try:
    from nanovllm.models.qwen2_vl import load_qwen2_vl_model
    QWEN2_VL_AVAILABLE = True
except ImportError:
    QWEN2_VL_AVAILABLE = False
    print("[ModelRunner] qwen2_vl module unavailable")

try:
    from nanovllm.models.qwen2_5_vl import load_qwen2_5_vl_model
    QWEN2_5_VL_AVAILABLE = True
except ImportError:
    QWEN2_5_VL_AVAILABLE = False
    print("[ModelRunner] qwen2_5_vl module unavailable")

try:
    from nanovllm.models.llavanext import load_llavanext_model
    LLAVANEXT_AVAILABLE = True
except ImportError:
    LLAVANEXT_AVAILABLE = False
    print("[ModelRunner] llavanext module unavailable")

MULTIMODAL_AVAILABLE = (
    QWEN3_VL_AVAILABLE or QWEN2_VL_AVAILABLE or
    QWEN2_5_VL_AVAILABLE or LLAVANEXT_AVAILABLE
)

from nanovllm.models.qwen3 import Qwen3ForCausalLM

try:
    from nanovllm.models.qwen3_reranker import Qwen3Reranker
    RERANKER_AVAILABLE = True
except ImportError:
    RERANKER_AVAILABLE = False
    print("[ModelRunner] reranker module unavailable")

try:
    from nanovllm.models.gemma_reranker import GemmaReranker
    from nanovllm.models.gemma import GemmaForCausalLM
    GEMMA_AVAILABLE = True
except ImportError:
    GEMMA_AVAILABLE = False
    print("[ModelRunner] gemma module unavailable")

try:
    from nanovllm.models.jina_reranker_v3 import JinaRerankerV3
    JINA_V3_AVAILABLE = True
except ImportError:
    JINA_V3_AVAILABLE = False
    print("[ModelRunner] jina-reranker-v3 module unavailable")

try:
    from nanovllm.models.gemma2_embedding import Gemma2Embedding
    from nanovllm.models.gemma2 import Gemma2ForCausalLM
    try:
        from transformers import Gemma2Config
    except ImportError:
        # Fallback to GemmaConfig if Gemma2Config is not available
        from transformers import GemmaConfig as Gemma2Config
    GEMMA2_AVAILABLE = True
except ImportError as e:
    GEMMA2_AVAILABLE = False
    print(f"[ModelRunner] gemma2 embedding module unavailable: {e}")

try:
    from nanovllm.models.qwen3_embedding import Qwen3Embedding
    QWEN3_EMBEDDING_AVAILABLE = True
except ImportError:
    QWEN3_EMBEDDING_AVAILABLE = False
    print("[ModelRunner] qwen3 embedding module unavailable")

try:
    from nanovllm.models.llavanext_embedding import LLaVANextEmbedding
    LLAVANEXT_EMBEDDING_AVAILABLE = True
except ImportError:
    LLAVANEXT_EMBEDDING_AVAILABLE = False
    print("[ModelRunner] llavanext embedding module unavailable")

try:
    from nanovllm.models.qwen2_vl_gme_embedding import Qwen2VLGmeEmbedding, GmeQwen2VLConfig
    QWEN2VL_GME_AVAILABLE = True
except ImportError as e:
    QWEN2VL_GME_AVAILABLE = False
    print(f"[ModelRunner] qwen2-vl-gme embedding module unavailable: {e}")

try:
    from nanovllm.models.jina_v4_embedding import JinaEmbeddingsV4
    JINA_V4_AVAILABLE = True
except ImportError as e:
    JINA_V4_AVAILABLE = False
    print(f"[ModelRunner] jina-v4 embedding module unavailable: {e}")

try:
    from nanovllm.models.jina_m0_reranker import JinaRerankerM0
    JINA_M0_AVAILABLE = True
except ImportError as e:
    JINA_M0_AVAILABLE = False
    print(f"[ModelRunner] jina-m0 reranker module unavailable: {e}")


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        global torch  # Ensure torch is treated as a global variable
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        # Disable torch.compile for layernorm if enforce_eager is True
        # This prevents inplace operation errors with qwen2.5vl and other models
        if self.enforce_eager:
            import torch._dynamo
            torch._dynamo.config.suppress_errors = True
            # Set global flag for layernorm to use non-inplace operations
            from nanovllm.layers import layernorm
            layernorm._ENFORCE_EAGER_MODE = True
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        # Check if process group is already initialized
        if not dist.is_initialized():
            # Always initialize process group (even for world_size=1) because
            # code components like VocabParallelEmbedding call dist.get_rank()
            # Use random port to avoid port conflicts when testing multiple models
            # Port range: 10000-65535 (avoiding well-known ports)
            port = random.randint(10000, 65535)
            dist.init_process_group(
                "nccl",
                f"tcp://localhost:{port}",
                world_size=self.world_size,
                rank=rank,
            )
        elif self.world_size > 1:
            # If already initialized, verify it matches our requirements
            current_world_size = dist.get_world_size()
            if current_world_size != self.world_size:
                raise RuntimeError(
                    f"Process group already initialized with "
                    f"world_size={current_world_size}, but config requires "
                    f"world_size={self.world_size}"
                )
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
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
        if torch_dtype is None:
            torch_dtype = torch.float16
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device("cuda")

        # Embedding support (check first, as embedding models can be multimodal)
        self.is_embedding = (
            getattr(config, "is_embedding", False)
        )
        embedding_type = getattr(config, "embedding_type", None)
        pooling_type = getattr(config, "pooling_type", "LAST")
        normalize_embeddings = getattr(config, "normalize_embeddings", True)
        
        # Reranker support (check second, as reranker models can be multimodal)
        self.is_reranker = (
            getattr(config, "is_reranker", False) and RERANKER_AVAILABLE
        )
        reranker_type = getattr(config, "reranker_type", None)
                
        # Multimodal support is optional; fall back to text-only runner when
        # the extended VLM stack is not available.
        # Note: Only set is_multimodal if NOT embedding/reranker (they handle multimodal separately)
        self.is_multimodal = (
            getattr(config, "is_multimodal", False) 
            and MULTIMODAL_AVAILABLE
            and not self.is_embedding
            and not self.is_reranker
        )
        multimodal_model_type = getattr(config, "multimodal_model_type", "qwen3_vl")
        
        # Determine target dtype for embedding/reranker models (before model creation)
        # This ensures load_model loads weights directly in the target dtype
        target_dtype = None
        if self.is_embedding or self.is_reranker:
            torch_dtype = getattr(hf_config, "torch_dtype", None)
            if torch_dtype is None and hasattr(hf_config, "text_config"):
                torch_dtype = getattr(hf_config.text_config, "torch_dtype", None)
            
            # Convert string dtype to torch.dtype if needed
            if isinstance(torch_dtype, str):
                if torch_dtype == "float16":
                    torch_dtype = torch.float16
                elif torch_dtype == "bfloat16":
                    torch_dtype = torch.bfloat16
                elif torch_dtype == "float32":
                    torch_dtype = torch.float32
                else:
                    torch_dtype = None
            
            # If original dtype is float32 or None, use float16 for FlashAttention
            if torch_dtype == torch.float32 or torch_dtype is None:
                target_dtype = torch.float16
            elif torch_dtype in (torch.float16, torch.bfloat16):
                target_dtype = torch_dtype
            else:
                target_dtype = torch.float16  # Default to float16
            
            # Update default dtype to target_dtype for embedding/reranker models
            # This ensures model parameters are created with the correct dtype
            torch.set_default_dtype(target_dtype)
        
        if self.is_embedding and embedding_type:
            # Load embedding models
            if embedding_type == "gemma2" and GEMMA2_AVAILABLE:
                try:
                    gemma2_config = Gemma2Config.from_pretrained(config.model)
                except Exception:
                    # Fallback: use hf_config directly if Gemma2Config fails
                    gemma2_config = hf_config
                self.model = Gemma2Embedding(
                    gemma2_config,
                    pooling_type=pooling_type,
                    normalize=normalize_embeddings,
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
            elif embedding_type == "qwen3" and QWEN3_EMBEDDING_AVAILABLE:
                text_config = getattr(hf_config, "text_config", hf_config)
                from transformers import Qwen3Config
                qwen3_config = Qwen3Config.from_dict(text_config.to_dict())
                self.model = Qwen3Embedding(
                    qwen3_config,
                    pooling_type=pooling_type,
                    normalize=normalize_embeddings,
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
            elif embedding_type == "llavanext" and LLAVANEXT_EMBEDDING_AVAILABLE:
                self.model = LLaVANextEmbedding(
                    hf_config,
                    pooling_type=pooling_type,
                    normalize=normalize_embeddings,
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
            elif embedding_type == "qwen2_vl_gme" and QWEN2VL_GME_AVAILABLE:
                from transformers import AutoConfig
                gme_config = GmeQwen2VLConfig.from_pretrained(config.model, trust_remote_code=True)
                self.model = Qwen2VLGmeEmbedding(
                    gme_config,
                    pooling_type=pooling_type,
                    normalize=normalize_embeddings,
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
            elif embedding_type == "jina_v4" and JINA_V4_AVAILABLE:
                from transformers import AutoConfig
                # JinaEmbeddingsV4Config extends Qwen2_5_VLConfig, use AutoConfig
                jina_v4_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
                self.model = JinaEmbeddingsV4(
                    jina_v4_config,
                    pooling_type=pooling_type,
                    normalize=normalize_embeddings,
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
            else:
                raise ValueError(f"Unsupported embedding type: {embedding_type}")
        elif self.is_reranker and reranker_type:
            text_config = getattr(hf_config, "text_config", hf_config)
            is_original = getattr(config, "is_original_qwen3_reranker", False)
            classifier_tokens = getattr(config, "classifier_from_token", None)
            
            if reranker_type == "qwen3":
                self.model = Qwen3Reranker(
                    text_config,
                    is_original_reranker=is_original,
                    classifier_from_token=classifier_tokens,
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
                
                # Convert original reranker weights if needed
                if is_original:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(config.model)
                    self.model.convert_from_original_reranker(tokenizer)
            elif reranker_type == "gemma" and GEMMA_AVAILABLE:
                self.model = GemmaReranker(
                    text_config,
                    is_original_reranker=is_original,
                    classifier_from_token=classifier_tokens or ["Yes"],
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
                
                # Convert original reranker weights if needed
                if is_original:
                    from transformers import AutoTokenizer
                    tokenizer = AutoTokenizer.from_pretrained(config.model)
                    self.model.convert_from_original_reranker(tokenizer)
            elif reranker_type == "jina_v3" and JINA_V3_AVAILABLE:
                projector_dim = getattr(config, "projector_dim", 512)
                use_flex_attention = getattr(config, "use_flex_attention", True)
                self.model = JinaRerankerV3(
                    text_config,
                    projector_dim=projector_dim,
                    use_flex_attention=use_flex_attention,
                )
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
            elif reranker_type == "jina_m0" and JINA_M0_AVAILABLE:
                from transformers import AutoConfig
                jina_m0_config = AutoConfig.from_pretrained(config.model, trust_remote_code=True)
                self.model = JinaRerankerM0(jina_m0_config)
                # Convert to target dtype before loading weights
                if target_dtype is not None:
                    self.model = self.model.to(target_dtype)
                load_model(self.model, config.model)
            else:
                raise ValueError(f"Unsupported reranker type: {reranker_type}")
        elif self.is_multimodal:
            if multimodal_model_type == "qwen3_vl" and QWEN3_VL_AVAILABLE:
                self.model = load_qwen3_vl_model(config.model, config)
            elif multimodal_model_type == "qwen2_vl" and QWEN2_VL_AVAILABLE:
                self.model = load_qwen2_vl_model(config.model, config)
            elif multimodal_model_type == "qwen2_5_vl" and QWEN2_5_VL_AVAILABLE:
                self.model = load_qwen2_5_vl_model(config.model, config)
            elif multimodal_model_type == "llavanext" and LLAVANEXT_AVAILABLE:
                self.model = load_llavanext_model(config.model, config)
            else:
                raise ValueError(
                    f"Unsupported multimodal_model_type: {multimodal_model_type} "
                    f"or model not available"
                )
        else:
            # Check if it's a Gemma model
            if hasattr(hf_config, 'model_type') and hf_config.model_type == 'gemma' and GEMMA_AVAILABLE:
                self.model = GemmaForCausalLM(hf_config)
                load_model(self.model, config.model)
            else:
                text_config = getattr(hf_config, "text_config", hf_config)
                self.model = Qwen3ForCausalLM(text_config)
                load_model(self.model, config.model)

        embed_module = getattr(self.model, "language_model", self.model)
        if hasattr(embed_module, "model"):
            embed_module = embed_module.model
        # Keep a reference dtype so that cached vision embeddings can be copied
        # back to the GPU without hitting dtype mismatches.
        self.model_dtype = embed_module.embed_tokens.weight.dtype
        self.sampler = Sampler()
        
        # Model dtype should already be correct (converted before load_model)
        # But verify for safety (for embedding/reranker models)
        if self.is_embedding or self.is_reranker:
            model_dtype = next(self.model.parameters()).dtype
            if model_dtype == torch.float32:
                print(f"[WARNING] Model dtype is still {model_dtype} after loading, converting to float16 for FlashAttention compatibility")
                self.model = self.model.to(torch.float16)
                # Explicitly clear cache to free the float32 model memory
                torch.cuda.empty_cache()
        
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                try:
                    self.shm = SharedMemory(
                        name="nanovllm",
                        create=True,
                        size=2**20,
                    )
                except FileExistsError:
                    # Shared memory already exists, try to open it
                    self.shm = SharedMemory(name="nanovllm")
                if dist.is_initialized():
                    dist.barrier()
            else:
                if dist.is_initialized():
                    dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            if hasattr(self, "shm"):
                self.shm.close()
            if dist.is_initialized():
                dist.barrier()
            if self.rank == 0 and hasattr(self, "shm"):
                try:
                    self.shm.unlink()
                except FileNotFoundError:
                    pass  # Already unlinked
        if not self.enforce_eager and hasattr(self, 'graphs'):
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        if dist.is_initialized():
            try:
                dist.destroy_process_group()
            except Exception:
                pass  # Process group may have been destroyed already

    def loop(self):
        while True:
            method_name, args, kwargs = self.read_shm()
            self.call(method_name, *args, **kwargs)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        data = pickle.loads(self.shm.buf[4:n+4])
        method_name = data[0]
        # Last element is kwargs if it's a dict, otherwise all remaining are args
        if len(data) > 1 and isinstance(data[-1], dict):
            args = tuple(data[1:-1])
            kwargs = data[-1]
        else:
            args = tuple(data[1:])
            kwargs = {}
        self.event.clear()
        return method_name, args, kwargs

    def write_shm(self, method_name, *args, **kwargs):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args, kwargs])
        n = len(data)
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args, **kwargs):
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args, **kwargs)
        method = getattr(self, method_name, None)
        return method(*args, **kwargs)

    def warmup_model(self):
        # Skip warmup for embedding and reranker models as they don't use
        # sampling
        if self.is_embedding or self.is_reranker:
            return
        # Skip warmup for multimodal models as they require pixel_values
        # which are not available during warmup
        if self.is_multimodal:
            return
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens = self.config.max_num_batched_tokens
        max_model_len = self.config.max_model_len
        num_seqs = min(
            max_num_batched_tokens // max_model_len,
            self.config.max_num_seqs,
        )
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        # Reranker and embedding models don't need KV cache (prefill-only)
        # Check both config flags and instance flags for safety
        config_is_reranker = getattr(self.config, "is_reranker", False)
        config_is_embedding = getattr(self.config, "is_embedding", False)
        if config_is_reranker or config_is_embedding or self.is_reranker or self.is_embedding:
            self.kv_cache = None
            # Explicitly set num_kvcache_blocks to 0 for reranker/embedding models
            # to prevent any downstream code from trying to use it
            self.config.num_kvcache_blocks = 0
            return
        
        # Prefill-only single-token generation doesn't need KV cache
        # Since we only generate one token, we don't need decode-phase KV cache
        # Prefill phase can use temporary cache without pre-allocation
        config_prefill_only = getattr(self.config, "prefill_only_mode", False)
        config_single_token = getattr(self.config, "single_token_mode", False)
        if config_prefill_only and config_single_token:
            self.kv_cache = None
            # Explicitly set num_kvcache_blocks to 0 for prefill-only single-token generation
            self.config.num_kvcache_blocks = 0
            print("[KV Cache] Skipping KV cache allocation for prefill-only single-token generation")
            return
        
        config = self.config
        hf_config = config.hf_config
        text_config = getattr(hf_config, "text_config", hf_config)
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        # Handle models that may not have num_key_value_heads (e.g., some older models)
        # For llavanext (Mistral), check both text_config and direct config
        num_attention_heads = getattr(text_config, "num_attention_heads", None)
        if num_attention_heads is None:
            # Try to get from direct config (for llavanext)
            num_attention_heads = getattr(hf_config, "num_attention_heads", None)
        
        # Ensure num_attention_heads is not None before using it
        if num_attention_heads is None:
            raise ValueError(
                "Model config missing num_attention_heads. "
                f"text_config: {type(text_config)}, hf_config: {type(hf_config)}"
            )
        
        num_key_value_heads = getattr(text_config, "num_key_value_heads", None)
        if num_key_value_heads is None:
            # Fallback to num_attention_heads if num_key_value_heads is not available
            num_key_value_heads = num_attention_heads
        num_kv_heads = num_key_value_heads // self.world_size
        
        # Calculate head_dim
        # Get hidden_size first
        hidden_size = getattr(text_config, "hidden_size", None)
        if hidden_size is None:
            hidden_size = getattr(hf_config, "hidden_size", None)
        if hidden_size is None:
            raise ValueError(
                "Model config missing hidden_size. "
                f"text_config: {type(text_config)}, hf_config: {type(hf_config)}"
            )
        
        head_dim = getattr(
            text_config,
            "head_dim",
            hidden_size // num_attention_heads if num_attention_heads > 0 else 128,
        )
        # Ensure head_dim is not None
        if head_dim is None:
            head_dim = 128  # Default fallback
        dtype = getattr(
            text_config,
            "torch_dtype",
            getattr(hf_config, "torch_dtype", torch.float16),
        )
        # Handle different dtype formats
        if dtype is None:
            dtype = torch.float16  # Default fallback
        elif isinstance(dtype, str):
            # Convert string dtype to torch.dtype
            # Handle special cases like "auto", "float16", "bfloat16", etc.
            dtype_str = dtype  # Save original string
            if dtype_str == "auto":
                dtype = torch.float16  # Default for "auto"
            else:
                # Try to get from torch module first
                dtype = getattr(torch, dtype_str, None)
                if dtype is None or not isinstance(dtype, torch.dtype):
                    # Try with different naming (e.g., "float16" -> torch.float16)
                    dtype_map = {
                        "float16": torch.float16,
                        "bfloat16": torch.bfloat16,
                        "float32": torch.float32,
                    }
                    dtype = dtype_map.get(dtype_str.lower(), torch.float16)
        # Ensure dtype is a valid torch.dtype, not None
        if dtype is None or not isinstance(dtype, torch.dtype):
            dtype = torch.float16  # Final fallback
        
        # Get num_hidden_layers, handle llavanext which may not have it in
        # text_config
        num_hidden_layers = getattr(text_config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            num_hidden_layers = getattr(hf_config, "num_hidden_layers", None)
        if num_hidden_layers is None:
            raise ValueError("Model config missing num_hidden_layers")
        
        block_bytes = (
            2
            * num_hidden_layers
            * self.block_size
            * num_kv_heads
            * head_dim
            * dtype.itemsize
        )
        available_memory = total * config.gpu_memory_utilization - used - peak + current
        # For multimodal models, use smaller memory utilization to account for vision encoder
        if self.is_multimodal:
            available_memory = available_memory * 0.5  # Reserve more for vision encoder
        
        config.num_kvcache_blocks = int(available_memory) // block_bytes
        if config.num_kvcache_blocks <= 0:
            raise RuntimeError(
                f"Failed to allocate KV cache: num_kvcache_blocks={config.num_kvcache_blocks}, "
                f"block_bytes={block_bytes}, available_memory={available_memory:.2f}, "
                f"total={total}, used={used}, peak={peak}, current={current}"
            )
        self.kv_cache = torch.empty(
            2,
            num_hidden_layers,
            config.num_kvcache_blocks,
            self.block_size,
            num_kv_heads,
            head_dim,
            dtype=dtype,
            device="cuda",
        )
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1

    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs) if seqs else 0
        if max_len == 0:
            # No KV cache blocks, return None
            return None
        block_tables = [
            seq.block_table + [-1] * (max_len - len(seq.block_table))
            for seq in seqs
        ]
        block_tables = (
            torch.tensor(block_tables, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        return block_tables

    def prepare_prefill(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        
        # Check if seqs is empty
        if not seqs:
            raise ValueError("prepare_prefill: seqs cannot be empty")
        
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not seq.block_table:    # warmup or no KV cache
                continue
            for i in range(seq.num_cached_blocks, seq.num_blocks):
                start = seq.block_table[i] * self.block_size
                if i != seq.num_blocks - 1:
                    end = start + self.block_size
                else:
                    end = start + seq.last_block_num_tokens
                slot_mapping.extend(list(range(start, end)))
        
        # Ensure cu_seqlens_q and cu_seqlens_k have at least 2 elements (batch_size >= 1)
        # FlashAttention requires batch_size > 0
        if len(cu_seqlens_q) < 2:
            raise ValueError(
                f"prepare_prefill: cu_seqlens_q must have at least 2 elements, "
                f"got {len(cu_seqlens_q)}. This indicates no valid sequences or "
                f"all sequences have seqlen_q = 0."
            )
        
        # Debug logging for troubleshooting
        if not hasattr(self, '_debug_prepare_prefill_logged'):
            print(f"[DEBUG prepare_prefill] seqs length: {len(seqs)}")
            if seqs:
                print(f"[DEBUG prepare_prefill] first seq len: {len(seqs[0])}, num_cached_tokens: {seqs[0].num_cached_tokens}")
                print(f"[DEBUG prepare_prefill] cu_seqlens_q: {cu_seqlens_q}")
                print(f"[DEBUG prepare_prefill] cu_seqlens_k: {cu_seqlens_k}")
                print(f"[DEBUG prepare_prefill] max_seqlen_q: {max_seqlen_q}, max_seqlen_k: {max_seqlen_k}")
                print(f"[DEBUG prepare_prefill] input_ids length: {len(input_ids)}")
                print(f"[DEBUG prepare_prefill] slot_mapping length: {len(slot_mapping)}")
                # Check input_ids at key positions (last tokens)
                if len(input_ids) > 515:
                    print(f"[DEBUG prepare_prefill] input_ids[514]: {input_ids[514]}")
                    print(f"[DEBUG prepare_prefill] input_ids[515]: {input_ids[515]} (first seq last)")
                    if len(input_ids) > 516:
                        print(f"[DEBUG prepare_prefill] input_ids[516]: {input_ids[516]} (second seq start)")
                if len(input_ids) > 941:
                    print(f"[DEBUG prepare_prefill] input_ids[940]: {input_ids[940]}")
                    print(f"[DEBUG prepare_prefill] input_ids[941]: {input_ids[941]} (second seq last)")
                    if len(input_ids) > 942:
                        print(f"[DEBUG prepare_prefill] input_ids[942]: {input_ids[942]} (third seq start)")
                if len(input_ids) > 3914:
                    print(f"[DEBUG prepare_prefill] input_ids[3913]: {input_ids[3913]}")
                    print(f"[DEBUG prepare_prefill] input_ids[3914]: {input_ids[3914]} (last seq last)")
                # Check if we have image_token_id to compare
                if hasattr(self.config, 'hf_config') and hasattr(self.config.hf_config, 'image_token_id'):
                    image_token_id = self.config.hf_config.image_token_id
                    print(f"[DEBUG prepare_prefill] image_token_id: {image_token_id}")
                    # Count image tokens in input_ids
                    image_token_count = sum(1 for tid in input_ids if tid == image_token_id)
                    print(f"[DEBUG prepare_prefill] image_token_count in input_ids: {image_token_count}")
                    # Check if last token positions are image tokens
                    if len(input_ids) > 515:
                        is_img_515 = input_ids[515] == image_token_id
                        print(f"[DEBUG prepare_prefill] input_ids[515] is image_token: {is_img_515}")
                    if len(input_ids) > 941:
                        is_img_941 = input_ids[941] == image_token_id
                        print(f"[DEBUG prepare_prefill] input_ids[941] is image_token: {is_img_941}")
            self._debug_prepare_prefill_logged = True
        
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(seqs)
        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        positions = (
            torch.tensor(positions, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        cu_seqlens_q = (
            torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        cu_seqlens_k = (
            torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        # Handle empty slot_mapping (when no KV cache)
        if len(slot_mapping) == 0:
            slot_mapping = torch.empty(0, dtype=torch.int32, device="cuda")
        else:
            slot_mapping = (
                torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True)
                .cuda(non_blocking=True)
            )
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            block_tables,
        )
        return input_ids, positions

    def prepare_decode(self, seqs: list[Sequence]):
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq) - 1)
            context_lens.append(len(seq))
            slot_mapping.append(
                seq.block_table[-1] * self.block_size
                + seq.last_block_num_tokens
                - 1
            )
        input_ids = (
            torch.tensor(input_ids, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        positions = (
            torch.tensor(positions, dtype=torch.int64, pin_memory=True)
            .cuda(non_blocking=True)
        )
        slot_mapping = (
            torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        context_lens = (
            torch.tensor(context_lens, dtype=torch.int32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        block_tables = self.prepare_block_tables(seqs)
        set_context(
            False,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return input_ids, positions

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = (
            torch.tensor(temperatures, dtype=torch.float32, pin_memory=True)
            .cuda(non_blocking=True)
        )
        return temperatures
    
    def expand_batch_to_tokens(
        self,
        x: torch.Tensor,  # [batch_size]
        sequence_lengths: list[int],  # [batch_size]
    ) -> torch.Tensor:
        """Expand [batch_size] tensor to [num_tokens] tensor based on sequence_lengths.
        
        For example, if x = [a, b, c] and sequence_lengths = [2, 3, 1], then
        num_tokens = 6, and expanded_x = [a, a, b, b, b, c].
        """
        batch_size = x.shape[0]
        assert len(sequence_lengths) == batch_size
        num_tokens = sum(sequence_lengths)
        expanded_x = x.new_empty(num_tokens)
        
        # Calculate cumulative sequence lengths
        cu_seqlens = [0]
        for seq_len in sequence_lengths:
            cu_seqlens.append(cu_seqlens[-1] + seq_len)
        
        # Expand: for each sequence, repeat its value for sequence_length times
        for seq_idx in range(batch_size):
            seq_start = cu_seqlens[seq_idx]
            seq_end = cu_seqlens[seq_idx + 1]
            expanded_x[seq_start:seq_end] = x[seq_idx]
        
        return expanded_x

    @torch.inference_mode()
    def run_model(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        is_prefill: bool,
        sequence_lengths: list[int] | None = None,
        vision_slices_per_seq: list[list[dict]] | None = None,
        seqs: list[Sequence] | None = None,
    ):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            model_kwargs = {}
            if self.is_multimodal:
                # For llavanext (no visual method), pass pixel_values directly
                # llavanext processes pixel_values in forward, not via vision cache
                # llavanext doesn't use vision_slices_per_seq or sequence_lengths
                if not hasattr(self.model, 'visual') and is_prefill and seqs:
                    pixel_values_list = []
                    image_grid_thw_list = []
                    for seq in seqs:
                        if seq.pixel_values is not None:
                            pv = seq.pixel_values
                            # Handle 5D tensor: (batch=1, num_patches, c, h, w)
                            # Reshape to (num_patches, c, h, w) for llavanext
                            if isinstance(pv, torch.Tensor) and pv.dim() == 5:
                                # Shape: (1, num_patches, c, h, w) -> (num_patches, c, h, w)
                                b, num_patches, c, h, w = pv.shape
                                if b == 1:
                                    pv = pv.squeeze(0)  # Remove batch dimension
                                else:
                                    # Multiple batches, reshape to (b * num_patches, c, h, w)
                                    pv = pv.view(b * num_patches, c, h, w)
                            pixel_values_list.append(pv)
                        else:
                            pixel_values_list.append(None)
                        if seq.image_grid_thw is not None:
                            image_grid_thw_list.append(seq.image_grid_thw)
                        else:
                            image_grid_thw_list.append(None)
                    
                    # Debug log for first call
                    if not hasattr(self, '_debug_llavanext_pixel_values_logged'):
                        print("\n[DEBUG model_runner.run_model (llavanext)]")
                        print(f"  Number of sequences: {len(seqs)}")
                        print(f"  pixel_values_list length: {len(pixel_values_list)}")
                        for i, pv in enumerate(pixel_values_list):
                            if pv is not None:
                                if isinstance(pv, list):
                                    print(f"  seq[{i}].pixel_values: list with {len(pv)} elements")
                                    if len(pv) > 0:
                                        print(f"    First element type: {type(pv[0])}")
                                        if isinstance(pv[0], torch.Tensor):
                                            print(f"    First element shape: {pv[0].shape}")
                                elif isinstance(pv, torch.Tensor):
                                    print(f"  seq[{i}].pixel_values: tensor, shape: {pv.shape}, dim: {pv.dim()}")
                                else:
                                    print(f"  seq[{i}].pixel_values: {type(pv)}")
                            else:
                                print(f"  seq[{i}].pixel_values: None")
                        self._debug_llavanext_pixel_values_logged = True
                    
                    # If all seqs have pixel_values, pass to model
                    if all(pv is not None for pv in pixel_values_list):
                        # Get model device to move pixel_values to GPU
                        model_device = next(self.model.parameters()).device
                        
                        # Move pixel_values to model device
                        if len(pixel_values_list) == 1:
                            # Single sequence, pass directly
                            pv = pixel_values_list[0]
                            if isinstance(pv, torch.Tensor):
                                pv = pv.to(model_device)
                            model_kwargs["pixel_values"] = pv
                            if image_grid_thw_list[0] is not None:
                                ig = image_grid_thw_list[0]
                                if isinstance(ig, torch.Tensor):
                                    ig = ig.to(model_device)
                                model_kwargs["image_grid_thw"] = ig
                        else:
                            # Multiple sequences, keep as list but move each to device
                            pv_list_device = []
                            for pv in pixel_values_list:
                                if isinstance(pv, torch.Tensor):
                                    pv_list_device.append(pv.to(model_device))
                                elif isinstance(pv, list):
                                    # Handle nested list
                                    pv_list_device.append([
                                        item.to(model_device) if isinstance(item, torch.Tensor) else item
                                        for item in pv
                                    ])
                                else:
                                    pv_list_device.append(pv)
                            model_kwargs["pixel_values"] = pv_list_device
                            if all(ig is not None for ig in image_grid_thw_list):
                                ig_list_device = []
                                for ig in image_grid_thw_list:
                                    if isinstance(ig, torch.Tensor):
                                        ig_list_device.append(ig.to(model_device))
                                    else:
                                        ig_list_device.append(ig)
                                model_kwargs["image_grid_thw"] = ig_list_device
                    
                    # Debug log for model_kwargs
                    if not hasattr(self, '_debug_llavanext_kwargs_logged'):
                        print(f"  model_kwargs keys: {list(model_kwargs.keys())}")
                        if "pixel_values" in model_kwargs:
                            pv = model_kwargs["pixel_values"]
                            if isinstance(pv, list):
                                print(f"  model_kwargs['pixel_values']: list with {len(pv)} elements")
                            elif isinstance(pv, torch.Tensor):
                                print(f"  model_kwargs['pixel_values']: tensor, shape: {pv.shape}, dim: {pv.dim()}")
                            else:
                                print(f"  model_kwargs['pixel_values']: {type(pv)}")
                        self._debug_llavanext_kwargs_logged = True
                    
                    # For llavanext, don't pass positions as positional argument
                    # llavanext's forward signature is: forward(input_ids, pixel_values, ...)
                    # So positions should be passed via kwargs or not at all
                    # llavanext uses position_ids, not positions
                    # Pass sequence_lengths if available for per-sequence processing
                    if sequence_lengths is not None:
                        model_kwargs["sequence_lengths"] = sequence_lengths
                    outputs = self.model(input_ids, **model_kwargs)
                else:
                    # For other multimodal models (qwen2vl, qwen2.5vl, qwen3vl)
                    model_kwargs["sequence_lengths"] = sequence_lengths
                    
                    # Check if we should use vision_slices_per_seq or pixel_values fallback
                    config_prefill_only = getattr(self.config, "prefill_only_mode", False)
                    config_single_token = getattr(self.config, "single_token_mode", False)
                    use_pixel_values_fallback = (
                        config_prefill_only 
                        and config_single_token 
                        and (vision_slices_per_seq is None or not any(vision_slices_per_seq))
                        and seqs is not None
                    )
                    
                    if use_pixel_values_fallback:
                        # For prefill-only single-token without vision cache,
                        # pass pixel_values directly to model (fallback path)
                        # Collect pixel_values and image_grid_thw from seqs
                        # Also record which images belong to which sequence
                        pixel_values_list = []
                        image_grid_thw_list = []
                        seq_image_indices = []  # Record image index range for each sequence
                        seq_vision_placeholders = []  # Record vision_placeholders for each sequence
                        current_image_idx = 0
                        
                        for seq_idx, seq in enumerate(seqs):
                            seq_start_image_idx = current_image_idx
                            if seq.pixel_values is not None:
                                pixel_values_list.append(seq.pixel_values)
                                # Debug: log pixel_values format
                                if not hasattr(self, '_debug_pixel_values_fallback_logged'):
                                    print(
                                        f"[DEBUG model_runner.run_model (fallback)] "
                                        f"seq[{seq_idx}].pixel_values shape: {seq.pixel_values.shape}, "
                                        f"dim: {seq.pixel_values.dim()}, "
                                        f"dtype: {seq.pixel_values.dtype}"
                                    )
                                    if seq.image_grid_thw is not None:
                                        print(
                                            f"[DEBUG model_runner.run_model (fallback)] "
                                            f"seq[{seq_idx}].image_grid_thw shape: {seq.image_grid_thw.shape}, "
                                            f"dim: {seq.image_grid_thw.dim()}"
                                        )
                                    if hasattr(seq, 'vision_placeholders'):
                                        print(
                                            f"[DEBUG model_runner.run_model (fallback)] "
                                            f"seq[{seq_idx}].vision_placeholders: {seq.vision_placeholders}"
                                        )
                                    self._debug_pixel_values_fallback_logged = True
                            else:
                                pixel_values_list.append(None)
                            
                            # Collect vision_placeholders for this sequence
                            if hasattr(seq, 'vision_placeholders') and seq.vision_placeholders:
                                seq_vision_placeholders.append(seq.vision_placeholders)
                            else:
                                seq_vision_placeholders.append([])
                            
                            if seq.image_grid_thw is not None:
                                image_grid_thw_list.append(seq.image_grid_thw)
                                # Count number of images in this sequence
                                if seq.image_grid_thw.dim() == 1:
                                    num_images = 1
                                elif seq.image_grid_thw.dim() == 2:
                                    num_images = seq.image_grid_thw.shape[0]
                                else:
                                    num_images = seq.image_grid_thw.shape[0] if seq.image_grid_thw.dim() == 3 else 1
                                seq_image_indices.append((seq_start_image_idx, current_image_idx + num_images))
                                current_image_idx += num_images
                            else:
                                image_grid_thw_list.append(None)
                                seq_image_indices.append((seq_start_image_idx, current_image_idx))
                        
                        # Move to model device and convert to model dtype
                        model_device = next(self.model.parameters()).device
                        model_dtype = next(self.model.parameters()).dtype
                        
                        # Debug: log model dtype
                        if not hasattr(self, '_debug_model_dtype_logged'):
                            print(
                                f"[DEBUG model_runner.run_model (fallback)] "
                                f"Model device: {model_device}, dtype: {model_dtype}"
                            )
                            self._debug_model_dtype_logged = True
                        
                        if all(pv is not None for pv in pixel_values_list):
                            if len(pixel_values_list) == 1:
                                # Single sequence
                                pv = pixel_values_list[0]
                                if isinstance(pv, torch.Tensor):
                                    # Log original dtype before conversion
                                    if not hasattr(self, '_debug_pv_dtype_conversion_logged'):
                                        print(
                                            f"[DEBUG model_runner.run_model (fallback)] "
                                            f"Converting pixel_values from {pv.dtype} to {model_dtype}"
                                        )
                                        self._debug_pv_dtype_conversion_logged = True
                                    pv = pv.to(model_device).to(model_dtype)
                                model_kwargs["pixel_values"] = pv
                                if image_grid_thw_list[0] is not None:
                                    ig = image_grid_thw_list[0]
                                    if isinstance(ig, torch.Tensor):
                                        ig = ig.to(model_device)
                                    model_kwargs["image_grid_thw"] = ig
                            else:
                                # Multiple sequences - batch them
                                # For qwen3vl, pixel_values should be a tensor (not list)
                                # According to vLLM, use torch.cat along dim=0 for flat_from_sizes format
                                pv_tensors = []
                                for i, pv in enumerate(pixel_values_list):
                                    if isinstance(pv, torch.Tensor):
                                        # Log each tensor's original dtype and shape
                                        if not hasattr(self, '_debug_pv_batch_logged'):
                                            print(
                                                f"[DEBUG model_runner.run_model (fallback)] "
                                                f"Batch pixel_values[{i}]: shape={pv.shape}, "
                                                f"dim={pv.dim()}, dtype={pv.dtype}"
                                            )
                                        # Convert to model dtype (critical fix!)
                                        pv_converted = pv.to(model_device).to(model_dtype)
                                        pv_tensors.append(pv_converted)
                                    else:
                                        raise ValueError(
                                            f"pixel_values_list[{i}] is not a torch.Tensor, "
                                            f"got {type(pv)}"
                                        )
                                
                                if not hasattr(self, '_debug_pv_batch_logged'):
                                    print(
                                        f"[DEBUG model_runner.run_model (fallback)] "
                                        f"Converting {len(pv_tensors)} pixel_values tensors "
                                        f"from original dtype to {model_dtype}"
                                    )
                                    self._debug_pv_batch_logged = True
                                
                                # Check if all tensors have the same shape
                                if all(isinstance(pv, torch.Tensor) for pv in pv_tensors):
                                    shapes = [pv.shape for pv in pv_tensors]
                                    if len(set(shapes)) == 1:
                                        # All same shape - can stack (unlikely for qwen3vl)
                                        if not hasattr(self, '_debug_pv_stack_logged'):
                                            print(
                                                f"[DEBUG model_runner.run_model (fallback)] "
                                                f"All pixel_values have same shape {shapes[0]}, using torch.stack"
                                            )
                                            self._debug_pv_stack_logged = True
                                        model_kwargs["pixel_values"] = torch.stack(pv_tensors, dim=0)
                                    else:
                                        # Different shapes - concatenate along first dimension
                                        # This is for flattened/varlen format (vLLM's flat_from_sizes)
                                        if not hasattr(self, '_debug_pv_cat_logged'):
                                            print(
                                                f"[DEBUG model_runner.run_model (fallback)] "
                                                f"pixel_values have different shapes: {shapes}, "
                                                f"using torch.cat along dim=0 (vLLM flat_from_sizes format)"
                                            )
                                            self._debug_pv_cat_logged = True
                                        model_kwargs["pixel_values"] = torch.cat(pv_tensors, dim=0)
                                else:
                                    # Some are not tensors - this shouldn't happen, but handle it
                                    raise ValueError(
                                        "pixel_values_list contains non-tensor elements. "
                                        "All elements must be torch.Tensor."
                                    )
                                
                                if all(ig is not None for ig in image_grid_thw_list):
                                    ig_tensors = [
                                        ig.to(model_device) if isinstance(ig, torch.Tensor) else ig
                                        for ig in image_grid_thw_list
                                    ]
                                    # image_grid_thw format: [num_images, 3] (2D)
                                    # When batching, we need to concatenate all images
                                    # (not stack, because we're flattening across sequences)
                                    if all(isinstance(ig, torch.Tensor) for ig in ig_tensors):
                                        # Ensure all are 2D [num_images, 3]
                                        ig_tensors_2d = []
                                        for ig in ig_tensors:
                                            if ig.dim() == 1:
                                                # [3] -> [1, 3]
                                                ig = ig.unsqueeze(0)
                                            elif ig.dim() == 3:
                                                # [batch, num_images, 3] -> [batch*num_images, 3]
                                                ig = ig.view(-1, 3)
                                            # Now ig is [num_images, 3]
                                            ig_tensors_2d.append(ig)
                                        # Concatenate all images: [total_images, 3]
                                        model_kwargs["image_grid_thw"] = torch.cat(ig_tensors_2d, dim=0)
                                    else:
                                        raise ValueError(
                                            "image_grid_thw_list contains non-tensor elements. "
                                            "All elements must be torch.Tensor."
                                        )
                        
                        # Don't pass vision_slices_per_seq when using fallback
                        model_kwargs["vision_slices_per_seq"] = None
                        
                        # Pass sequence-to-image mapping for correct image_chunks allocation
                        model_kwargs["seq_image_indices"] = seq_image_indices
                        
                        # Pass vision_placeholders for correct target_offset placement
                        model_kwargs["seq_vision_placeholders"] = seq_vision_placeholders
                        
                        # Debug: log what we're passing to model
                        if not hasattr(self, '_debug_model_kwargs_fallback_logged'):
                            print(
                                f"[DEBUG model_runner.run_model (fallback)] "
                                f"model_kwargs keys: {list(model_kwargs.keys())}"
                            )
                            if "pixel_values" in model_kwargs:
                                pv = model_kwargs["pixel_values"]
                                print(
                                    f"[DEBUG model_runner.run_model (fallback)] "
                                    f"model_kwargs['pixel_values'] shape: {pv.shape}, "
                                    f"dim: {pv.dim()}, dtype: {pv.dtype}"
                                )
                            if "image_grid_thw" in model_kwargs:
                                ig = model_kwargs["image_grid_thw"]
                                print(
                                    f"[DEBUG model_runner.run_model (fallback)] "
                                    f"model_kwargs['image_grid_thw'] shape: {ig.shape}, "
                                    f"dim: {ig.dim()}"
                                )
                            if "seq_image_indices" in model_kwargs:
                                print(
                                    f"[DEBUG model_runner.run_model (fallback)] "
                                    f"model_kwargs['seq_image_indices']: {model_kwargs['seq_image_indices']} "
                                    f"(maps each sequence to image index range)"
                                )
                            if "seq_vision_placeholders" in model_kwargs:
                                print(
                                    f"[DEBUG model_runner.run_model (fallback)] "
                                    f"model_kwargs['seq_vision_placeholders']: {model_kwargs['seq_vision_placeholders']} "
                                    f"(maps each sequence to vision placeholder offsets)"
                                )
                            self._debug_model_kwargs_fallback_logged = True
                    else:
                        # Normal path: use vision_slices_per_seq (chunked prefill)
                        # Prefill can stream only part of the visual tokens. Pass
                        # slice metadata so the forward pass knows which cached chunks
                        # to use.
                        model_kwargs["vision_slices_per_seq"] = vision_slices_per_seq
                    
                    outputs = self.model(input_ids, positions, **model_kwargs)
            else:
                # Non-multimodal models
                outputs = self.model(input_ids, positions, **model_kwargs)
            
            # For embedding models, forward() returns embeddings directly
            # For reranker models, forward() returns hidden_states (no logits)
            # For language models, forward() returns hidden_states and needs
            # compute_logits
            if self.is_embedding or self.is_reranker:
                return outputs
            elif hasattr(self.model, 'compute_logits'):
                return self.model.compute_logits(outputs)
            else:
                return outputs
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph_idx = next(x for x in self.graph_bs if x >= bs)
            graph = self.graphs[graph_idx]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = (
                context.block_tables
            )
            graph.replay()
            # For embedding models, forward() returns embeddings directly
            # For reranker models, forward() returns hidden_states (no logits)
            # For language models, forward() returns hidden_states and needs
            # compute_logits
            if self.is_embedding or self.is_reranker:
                return graph_vars["outputs"][:bs]
            elif hasattr(self.model, 'compute_logits'):
                return self.model.compute_logits(
                    graph_vars["outputs"][:bs])
            else:
                return graph_vars["outputs"][:bs]


    def _batch_to_varlen(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
        """Convert batch format to varlen format for FlashAttention.
        
        Args:
            input_ids: [batch_size, seq_len]
            positions: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len] or None
            
        Returns:
            input_ids_flat: [total_tokens] - flattened input_ids (excluding padding)
            positions_flat: [total_tokens] - flattened positions
            cu_seqlens: [batch_size + 1] - cumulative sequence lengths
            max_seqlen: int - maximum sequence length
            seq_lens: [batch_size] - actual sequence lengths
        """
        batch_size, seq_len = input_ids.shape
        
        # Compute actual sequence lengths from attention_mask
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1).cpu().tolist()
        else:
            seq_lens = [seq_len] * batch_size
        
        # Create cumulative sequence lengths
        cu_seqlens = torch.zeros(
            batch_size + 1, dtype=torch.int32, device=input_ids.device
        )
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=input_ids.device), dim=0
        )
        max_seqlen = max(seq_lens) if seq_lens else seq_len
        
        # Extract only actual tokens (excluding padding) for varlen
        if attention_mask is not None:
            mask = attention_mask.bool()
            input_ids_flat = input_ids[mask]
            positions_flat = positions[mask]
        else:
            input_ids_flat = input_ids.flatten()
            positions_flat = positions.flatten()
        
        return input_ids_flat, positions_flat, cu_seqlens, max_seqlen, seq_lens
    
    def _varlen_to_batch(
        self,
        hidden_states_varlen: torch.Tensor,
        batch_size: int,
        seq_len: int,
        attention_mask: torch.Tensor | None = None,
        seq_lens: list[int] | None = None,
    ) -> torch.Tensor:
        """Convert varlen format back to batch format.
        
        Args:
            hidden_states_varlen: [total_tokens, hidden_size]
            batch_size: int
            seq_len: int
            attention_mask: [batch_size, seq_len] or None
            seq_lens: [batch_size] or None
            
        Returns:
            hidden_states: [batch_size, seq_len, hidden_size]
        """
        hidden_size = hidden_states_varlen.shape[-1]
        hidden_states = torch.zeros(
            batch_size, seq_len, hidden_size,
            dtype=hidden_states_varlen.dtype,
            device=hidden_states_varlen.device
        )
        
        # Fill in the actual tokens (non-padding positions)
        if attention_mask is not None:
            mask = attention_mask.bool()
            # hidden_states_varlen is ordered as: seq0_tokens, seq1_tokens, ...
            varlen_idx = 0
            for batch_idx in range(batch_size):
                seq_len_actual = seq_lens[batch_idx] if seq_lens else seq_len
                for pos_idx in range(seq_len_actual):
                    if mask[batch_idx, pos_idx]:
                        hidden_states[batch_idx, pos_idx] = hidden_states_varlen[varlen_idx]
                        varlen_idx += 1
        else:
            # If no attention_mask, reshape directly
            hidden_states = hidden_states_varlen.view(batch_size, seq_len, -1)
        
        return hidden_states

    @torch.inference_mode()
    def embed(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate embeddings for input texts.
        
        Args:
            input_ids: Token IDs, shape [batch_size, seq_len]
            positions: Position IDs, shape [batch_size, seq_len]
            attention_mask: Optional attention mask, shape [batch_size, seq_len]
                          
        Returns:
            embeddings: [batch_size, hidden_size]
        """
        # Check if model supports embedding
        if not self.is_embedding:
            raise ValueError("Model does not support embedding.")
        
        # Get model device and dtype
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Check if model dtype is supported by FlashAttention
        if model_dtype not in (torch.float16, torch.bfloat16):
            # Try to convert model to fp16 if it's fp32
            if model_dtype == torch.float32:
                print(f"[WARNING] Model dtype is {model_dtype}, converting to float16 for FlashAttention compatibility")
                self.model = self.model.to(torch.float16)
                model_dtype = torch.float16
                # Explicitly clear cache to free the float32 model memory
                torch.cuda.empty_cache()
            else:
                raise ValueError(
                    f"Model dtype {model_dtype} is not supported by FlashAttention. "
                    f"Model weights should be fp16 or bf16, but got {model_dtype}."
                )
        
        # Move inputs to model device
        input_ids = input_ids.to(model_device)
        positions = positions.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(model_device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(model_device)
        
        # Check if this is a multimodal embedding (has pixel_values)
        # Note: We check pixel_values instead of self.is_multimodal because
        # embedding models set self.is_multimodal=False in __init__
        is_multimodal_embedding = (
            self.is_embedding
            and (pixel_values is not None)
        )
        
        # For multimodal embeddings, try to use varlen optimization
        # Convert batch to varlen, process, then convert back to batch
        if is_multimodal_embedding:
            # Check if this is a transformers-based model
            is_transformers_model = hasattr(self.model, '__class__') and (
                'transformers' in str(type(self.model)) or
                'JinaEmbeddingsV4' in str(type(self.model))
            )
            
            if is_transformers_model:
                # For transformers models, we need to use batch format for the final call
                # But we can optimize by using varlen for text-only parts if the model supports it
                # For now, use batch format but ensure proper attention_mask usage
                if not hasattr(self, '_multimodal_embed_logged'):
                    print(
                        f"[DEBUG Multimodal Embedding] Using batch format for "
                        f"transformers model: {type(self.model).__name__}"
                    )
                    if attention_mask is not None:
                        seq_lens = attention_mask.sum(dim=1).cpu().tolist()
                        print(
                            f"[DEBUG Multimodal Embedding] Sequence lengths: "
                            f"{seq_lens}"
                        )
                    self._multimodal_embed_logged = True
                
                kwargs = {
                    "input_ids": input_ids,
                    "positions": positions,
                    "attention_mask": attention_mask,
                }
                if pixel_values is not None:
                    kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    kwargs["image_grid_thw"] = image_grid_thw
                embeddings = self.model(**kwargs)
                from nanovllm.utils.context import reset_context
                reset_context()
                return embeddings
            else:
                # Native nano-vllm model - can use varlen optimization
                # Convert batch to varlen
                batch_size, seq_len = input_ids.shape
                input_ids_flat, positions_flat, cu_seqlens, max_seqlen, seq_lens = (
                    self._batch_to_varlen(input_ids, positions, attention_mask)
                )
                
                # Set context for varlen FlashAttention
                slot_mapping = torch.empty(0, dtype=torch.int32, device=model_device)
                from nanovllm.utils.context import set_context, reset_context
                set_context(
                    True,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    slot_mapping=slot_mapping,
                )
                
                try:
                    # Forward pass with varlen format
                    # Note: For multimodal, we need to handle pixel_values and image_grid_thw
                    # This requires model support for varlen multimodal inputs
                    # For now, fall back to batch format for native models too
                    kwargs = {
                        "input_ids": input_ids,
                        "positions": positions,
                        "attention_mask": attention_mask,
                    }
                    if pixel_values is not None:
                        kwargs["pixel_values"] = pixel_values
                    if image_grid_thw is not None:
                        kwargs["image_grid_thw"] = image_grid_thw
                    embeddings = self.model(**kwargs)
                finally:
                    reset_context()
                
                return embeddings
        
        # For text-only embeddings, use varlen format for optimization
        # Flatten inputs for varlen format
        batch_size = input_ids.shape[0]
        seq_len = input_ids.shape[1] if len(input_ids.shape) > 1 else input_ids.shape[0]
        
        # Compute actual sequence lengths from attention_mask (vLLM approach)
        # This excludes padding tokens from flash-attn computation
        if attention_mask is not None:
            seq_lens = attention_mask.sum(dim=1).cpu().tolist()  # [batch_size]
        else:
            # If no attention_mask, assume all tokens are valid
            seq_lens = [seq_len] * batch_size
        
        # Create cumulative sequence lengths using actual lengths (excluding padding)
        cu_seqlens = torch.zeros(batch_size + 1, dtype=torch.int32, device=model_device)
        cu_seqlens[1:] = torch.cumsum(
            torch.tensor(seq_lens, dtype=torch.int32, device=model_device), dim=0
        )
        max_seqlen = max(seq_lens) if seq_lens else seq_len
        
        # Extract only actual tokens (excluding padding) for flash-attn
        if attention_mask is not None:
            # Create mask for non-padding tokens
            mask = attention_mask.bool()  # [batch_size, seq_len]
            # Extract non-padding tokens
            input_ids_flat = input_ids[mask]  # [num_non_pad_tokens]
            positions_flat = positions[mask]  # [num_non_pad_tokens]
        else:
            # If no attention_mask, use all tokens
            input_ids_flat = input_ids.flatten()
            positions_flat = positions.flatten()
        
        # Create empty slot_mapping (embedding models don't use KV cache)
        slot_mapping = torch.empty(0, dtype=torch.int32, device=model_device)
        
        # Set context for prefill
        from nanovllm.utils.context import set_context, reset_context
        set_context(
            True,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            slot_mapping=slot_mapping,
        )
        
        try:
            # Forward pass (only non-padding tokens)
            # For embedding models, we need to get hidden_states from the base model,
            # then pad back and pool
            if hasattr(self.model, 'model'):
                # The embedding model wraps the base model
                # Get hidden_states from base model first (only non-padding tokens)
                hidden_states_varlen = self.model.model(
                    input_ids_flat, positions_flat
                )
            else:
                # Direct model (shouldn't happen for embedding models, but handle it)
                hidden_states_varlen = self.model(
                    input_ids_flat, positions_flat
                )
            
            # hidden_states_varlen shape: [num_non_pad_tokens, hidden_size]
            # Now we need to pad it back to [batch_size, seq_len, hidden_size] for pooling
            # This matches vLLM's approach: "varlen排除 + 输出补零"
            hidden_size = hidden_states_varlen.shape[-1]
            hidden_states = torch.zeros(
                batch_size, seq_len, hidden_size,
                dtype=hidden_states_varlen.dtype,
                device=hidden_states_varlen.device
            )
            
            # Fill in the actual tokens (non-padding positions)
            if attention_mask is not None:
                mask = attention_mask.bool()  # [batch_size, seq_len]
                hidden_states[mask] = hidden_states_varlen
            else:
                # If no attention_mask, reshape directly
                hidden_states = hidden_states_varlen.view(batch_size, seq_len, -1)
            
            # Now call the embedding model's pooler with the padded hidden_states
            # and normalize if requested
            if hasattr(self.model, 'pooler'):
                # Direct pooling if pooler is accessible
                embeddings = self.model.pooler(hidden_states, attention_mask)
                # Normalize if requested
                if getattr(self.model, 'normalize', False):
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            else:
                # Fallback: call forward method with original input_ids
                # But this would re-process tokens, so it's less efficient
                # For now, raise error to force proper implementation
                raise ValueError("Embedding model does not have pooler attribute. Cannot pool hidden_states directly.")
            
            return embeddings
        finally:
            reset_context()
    
    @torch.inference_mode()
    def rerank(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        token_indices: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        use_flex_attention: bool = True,
        pixel_values: torch.Tensor | None = None,
        image_grid_thw: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rerank documents by computing scores.
        
        Args:
            input_ids: Token IDs, shape [batch_size, seq_len]
            positions: Position IDs, shape [batch_size, seq_len]
            token_indices: Indices of tokens to extract scores from (within each sequence).
                          If None, uses last token for each sequence (pointwise) or special tokens (listwise).
                          Shape: [batch_size] with values in [0, seq_len-1]
            attention_mask: Optional custom attention mask for listwise reranking (deprecated, use block_mask instead).
            use_flex_attention: Whether to use FlexAttention for listwise rerankers (jina_v3).
                          
        Returns:
            For pointwise rerankers: scores, shape [batch_size]
            For listwise rerankers (jina_v3): (scores, query_embeds, doc_embeds)
        """
        # Import context utilities at the start to avoid scope issues
        from nanovllm.utils.context import set_context, reset_context
        
        # Check if model supports reranking
        # Some rerankers (like JinaRerankerM0) use forward() directly instead of
        # compute_score/compute_scores
        has_compute_score = hasattr(self.model, 'compute_score')
        has_compute_scores = hasattr(self.model, 'compute_scores')
        has_forward_rerank = (
            self.is_reranker
            and hasattr(self.model, 'forward')
            and not has_compute_score
            and not has_compute_scores
        )
        
        if not (has_compute_score or has_compute_scores or has_forward_rerank):
            raise ValueError("Model does not support reranking.")
        
        # Get model device and dtype
        model_device = next(self.model.parameters()).device
        model_dtype = next(self.model.parameters()).dtype
        
        # Check if model dtype is supported by FlashAttention
        if model_dtype not in (torch.float16, torch.bfloat16):
            # Try to convert model to fp16 if it's fp32
            if model_dtype == torch.float32:
                print(f"[WARNING] Model dtype is {model_dtype}, converting to float16 for FlashAttention compatibility")
                self.model = self.model.to(torch.float16)
                model_dtype = torch.float16
                # Explicitly clear cache to free the float32 model memory
                torch.cuda.empty_cache()
            else:
                raise ValueError(
                    f"Model dtype {model_dtype} is not supported by FlashAttention. "
                    f"Model weights should be fp16 or bf16, but got {model_dtype}."
                )
        
        # Move inputs to model device
        input_ids = input_ids.to(model_device)
        positions = positions.to(model_device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(model_device)
        if pixel_values is not None:
            pixel_values = pixel_values.to(model_device)
        if image_grid_thw is not None:
            image_grid_thw = image_grid_thw.to(model_device)
        
        # Check if this is a multimodal reranker (has pixel_values)
        # Note: We check pixel_values instead of self.is_multimodal because
        # reranker models set self.is_multimodal=False in __init__
        is_multimodal_reranker_forward = (
            has_forward_rerank
            and self.is_reranker
            and (pixel_values is not None)
        )
        
        # For multimodal rerankers, use batch format (transformers models require it)
        # But we can still optimize by ensuring proper attention_mask usage
        if is_multimodal_reranker_forward:
            # Check if this is a transformers-based model
            is_transformers_model = hasattr(self.model, '__class__') and (
                'transformers' in str(type(self.model)) or
                'Qwen2VLForConditionalGeneration' in str(type(self.model))
            )
            
            if is_transformers_model:
                # Transformers models need batch format, but we can optimize:
                # 1. Ensure left padding for consistent last token extraction
                # 2. Use proper attention_mask to avoid computing on padding
                # 3. The model's internal attention should handle this efficiently
                # Note: We cannot use varlen directly for transformers models
                # because they use their own attention implementation
                if not hasattr(self, '_multimodal_rerank_logged'):
                    print(
                        f"[DEBUG Multimodal Reranker] Using batch format for "
                        f"transformers model: {type(self.model).__name__}"
                    )
                    if attention_mask is not None:
                        seq_lens = attention_mask.sum(dim=1).cpu().tolist()
                        print(
                            f"[DEBUG Multimodal Reranker] Sequence lengths: "
                            f"{seq_lens}"
                        )
                    self._multimodal_rerank_logged = True
                
                kwargs = {
                    "input_ids": input_ids,
                    "positions": positions,
                    "attention_mask": attention_mask,
                }
                if pixel_values is not None:
                    kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    kwargs["image_grid_thw"] = image_grid_thw
                
                # Debug: print input info (only for first call)
                if not hasattr(self, '_debug_rerank_input_logged'):
                    print(f"\n[DEBUG model_runner.rerank (Multimodal)]")
                    print(f"  input_ids shape: {input_ids.shape}")
                    if input_ids.numel() > 0:
                        print(f"  input_ids (first 20): {input_ids[0, :20].tolist()}")
                        print(f"  input_ids (last 20): {input_ids[0, -20:].tolist()}")
                        if input_ids.shape[0] > 1:
                            print(f"  input_ids seq1 (first 20): {input_ids[1, :20].tolist()}")
                            print(f"  input_ids seq1 (last 20): {input_ids[1, -20:].tolist()}")
                    if attention_mask is not None:
                        print(f"  attention_mask shape: {attention_mask.shape}")
                        print(f"  attention_mask (first 20): {attention_mask[0, :20].tolist()}")
                        print(f"  attention_mask (last 20): {attention_mask[0, -20:].tolist()}")
                        print(f"  attention_mask sum: {attention_mask.sum(dim=1).tolist()}")
                        # Check if padding is on left or right
                        first_nonzero = [
                            (attention_mask[i] != 0).nonzero(as_tuple=True)[0][0].item()
                            if (attention_mask[i] != 0).any() else -1
                            for i in range(attention_mask.shape[0])
                        ]
                        print(f"  first_nonzero (padding side check): {first_nonzero}")
                    if pixel_values is not None:
                        print(f"  pixel_values shape: {pixel_values.shape}")
                    if image_grid_thw is not None:
                        print(f"  image_grid_thw: {image_grid_thw.tolist()}")
                    self._debug_rerank_input_logged = True
                
                scores = self.model(**kwargs)
                
                # Debug: print output scores
                if not hasattr(self, '_debug_rerank_output_logged'):
                    print(f"  output scores: {scores.tolist() if hasattr(scores, 'tolist') else scores}")
                    self._debug_rerank_output_logged = True
                
                reset_context()
                return scores
            else:
                # Native nano-vllm model - can use varlen optimization
                # Convert batch to varlen
                batch_size, seq_len = input_ids.shape
                input_ids_flat, positions_flat, cu_seqlens, max_seqlen, seq_lens = (
                    self._batch_to_varlen(input_ids, positions, attention_mask)
                )
                
                # Set context for varlen FlashAttention
                slot_mapping = torch.empty(0, dtype=torch.int32, device=model_device)
                set_context(
                    True,
                    cu_seqlens_q=cu_seqlens,
                    cu_seqlens_k=cu_seqlens,
                    max_seqlen_q=max_seqlen,
                    max_seqlen_k=max_seqlen,
                    slot_mapping=slot_mapping,
                )
                
                try:
                    # Forward pass with varlen format
                    # Note: For multimodal, we need to handle pixel_values and image_grid_thw
                    # This requires model support for varlen multimodal inputs
                    # For now, fall back to batch format for native models too
                    kwargs = {
                        "input_ids": input_ids,
                        "positions": positions,
                        "attention_mask": attention_mask,
                    }
                    if pixel_values is not None:
                        kwargs["pixel_values"] = pixel_values
                    if image_grid_thw is not None:
                        kwargs["image_grid_thw"] = image_grid_thw
                    scores = self.model(**kwargs)
                finally:
                    reset_context()
                
                return scores
        
        # For text-only rerankers, use varlen format for optimization
        # Handle batched inputs
        if input_ids.dim() == 2:
            batch_size, seq_len = input_ids.shape
            
            # Compute actual sequence lengths from attention_mask (vLLM approach)
            # This excludes padding tokens from flash-attn computation
            if attention_mask is not None:
                seq_lens = attention_mask.sum(dim=1).cpu().tolist()  # [batch_size]
                print(f"[DEBUG model_runner] Using attention_mask to compute cu_seqlens_q: seq_lens={seq_lens}")
            else:
                # If no attention_mask, assume all tokens are valid
                seq_lens = [seq_len] * batch_size
            
            # Create cumulative sequence lengths using actual lengths (excluding padding)
            # This matches vLLM's approach: pad tokens don't enter flash-attn kernel
            cu_seqlens_q = torch.zeros(
                batch_size + 1, dtype=torch.int32, device=model_device
            )
            cu_seqlens_q[1:] = torch.cumsum(
                torch.tensor(seq_lens, dtype=torch.int32, device=model_device), dim=0
            )
            cu_seqlens_k = cu_seqlens_q.clone()
            max_seqlen_q = max(seq_lens) if seq_lens else seq_len
            max_seqlen_k = max_seqlen_q
            
            # Extract only actual tokens (excluding padding) for flash-attn
            # Flatten for varlen format, but only include non-padding tokens
            if attention_mask is not None:
                # Create mask for non-padding tokens
                mask = attention_mask.bool()  # [batch_size, seq_len]
                # Extract non-padding tokens
                input_ids_flat = input_ids[mask]  # [num_non_pad_tokens]
                positions_flat = positions[mask]  # [num_non_pad_tokens]
            else:
                # If no attention_mask, use all tokens
                input_ids_flat = input_ids.flatten()
                positions_flat = positions.flatten()
            
            # Token indices for extracting scores (last token of each sequence for pointwise)
            # Don't set token_indices here - let compute_score use attention_mask if available
            # if token_indices is None and hasattr(self.model, 'compute_score'):
            #     token_indices = torch.full(
            #         (batch_size,), seq_len - 1,
            #         dtype=torch.int64,
            #         device=input_ids.device,
            #     )
        else:
            raise NotImplementedError("Only batched input format supported for rerank")
        
        # Create dummy slot_mapping (not used for reranking without KV cache)
        slot_mapping = torch.empty(0, dtype=torch.int32, device=model_device)
        
        set_context(
            True,
            cu_seqlens_q,
            cu_seqlens_k,
            max_seqlen_q,
            max_seqlen_k,
            slot_mapping,
            None,
            None,
        )
        
        # Forward pass (only prefill, no decode)
        # Use standard Flash Attention with causal mask (no FlexAttention block_mask needed)
        hidden_states_varlen = self.model(input_ids_flat, positions_flat)
        
        # Process hidden_states (text-only rerankers only, multimodal already handled above)
        # hidden_states_varlen shape: [num_non_pad_tokens, hidden_size]
        # Now we need to pad it back to [batch_size, seq_len, hidden_size] to match Transformers
        # This matches vLLM's approach: "varlen排除 + 输出补零"
        hidden_size = hidden_states_varlen.shape[-1]
        hidden_states = torch.zeros(
            batch_size, seq_len, hidden_size,
            dtype=hidden_states_varlen.dtype,
            device=hidden_states_varlen.device
        )
        
        # Fill in the actual tokens (non-padding positions)
        # IMPORTANT: hidden_states_varlen is in varlen format (concatenated sequences),
        # so we need to fill it according to cu_seqlens_q, not just using mask
        if attention_mask is not None:
            mask = attention_mask.bool()  # [batch_size, seq_len]
            # Fill hidden_states according to cu_seqlens_q order
            # hidden_states_varlen is ordered as: seq0_tokens, seq1_tokens, seq2_tokens, ...
            varlen_idx = 0
            for batch_idx in range(batch_size):
                seq_len_actual = seq_lens[batch_idx]
                # Fill only the actual (non-padding) positions for this sequence
                for pos_idx in range(seq_len_actual):
                    if mask[batch_idx, pos_idx]:
                        hidden_states[batch_idx, pos_idx] = hidden_states_varlen[varlen_idx]
                        varlen_idx += 1
        else:
            # If no attention_mask, reshape directly
            hidden_states = hidden_states_varlen.view(batch_size, seq_len, -1)
        
        # Debug: print hidden_states at last position for comparison with Transformers
        print(f"[DEBUG model_runner] hidden_states shape: {hidden_states.shape}")
        print(f"[DEBUG model_runner] hidden_states at last position mean: {hidden_states[:, -1, :].mean().item():.6f}, std: {hidden_states[:, -1, :].std().item():.6f}")
        print(f"[DEBUG model_runner] hidden_states first seq at last position: {hidden_states[0, -1, :10].tolist()}... (first 10)")
        if batch_size > 1:
            print(f"[DEBUG model_runner] hidden_states second seq at last position: {hidden_states[1, -1, :10].tolist()}... (first 10)")
        
        # Compute scores
        if hasattr(self.model, 'compute_scores'):
            # Listwise reranker (jina_v3)
            scores, query_embeds, doc_embeds = self.model.compute_scores(
                hidden_states, input_ids
            )
            reset_context()
            return scores, query_embeds, doc_embeds
        elif hasattr(self.model, 'compute_score'):
            # Pointwise reranker with compute_score method
            print(f"[DEBUG model_runner] token_indices: {token_indices}")
            print(f"[DEBUG model_runner] attention_mask: {attention_mask}")
            if attention_mask is not None:
                print(
                    f"[DEBUG model_runner] attention_mask shape: "
                    f"{attention_mask.shape}"
                )
            # Pass seq_lens to compute_score so it can calculate correct token_indices
            # if token_indices is None
            scores = self.model.compute_score(
                hidden_states, token_indices, attention_mask
            )
            reset_context()
            return scores
        elif has_forward_rerank:
            # Reranker that uses forward() directly (e.g., JinaRerankerM0)
            # Note: Multimodal rerankers are already handled above (before varlen processing)
            # This branch should only be reached for text-only rerankers using forward()
            # which is not currently supported
            raise NotImplementedError(
                "Text-only rerankers using forward() not yet supported. "
                "Multimodal rerankers should be handled before varlen processing."
            )
        else:
            raise ValueError("Model does not support reranking.")

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        """Run model forward pass for generation.
        
        Note: This method already uses varlen format for prefill (via prepare_prefill),
        which converts batch format to varlen format for FlashAttention optimization.
        For multimodal models, it uses vision_slices_per_seq to handle image tokens.
        """
        if is_prefill:
            # prepare_prefill converts batch to varlen format automatically
            input_ids, positions = self.prepare_prefill(seqs)
        else:
            input_ids, positions = self.prepare_decode(seqs)

        # Track how many freshly decoded tokens each sequence contributes; the
        # model uses these lengths to align partial vision slices with text.
        sequence_lengths = (
            [len(seq) - seq.num_cached_tokens for seq in seqs]
            if is_prefill
            else None
        )
        vision_slices_per_seq = None

        if is_prefill and self.is_multimodal:
            vision_slices_per_seq = []
            has_slices = False

            for seq in seqs:
                # Cache the full vision tower output once; subsequent prefill
                # steps only read the portions still needed for this sequence.
                # Note: For prefill-only single-token, we skip caching to save
                # memory and will pass pixel_values directly to model (fallback
                # path)
                self._ensure_vision_cache(seq)
                slices_for_seq: list[dict] = []
                window_start = seq.num_cached_tokens
                window_end = len(seq)

                for placeholder_idx, (offset, length) in enumerate(
                    seq.vision_placeholders
                ):
                    if placeholder_idx >= len(seq.vision_counts):
                        continue
                    consumed = seq.vision_consumed[placeholder_idx]
                    total_len = length
                    if consumed >= total_len:
                        continue

                    range_start = offset
                    range_end = offset + total_len

                    overlap_start = max(range_start, window_start)
                    overlap_end = min(range_end, window_end)
                    if overlap_end <= overlap_start:
                        continue

                    slice_offset = max(consumed, overlap_start - range_start)
                    remaining = total_len - slice_offset
                    overlap_available = overlap_end - overlap_start
                    take = min(remaining, overlap_available)
                    if take <= 0:
                        continue

                    target_offset = overlap_start - window_start

                    # Check if cached_vision_tokens has enough elements
                    if (not seq.cached_vision_tokens or 
                        placeholder_idx >= len(seq.cached_vision_tokens)):
                        continue

                    chunk_tokens = seq.cached_vision_tokens[placeholder_idx]
                    token_slice = (
                        chunk_tokens[slice_offset:slice_offset + take]
                        .to(
                            device="cuda",
                            dtype=self.model_dtype,
                            non_blocking=True,
                        )
                        .contiguous()
                    )

                    deepstack_slice: list[torch.Tensor] | None = None
                    if seq.cached_deepstack_tokens:
                        deepstack_slice = []
                        for layer_tokens in seq.cached_deepstack_tokens:
                            if placeholder_idx >= len(layer_tokens):
                                deepstack_slice.append(None)
                                continue
                            layer_slice = (
                                layer_tokens[placeholder_idx][
                                    slice_offset:slice_offset + take
                                ]
                                .to(
                                    device="cuda",
                                    dtype=self.model_dtype,
                                    non_blocking=True,
                                )
                                .contiguous()
                            )
                            deepstack_slice.append(layer_slice)

                    slices_for_seq.append(
                        {
                            "tokens": token_slice,
                            "deepstack": deepstack_slice,
                            "length": take,
                            "target_offset": target_offset,
                            "placeholder_idx": placeholder_idx,
                        }
                    )
                    has_slices = True

                vision_slices_per_seq.append(slices_for_seq)

            if not has_slices:
                vision_slices_per_seq = None

        def _advance_vision_offsets():
            if not is_prefill or not self.is_multimodal:
                return
            if vision_slices_per_seq is None:
                return
            for seq, slices in zip(seqs, vision_slices_per_seq):
                for slice_info in slices:
                    length = slice_info["length"]
                    placeholder_idx = slice_info["placeholder_idx"]
                    if placeholder_idx < len(seq.vision_consumed):
                        span = seq.vision_placeholders[placeholder_idx][1]
                        seq.vision_consumed[placeholder_idx] += length
                        seq.vision_consumed[placeholder_idx] = min(
                            seq.vision_consumed[placeholder_idx],
                            span,
                        )
                if seq.vision_placeholders:
                    # Once every placeholder has been consumed we can drop the
                    # cached tensors to release CPU memory.
                    all_consumed = all(
                        seq.vision_consumed[idx] >= span
                        for idx, (_, span) in enumerate(
                            seq.vision_placeholders
                        )
                    )
                else:
                    all_consumed = True
                if all_consumed:
                    seq.cached_vision_tokens = None
                    seq.cached_deepstack_tokens = None

        temperatures = self.prepare_sample(seqs) if self.rank == 0 else None
        logits = self.run_model(
            input_ids,
            positions,
            is_prefill,
            sequence_lengths=sequence_lengths,
            vision_slices_per_seq=vision_slices_per_seq,
            seqs=seqs,  # Pass seqs for llavanext pixel_values handling
        )
        _advance_vision_offsets()
        
        # For prefill-only single-token generation, explicitly clear GPU and CPU
        # memory used by vision cache to reduce peak memory usage
        config_prefill_only = getattr(self.config, "prefill_only_mode", False)
        config_single_token = getattr(self.config, "single_token_mode", False)
        if config_prefill_only and config_single_token:
            # Record memory before cleanup for logging
            if torch.cuda.is_available():
                mem_before = torch.cuda.memory_allocated() / 1024**2
                stats_before = torch.cuda.memory_stats()
                reserved_before = stats_before.get(
                    "reserved_bytes.all.current", 0
                ) / 1024**2
            
            # Clear GPU memory used by vision slices immediately after use
            if vision_slices_per_seq:
                for slices in vision_slices_per_seq:
                    for slice_info in slices:
                        if "tokens" in slice_info and slice_info["tokens"] is not None:
                            del slice_info["tokens"]
                        if "deepstack" in slice_info and slice_info["deepstack"]:
                            for layer_slice in slice_info["deepstack"]:
                                if layer_slice is not None:
                                    del layer_slice
                            slice_info["deepstack"] = None
                vision_slices_per_seq = None
            
            # Clear CPU vision cache immediately after use to save memory
            # This is important because vision tokens can be very large
            vision_mem_mb = 0
            for seq in seqs:
                if seq.cached_vision_tokens is not None:
                    # Calculate memory saved (for logging)
                    if isinstance(seq.cached_vision_tokens, list):
                        for emb in seq.cached_vision_tokens:
                            if isinstance(emb, torch.Tensor):
                                vision_mem_mb += (
                                    emb.numel() * emb.element_size() / 1024**2
                                )
                    seq.cached_vision_tokens = None
                    
                if seq.cached_deepstack_tokens is not None:
                    if isinstance(seq.cached_deepstack_tokens, list):
                        for layer_tokens in seq.cached_deepstack_tokens:
                            if isinstance(layer_tokens, list):
                                for feat in layer_tokens:
                                    if isinstance(feat, torch.Tensor):
                                        vision_mem_mb += (
                                            feat.numel() * feat.element_size() / 1024**2
                                        )
                    seq.cached_deepstack_tokens = None
            
            # Clear GPU cache
            torch.cuda.empty_cache()
            
            # Log memory cleanup if significant memory was freed
            if torch.cuda.is_available():
                mem_after = torch.cuda.memory_allocated() / 1024**2
                stats_after = torch.cuda.memory_stats()
                reserved_after = stats_after.get(
                    "reserved_bytes.all.current", 0
                ) / 1024**2
                
                gpu_freed = mem_before - mem_after
                reserved_freed = reserved_before - reserved_after
                
                # Log if significant memory was freed or vision cache was large
                if vision_mem_mb > 10 or gpu_freed > 10 or reserved_freed > 10:
                    print(
                        f"[Memory] Cleaned vision cache: "
                        f"GPU allocated freed {gpu_freed:.2f} MB, "
                        f"GPU reserved freed {reserved_freed:.2f} MB, "
                        f"CPU vision cache ~{vision_mem_mb:.2f} MB"
                    )
        if self.rank == 0:
            # Debug logits information
            if not hasattr(self, '_debug_logits_info_logged'):
                print(f"\n[DEBUG model_runner.run] Logits information:")
                print(f"  logits shape: {logits.shape}")
                print(f"  logits dtype: {logits.dtype}")
                print(f"  logits device: {logits.device}")
                print(f"  temperatures shape: {temperatures.shape}")
                print(f"  temperatures dtype: {temperatures.dtype}")
                if logits.numel() > 0:
                    print(f"  logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
                    print(f"  logits mean: {logits.mean().item():.4f}")
                    print(f"  logits has NaN: {torch.isnan(logits).any().item()}")
                    print(f"  logits has Inf: {torch.isinf(logits).any().item()}")
                    print(f"  logits all zeros: {(logits == 0).all().item()}")
                    # Check first few logits values
                    if logits.shape[0] > 0 and logits.shape[1] > 0:
                        print(f"  logits[0, :5]: {logits[0, :5].tolist()}")
                        print(f"  logits[-1, :5]: {logits[-1, :5].tolist()}")
                print(f"  is_prefill: {is_prefill}")
                print(f"  sequence_lengths: {sequence_lengths}")
                if seqs:
                    print(f"  num_seqs: {len(seqs)}")
                    print(f"  first seq len: {len(seqs[0])}")
                self._debug_logits_info_logged = True
            
            # Expand temperatures from batch format to varlen format if needed
            # Check if logits and temperatures shapes are compatible
            if logits.dim() == 2 and temperatures.dim() == 1:
                logits_num_tokens = logits.shape[0]
                temperatures_batch_size = temperatures.shape[0]
                
                # If shapes don't match, we need to expand temperatures
                if logits_num_tokens != temperatures_batch_size:
                    # Get sequence_lengths if not available
                    if sequence_lengths is None and is_prefill:
                        sequence_lengths = [
                            len(seq) - seq.num_cached_tokens for seq in seqs
                        ]
                    
                    # Check if we can expand
                    if sequence_lengths is not None:
                        expected_total = sum(sequence_lengths)
                        if expected_total == logits_num_tokens:
                            # Expand temperatures to varlen
                            temperatures_varlen = self.expand_batch_to_tokens(
                                temperatures, sequence_lengths
                            )
                            # Debug log
                            if not hasattr(self, '_debug_temp_expand_logged'):
                                print(f"\n[DEBUG model_runner.run] Temperature expansion:")
                                print(f"  temperatures (batch) shape: {temperatures.shape}")
                                print(f"  temperatures_varlen shape: {temperatures_varlen.shape}")
                                print(f"  logits shape: {logits.shape}")
                                print(f"  sequence_lengths: {sequence_lengths}")
                                self._debug_temp_expand_logged = True
                            token_ids = self.sampler(logits, temperatures_varlen).tolist()
                        else:
                            # Shape mismatch - this is an error condition
                            raise ValueError(
                                f"logits shape {logits.shape} doesn't match "
                                f"expected total tokens {expected_total} from "
                                f"sequence_lengths {sequence_lengths}"
                            )
                    else:
                        # Cannot expand without sequence_lengths
                        raise ValueError(
                            f"Cannot expand temperatures: logits has {logits_num_tokens} "
                            f"tokens but temperatures has {temperatures_batch_size} values, "
                            f"and sequence_lengths is not available"
                        )
                else:
                    # Shapes match - use as-is
                    if not hasattr(self, '_debug_sampler_input_logged'):
                        print(f"\n[DEBUG model_runner.run] Sampler input (shapes match):")
                        print(f"  logits shape: {logits.shape}")
                        print(f"  temperatures shape: {temperatures.shape}")
                        print(f"  logits[0, :10]: {logits[0, :10].tolist()}")
                        self._debug_sampler_input_logged = True
                    token_ids = self.sampler(logits, temperatures).tolist()
                    if not hasattr(self, '_debug_sampler_output_logged'):
                        print(f"\n[DEBUG model_runner.run] Sampler output:")
                        print(f"  token_ids: {token_ids}")
                        print(f"  token_ids type: {type(token_ids)}")
                        print(f"  token_ids length: {len(token_ids) if isinstance(token_ids, list) else 'N/A'}")
                        self._debug_sampler_output_logged = True
            else:
                # Unexpected dimensions - use as-is (fallback)
                token_ids = self.sampler(logits, temperatures).tolist()
        else:
            token_ids = None
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        # Skip cudagraph capture for embedding and reranker models
        if self.is_embedding or self.is_reranker:
            return
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (
            config.max_model_len + self.block_size - 1
        ) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(
                False,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            warmup_out = self.model(input_ids[:bs], positions[:bs])
            outputs[:bs] = warmup_out  # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                capture_out = self.model(input_ids[:bs], positions[:bs])
                outputs[:bs] = capture_out  # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )

    def _ensure_vision_cache(self, seq: Sequence):
        if seq.cached_vision_tokens is not None:
            return
        if seq.pixel_values is None or seq.image_grid_thw is None:
            seq.cached_vision_tokens = []
            seq.cached_deepstack_tokens = []
            return

        # For prefill-only single-token generation, skip vision cache to save memory.
        # Since sequences are short, we don't need chunked prefill, so vision cache
        # is unnecessary. Instead, we'll pass pixel_values directly to the model
        # which will compute vision tokens on-the-fly (fallback path).
        config_prefill_only = getattr(self.config, "prefill_only_mode", False)
        config_single_token = getattr(self.config, "single_token_mode", False)
        if config_prefill_only and config_single_token:
            # Don't cache vision tokens - save memory by avoiding GPU->CPU->GPU transfer
            # Keep pixel_values and image_grid_thw so they can be passed to model
            # The model's forward method has a fallback path that accepts pixel_values
            seq.cached_vision_tokens = []
            seq.cached_deepstack_tokens = []
            # IMPORTANT: Do NOT set seq.pixel_values = None here!
            # We need to keep pixel_values so it can be passed to model.forward()
            if not hasattr(self, '_debug_skip_vision_cache_logged'):
                print(
                    "[Memory] Skipping vision cache for prefill-only "
                    "single-token generation (will use pixel_values fallback path)"
                )
                self._debug_skip_vision_cache_logged = True
            return

        # Check if model has visual method (qwen2vl, qwen2.5vl, qwen3vl)
        # llavanext doesn't have visual method, it processes pixel_values in forward
        if not hasattr(self.model, 'visual'):
            # For llavanext, we don't cache vision tokens separately
            # The model will process pixel_values directly in forward
            # Log this for debugging
            if not hasattr(self, '_debug_llavanext_no_visual_logged'):
                print(f"\n[DEBUG model_runner._ensure_vision_cache (llavanext)]")
                print(f"  Model does not have 'visual' method (llavanext)")
                print(f"  seq.pixel_values type: {type(seq.pixel_values)}")
                if isinstance(seq.pixel_values, list):
                    print(f"  seq.pixel_values is list with {len(seq.pixel_values)} elements")
                elif isinstance(seq.pixel_values, torch.Tensor):
                    print(f"  seq.pixel_values is tensor, shape: {seq.pixel_values.shape}, dim: {seq.pixel_values.dim()}")
                print(f"  Skipping vision cache, pixel_values will be passed directly to forward()")
                self._debug_llavanext_no_visual_logged = True
            seq.cached_vision_tokens = []
            seq.cached_deepstack_tokens = []
            return

        # Handle pixel_values format
        # For llavanext, pixel_values might be a list of tensors
        # For other models (qwen2vl, qwen2.5vl, qwen3vl), it's a tensor
        if isinstance(seq.pixel_values, list):
            # For llavanext with list format, we can't use visual method
            # This shouldn't happen if model has visual method, but handle it anyway
            seq.cached_vision_tokens = []
            seq.cached_deepstack_tokens = []
            return

        # Run the vision encoder once on the GPU and stash the outputs on CPU.
        # Later prefill iterations reuse these tensors without recomputing the
        # expensive 3D convolutions.
        pixel = seq.pixel_values.to(
            device="cuda",
            dtype=self.model_dtype,
            non_blocking=True,
        ).contiguous()
        grid = seq.image_grid_thw.to(
            device="cuda",
            dtype=torch.int32,
            non_blocking=True,
        ).contiguous()

        visual_output = self.model.visual(pixel, grid)
        # Handle different return types:
        # - qwen2vl: returns list[torch.Tensor] (only image_embeds)
        # - qwen2.5vl/qwen3vl: returns (image_embeds, deepstack_features)
        if isinstance(visual_output, tuple):
            image_embeds, deepstack_features = visual_output
        else:
            # qwen2vl returns only image_embeds as a list
            image_embeds = visual_output
            deepstack_features = None

        seq.cached_vision_tokens = [emb.detach().cpu() for emb in image_embeds]
        if deepstack_features:
            cached_deepstack = []
            for layer_tokens in deepstack_features:
                cached_layer = [feat.detach().cpu() for feat in layer_tokens]
                cached_deepstack.append(cached_layer)
            seq.cached_deepstack_tokens = cached_deepstack
        else:
            seq.cached_deepstack_tokens = []

        seq.pixel_values = None
        seq.image_grid_thw = None