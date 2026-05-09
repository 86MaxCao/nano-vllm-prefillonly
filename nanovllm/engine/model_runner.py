import pickle
import random
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.model_loader import (
    ModelLoader,
    get_torch_dtype,
    get_target_dtype_for_embedding_reranker,
    infer_embedding_type,
    infer_reranker_type,
    infer_multimodal_model_type,
    MULTIMODAL_AVAILABLE,
)
from nanovllm.engine.multimodal_handler import MultimodalHandler
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import set_context, get_context, reset_context


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
        torch_dtype = get_torch_dtype(hf_config)
        torch.set_default_dtype(torch_dtype)
        torch.set_default_device("cuda")

        # Embedding support (check first, as embedding models can be multimodal)
        self.is_embedding = getattr(config, "is_embedding", False)
        embedding_type = getattr(config, "embedding_type", None)
        pooling_type = getattr(config, "pooling_type", "LAST")
        normalize_embeddings = getattr(config, "normalize_embeddings", True)
        
        # Reranker support (check second, as reranker models can be multimodal)
        self.is_reranker = getattr(config, "is_reranker", False)
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
        
        # Determine target dtype for embedding/reranker models (before model creation)
        target_dtype = None
        if self.is_embedding or self.is_reranker:
            target_dtype = get_target_dtype_for_embedding_reranker(hf_config)
            torch.set_default_dtype(target_dtype)
        
        # Auto-infer model types if not specified
        if self.is_embedding and embedding_type is None:
            embedding_type = infer_embedding_type(config, hf_config)
            if embedding_type is None:
                raise ValueError("Cannot infer embedding_type. Please specify it explicitly.")
        
        if self.is_reranker and reranker_type is None:
            reranker_type = infer_reranker_type(config, hf_config)
            if reranker_type is None:
                raise ValueError("Cannot infer reranker_type. Please specify it explicitly.")
        
        multimodal_model_type = getattr(config, "multimodal_model_type", None)
        if self.is_multimodal and multimodal_model_type is None:
            multimodal_model_type = infer_multimodal_model_type(config, hf_config)
            if multimodal_model_type is None:
                multimodal_model_type = "qwen3_vl"  # default fallback
        
        # Load model based on type
        if self.is_embedding:
            self.model = ModelLoader.load_embedding_model(
                config, hf_config, embedding_type, pooling_type,
                normalize_embeddings, target_dtype
            )
        elif self.is_reranker:
            self.model = ModelLoader.load_reranker_model(
                config, hf_config, reranker_type, target_dtype
            )
        elif self.is_multimodal:
            self.model = ModelLoader.load_multimodal_model(
                config, multimodal_model_type
            )
        else:
            self.model = ModelLoader.load_text_model(config, hf_config)

        embed_module = getattr(self.model, "language_model", self.model)
        if hasattr(embed_module, "model"):
            embed_module = embed_module.model
        if not hasattr(embed_module, "embed_tokens"):
            # Handle transformers Qwen2VLForConditionalGeneration hierarchy:
            # model.model.language_model.embed_tokens
            if hasattr(self.model, "model"):
                inner = getattr(self.model, "model", None)
                if inner is not None and hasattr(inner, "language_model"):
                    embed_module = inner.language_model
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
            
            # Determine if we should process multimodal inputs
            has_multimodal_input = False
            if seqs is not None:
                has_multimodal_input = any(
                    seq.pixel_values is not None or 
                    (hasattr(seq, 'vision_placeholders') and seq.vision_placeholders)
                    for seq in seqs
                )
            
            if self.is_multimodal and has_multimodal_input:
                model_kwargs = self._prepare_multimodal_inputs(
                    is_prefill, sequence_lengths, vision_slices_per_seq, seqs
                )
                if not hasattr(self.model, 'visual') and is_prefill and seqs:
                    outputs = self.model(input_ids, **model_kwargs)
                else:
                    outputs = self.model(input_ids, positions, **model_kwargs)
            else:
                if (self.is_embedding or self.is_reranker) and sequence_lengths is not None:
                    model_kwargs["sequence_lengths"] = sequence_lengths
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


    def _prepare_multimodal_inputs(
        self,
        is_prefill: bool,
        sequence_lengths: list[int] | None,
        vision_slices_per_seq: list[list[dict]] | None,
        seqs: list[Sequence] | None,
    ) -> dict:
        """Prepare multimodal model inputs using MultimodalHandler."""
        model_kwargs = {}
        
        # Handle llavanext models (no visual method)
        if not hasattr(self.model, 'visual') and is_prefill and seqs:
            model_kwargs = MultimodalHandler.prepare_llavanext_inputs(self.model, seqs)
            if sequence_lengths is not None:
                model_kwargs["sequence_lengths"] = sequence_lengths
            return model_kwargs
        
        model_kwargs["sequence_lengths"] = sequence_lengths
        
        config_prefill_only = getattr(self.config, "prefill_only_mode", False)
        config_single_token = getattr(self.config, "single_token_mode", False)
        
        if config_prefill_only and config_single_token and seqs is not None:
            model_kwargs.update(MultimodalHandler.prepare_prefill_only_inputs(self.model, seqs))
            model_kwargs["sequence_lengths"] = sequence_lengths
        elif vision_slices_per_seq is not None:
            model_kwargs["vision_slices_per_seq"] = vision_slices_per_seq
        
        return model_kwargs

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
                kwargs = {
                    "input_ids": input_ids,
                    "positions": positions,
                    "attention_mask": attention_mask,
                }
                if pixel_values is not None:
                    kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    kwargs["image_grid_thw"] = image_grid_thw
                # Pass sequence_lengths for VL rerankers that need it
                if attention_mask is not None:
                    kwargs["sequence_lengths"] = attention_mask.sum(dim=1).cpu().tolist()
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
        
        # For transformers-based forward rerankers (e.g., JinaRerankerM0),
        # always use batch format — they can't handle varlen input.
        # Handle both text-only and multimodal cases early, before varlen conversion.
        if has_forward_rerank:
            try:
                from transformers import Qwen2VLForConditionalGeneration as _Qwen2VL
                is_transformers_model = isinstance(self.model, _Qwen2VL)
            except ImportError:
                is_transformers_model = False
            if is_transformers_model:
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
                reset_context()
                return scores
        
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
                kwargs = {
                    "input_ids": input_ids,
                    "positions": positions,
                    "attention_mask": attention_mask,
                }
                if pixel_values is not None:
                    kwargs["pixel_values"] = pixel_values
                if image_grid_thw is not None:
                    kwargs["image_grid_thw"] = image_grid_thw
                # Pass sequence_lengths for VL rerankers that need it
                if attention_mask is not None:
                    kwargs["sequence_lengths"] = attention_mask.sum(dim=1).cpu().tolist()
                
                scores = self.model(**kwargs)
                
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
            scores = self.model.compute_score(
                hidden_states, token_indices, attention_mask
            )
            reset_context()
            return scores
        elif has_forward_rerank:
            # Native nano-vllm forward rerankers (non-transformers)
            raise NotImplementedError(
                "Native nano-vllm forward-rerankers not yet supported."
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
                    token_ids = self.sampler(logits, temperatures).tolist()
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