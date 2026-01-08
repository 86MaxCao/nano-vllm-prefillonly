import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3Model, Qwen3ForCausalLM


class Qwen3Reranker(Qwen3ForCausalLM):
    """Qwen3-Reranker model for sequence classification/reranking tasks.
    
    Supports both:
    1. Original Qwen3-Reranker (uses "no"/"yes" token logits)
    2. Converted sequence classification model (has score_head)
    """
    packed_modules_mapping = Qwen3ForCausalLM.packed_modules_mapping

    def __init__(
        self,
        config: Qwen3Config,
        is_original_reranker: bool = False,
        classifier_from_token: list[str] | None = None,
    ) -> None:
        super().__init__(config)
        
        # Save config for use in convert_from_original_reranker
        self.config = config
        
        self.is_original_reranker = is_original_reranker
        self.classifier_from_token = classifier_from_token or []
        
        # For reranker, we replace lm_head with score_head
        # But for original reranker, we'll keep lm_head temporarily for weight loading
        if not is_original_reranker:
            # Converted model: directly use score_head
            self.score_head = ReplicatedLinear(
                config.hidden_size,
                1,  # num_labels for reranker
                bias=False,
            )
            # Remove lm_head if it exists
            if hasattr(self, 'lm_head'):
                delattr(self, 'lm_head')
        else:
            # Original reranker: keep lm_head for weight loading, will be converted later
            assert len(self.classifier_from_token) == 2, \
                "Original Qwen3-Reranker requires exactly 2 tokens (e.g., ['no', 'yes'])"
            # Will create score_head after loading weights

    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for reranking.
        
        Returns:
            hidden_states: Hidden states from the model backbone
        """
        return self.model(input_ids, positions)

    def compute_score(
        self,
        hidden_states: torch.Tensor,
        token_indices: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute reranking scores from hidden states.
        
        Args:
            hidden_states: Hidden states from forward pass, shape [batch_size, seq_len, hidden_size]
            token_indices: Indices of tokens to extract scores from (within each sequence).
                          Shape: [batch_size] with values in [0, seq_len-1]
                          If None, uses last token for each sequence.
            attention_mask: Attention mask, shape [batch_size, seq_len]. If provided, uses it
                          to find the last real token for each sequence.
            
        Returns:
            scores: Reranking scores, shape [batch_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        print(f"[DEBUG compute_score] token_indices: {token_indices}")
        print(f"[DEBUG compute_score] attention_mask: {attention_mask}")
        if attention_mask is not None:
            print(f"[DEBUG compute_score] attention_mask shape: {attention_mask.shape}")
        
        if token_indices is None:
            # Use the actual last token position for each sequence (not padding position)
            # Transformers uses logits[:, -1, :], but for padded sequences, we need to use
            # the actual last token position (seq_len_actual - 1) instead of seq_len - 1
            # to avoid getting padding positions (which are zeros in our implementation)
            if attention_mask is not None:
                # Calculate actual sequence lengths from attention_mask
                seq_lens_actual = attention_mask.sum(dim=1)  # [batch_size]
                # Use actual last token position for each sequence
                token_indices = (seq_lens_actual - 1).to(torch.int64)
                print(f"[DEBUG] Using actual last token positions: {token_indices.tolist()} (from seq_lens: {seq_lens_actual.tolist()})")
            else:
                # If no attention_mask, assume all sequences have the same length
                token_indices = torch.full(
                    (batch_size,), seq_len - 1,
                dtype=torch.int64,
                device=hidden_states.device,
            )
                print(f"[DEBUG] Using seq_len - 1 = {seq_len - 1} as token_indices (no attention_mask)")
        
        # Extract hidden states at specified positions
        # token_indices is [batch_size] with seq positions
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        selected_states = hidden_states[batch_indices, token_indices]  # [batch_size, hidden_size]
        print(f"[DEBUG] selected_states shape: {selected_states.shape}")
        print(f"[DEBUG] selected_states mean: {selected_states.mean().item():.6f}, std: {selected_states.std().item():.6f}")
        print(f"[DEBUG] selected_states first seq: {selected_states[0, :10].tolist()}... (first 10)")
        print(f"[DEBUG] selected_states second seq: {selected_states[1, :10].tolist()}... (first 10)")
        
        # Compute raw score (logit(yes) - logit(no))
        raw_scores = self.score_head(selected_states).squeeze(-1)
        print(f"[DEBUG] raw_scores: {raw_scores.tolist()}")
        print(f"[DEBUG] score_head weight mean: {self.score_head.weight.data.mean().item():.6f}, std: {self.score_head.weight.data.std().item():.6f}")
        
        # Apply sigmoid to convert to probability (matching transformers
        # baseline). This is equivalent to:
        # exp(log_softmax([logit(no), logit(yes)])[1])
        scores = torch.sigmoid(raw_scores)
        print(f"[DEBUG] scores after sigmoid: {scores.tolist()}")
        return scores

    def convert_from_original_reranker(self, tokenizer):
        """Convert original Qwen3-Reranker weights to score_head.
        
        This should be called after loading weights (including lm_head).
        Reference: https://huggingface.co/Qwen/Qwen3-Reranker-0.6B/discussions/3
        """
        if not self.is_original_reranker or not hasattr(self, 'lm_head'):
            return
        
        # Extract token IDs
        false_token, true_token = self.classifier_from_token
        false_id = tokenizer.convert_tokens_to_ids(false_token)
        true_id = tokenizer.convert_tokens_to_ids(true_token)
        
        # Get the dtype and device of the model weights
        model_dtype = self.lm_head.weight.data.dtype
        model_device = self.lm_head.weight.data.device

        # For ParallelLMHead, weights are always sharded (even when tp_size == 1)
        # Need to gather all shards to access arbitrary token IDs
        import torch.distributed as dist
        tp_size = getattr(self.lm_head, 'tp_size', 1)
        
        if tp_size > 1:
            # Multi-GPU case: gather all weight shards
            all_weights = [
                torch.empty_like(self.lm_head.weight.data)
                for _ in range(tp_size)
            ]
            dist.all_gather(all_weights, self.lm_head.weight.data)
            if dist.get_rank() == 0:
                # Concatenate all shards to get full weight matrix
                full_weight = torch.cat(all_weights, dim=0)
                # [vocab_size, hidden_size]
                # Use [[true_id]] and [[false_id]] to keep 2D shape, matching vLLM
                weight_true = full_weight[[true_id]].to(torch.float32)  # [1, hidden_size]
                weight_false = full_weight[[false_id]].to(torch.float32)  # [1, hidden_size]
                # [1, hidden_size]
                score_weight = (weight_true - weight_false).to(model_dtype)
            else:
                score_weight = torch.zeros(
                    1, self.config.hidden_size,
                    dtype=model_dtype,
                    device=model_device
                )
            # Broadcast from rank 0 to all ranks
            dist.broadcast(score_weight, src=0)
        else:
            # Single GPU case: weights are still sharded (ParallelLMHead structure)
            # Need to check if token IDs are in the current shard range
            vocab_start_idx = self.lm_head.vocab_start_idx
            vocab_end_idx = self.lm_head.vocab_end_idx
            
            # Check if both tokens are in the current shard
            if (vocab_start_idx <= true_id < vocab_end_idx and 
                vocab_start_idx <= false_id < vocab_end_idx):
                # Both tokens are in current shard, can access directly
                local_true_id = true_id - vocab_start_idx
                local_false_id = false_id - vocab_start_idx
                weight_true = self.lm_head.weight.data[[local_true_id]].to(torch.float32)
                weight_false = self.lm_head.weight.data[[local_false_id]].to(torch.float32)
                score_weight = (weight_true - weight_false).to(model_dtype)
            else:
                # Need to gather weights (shouldn't happen with tp_size == 1, but handle gracefully)
                # For tp_size == 1, vocab_start_idx should be 0 and vocab_end_idx should be vocab_size
                # So this branch should rarely be hit, but we handle it for safety
                all_weights = [self.lm_head.weight.data]
                full_weight = torch.cat(all_weights, dim=0)
                weight_true = full_weight[[true_id]].to(torch.float32)
                weight_false = full_weight[[false_id]].to(torch.float32)
                score_weight = (weight_true - weight_false).to(model_dtype)
            
            print(f"[DEBUG] Weight conversion:")
            print(f"[DEBUG]   true_id: {true_id}, false_id: {false_id}")
            print(f"[DEBUG]   vocab_start_idx: {vocab_start_idx}, vocab_end_idx: {vocab_end_idx}")
            print(f"[DEBUG]   weight_true shape: {weight_true.shape}, mean: {weight_true.mean().item():.6f}, std: {weight_true.std().item():.6f}")
            print(f"[DEBUG]   weight_false shape: {weight_false.shape}, mean: {weight_false.mean().item():.6f}, std: {weight_false.std().item():.6f}")
            print(f"[DEBUG]   score_weight shape: {score_weight.shape}, mean: {score_weight.mean().item():.6f}, std: {score_weight.std().item():.6f}")
        
        # Create score_head and load the converted weight
        # vLLM uses weight_loader, but we can directly copy since score_weight is already [1, hidden_size]
        self.score_head = ReplicatedLinear(
            self.config.hidden_size,
            1,
            bias=False,
        )
        # Ensure score_head is on the same device as the model
        self.score_head = self.score_head.to(model_device)
        # score_weight is already [1, hidden_size], matching score_head.weight shape
        self.score_head.weight.data.copy_(score_weight)
        
        # Remove lm_head
        delattr(self, 'lm_head')
