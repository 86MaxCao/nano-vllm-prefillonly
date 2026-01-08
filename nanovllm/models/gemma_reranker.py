import torch
from torch import nn
from transformers import GemmaConfig

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.gemma import GemmaModel, GemmaForCausalLM


class GemmaReranker(GemmaForCausalLM):
    """Gemma-Reranker model for sequence classification/reranking tasks.
    
    Supports bge-reranker-v2-gemma which uses "Yes" token logits.
    """
    packed_modules_mapping = GemmaForCausalLM.packed_modules_mapping
    
    def __init__(
        self,
        config: GemmaConfig,
        is_original_reranker: bool = False,
        classifier_from_token: list[str] | None = None,
    ) -> None:
        super().__init__(config)
        
        # Save config for use in convert_from_original_reranker
        self.config = config
        
        self.is_original_reranker = is_original_reranker
        self.classifier_from_token = classifier_from_token or []
        
        # For reranker, we replace lm_head with score_head
        if not is_original_reranker:
            # Converted model: directly use score_head
            self.score_head = ReplicatedLinear(
                config.hidden_size,
                1,
                bias=False,
            )
        else:
            # Original reranker: keep lm_head for weight loading, will be converted later
            assert len(self.classifier_from_token) == 1, \
                "Gemma-Reranker requires exactly 1 token (e.g., ['Yes'])"
    
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
                          If None, uses last non-padding token for each sequence.
            attention_mask: Attention mask, shape [batch_size, seq_len]. If provided, uses it
                          to find the last non-padding token for each sequence.
            
        Returns:
            scores: Reranking scores, shape [batch_size]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
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
            else:
                # If no attention_mask, assume all sequences have the same length
                token_indices = torch.full(
                    (batch_size,), seq_len - 1,
                    dtype=torch.int64,
                    device=hidden_states.device,
                )
        
        # Extract hidden states at specified positions
        batch_indices = torch.arange(batch_size, device=hidden_states.device)
        selected_states = hidden_states[batch_indices, token_indices]
        
        # Compute raw score (logit(Yes))
        raw_scores = self.score_head(selected_states).squeeze(-1)
        
        # Apply sigmoid to convert to probability (matching transformers
        # baseline)
        scores = torch.sigmoid(raw_scores)
        return scores
    
    def convert_from_original_reranker(self, tokenizer):
        """Convert original Gemma-Reranker weights to score_head.
        
        This should be called after loading weights (including lm_head).
        Uses "no_post_processing" method: directly extract token logit.
        """
        if not self.is_original_reranker:
            return
        
        # Extract token ID
        token = self.classifier_from_token[0]
        token_id = tokenizer.convert_tokens_to_ids(token)
        
        # For Gemma, lm_head is embed_tokens (tie_word_embeddings=True)
        # Extract score_weight = embed_tokens[token_id]
        embed_weight = self.model.embed_tokens.weight.data
        # Get the dtype and device of the model weights
        model_dtype = embed_weight.dtype
        model_device = embed_weight.device
        # Use [[token_id]] to keep 2D shape, matching vLLM approach
        score_weight = embed_weight[[token_id]].to(torch.float32).to(model_dtype)  # [1, hidden_size]
        
        # Create score_head and load the converted weight
        self.score_head = ReplicatedLinear(
            self.config.hidden_size,
            1,
            bias=False,
        )
        # Ensure score_head is on the same device as the model
        self.score_head = self.score_head.to(model_device)
        # score_weight is already [1, hidden_size], matching score_head.weight shape
        self.score_head.weight.data.copy_(score_weight)
