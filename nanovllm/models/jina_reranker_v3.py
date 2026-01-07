import torch
from torch import nn
from transformers import Qwen3Config

from nanovllm.layers.linear import ReplicatedLinear
from nanovllm.models.qwen3 import Qwen3ForCausalLM
from nanovllm.models.jina_reranker_v3_model import JinaRerankerV3Model
from nanovllm.layers.flex_attention import parse_jina_v3_structure


class JinaRerankerV3(nn.Module):
    """Jina-Reranker-V3 model for listwise reranking.
    
    This is a listwise reranker that processes query + multiple docs together.
    Uses special tokens: <|rerank_token|> (151671) for query, <|embed_token|> (151670) for docs.
    """
    # Reuse packed_modules_mapping from Qwen3ForCausalLM for weight loading
    packed_modules_mapping = {
        "q_proj": ("qkv_proj", "q"),
        "k_proj": ("qkv_proj", "k"),
        "v_proj": ("qkv_proj", "v"),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }
    
    def __init__(
        self,
        config: Qwen3Config,
        projector_dim: int = 512,
        use_flex_attention: bool = True,
    ) -> None:
        # Don't call super().__init__() - we'll create our own model
        nn.Module.__init__(self)
        self.config = config
        
        # Use JinaRerankerV3Model with FlexAttention support
        self.model = JinaRerankerV3Model(config, use_flex_attention=use_flex_attention)
        
        # Replace lm_head with Identity (not used for reranking)
        self.lm_head = nn.Identity()
        
        # Projector: hidden_size -> projector_dim
        self.projector = nn.Sequential(
            ReplicatedLinear(config.hidden_size, config.hidden_size // 2, bias=False),
            nn.ReLU(),
            ReplicatedLinear(config.hidden_size // 2, projector_dim, bias=False),
        )
        
        # Special token IDs
        self.doc_embed_token_id = 151670
        self.query_embed_token_id = 151671
        
        self.projector_dim = projector_dim
        self.use_flex_attention = use_flex_attention
    
    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """Dummy method for compatibility - not used for reranking."""
        return hidden_states
    
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        block_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass for listwise reranking.
        
        Args:
            input_ids: Token IDs, shape [batch_size, seq_len] or flattened
            positions: Position IDs, shape [batch_size, seq_len] or flattened
            block_mask: Optional BlockMask for FlexAttention listwise mask.
                       If None and use_flex_attention=True, will be created automatically.
        
        Returns:
            hidden_states: Hidden states from the model backbone
        """
        return self.model(input_ids, positions, block_mask=block_mask)
    
    def compute_scores(
        self,
        hidden_states: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute reranking scores from hidden states.
        
        Args:
            hidden_states: Hidden states from forward pass, shape [batch_size, seq_len, hidden_size]
            input_ids: Token IDs, shape [batch_size, seq_len]
            
        Returns:
            scores: Cosine similarity scores, shape [batch_size, num_docs]
            query_embeds: Query embeddings, shape [batch_size, projector_dim]
            doc_embeds: Document embeddings, shape [batch_size, num_docs, projector_dim]
        """
        batch_size, seq_len, hidden_size = hidden_states.shape
        
        # Find special token positions per batch
        query_positions = []
        doc_positions = []
        for b in range(batch_size):
            batch_query_pos = []
            batch_doc_pos = []
            for s in range(seq_len):
                if input_ids[b, s] == self.query_embed_token_id:
                    batch_query_pos.append(s)
                elif input_ids[b, s] == self.doc_embed_token_id:
                    batch_doc_pos.append(s)
            query_positions.append(batch_query_pos)
            doc_positions.append(batch_doc_pos)
        
        # Extract embeddings per batch
        query_embeds_list = []
        doc_embeds_list = []
        
        for b in range(batch_size):
            batch_query_positions = query_positions[b]  # Already per-batch positions
            batch_doc_positions = doc_positions[b]  # Already per-batch positions
            
            if len(batch_query_positions) > 0:
                # Extract query embeddings for this batch
                batch_query_embeds = hidden_states[b, batch_query_positions, :]  # [num_query_tokens, hidden_size]
                # Average if multiple query tokens
                query_embeds_list.append(batch_query_embeds.mean(dim=0))  # [hidden_size]
            else:
                # Fallback: use last token
                query_embeds_list.append(hidden_states[b, -1, :])
            
            if len(batch_doc_positions) > 0:
                # Extract doc embeddings for this batch
                batch_doc_embeds = hidden_states[b, batch_doc_positions, :]  # [num_doc_tokens, hidden_size]
                doc_embeds_list.append(batch_doc_embeds)  # [num_docs, hidden_size]
            else:
                # Fallback: use a dummy embedding
                doc_embeds_list.append(hidden_states[b, -1:, :])  # [1, hidden_size]
        
        # Stack embeddings
        query_embeds_raw = torch.stack(query_embeds_list, dim=0)  # [batch_size, hidden_size]
        
        # Pad doc embeddings to same length
        max_docs = max(doc_emb.shape[0] for doc_emb in doc_embeds_list)
        padded_doc_embeds = []
        for doc_emb in doc_embeds_list:
            if doc_emb.shape[0] < max_docs:
                padding = torch.zeros(max_docs - doc_emb.shape[0], doc_emb.shape[1], device=doc_emb.device, dtype=doc_emb.dtype)
                doc_emb = torch.cat([doc_emb, padding], dim=0)
            padded_doc_embeds.append(doc_emb)
        doc_embeds_raw = torch.stack(padded_doc_embeds, dim=0)  # [batch_size, max_docs, hidden_size]
        
        # Project embeddings
        query_embeds = self.projector(query_embeds_raw)  # [batch_size, projector_dim]
        doc_embeds = self.projector(doc_embeds_raw.view(-1, hidden_size)).view(
            batch_size, -1, self.projector_dim
        )  # [batch_size, num_docs, projector_dim]
        
        # Compute cosine similarity
        query_embeds_norm = query_embeds / (query_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        doc_embeds_norm = doc_embeds / (doc_embeds.norm(dim=-1, keepdim=True) + 1e-8)
        
        # Expand query for broadcasting: [batch_size, 1, projector_dim]
        query_embeds_expanded = query_embeds_norm.unsqueeze(1)  # [batch_size, 1, projector_dim]
        
        # Cosine similarity: [batch_size, num_docs]
        scores = (doc_embeds_norm * query_embeds_expanded).sum(dim=-1)
        
        return scores, query_embeds, doc_embeds
