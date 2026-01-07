"""FlexAttention implementation for listwise reranking.

This module provides FlexAttention support for custom attention masks,
specifically for listwise rerankers like jina-reranker-v3.
"""
import torch
from torch import nn
from typing import Optional, Callable

try:
    from torch.nn.attention.flex_attention import flex_attention, create_block_mask
    FLEX_ATTENTION_AVAILABLE = True
except ImportError:
    FLEX_ATTENTION_AVAILABLE = False
    print("[FlexAttention] torch.nn.attention.flex_attention not available. Using fallback.")


class ListwiseAttentionMask:
    """Listwise attention mask for jina-reranker-v3.
    
    Mask logic:
    - Query tokens: can attend to all tokens
    - Doc tokens: can only attend to query tokens + tokens in the same doc
    """
    
    def __init__(
        self,
        query_start_idx: int,
        query_end_idx: int,
        doc_boundaries: list[tuple[int, int]],  # List of (start, end) for each doc
        seq_len: int,
    ):
        """
        Args:
            query_start_idx: Start index of query tokens
            query_end_idx: End index of query tokens (exclusive)
            doc_boundaries: List of (start, end) tuples for each document
            seq_len: Total sequence length
        """
        self.query_start_idx = query_start_idx
        self.query_end_idx = query_end_idx
        self.doc_boundaries = doc_boundaries
        self.seq_len = seq_len
        
        # Create a mapping from token index to doc index (-1 for query)
        # Note: token_to_doc is created on CPU because vmap requires CPU tensors
        # The mask will be created on CPU, then block_mask will be moved to GPU if needed
        # Initialize all as a special value (we'll mark docs and query explicitly)
        # Use -2 as initial value to detect unmarked tokens
        self.token_to_doc = torch.full((seq_len,), -2, dtype=torch.long, device='cpu')
        # Mark document tokens first
        for doc_idx, (doc_start, doc_end) in enumerate(doc_boundaries):
            self.token_to_doc[doc_start:doc_end] = doc_idx
        # Mark query tokens as -1 (query comes after docs)
        self.token_to_doc[query_start_idx:query_end_idx] = -1
        # Any remaining -2 tokens are likely padding tokens (after query_end)
        # These should not be accessible in attention, so we mark them as a special value
        # We'll ensure mask logic excludes them
        unmarked = (self.token_to_doc == -2)
        if unmarked.any():
            unmarked_count = unmarked.sum().item()
            unmarked_indices = torch.where(unmarked)[0].tolist()
            print(f"[FlexAttention] WARNING: Found {unmarked_count} unmarked tokens (likely padding) at indices: {unmarked_indices[:10]}...")
            # Mark them as -3 (padding/unknown) - these should be excluded from attention
            # Don't mark as query, as they're not part of the query
            self.token_to_doc[unmarked] = -3
        
        # Debug: Verify token_to_doc mapping (only once)
        if not hasattr(self, '_debug_logged'):
            query_tokens_in_range = (
                (torch.arange(seq_len) >= query_start_idx) &
                (torch.arange(seq_len) < query_end_idx)
            ).sum().item()
            query_tokens_marked = (self.token_to_doc == -1).sum().item()
            q_end_sample = min(query_end_idx, query_start_idx + 20)
            # Count doc tokens
            doc_token_counts = {}
            for doc_idx in range(len(self.doc_boundaries)):
                doc_count = (self.token_to_doc == doc_idx).sum().item()
                doc_token_counts[doc_idx] = doc_count
            print(f"[FlexAttention] token_to_doc debug: query range "
                  f"[{query_start_idx}:{query_end_idx}] has "
                  f"{query_tokens_in_range} tokens")
            print(f"[FlexAttention] token_to_doc debug: {query_tokens_marked} "
                  f"tokens marked as -1 (query) - EXPECTED: {query_tokens_in_range}")
            if query_tokens_marked != query_tokens_in_range:
                print(f"[FlexAttention] token_to_doc debug: ⚠️  MISMATCH: "
                      f"{query_tokens_marked - query_tokens_in_range} extra tokens marked as query!")
                # Find where the extra -1 tokens are
                all_query_indices = torch.where(self.token_to_doc == -1)[0].tolist()
                expected_range = set(range(query_start_idx, query_end_idx))
                extra_indices = [idx for idx in all_query_indices if idx not in expected_range]
                if extra_indices:
                    print(f"[FlexAttention] token_to_doc debug: Extra query indices: {extra_indices[:20]}")
            print(f"[FlexAttention] token_to_doc debug: Doc token counts: {doc_token_counts}")
            print(f"[FlexAttention] token_to_doc debug: token_to_doc sample "
                  f"[0:20]={self.token_to_doc[:20].tolist()}")
            print(f"[FlexAttention] token_to_doc debug: token_to_doc sample "
                  f"[{query_start_idx}:{q_end_sample}]="
                  f"{self.token_to_doc[query_start_idx:q_end_sample].tolist()}")
            self._debug_logged = True
    
    @staticmethod
    def _move_block_mask_tensors(block_mask, target_device):
        """Recursively move all tensor attributes in block_mask to target_device.
        
        This is necessary because BlockMask may have nested tensor attributes
        that the .to() method doesn't handle automatically.
        """
        moved_count = 0
        
        # Move all direct tensor attributes (including those starting with _)
        # Some internal attributes might be important
        for attr_name in dir(block_mask):
            try:
                attr = getattr(block_mask, attr_name)
                if isinstance(attr, torch.Tensor) and attr.device != target_device:
                    try:
                        setattr(block_mask, attr_name, attr.to(target_device))
                        moved_count += 1
                    except (AttributeError, TypeError):
                        # Skip read-only attributes
                        pass
            except (AttributeError, TypeError):
                # Skip non-attributes or properties that raise errors
                pass
        
        # Also check __dict__ for any additional attributes
        if hasattr(block_mask, '__dict__'):
            for key, value in block_mask.__dict__.items():
                if isinstance(value, torch.Tensor) and value.device != target_device:
                    block_mask.__dict__[key] = value.to(target_device)
                    moved_count += 1
                elif isinstance(value, (list, tuple)):
                    # Check if list/tuple contains tensors
                    new_value = []
                    for item in value:
                        if isinstance(item, torch.Tensor) and item.device != target_device:
                            new_value.append(item.to(target_device))
                            moved_count += 1
                        else:
                            new_value.append(item)
                    block_mask.__dict__[key] = type(value)(new_value)
                elif isinstance(value, dict):
                    # Check if dict contains tensors
                    new_dict = {}
                    for k, v in value.items():
                        if isinstance(v, torch.Tensor) and v.device != target_device:
                            new_dict[k] = v.to(target_device)
                            moved_count += 1
                        else:
                            new_dict[k] = v
                    block_mask.__dict__[key] = new_dict
        
        # Also try to access common BlockMask attributes directly
        # Based on PyTorch's BlockMask implementation, these are common attributes
        common_attrs = [
            'kv_num_blocks', 'kv_indices', 'full_kv_num_blocks', 'full_kv_indices',
            'q_num_blocks', 'q_indices', 'full_q_num_blocks', 'full_q_indices',
            'seq_lengths', 'BLOCK_SIZE', 'mask_mod'
        ]
        for attr_name in common_attrs:
            try:
                attr = getattr(block_mask, attr_name, None)
                if isinstance(attr, torch.Tensor) and attr.device != target_device:
                    setattr(block_mask, attr_name, attr.to(target_device))
                    moved_count += 1
            except (AttributeError, TypeError):
                pass
        
        return moved_count
    
    def get_mask_function(self):
        """Get mask function for FlexAttention using pure tensor operations.
        
        This implementation uses only tensor operations (no if statements or .item())
        so it's compatible with vmap. The mask logic:
        - Query tokens (query_start <= idx < query_end) can attend to all tokens
        - Doc tokens can attend to query tokens + tokens in the same doc
        
        Strategy: Pre-compute the full mask tensor, then mask_fn just indexes into it.
        This avoids all data-dependent control flow.
        """
        query_start = self.query_start_idx
        query_end = self.query_end_idx
        token_to_doc = self.token_to_doc
        seq_len = self.seq_len
        device = token_to_doc.device
        
        # Pre-compute the full attention mask tensor [seq_len, seq_len]
        # This avoids any data-dependent control flow in mask_fn
        q_indices = torch.arange(seq_len, device=device).unsqueeze(1)  # [seq_len, 1]
        kv_indices = torch.arange(seq_len, device=device).unsqueeze(0)  # [1, seq_len]
        
        # Expand token_to_doc to 2D for broadcasting
        q_docs = token_to_doc.unsqueeze(1)  # [seq_len, 1]
        kv_docs = token_to_doc.unsqueeze(0)  # [1, seq_len]
        
        # TEMPORARY: Use causal mask to match Transformers baseline behavior
        # TODO: Once we confirm causal mask works, we can implement proper listwise mask
        # Causal mask: token i can attend to token j if j <= i (lower triangular)
        mask = torch.tril(torch.ones((seq_len, seq_len), dtype=torch.bool, device=device))
        
        # Exclude padding tokens from attention (they shouldn't be attended to or attend to anything)
        is_padding_kv = (token_to_doc == -3).unsqueeze(0)  # [1, seq_len] - padding tokens
        is_padding_q = (token_to_doc == -3).unsqueeze(1)  # [seq_len, 1] - padding tokens as query
        
        # Mask out padding tokens: they can't attend to anything and nothing can attend to them
        mask = mask & ~is_padding_kv.expand(seq_len, seq_len)  # Don't attend to padding
        mask = mask & ~is_padding_q.expand(seq_len, seq_len)  # Padding doesn't attend
        
        # Store mask for debug logging
        causal_mask = mask.clone()
        
        # Debug: Verify mask logic (only once)
        if not hasattr(self, '_mask_debug_logged'):
            query_mask_row = (
                mask[query_start, :].cpu()
                if query_start < seq_len else None
            )
            doc_token_idx = (
                self.doc_boundaries[0][0]
                if self.doc_boundaries and
                self.doc_boundaries[0][0] < seq_len else None
            )
            doc_mask_row = (
                mask[doc_token_idx, :].cpu()
                if doc_token_idx is not None else None
            )
            print(f"[FlexAttention] Mask debug: query_start={query_start}, "
                  f"query_end={query_end}")
            if query_mask_row is not None:
                q_attend_count = query_mask_row.sum().item()
                print(f"[FlexAttention] Mask debug: query token at "
                      f"{query_start} can attend to {q_attend_count}/{seq_len} "
                      f"tokens")
                print(f"[FlexAttention] Mask debug: query mask row sample "
                      f"[0:20]={query_mask_row[:20].tolist()}")
            if doc_mask_row is not None and doc_token_idx is not None:
                d_attend_count = doc_mask_row.sum().item()
                # Check what tokens doc can attend to
                doc_attend_indices = torch.where(doc_mask_row)[0].tolist()
                # Count how many are query tokens vs same doc tokens
                # Note: query tokens are marked as -1 in token_to_doc
                query_attend_count = sum(1 for idx in doc_attend_indices 
                                       if self.token_to_doc[idx] == -1)
                same_doc_attend_count = sum(1 for idx in doc_attend_indices 
                                           if self.token_to_doc[idx] == self.token_to_doc[doc_token_idx])
                # Check for other docs
                other_doc_attend_list = [idx for idx in doc_attend_indices 
                                        if (self.token_to_doc[idx] != self.token_to_doc[doc_token_idx] 
                                            and self.token_to_doc[idx] != -1)]
                other_doc_attend_count = len(other_doc_attend_list)
                print(f"[FlexAttention] Mask debug: doc token at "
                      f"{doc_token_idx} can attend to {d_attend_count}/{seq_len} "
                      f"tokens")
                print(f"[FlexAttention] Mask debug:   - Query tokens: {query_attend_count}, "
                      f"Same doc tokens: {same_doc_attend_count}, "
                      f"Other docs: {other_doc_attend_count}")
                if other_doc_attend_count > 0:
                    print(f"[FlexAttention] Mask debug:   ⚠️  ERROR: Doc token can attend to "
                          f"{other_doc_attend_count} tokens from OTHER docs! Indices: {other_doc_attend_list[:10]}")
                print(f"[FlexAttention] Mask debug: doc mask row sample "
                      f"[0:20]={doc_mask_row[:20].tolist()}")
                # Debug: Check the exact breakdown
                expected_count = query_attend_count + same_doc_attend_count + other_doc_attend_count
                if d_attend_count != expected_count:
                    print(f"[FlexAttention] Mask debug: ⚠️  Count mismatch: "
                          f"expected {expected_count} (query:{query_attend_count} + same_doc:{same_doc_attend_count} + other:{other_doc_attend_count}), "
                          f"got {d_attend_count}, diff={d_attend_count - expected_count}")
            # Check mask type (causal vs listwise)
            if query_start < query_end:
                # For causal mask, query tokens can attend to tokens up to their position
                print(f"[FlexAttention] Mask debug: Using CAUSAL mask (token i can attend to j if j <= i)")
            self._mask_debug_logged = True
        
        # Store the original mask on its device (CPU for vmap) for later use
        self._precomputed_mask = mask
        
        # Create a CPU copy of mask for mask_fn (vmap requires CPU tensors)
        # vmap internally runs on CPU, so mask_fn must work with CPU tensors
        # IMPORTANT: Keep mask_cpu on CPU for vmap, but we'll need to handle device
        # conversion if flex_attention calls mask_mod at runtime
        mask_cpu = mask.cpu()
        
        # Store mask_cpu as an instance variable so we can move it to GPU later if needed
        self._mask_cpu = mask_cpu
        
        def mask_fn(b: int, h: int, q_idx: int, kv_idx: int):
            """Mask function for FlexAttention using pure tensor operations.
            
            ✅ This implementation is vmap-compatible:
            - No data-dependent if statements (all conditions pre-computed)
            - No .item() calls
            - Simple tensor indexing only
            
            Note: vmap runs on CPU, so mask_cpu is used here.
            However, if flex_attention calls this at runtime (after block_mask creation),
            it might expect GPU tensors. We'll handle this by checking the device of
            the indices and using the appropriate mask.
            
            Args:
                b: Batch index (unused)
                h: Head index (unused)
                q_idx: Query token index (0-based, can be int or tensor scalar)
                kv_idx: Key-Value token index (0-based, can be int or tensor scalar)
                
            Returns:
                Tensor scalar (bool): True if q_idx can attend to kv_idx, False otherwise
            """
            # Convert indices to CPU tensors (vmap requires CPU)
            # Always use CPU for vmap compatibility - vmap passes CPU integers
            # Use torch.as_tensor which handles both int and tensor inputs
            q_idx_tensor = torch.as_tensor(q_idx, device='cpu')
            kv_idx_tensor = torch.as_tensor(kv_idx, device='cpu')
            
            # Clamp indices to valid range [0, seq_len) to avoid IndexError
            # vmap may test indices that are out of bounds
            # Using torch.clamp is a pure tensor operation compatible with vmap
            max_idx = mask_cpu.shape[0] - 1  # seq_len - 1
            q_idx_tensor = torch.clamp(q_idx_tensor, 0, max_idx)
            kv_idx_tensor = torch.clamp(kv_idx_tensor, 0, max_idx)
            
            # Always use CPU mask (vmap requires CPU)
            # Note: Once block_mask is created, flex_attention shouldn't call mask_mod again
            # The block_mask itself should contain all needed info, so mask_mod is only
            # used during block_mask creation (which happens on CPU via vmap)
            mask_to_use = mask_cpu
            
            # Simply index into the pre-computed mask tensor
            # This is a pure tensor operation with no control flow
            # NOTE: Do NOT use print() or any operation that calls .item() inside mask_fn
            # vmap does not support .item() calls, and print() on tensors triggers .item()
            # Use proper indexing: mask[q_idx, kv_idx] means "can q_idx attend to kv_idx?"
            result = mask_to_use[q_idx_tensor, kv_idx_tensor]
            return result
        
        return mask_fn
    
    def create_block_mask(self, block_size: int = 16, target_device: torch.device | None = None):
        """Create BlockMask for FlexAttention.
        
        Args:
            block_size: Block size for block mask
            target_device: Target device to move block_mask to after creation (default: None, keep on CPU)
        """
        if not FLEX_ATTENTION_AVAILABLE:
            return None
        
        mask_fn = self.get_mask_function()
        mask_device = self._precomputed_mask.device
        
        try:
            # Try different API signatures for create_block_mask
            try:
                # Newer API: create_block_mask(mask_fn, Q_LEN, KV_LEN, block_size, B, H)
                block_mask = create_block_mask(
                    mask_fn,
                    Q_LEN=self.seq_len,
                    KV_LEN=self.seq_len,
                    block_size=block_size,
                    B=None,  # Broadcast over batch
                    H=None,  # Broadcast over heads
                )
            except TypeError:
                # Older API: create_block_mask(mask_fn, B, H, Q_LEN, KV_LEN, block_size)
                try:
                    block_mask = create_block_mask(
                        mask_fn,
                        B=None,
                        H=None,
                        Q_LEN=self.seq_len,
                        KV_LEN=self.seq_len,
                        block_size=block_size,
                    )
                except TypeError:
                    # Fallback: try without block_size
                    block_mask = create_block_mask(
                        mask_fn,
                        B=None,
                        H=None,
                        Q_LEN=self.seq_len,
                        KV_LEN=self.seq_len,
                    )
            
            # After creating block_mask, try to move it to the same device as the mask
            # BlockMask may have internal tensors that need device alignment
            if block_mask is not None:
                # Move block_mask to target device if specified
                # If not specified, try to move to mask_device (though mask is on CPU for vmap)
                target = target_device if target_device is not None else mask_device
                try:
                    # Try to move block_mask to target device
                    if hasattr(block_mask, 'to') and target is not None:
                        block_mask = block_mask.to(target)
                        # Also manually move all tensor attributes to ensure everything is on the same device
                        # BlockMask may have nested tensor attributes that .to() doesn't handle
                        ListwiseAttentionMask._move_block_mask_tensors(block_mask, target)
                        
                        # If block_mask has a mask_mod attribute (the original mask function),
                        # we need to ensure it doesn't reference CPU tensors
                        # Create a new mask_mod that uses GPU mask if needed
                        if hasattr(block_mask, 'mask_mod') and target.type == 'cuda':
                            # Create a GPU version of the mask for mask_mod
                            mask_gpu = self._precomputed_mask.to(target) if self._precomputed_mask.device != target else self._precomputed_mask
                            
                            # Create a new mask_fn that uses GPU mask
                            def gpu_mask_fn(b: int, h: int, q_idx: int, kv_idx: int):
                                q_idx_tensor = torch.as_tensor(q_idx, device=target)
                                kv_idx_tensor = torch.as_tensor(kv_idx, device=target)
                                max_idx = mask_gpu.shape[0] - 1
                                q_idx_tensor = torch.clamp(q_idx_tensor, 0, max_idx)
                                kv_idx_tensor = torch.clamp(kv_idx_tensor, 0, max_idx)
                                return mask_gpu[q_idx_tensor, kv_idx_tensor]
                            
                            # Replace mask_mod with GPU version
                            try:
                                block_mask.mask_mod = gpu_mask_fn
                            except (AttributeError, TypeError):
                                # mask_mod might be read-only, skip
                                pass
                except Exception as move_error:
                    # If moving fails, log and continue (will fallback to flash attention later)
                    print(f"[FlexAttention] Warning: Failed to move block_mask to {target}: {move_error}")
                    import traceback
                    traceback.print_exc()
            
            return block_mask
        except Exception as e:
            # FlexAttention block mask creation failed (e.g., vmap limitation with data-dependent control flow)
            # This is expected for complex masks - we'll fall back to flash attention
            # Only print error once to avoid spam
            if not hasattr(self, '_block_mask_error_logged'):
                print(f"[FlexAttention] Block mask creation failed (will use flash attention): {type(e).__name__}: {e}")
                self._block_mask_error_logged = True
            return None


def parse_jina_v3_structure(input_ids: torch.Tensor, query_token_id: int = 151671, doc_token_id: int = 151670):
    """Parse jina-reranker-v3 input structure to find query and doc boundaries.
    
    Note: parse_jina_v3_structure._debug_logged is reset for each new call.
    
    Args:
        input_ids: Token IDs, shape [batch_size, seq_len] or [seq_len]
        query_token_id: Token ID for query embedding token (<|rerank_token|>)
        doc_token_id: Token ID for doc embedding token (<|embed_token|>)
        
    Returns:
        List of ListwiseAttentionMask objects (one per batch item)
    """
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)
    
    batch_size, seq_len = input_ids.shape
    masks = []
    
    # Reset debug flag for each call
    parse_jina_v3_structure._debug_logged = False
    
    for b in range(batch_size):
        seq = input_ids[b].cpu().tolist()
        
        # Find query token position (should be near the end)
        query_positions = [i for i, tid in enumerate(seq) if tid == query_token_id]
        if not query_positions:
            # Fallback: assume query is in the last 20% of sequence
            query_start = int(seq_len * 0.8)
            query_end = seq_len
        else:
            # Query token marks the end of the query
            # Query starts after all documents
            # We need to find where the last document ends
            first_query_pos = query_positions[0]
            
            # Query starts right before the query token
            # For jina-reranker-v3 format: "<query>\n{query_text}<|rerank_token|>"
            # The query_start should be the position of "<query>\n"
            # Look backwards for potential query start markers
            # Since we don't have tokenizer context, we'll use the last doc end + 1
            # Or look back a small amount (query is typically short, ~10-30 tokens)
            # Query starts after the last document ends
            # We'll compute doc_boundaries first, then set query_start to be after last doc
            # For now, use a conservative estimate: look back less to avoid overlapping with docs
            query_start = max(0, first_query_pos - 15)  # Look back up to 15 tokens
            query_end = first_query_pos + 1  # Include query token itself
        
        # Find doc token positions
        doc_positions = [i for i, tid in enumerate(seq) if tid == doc_token_id]
        
        # Group doc tokens into documents
        # Documents are separated by passage tags, we'll use doc tokens as markers
        # Each <|embed_token|> marks the end of a document
        # Important: Documents come BEFORE the query, so doc_positions should all be < query_positions[0]
        doc_boundaries = []
        if doc_positions:
            # Filter: only include doc positions before the first query token
            # (not query_start which is approximate)
            first_query_token_pos = query_positions[0] if query_positions else seq_len
            valid_doc_positions = [
                pos for pos in doc_positions 
                if pos < first_query_token_pos
            ]
            
            if valid_doc_positions:
                prev_end = 0
                for doc_pos in valid_doc_positions:
                    # Each doc token marks the end of a document
                    # Find the start of this document (end of previous doc or 0)
                    doc_start = prev_end
                    
                    # Document ends at doc_pos + 1 (inclusive of doc token)
                    # Don't cap at query_start - let each doc use its actual end position
                    # The query_start is just approximate, we rely on first_query_token_pos
                    # to ensure docs don't overlap with query
                    doc_end = doc_pos + 1
                    
                    # Ensure doc doesn't extend beyond first query token
                    if doc_end <= first_query_token_pos and doc_end > doc_start:
                        doc_boundaries.append((doc_start, doc_end))
                    prev_end = doc_end
        
        # If no doc boundaries found, create a single doc from start to query
        if not doc_boundaries:
            doc_boundaries = [(0, query_start)] if query_start > 0 else [(0, seq_len)]
        else:
            # Adjust query_start to be after the last document
            # This ensures query doesn't overlap with documents
            last_doc_end = doc_boundaries[-1][1]
            if query_start < last_doc_end:
                # Query should start after last doc ends
                query_start = last_doc_end
        
        # Debug: Print boundaries (only once per call)
        if not parse_jina_v3_structure._debug_logged:
            print(f"[FlexAttention] === parse_jina_v3_structure Debug ===")
            print(f"[FlexAttention] Batch {b}: seq_len={seq_len}, query_token_id={query_token_id}, doc_token_id={doc_token_id}")
            print(f"[FlexAttention] Batch {b}: "
                  f"query_positions={query_positions}")
            print(f"[FlexAttention] Batch {b}: query_start={query_start}, "
                  f"query_end={query_end} (computed from positions)")
            print(f"[FlexAttention] Batch {b}: "
                  f"doc_positions={doc_positions[:10]}... (first 10)")
            print(f"[FlexAttention] Batch {b}: "
                  f"doc_boundaries={doc_boundaries}")
            print(f"[FlexAttention] Batch {b}: "
                  f"Total docs={len(doc_boundaries)}")
            parse_jina_v3_structure._debug_logged = True
        
        mask = ListwiseAttentionMask(
            query_start_idx=query_start,
            query_end_idx=query_end,
            doc_boundaries=doc_boundaries,
            seq_len=seq_len,
        )
        
        masks.append(mask)
    
    return masks


class FlexAttention(nn.Module):
    """FlexAttention wrapper that supports custom masks."""
    
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        use_flex_attention: bool = True,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.use_flex_attention = use_flex_attention and FLEX_ATTENTION_AVAILABLE
        
        # Fallback to standard attention if FlexAttention not available
        if not self.use_flex_attention:
            from flash_attn import flash_attn_varlen_func
            self.fallback_attention = flash_attn_varlen_func
    
    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: Optional[torch.Tensor] = None,
        cu_seqlens_q: Optional[torch.Tensor] = None,
        cu_seqlens_k: Optional[torch.Tensor] = None,
        max_seqlen_q: Optional[int] = None,
        max_seqlen_k: Optional[int] = None,
    ) -> torch.Tensor:
        """Forward pass with optional FlexAttention block mask.
        
        Args:
            q: Query tensor, shape [total_q_tokens, num_heads, head_dim]
            k: Key tensor, shape [total_kv_tokens, num_kv_heads, head_dim]
            v: Value tensor, shape [total_kv_tokens, num_kv_heads, head_dim]
            block_mask: Optional BlockMask for FlexAttention
            cu_seqlens_q: Cumulative sequence lengths for queries
            cu_seqlens_k: Cumulative sequence lengths for keys
            max_seqlen_q: Maximum query sequence length
            max_seqlen_k: Maximum key sequence length
        """
        if self.use_flex_attention and block_mask is not None and FLEX_ATTENTION_AVAILABLE:
            # Use FlexAttention with block mask
            # Reshape for FlexAttention: [batch, num_heads, seq_len, head_dim]
            if cu_seqlens_q is not None and cu_seqlens_k is not None:
                # Varlen format: reshape per sequence
                batch_size = cu_seqlens_q.size(0) - 1
                if batch_size == 0:
                    batch_size = 1
                seq_len = max_seqlen_q if max_seqlen_q else (cu_seqlens_q[-1].item() // batch_size if batch_size > 0 else cu_seqlens_q[-1].item())
                
                if seq_len == 0:
                    seq_len = q.size(0)
                
                # Reshape q, k, v from [total_tokens, num_heads, head_dim] to [batch, seq_len, num_heads, head_dim]
                # Then transpose to [batch, num_heads, seq_len, head_dim]
                total_tokens = q.size(0)
                if total_tokens == batch_size * seq_len:
                    q_reshaped = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                    k_reshaped = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    v_reshaped = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                    
                    # Repeat k, v heads if needed (GQA)
                    if self.num_kv_heads < self.num_heads:
                        repeat_factor = self.num_heads // self.num_kv_heads
                        k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
                        v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)
                    
                    try:
                        # Ensure block_mask is on the same device as q, k, v
                        # BlockMask might have internal tensors that need device alignment
                        if block_mask is not None:
                            target_device = q_reshaped.device
                            # First try .to() method
                            if hasattr(block_mask, 'to'):
                                block_mask = block_mask.to(target_device)
                            
                            # Manually move all tensor attributes to ensure everything is on the same device
                            # BlockMask may have nested tensor attributes that .to() doesn't handle
                            moved_count = ListwiseAttentionMask._move_block_mask_tensors(block_mask, target_device)
                            if moved_count > 0 and not hasattr(self, '_tensor_move_logged'):
                                print(f"[FlexAttention] Moved {moved_count} tensor attributes to {target_device}")
                                self._tensor_move_logged = True
                            
                            # Double-check: verify all tensor attributes are on the correct device
                            # This helps catch any attributes that might have been missed
                            moved_tensors = []
                            cpu_tensors = []
                            for attr_name in dir(block_mask):
                                if not attr_name.startswith('_'):
                                    try:
                                        attr = getattr(block_mask, attr_name)
                                        if isinstance(attr, torch.Tensor):
                                            if attr.device != target_device:
                                                # Force move any remaining CPU tensors
                                                setattr(block_mask, attr_name, attr.to(target_device))
                                                moved_tensors.append(attr_name)
                                            elif attr.device.type == 'cpu':
                                                cpu_tensors.append(f"{attr_name} (CPU)")
                                    except (AttributeError, TypeError):
                                        pass
                            
                            # Also check __dict__ for any CPU tensors
                            if hasattr(block_mask, '__dict__'):
                                for key, value in block_mask.__dict__.items():
                                    if isinstance(value, torch.Tensor) and value.device.type == 'cpu':
                                        block_mask.__dict__[key] = value.to(target_device)
                                        if key not in moved_tensors:
                                            moved_tensors.append(key)
                            
                            # Device check logging disabled to reduce noise
                            # Log detailed device information only if needed for debugging
                            # if not hasattr(self, '_device_check_logged'):
                            #     ... (device check code) ...
                            #     self._device_check_logged = True
                        
                        # Apply FlexAttention
                        output = flex_attention(
                            q_reshaped,
                            k_reshaped,
                            v_reshaped,
                            block_mask=block_mask,
                        )
                        
                        # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch*seq_len, num_heads, head_dim]
                        output = output.transpose(1, 2).contiguous().view(-1, self.num_heads, self.head_dim)
                        
                        return output
                    except Exception as e:
                        # Print detailed error information (only once)
                        if not hasattr(self, '_flex_attention_error_logged'):
                            error_msg = str(e)
                            print(f"[FlexAttention] Error during flex_attention call: {error_msg}")
                            
                            # If it's a device mismatch error, try to find which tensor is on CPU
                            if "same device" in error_msg.lower() and "cpu" in error_msg.lower():
                                print(f"[FlexAttention] Device mismatch detected. Checking block_mask for CPU tensors...")
                                cpu_tensors_found = []
                                for attr_name in dir(block_mask):
                                    try:
                                        attr = getattr(block_mask, attr_name)
                                        if isinstance(attr, torch.Tensor) and attr.device.type == 'cpu':
                                            cpu_tensors_found.append(f"{attr_name}: {attr.shape} on {attr.device}")
                                    except:
                                        pass
                                if cpu_tensors_found:
                                    print(f"[FlexAttention] Found CPU tensors in block_mask: {cpu_tensors_found[:5]}")
                                else:
                                    print(f"[FlexAttention] No CPU tensors found in block_mask attributes")
                                    print(f"[FlexAttention] Block_mask type: {type(block_mask)}")
                                    print(f"[FlexAttention] Block_mask __dict__ keys: {list(block_mask.__dict__.keys()) if hasattr(block_mask, '__dict__') else 'N/A'}")
                            
                            self._flex_attention_error_logged = True
                        # Fall through to flash attention
                else:
                    print(f"[FlexAttention] Shape mismatch: total_tokens={total_tokens}, batch*seq={batch_size*seq_len}, falling back")
                    # Fall through to flash attention
        
        # Fallback: Use PyTorch's scaled_dot_product_attention with explicit mask
        # This is necessary for listwise rerankers where documents cannot attend to each other
        # Flash attention with causal=True would incorrectly allow cross-document attention
        
        if cu_seqlens_q is not None and cu_seqlens_k is not None:
            # Varlen format: reshape per sequence
            batch_size = cu_seqlens_q.size(0) - 1
            if batch_size == 0:
                batch_size = 1
            seq_len = max_seqlen_q if max_seqlen_q else (cu_seqlens_q[-1].item() // batch_size if batch_size > 0 else cu_seqlens_q[-1].item())
            
            if seq_len == 0:
                seq_len = q.size(0)
            
            total_tokens = q.size(0)
            if total_tokens == batch_size * seq_len:
                # Reshape to [batch, seq_len, num_heads, head_dim]
                q_reshaped = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
                k_reshaped = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                v_reshaped = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
                
                # Repeat k, v heads if needed (GQA)
                if self.num_kv_heads < self.num_heads:
                    repeat_factor = self.num_heads // self.num_kv_heads
                    k_reshaped = k_reshaped.repeat_interleave(repeat_factor, dim=1)
                    v_reshaped = v_reshaped.repeat_interleave(repeat_factor, dim=1)
                
                # Use PyTorch's scaled_dot_product_attention (supports custom masks)
                # Note: This is slower than flash attention but necessary for complex masks
                # For now, we use causal mask as fallback - proper listwise mask requires
                # additional information (query/doc boundaries) which is not available here
                # Only print warning once to avoid spam
                if not hasattr(self, '_fallback_warning_logged'):
                    print(f"[FlexAttention] Using PyTorch scaled_dot_product_attention fallback (causal mask - may not be correct for listwise reranking)")
                    self._fallback_warning_logged = True
                output = torch.nn.functional.scaled_dot_product_attention(
                    q_reshaped,
                    k_reshaped,
                    v_reshaped,
                    scale=self.scale,
                    is_causal=True,  # Fallback to causal - not ideal for listwise reranking
                )
                
                # Reshape back: [batch, num_heads, seq_len, head_dim] -> [batch*seq_len, num_heads, head_dim]
                output = output.transpose(1, 2).contiguous().view(-1, self.num_heads, self.head_dim)
                print("[FlexAttention] Using PyTorch scaled_dot_product_attention fallback (causal mask - may not be correct for listwise reranking)")
                return output
        
        # Final fallback: use flash attention with causal (not ideal for listwise reranking)
        from nanovllm.utils.context import get_context
        context = get_context()
        
        from flash_attn import flash_attn_varlen_func
        print("[FlexAttention] WARNING: Using flash attention with causal mask - this may be incorrect for listwise reranking models!")
        return flash_attn_varlen_func(
            q, k, v,
            max_seqlen_q=max_seqlen_q or context.max_seqlen_q,
            cu_seqlens_q=cu_seqlens_q or context.cu_seqlens_q,
            max_seqlen_k=max_seqlen_k or context.max_seqlen_k,
            cu_seqlens_k=cu_seqlens_k or context.cu_seqlens_k,
            softmax_scale=self.scale,
            causal=True,
        )
