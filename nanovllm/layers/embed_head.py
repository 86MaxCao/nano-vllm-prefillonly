import torch
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist

from nanovllm.utils.context import get_context


class VocabParallelEmbedding(nn.Module):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
    ):
        super().__init__()
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        assert num_embeddings % self.tp_size == 0
        self.num_embeddings = num_embeddings
        self.num_embeddings_per_partition = self.num_embeddings // self.tp_size
        self.vocab_start_idx = self.num_embeddings_per_partition * self.tp_rank
        self.vocab_end_idx = self.vocab_start_idx + self.num_embeddings_per_partition
        self.weight = nn.Parameter(torch.empty(self.num_embeddings_per_partition, embedding_dim))
        self.weight.weight_loader = self.weight_loader

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(0)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(0, start_idx, shard_size)
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor):
        if self.tp_size > 1:
            mask = (x >= self.vocab_start_idx) & (x < self.vocab_end_idx)
            x = mask * (x - self.vocab_start_idx)
        y = F.embedding(x, self.weight)
        if self.tp_size > 1:
            y = mask.unsqueeze(1) * y
            dist.all_reduce(y)
        return y


class ParallelLMHead(VocabParallelEmbedding):

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        bias: bool = False,
    ):
        assert not bias
        super().__init__(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor):
        context = get_context()
        if context.is_prefill:
            last_indices = context.cu_seqlens_q[1:] - 1
            if not hasattr(self, '_debug_lm_head_prefill_logged'):
                print(f"\n[DEBUG ParallelLMHead.forward] Prefill mode:")
                print(f"  input x shape: {x.shape}")
                print(f"  input x dtype: {x.dtype}")
                # Check input x values
                if x.numel() > 0:
                    print(f"  input x min/max: {x.min().item():.4f} / {x.max().item():.4f}")
                    print(f"  input x mean: {x.mean().item():.4f}")
                    print(f"  input x has NaN: {torch.isnan(x).any().item()}")
                    print(f"  input x has Inf: {torch.isinf(x).any().item()}")
                    print(f"  input x all zeros: {(x == 0).all().item()}")
                    # Check values at indices we're about to extract
                    if isinstance(last_indices, torch.Tensor) and len(last_indices) > 0:
                        first_idx = last_indices[0].item()
                        if first_idx < x.shape[0]:
                            print(f"  input x[{first_idx}, :5]: {x[first_idx, :5].tolist()}")
                        last_idx = last_indices[-1].item()
                        if last_idx < x.shape[0]:
                            print(f"  input x[{last_idx}, :5]: {x[last_idx, :5].tolist()}")
                print(f"  context.cu_seqlens_q: {context.cu_seqlens_q}")
                print(f"  last_indices: {last_indices}")
                print(f"  last_indices type: {type(last_indices)}")
                if isinstance(last_indices, torch.Tensor):
                    last_indices_list = last_indices.tolist()
                    print(f"  last_indices values: {last_indices_list}")
                    # Check if last_indices match expected positions (515, 941, etc.)
                    print(f"  Expected last token positions (from cu_seqlens_q): {[context.cu_seqlens_q[i].item() - 1 for i in range(1, len(context.cu_seqlens_q))]}")
                    # Check if there's a mismatch
                    expected_positions = [context.cu_seqlens_q[i].item() - 1 for i in range(1, len(context.cu_seqlens_q))]
                    if len(last_indices_list) == len(expected_positions):
                        mismatches = [i for i, (actual, expected) in enumerate(zip(last_indices_list, expected_positions)) if actual != expected]
                        if mismatches:
                            print(f"  WARNING: last_indices mismatch at sequence indices: {mismatches}")
                        else:
                            print(f"  last_indices match expected positions")
                else:
                    print(f"  last_indices values: {last_indices}")
                # Additional check: verify that x at last_indices is not all zeros before indexing
                if isinstance(last_indices, torch.Tensor) and len(last_indices) > 0:
                    for i, idx in enumerate(last_indices):
                        idx_val = idx.item() if isinstance(idx, torch.Tensor) else idx
                        if idx_val < x.shape[0]:
                            x_at_idx = x[idx_val]
                            is_zero = (x_at_idx == 0).all().item() if x_at_idx.numel() > 0 else True
                            if is_zero:
                                print(f"  WARNING: x[{idx_val}] (sequence {i} last token) is all zeros before indexing!")
                            else:
                                print(f"  x[{idx_val}] (sequence {i} last token) is NOT all zeros, mean: {x_at_idx.mean().item():.4f}")
                self._debug_lm_head_prefill_logged = True
            x = x[last_indices].contiguous()
            if not hasattr(self, '_debug_lm_head_after_index_logged'):
                print(f"\n[DEBUG ParallelLMHead.forward] After indexing:")
                print(f"  x shape after indexing: {x.shape}")
                print(f"  x dtype: {x.dtype}")
                print(f"  self.weight shape: {self.weight.shape}")
                print(f"  self.weight dtype: {self.weight.dtype}")
                # Check x values after indexing
                if x.numel() > 0:
                    print(f"  x min/max: {x.min().item():.4f} / {x.max().item():.4f}")
                    print(f"  x mean: {x.mean().item():.4f}")
                    print(f"  x has NaN: {torch.isnan(x).any().item()}")
                    print(f"  x has Inf: {torch.isinf(x).any().item()}")
                    print(f"  x all zeros: {(x == 0).all().item()}")
                    if x.shape[0] > 0:
                        print(f"  x[0, :5]: {x[0, :5].tolist()}")
                        if x.shape[0] > 1:
                            print(f"  x[-1, :5]: {x[-1, :5].tolist()}")
                # Check dtype compatibility
                if x.dtype != self.weight.dtype:
                    print(f"  WARNING: dtype mismatch! x.dtype={x.dtype}, weight.dtype={self.weight.dtype}")
                self._debug_lm_head_after_index_logged = True
        logits = F.linear(x, self.weight)
        if not hasattr(self, '_debug_lm_head_logits_logged'):
            print(f"\n[DEBUG ParallelLMHead.forward] After F.linear:")
            print(f"  logits shape: {logits.shape}")
            print(f"  logits dtype: {logits.dtype}")
            if logits.numel() > 0:
                print(f"  logits min/max: {logits.min().item():.4f} / {logits.max().item():.4f}")
                print(f"  logits has NaN: {torch.isnan(logits).any().item()}")
                print(f"  logits has Inf: {torch.isinf(logits).any().item()}")
                if logits.shape[0] > 0 and logits.shape[1] > 0:
                    print(f"  logits[0, :5]: {logits[0, :5].tolist()}")
            print(f"  tp_size: {self.tp_size}")
            self._debug_lm_head_logits_logged = True
        if self.tp_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(self.tp_size)] if self.tp_rank == 0 else None
            dist.gather(logits, all_logits, 0)
            logits = torch.cat(all_logits, -1) if self.tp_rank == 0 else None
            if not hasattr(self, '_debug_lm_head_tp_logged'):
                print(f"\n[DEBUG ParallelLMHead.forward] After TP gather:")
                print(f"  logits shape: {logits.shape if logits is not None else 'None'}")
                print(f"  tp_rank: {self.tp_rank}")
                self._debug_lm_head_tp_logged = True
        return logits
