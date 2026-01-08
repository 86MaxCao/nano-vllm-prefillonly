import torch
from torch import nn


class Sampler(nn.Module):

    def __init__(self):
        super().__init__()

    @torch.compile
    def forward(self, logits: torch.Tensor, temperatures: torch.Tensor):
        # Use argmax (greedy sampling) for deterministic results
        # Ignore temperature parameter and directly use argmax on logits
        sample_tokens = logits.float().argmax(dim=-1)
        return sample_tokens
