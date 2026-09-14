import torch
from torch import nn


class Sampler(nn.Module):
    """Temperature + nucleus sampling with a greedy path for temperature 0."""

    def forward(
        self,
        logits: torch.Tensor,
        temperatures: torch.Tensor,
        top_p: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = logits.float()
        greedy_tokens = logits.argmax(dim=-1)

        sampling_rows = temperatures > 0
        if not bool(sampling_rows.any()):
            return greedy_tokens

        # Scale only the stochastic rows. Dividing greedy rows by a tiny epsilon
        # would overflow the softmax to inf/nan and trip multinomial's checks.
        scale = torch.where(
            sampling_rows, temperatures, torch.ones_like(temperatures)
        ).unsqueeze(-1)
        probs = torch.softmax(logits / scale, dim=-1)

        if top_p is not None and bool((top_p < 1.0).any()):
            probs = _apply_top_p(probs, top_p)

        sampled_tokens = torch.multinomial(probs, num_samples=1).squeeze(-1)
        return torch.where(sampling_rows, sampled_tokens, greedy_tokens)


def _apply_top_p(probs: torch.Tensor, top_p: torch.Tensor) -> torch.Tensor:
    """Zero out the tail of the distribution beyond cumulative mass `top_p`."""
    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
    cumulative = sorted_probs.cumsum(dim=-1)
    # Keep the token that crosses the threshold so at least one always survives.
    drop = (cumulative - sorted_probs) >= top_p.unsqueeze(-1)
    sorted_probs = sorted_probs.masked_fill(drop, 0.0)
    probs = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs)
    return probs / probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
