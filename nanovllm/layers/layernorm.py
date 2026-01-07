import torch
from torch import nn

# Global flag to indicate if enforce_eager mode is enabled
# This is set by model_runner.py to avoid inplace operation issues with torch.compile
_ENFORCE_EAGER_MODE = False


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def _rms_forward_impl(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        """Implementation of RMS forward without inplace operations for enforce_eager mode."""
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # Use non-inplace operations to avoid torch.compile issues
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        orig_dtype = x.dtype
        x = x.float()
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x

    def _add_rms_forward_impl(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Implementation of add RMS forward without inplace operations for enforce_eager mode."""
        orig_dtype = x.dtype
        x = x.float() + residual.float()
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        # Use non-inplace operations to avoid torch.compile issues
        x = x * torch.rsqrt(var + self.eps)
        x = x.to(orig_dtype) * self.weight
        return x, residual

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        orig_dtype = x.dtype
        x = x.float().add_(residual.float())
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # Check if enforce_eager mode is enabled
        # If so, use non-inplace implementations to avoid compilation issues
        from nanovllm.layers.layernorm import _ENFORCE_EAGER_MODE
        if _ENFORCE_EAGER_MODE:
            if residual is None:
                return self._rms_forward_impl(x)
            else:
                return self._add_rms_forward_impl(x, residual)
        else:
            if residual is None:
                return self.rms_forward(x)
            else:
                return self.add_rms_forward(x, residual)
