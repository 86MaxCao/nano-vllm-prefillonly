from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0

    def __post_init__(self):
        if self.temperature < 0:
            raise ValueError(
                f"temperature must be non-negative, got {self.temperature}"
            )
        if self.max_tokens < 1:
            raise ValueError(
                f"max_tokens must be at least 1, got {self.max_tokens}"
            )
        if not 0 < self.top_p <= 1:
            raise ValueError(
                f"top_p must be in (0, 1], got {self.top_p}"
            )
