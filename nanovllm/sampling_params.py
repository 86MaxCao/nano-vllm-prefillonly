from dataclasses import dataclass


@dataclass
class SamplingParams:
    temperature: float = 1.0
    max_tokens: int = 64
    ignore_eos: bool = False
    top_p: float = 1.0

    def __post_init__(self):
        assert self.temperature >= 0, "temperature must be non-negative"
        assert self.max_tokens >= 1, "max_tokens must be at least 1"
        assert 0 < self.top_p <= 1, "top_p must be in (0, 1]"
