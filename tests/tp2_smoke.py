"""TP=2 GPU smoke test: the rendezvous fix must not deadlock initialization.

Run (from the repo root, two GPUs required):
    NANOVLLM_TEST_MODEL_ROOT=/path/to/hf_cache python tests/tp2_smoke.py
"""
import os
import sys
import time

MODEL_ROOT = os.environ.get("NANOVLLM_TEST_MODEL_ROOT", "")
MODEL = os.path.join(MODEL_ROOT, "Qwen3-0.6B")


def main() -> int:
    import torch

    if torch.cuda.device_count() < 2:
        print(f"SKIP: need 2 GPUs, found {torch.cuda.device_count()}")
        return 0

    from nanovllm import LLM, SamplingParams

    prompts = ["Is the Earth round? Answer Yes or No.", "The capital of France is"]

    t0 = time.perf_counter()
    llm = LLM(MODEL, tensor_parallel_size=2, enforce_eager=True)
    init_seconds = time.perf_counter() - t0
    print(f"init ok in {init_seconds:.1f}s (no deadlock)")

    try:
        # Single-token generation through the TP RPC path.
        tokens = llm.generate_single_token(
            prompts, SamplingParams(temperature=0.0, max_tokens=1)
        )
        print("single-token ok:", tokens)
        assert len(tokens) == len(prompts)

        # A larger batch exercises the shared-memory RPC loop further.
        # Multi-token decode is intentionally unsupported here: this
        # flash-attn build returns NaN for paged KV caches, so
        # max_tokens > 1 raises by design (see layers/attention.py).
        batch = [f"prompt number {i}: the capital of France is" for i in range(8)]
        tokens2 = llm.generate_single_token(
            batch, SamplingParams(temperature=0.0, max_tokens=1)
        )
        print("batched single-token ok:", tokens2)
        assert len(tokens2) == 8
    finally:
        llm.exit()

    print("TP=2 SMOKE PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
