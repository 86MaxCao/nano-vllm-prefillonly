"""Regression tests for the 2026-09-14 review fixes.

Covers three fixes (see docs/2026-09-14-review-fixes.md):
1. Mixed image + text batches no longer crash embed_batch.
2. MeanPool.forward_varlen accumulates in float32.
3. Global attention context is cleared even when a forward raises.

This file deliberately does NOT import ``tests.conftest``: on machines where
site-packages contains another ``tests`` package, that import is shadowed and
collection fails.
"""
import pytest

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.slow

MODEL_ROOT = "/mnt/nas-tbt/tbt/checkpoint/hf_cache"
QWEN3_VL_EMBED = f"{MODEL_ROOT}/Qwen3-VL-Embedding-2B"
QWEN3_EMBED = f"{MODEL_ROOT}/Qwen3-Embedding-0.6B"
QWEN3 = f"{MODEL_ROOT}/Qwen3-0.6B"

requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA is required"
)


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a, b = a.float().cpu(), b.float().cpu()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


def _requires_model(path: str):
    import os

    return pytest.mark.skipif(
        not os.path.isdir(path), reason=f"model not available: {path}"
    )


# ---------------------------------------------------------------- pooler
class TestMeanPoolVarlenPrecision:
    def test_bf16_accumulation_matches_float32(self):
        """Before the fix, bf16 index_add_ drifted ~1.7e-3 relative error."""
        from nanovllm.layers.pooler import MeanPool

        torch.manual_seed(0)
        length, hidden = 8192, 64
        h = (torch.randn(length, hidden) * 0.1).to(torch.bfloat16)
        cu = torch.tensor([0, length], dtype=torch.int32)

        pool = MeanPool()
        got = pool.forward_varlen(h, cu)
        ref = h.float().mean(dim=0)

        assert got.dtype == torch.float32
        rel_err = ((got - ref).norm() / ref.norm()).item()
        assert rel_err < 1e-5, f"relative error {rel_err:.2e} too large"

    def test_multi_sequence_split_matches_forward(self):
        """forward_varlen must agree with the padded forward() reference."""
        from nanovllm.layers.pooler import MeanPool

        torch.manual_seed(1)
        lens, hidden = [7, 1, 33], 16
        h = torch.randn(sum(lens), hidden).to(torch.bfloat16)
        cu = torch.tensor([0] + list(torch.cumsum(torch.tensor(lens), 0)), dtype=torch.int32)

        got = MeanPool().forward_varlen(h, cu)
        padded = torch.zeros(3, max(lens), hidden, dtype=torch.bfloat16)
        mask = torch.zeros(3, max(lens))
        for i, n in enumerate(lens):
            padded[i, :n] = h[cu[i]: cu[i + 1]]
            mask[i, :n] = 1
        ref = MeanPool().forward(padded, mask)

        err = (got - ref).abs().max().item()
        assert err < 1e-4, f"forward vs forward_varlen disagree by {err:.2e}"


# ------------------------------------------------------- mixed batches
@requires_cuda
@_requires_model(QWEN3_VL_EMBED)
class TestMixedBatchEmbedding:
    def _llm(self):
        from nanovllm import LLM

        return LLM(
            QWEN3_VL_EMBED,
            multimodal_model_type="qwen3_vl",
            is_embedding=True,
            embedding_type="qwen3_vl",
            enforce_eager=True,
        )

    def test_image_plus_text_in_one_batch(self):
        """The old merge branch indexed a 2-D pixel_values by sequence and
        crashed with split_with_sizes errors."""
        from PIL import Image

        llm = self._llm()
        try:
            img = Image.new("RGB", (224, 224), color=(200, 30, 30))
            img_prompt, text_prompt = "Describe the image.", "plain text query"

            alone_img = llm.embed_batch([img_prompt], images=[img]).float().cpu()
            together = llm.embed_batch(
                [img_prompt, text_prompt], images=[img]
            ).float().cpu()
        finally:
            llm.exit()

        assert together.shape == (2, alone_img.shape[1])
        sim = cosine(together[0], alone_img[0]).item()
        assert sim > 0.999, f"image request changed in mixed batch: {sim:.6f}"

    def test_none_image_entry_is_text_only(self):
        """images=[img, None] used to build a corrupt message with an image
        placeholder but no image."""
        from PIL import Image

        llm = self._llm()
        try:
            img = Image.new("RGB", (224, 224), color=(30, 30, 200))
            texts = ["Describe the image.", "plain text query"]

            got = llm.embed_batch(texts, images=[img, None]).float().cpu()
            # Reference goes through the same multimodal branch (chat
            # template applied); embed_batch without images would not.
            ref_text = (
                llm.embed_batch([texts[1]], images=[None]).float().cpu()
            )
        finally:
            llm.exit()

        # The text-only entry must not silently embed an image placeholder.
        sim = cosine(got[1], ref_text[0]).item()
        assert sim > 0.999, f"None-image entry drifted: {sim:.6f}"


# ------------------------------------------------- generation regression
@requires_cuda
@_requires_model(QWEN3)
def test_generation_single_token_still_matches_hf():
    """run() was split into run()/_run_forward(); greedy tokens must not
    change. Reference computed with transformers on the same GPU."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanovllm import LLM, SamplingParams

    prompts = [
        "Is the Earth round? Answer Yes or No.",
        "What is the capital of France?",
        "The quick brown fox jumps over the lazy",
    ]
    tok = AutoTokenizer.from_pretrained(QWEN3)
    hf = AutoModelForCausalLM.from_pretrained(
        QWEN3, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    expected = []
    with torch.inference_mode():
        for prompt in prompts:
            ids = tok(prompt, return_tensors="pt").to("cuda")
            expected.append(int(hf(**ids).logits[0, -1].argmax()))
    del hf
    torch.cuda.empty_cache()

    llm = LLM(QWEN3, enforce_eager=True)
    try:
        got = llm.generate_single_token(
            prompts, SamplingParams(temperature=0.0, max_tokens=1)
        )
    finally:
        llm.exit()

    assert got == expected, f"token mismatch: {got} != {expected}"


# ------------------------------------------------ context cleanup on error
@requires_cuda
@_requires_model(QWEN3)
def test_context_reset_after_forward_failure():
    """A forward that raises must not leak the attention context."""
    from nanovllm import LLM, SamplingParams
    from nanovllm.engine.sequence import Sequence
    from nanovllm.utils.context import reset_context

    llm = LLM(QWEN3, enforce_eager=True)
    try:
        reset_context()

        class _Boom(Exception):
            pass

        real_run_forward = llm.model_runner._run_forward

        def failing(*args, **kwargs):
            raise _Boom("simulated forward failure")

        llm.model_runner._run_forward = failing
        seq = Sequence([1] * 16)
        with pytest.raises(_Boom):
            llm.model_runner.run([seq], True)
        llm.model_runner._run_forward = real_run_forward

        from nanovllm.utils.context import get_context

        ctx = get_context()
        assert ctx.cu_seqlens_q is None and ctx.slot_mapping is None, (
            "attention context leaked after a failed forward: "
            f"cu_seqlens_q={ctx.cu_seqlens_q}, slot_mapping={ctx.slot_mapping}"
        )
    finally:
        llm.exit()
