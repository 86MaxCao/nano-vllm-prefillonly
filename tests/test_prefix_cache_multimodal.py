"""Multimodal coverage for the prefix KV cache APIs (P4).

An image-bearing prefix runs through the vision tower once; its per-layer
KV (rotated with M-RoPE positions) is cached, and text-only suffixes
continue the M-RoPE position sequence from the recorded ``mrope_end``.
These tests pin, on Qwen3-VL-2B (and the parity test on Qwen2.5-VL-3B):

- image prefix + text suffix vs one full multimodal prefill (parity),
- text-only prefixes on a VL model (no M-RoPE offset needed),
- the SemIf shared shape with vision prefixes (fork + batched suffixes),
- input validation (image tokens in suffixes, malformed request dicts).

The reference is the engine's own ``prefill_last_logits_multimodal`` on the
combined image+text request; HF parity of that base path is covered by
test_prefill_last_logits.py / test_hf_parity.py.

GPU + model-weight tests are marked `slow` and need
NANOVLLM_TEST_MODEL_ROOT pointing at /mnt/nas-tbt/tbt/checkpoint/hf_cache.
"""
import pytest

torch = pytest.importorskip("torch")

from tests.conftest import model_path, requires_cuda

pytestmark = pytest.mark.slow

CANDIDATES = [9454, 1406, 1917]

# bf16 forward + different kernel tilings (prefix+suffix varlen vs full
# prefill) shift logits by ~0.1-0.2, same scale as the parity suite.
TOL = {"atol": 0.25, "rtol": 0.05}

PREFIX_TEXT = "Look at the image carefully."
SUFFIX_TEXTS = [
    " What is the dominant color? Answer red or blue.",
    " How many objects do you see?",
    " Describe the image in one word.",
]


def _make_image(color=(200, 30, 30), size=(224, 224)):
    from PIL import Image

    return Image.new("RGB", size, color=color)


def _mm_request(text, image):
    """Raw text-shape request: no chat template, so the expanded id stream
    has no template tail and a prefix request is a true token-level prefix
    of the combined request."""
    return {
        "text": "<|vision_start|><|image_pad|><|vision_end|>" + text,
        "images": [image],
    }


@pytest.fixture
def llm(request):
    """Function-scoped: two live engines exhaust GPU memory, and VL engines
    are too heavy to share across parametrized model keys."""
    requires_cuda()
    from nanovllm import LLM

    model_key = request.param
    engine = LLM(
        model_path(model_key),
        multimodal_model_type=model_key,
        enforce_eager=True,
    )
    yield engine
    engine.exit()
    import gc

    gc.collect()
    torch.cuda.empty_cache()


def _full_prefill_logits(llm, request, candidates=CANDIDATES):
    result = llm.prefill_last_logits_multimodal([request], [list(candidates)])
    return result["logits"][0]


def _split_mm(llm, prefix_text, suffix_text, image):
    """Split an image+text request into (prefix_ids, suffix_ids) on the
    engine's own expanded id streams.

    The requests use the raw text shape (no chat template), so the full
    stream is literally prefix ids + suffix ids; the assert guards against
    a BPE merge across the split boundary.
    """
    full_ids, _, _ = llm._prepare_prefix_request(
        _mm_request(prefix_text + suffix_text, image)
    )
    prefix_ids, _, _ = llm._prepare_prefix_request(
        _mm_request(prefix_text, image)
    )
    assert full_ids[: len(prefix_ids)] == prefix_ids, (
        "tokenization merged across the split boundary"
    )
    return prefix_ids, full_ids[len(prefix_ids):]


@pytest.mark.parametrize("llm", ["qwen3_vl", "qwen2_5_vl"], indirect=True)
def test_mm_prefix_suffix_matches_full_prefill(llm):
    image = _make_image()
    prefix_ids, suffix_ids = _split_mm(llm, PREFIX_TEXT, SUFFIX_TEXTS[0], image)

    expected = _full_prefill_logits(
        llm, _mm_request(PREFIX_TEXT + SUFFIX_TEXTS[0], image)
    )
    handle = llm.prefill_prefix(_mm_request(PREFIX_TEXT, image))
    try:
        result = llm.prefill_suffix_logits([handle], [suffix_ids], [CANDIDATES])
    finally:
        llm.release_prefix(handle)
    assert result["prefix_lengths"] == [len(prefix_ids)]
    assert result["sequence_lengths"] == [len(suffix_ids)]
    torch.testing.assert_close(result["logits"][0], expected, **TOL)


@pytest.mark.parametrize("llm", ["qwen3_vl"], indirect=True)
def test_mm_text_prefix_on_vl_model(llm):
    """Text-only prefixes on a VL model: positions are linear, no M-RoPE
    offset; behaves exactly like a text model."""
    text = (
        "The quick brown fox jumps over the lazy dog. " * 4
        + "Is the Earth round? Answer Yes or No."
    )
    ids = llm.tokenizer.encode(text, add_special_tokens=False)
    split = len(ids) // 2
    prefix_ids, suffix_ids = ids[:split], ids[split:]

    expected = llm.prefill_last_logits([ids], [CANDIDATES])["logits"][0]
    handle = llm.prefill_prefix(prefix_ids)
    try:
        result = llm.prefill_suffix_logits([handle], [suffix_ids], [CANDIDATES])
    finally:
        llm.release_prefix(handle)
    assert result["prefix_lengths"] == [split]
    torch.testing.assert_close(result["logits"][0], expected, **TOL)


@pytest.mark.parametrize("llm", ["qwen3_vl"], indirect=True)
def test_mm_shared_shape_batch_of_forks(llm):
    """SemIf shared shape with a vision prefix: one image prefix forked N
    times, N text suffixes in a single batched suffix forward."""
    image = _make_image()
    splits = [_split_mm(llm, PREFIX_TEXT, text, image) for text in SUFFIX_TEXTS]
    prefix_ids = splits[0][0]
    suffix_ids = [suffix for _, suffix in splits]

    expected = [
        _full_prefill_logits(llm, _mm_request(PREFIX_TEXT + text, image))
        for text in SUFFIX_TEXTS
    ]
    handle = llm.prefill_prefix(_mm_request(PREFIX_TEXT, image))
    forks = []
    try:
        forks = llm.fork_prefix(handle, len(suffix_ids))
        result = llm.prefill_suffix_logits(
            forks, suffix_ids, [CANDIDATES] * len(suffix_ids)
        )
    finally:
        llm.release_prefix(handle, *forks)
    for i, exp in enumerate(expected):
        torch.testing.assert_close(result["logits"][i], exp, **TOL)


@pytest.mark.parametrize("llm", ["qwen3_vl"], indirect=True)
def test_mm_suffix_rejects_image_tokens(llm):
    image_token_id = getattr(llm.config.hf_config, "image_token_id", None)
    assert image_token_id is not None, "test expects a VL model"
    handle = llm.prefill_prefix([1, 2, 3])
    try:
        with pytest.raises(ValueError, match="text-only"):
            llm.prefill_suffix_logits(
                [handle], [[4, image_token_id, 5]], [CANDIDATES]
            )
    finally:
        llm.release_prefix(handle)


def test_mm_prefix_request_validation():
    """Malformed request dicts must fail before any model/processor load."""
    from nanovllm.engine.llm_engine import LLMEngine

    class FakeHFConfig:
        vocab_size = 100

    class FakeConfig:
        hf_config = FakeHFConfig()
        is_multimodal = False
        tensor_parallel_size = 1
        max_model_len = 4096
        model = "fake"
        model_revision = None

    engine = LLMEngine.__new__(LLMEngine)
    engine.config = FakeConfig

    with pytest.raises(ValueError, match="input_ids.*or"):
        engine._prepare_prefix_request({"foo": 1})
    with pytest.raises(ValueError, match="empty input_ids"):
        engine._prepare_prefix_request({"input_ids": []})
    with pytest.raises(ValueError, match="multimodal model"):
        engine._prepare_prefix_request({"text": "hi", "images": ["x"]})
