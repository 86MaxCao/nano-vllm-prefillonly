"""GDN (linear-attention) coverage for the prefix KV cache APIs (P3).

Qwen3.5-0.8B interleaves full-attention layers with gated delta net layers,
so prefix reuse must fork both the attention KV and the per-layer GDN state
(conv window + recurrent state). These tests pin:

- prefix/suffix split vs full prefill parity (state hand-off correctness),
- the conv-window boundary (prefix shorter than kernel_size - 1),
- the SemIf shared shape (one prefix, N forks, N suffixes, one batch),
- store immutability under replay (a handle can be reused repeatedly).

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

PROMPT = (
    "The quick brown fox jumps over the lazy dog. " * 6
    + "Is the Earth round? Answer Yes or No."
)


@pytest.fixture(scope="module")
def llm():
    """One engine per module: repeated LLM construction in a single process
    exhausts GPU memory on KV block allocation."""
    requires_cuda()
    from nanovllm import LLM

    engine = LLM(model_path("qwen3_5"), enforce_eager=True)
    yield engine
    engine.exit()
    del engine
    import gc

    gc.collect()
    torch.cuda.empty_cache()


@pytest.fixture(scope="module")
def tokenizer(llm):
    return llm.tokenizer


def _full_prefill_logits(llm, full_ids, candidates=CANDIDATES):
    result = llm.prefill_last_logits([full_ids], [list(candidates)])
    return result["logits"][0]


@pytest.mark.parametrize("split_fraction", [0.1, 0.5, 0.9])
def test_gdn_prefix_suffix_matches_full_prefill(llm, tokenizer, split_fraction):
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    split = max(1, min(len(ids) - 1, int(len(ids) * split_fraction)))
    prefix_ids, suffix_ids = ids[:split], ids[split:]

    expected = _full_prefill_logits(llm, ids)
    handle = llm.prefill_prefix(prefix_ids)
    try:
        result = llm.prefill_suffix_logits(
            [handle], [suffix_ids], [CANDIDATES]
        )
    finally:
        llm.release_prefix(handle)
    assert result["prefix_lengths"] == [split]
    assert result["sequence_lengths"] == [len(suffix_ids)]
    torch.testing.assert_close(result["logits"][0], expected, **TOL)


@pytest.mark.parametrize("prefix_len", [1, 2])
def test_gdn_short_prefix_conv_boundary(llm, tokenizer, prefix_len):
    """Prefix shorter than the GDN conv window (kernel_size - 1 = 3): the
    captured conv state is partly zero-filled and must still line up with
    the zero left-padding a full prefill sees."""
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    prefix_ids, suffix_ids = ids[:prefix_len], ids[prefix_len:]

    expected = _full_prefill_logits(llm, ids)
    handle = llm.prefill_prefix(prefix_ids)
    try:
        result = llm.prefill_suffix_logits(
            [handle], [suffix_ids], [CANDIDATES]
        )
    finally:
        llm.release_prefix(handle)
    torch.testing.assert_close(result["logits"][0], expected, **TOL)


def test_gdn_shared_shape_batch_of_forks(llm, tokenizer):
    """SemIf shared shape: one prefix forked N times, N different suffixes in
    a single batched suffix forward; each row matches its full prefill."""
    prefix_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    suffix_texts = [
        " Is the Earth round?",
        " What is the capital of France?",
        " How many legs does a cat have?",
        " Name one primary color.",
    ]
    suffix_ids = [
        tokenizer.encode(text, add_special_tokens=False) for text in suffix_texts
    ]

    expected = [
        _full_prefill_logits(llm, prefix_ids + suffix) for suffix in suffix_ids
    ]
    handle = llm.prefill_prefix(prefix_ids)
    forks = []
    try:
        forks = llm.fork_prefix(handle, len(suffix_ids))
        result = llm.prefill_suffix_logits(
            forks, suffix_ids, [CANDIDATES] * len(suffix_ids)
        )
    finally:
        llm.release_prefix(handle, *forks)
    assert result["prefix_lengths"] == [len(prefix_ids)] * len(suffix_ids)
    for i, exp in enumerate(expected):
        torch.testing.assert_close(result["logits"][i], exp, **TOL)


def test_gdn_replay_does_not_mutate_store(llm, tokenizer):
    """Reusing one handle across several suffix forwards must give identical
    results every time: replay reads the stored conv/recurrent state but
    must never write into it."""
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    prefix_ids, suffix_ids = ids[:-6], ids[-6:]

    expected = _full_prefill_logits(llm, ids)
    handle = llm.prefill_prefix(prefix_ids)
    try:
        first = llm.prefill_suffix_logits([handle], [suffix_ids], [CANDIDATES])
        second = llm.prefill_suffix_logits([handle], [suffix_ids], [CANDIDATES])
        # Batch the same handle with a fork of itself.
        (fork,) = llm.fork_prefix(handle, 1)
        try:
            mixed = llm.prefill_suffix_logits(
                [handle, fork], [suffix_ids, suffix_ids], [CANDIDATES] * 2
            )
        finally:
            llm.release_prefix(fork)
    finally:
        llm.release_prefix(handle)
    torch.testing.assert_close(first["logits"][0], expected, **TOL)
    torch.testing.assert_close(second["logits"][0], first["logits"][0], **TOL)
    torch.testing.assert_close(mixed["logits"][0], first["logits"][0], **TOL)
    torch.testing.assert_close(mixed["logits"][1], first["logits"][0], **TOL)


def test_gdn_fork_independence_after_release(llm, tokenizer):
    """Forks are physical copies (GDN state included): releasing the source
    handle must not affect suffix forwards on the fork."""
    ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    prefix_ids, suffix_ids = ids[:-4], ids[-4:]

    expected = _full_prefill_logits(llm, ids)
    handle = llm.prefill_prefix(prefix_ids)
    (fork,) = llm.fork_prefix(handle, 1)
    llm.release_prefix(handle)
    try:
        result = llm.prefill_suffix_logits([fork], [suffix_ids], [CANDIDATES])
    finally:
        llm.release_prefix(fork)
    torch.testing.assert_close(result["logits"][0], expected, **TOL)
