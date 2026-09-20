"""Tests for the prefix KV cache APIs (feat/prefill-kv-cache, P1/P2).

Coverage:
- PrefixCacheStore logic (CPU): create/get/release, fork deep-copy
  independence, quota accounting, error paths.
- _interleave_prefix_suffix assembly (CPU).
- Context capture/replay mutual exclusion (CPU).
- GPU (Qwen3-0.6B): prefix/suffix split vs full prefill parity, fork
  independence, batched shared-shape forward (N forks, N suffixes), mixed
  handles in one batch, permutation invariance, long-prefix bottom-right
  causal guard, API validation.

GPU + model-weight tests are marked `slow` and need
NANOVLLM_TEST_MODEL_ROOT pointing at /mnt/nas-tbt/tbt/checkpoint/hf_cache.
"""
import pytest

torch = pytest.importorskip("torch")

from nanovllm.engine.prefix_cache import PrefixCacheStore, PrefixEntry
from nanovllm.layers.attention import _interleave_prefix_suffix
from nanovllm.utils.context import set_context, reset_context
from tests.conftest import model_path, requires_cuda

CANDIDATES = [9454, 1406, 1917]

# bf16 forward + different kernel tilings (prefix+suffix varlen vs full
# prefill) shift logits by ~0.1-0.2, same scale as the parity suite.
TOL = {"atol": 0.25, "rtol": 0.05}


def _entry(prefix_len=4, layers=2, heads=2, head_dim=8, device="cpu"):
    kv = [
        (
            torch.randn(prefix_len, heads, head_dim, device=device),
            torch.randn(prefix_len, heads, head_dim, device=device),
        )
        for _ in range(layers)
    ]
    return PrefixEntry(prefix_len=prefix_len, kv=kv, gdn=[])


class TestPrefixCacheStore:
    def test_create_get_release(self):
        store = PrefixCacheStore(max_bytes=1 << 20)
        entry = _entry()
        handle = store.create(entry)
        assert store.get(handle) is entry
        assert store.used_bytes == entry.nbytes
        store.release(handle)
        assert store.used_bytes == 0
        assert len(store) == 0

    def test_unknown_handle_raises(self):
        store = PrefixCacheStore(max_bytes=1 << 20)
        with pytest.raises(ValueError, match="unknown prefix handle"):
            store.get(999)
        with pytest.raises(ValueError, match="unknown prefix handle"):
            store.release(999)
        with pytest.raises(ValueError, match="unknown prefix handle"):
            store.fork(999, 1)

    def test_fork_is_deep_copy(self):
        store = PrefixCacheStore(max_bytes=1 << 20)
        src = _entry()
        handle = store.create(src)
        (forked,) = store.fork(handle, 1)
        fork_entry = store.get(forked)
        assert fork_entry.prefix_len == src.prefix_len
        for (k_s, v_s), (k_f, v_f) in zip(src.kv, fork_entry.kv):
            assert torch.equal(k_s, k_f) and torch.equal(v_s, v_f)
            assert k_s.data_ptr() != k_f.data_ptr()
        # Mutating the source must not touch the fork.
        src.kv[0][0].fill_(123.0)
        assert not torch.equal(src.kv[0][0], fork_entry.kv[0][0])

    def test_fork_multiple_and_quota(self):
        entry = _entry()
        store = PrefixCacheStore(max_bytes=entry.nbytes * 3)
        handle = store.create(entry)  # budget left: 2 copies
        forks = store.fork(handle, 2)
        assert len(forks) == 2 and len(set(forks)) == 2
        assert store.used_bytes == entry.nbytes * 3
        with pytest.raises(RuntimeError, match="quota exceeded"):
            store.fork(handle, 1)
        with pytest.raises(RuntimeError, match="quota exceeded"):
            store.create(_entry())
        # Releasing one fork makes room for exactly one more.
        store.release(forks[0])
        store.fork(handle, 1)

    def test_failed_fork_is_atomic(self):
        entry = _entry()
        store = PrefixCacheStore(max_bytes=entry.nbytes * 2)
        handle = store.create(entry)
        with pytest.raises(RuntimeError, match="quota exceeded"):
            store.fork(handle, 5)
        assert len(store) == 1
        assert store.used_bytes == entry.nbytes

    def test_fork_count_validation(self):
        store = PrefixCacheStore(max_bytes=1 << 20)
        handle = store.create(_entry())
        with pytest.raises(ValueError, match="fork count"):
            store.fork(handle, 0)

    def test_store_quota_must_be_positive(self):
        with pytest.raises(ValueError, match="max_bytes"):
            PrefixCacheStore(max_bytes=0)


class TestInterleavePrefixSuffix:
    def test_two_sequences(self):
        # seq0: prefix 2 + suffix 1; seq1: prefix 1 + suffix 2
        k_p0 = torch.full((2, 1, 1), 10.0)
        v_p0 = torch.full((2, 1, 1), 20.0)
        k_p1 = torch.full((1, 1, 1), 30.0)
        v_p1 = torch.full((1, 1, 1), 40.0)
        k = torch.tensor([[[1.0]], [[2.0]], [[3.0]]])  # suffix varlen: [1] + [2,3]
        v = -k
        bounds = [0, 1, 3]
        k_out, v_out = _interleave_prefix_suffix(
            [(k_p0, v_p0), (k_p1, v_p1)], k, v, bounds
        )
        expected_k = torch.tensor([[[10.0]], [[10.0]], [[1.0]],
                                   [[30.0]], [[2.0]], [[3.0]]])
        expected_v = torch.tensor([[[20.0]], [[20.0]], [[-1.0]],
                                   [[40.0]], [[-2.0]], [[-3.0]]])
        assert torch.equal(k_out, expected_k)
        assert torch.equal(v_out, expected_v)

    def test_empty_prefix_list_raises(self):
        with pytest.raises(ValueError, match="non-empty"):
            _interleave_prefix_suffix([], torch.zeros(1, 1, 1), torch.zeros(1, 1, 1), [0, 1])


class TestContextGuard:
    def test_capture_and_replay_are_mutually_exclusive(self):
        with pytest.raises(AssertionError, match="mutually exclusive"):
            set_context(True, kv_capture=[], prefix_kv=[])
        reset_context()  # do not leak the partial state into other tests


# ---------------------------------------------------------------------------
# GPU tests (Qwen3-0.6B)
# ---------------------------------------------------------------------------

pytestmark_gpu = pytest.mark.slow

PROMPT = (
    "The quick brown fox jumps over the lazy dog. " * 6
    + "Is the Earth round? Answer Yes or No."
)
OTHER_PROMPT = "What is the capital of France? " * 10


@pytest.fixture(scope="module")
def llm():
    """One engine per module: repeated LLM construction in a single process
    exhausts GPU memory on KV block allocation."""
    requires_cuda()
    from nanovllm import LLM

    engine = LLM(model_path("qwen3"), enforce_eager=True)
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


@pytest.mark.slow
@pytest.mark.parametrize("split_fraction", [0.05, 0.5, 0.95])
def test_prefix_suffix_matches_full_prefill(llm, tokenizer, split_fraction):
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


@pytest.mark.slow
def test_long_prefix_short_suffix_bottom_right_causal(llm, tokenizer):
    """Guard the flash-attn bottom-right alignment assumption: a ~1.5k-token
    prefix with a 3-token suffix must still match the full prefill."""
    ids = tokenizer.encode(PROMPT * 20, add_special_tokens=False)
    prefix_ids, suffix_ids = ids[:-3], ids[-3:]
    assert 1000 < len(prefix_ids) < 4000  # stays under the default max_model_len

    expected = _full_prefill_logits(llm, ids)
    handle = llm.prefill_prefix(prefix_ids)
    try:
        result = llm.prefill_suffix_logits(
            [handle], [suffix_ids], [CANDIDATES]
        )
    finally:
        llm.release_prefix(handle)
    torch.testing.assert_close(result["logits"][0], expected, **TOL)


@pytest.mark.slow
def test_shared_shape_batch_of_forks(llm, tokenizer):
    """SemIf shared shape: one prefix forked N times, N different suffixes in
    a single batched suffix forward; each row matches its full prefill."""
    prefix_ids = tokenizer.encode(PROMPT, add_special_tokens=False)
    suffix_texts = [
        " Is the Earth round?",
        " What is the capital of France?",
        " How many legs does a cat have?",
        " Name one primary color.",
        " Is water wet?",
        " What is two plus two?",
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


@pytest.mark.slow
def test_mixed_handles_and_permutation_invariance(llm, tokenizer):
    """Two different prefixes in one batch; shuffling row order must not
    change per-row logits."""
    prefix_a = tokenizer.encode(PROMPT, add_special_tokens=False)
    prefix_b = tokenizer.encode(OTHER_PROMPT, add_special_tokens=False)
    suffix_a = tokenizer.encode(" Answer Yes or No.", add_special_tokens=False)
    suffix_b = tokenizer.encode(" Name the city.", add_special_tokens=False)

    expected_a = _full_prefill_logits(llm, prefix_a + suffix_a)
    expected_b = _full_prefill_logits(llm, prefix_b + suffix_b)
    handle_a = llm.prefill_prefix(prefix_a)
    handle_b = llm.prefill_prefix(prefix_b)
    try:
        forward = llm.prefill_suffix_logits(
            [handle_a, handle_b], [suffix_a, suffix_b], [CANDIDATES] * 2
        )
        shuffled = llm.prefill_suffix_logits(
            [handle_b, handle_a], [suffix_b, suffix_a], [CANDIDATES] * 2
        )
    finally:
        llm.release_prefix(handle_a, handle_b)
    torch.testing.assert_close(forward["logits"][0], expected_a, **TOL)
    torch.testing.assert_close(forward["logits"][1], expected_b, **TOL)
    # Same rows, different batch order.
    torch.testing.assert_close(shuffled["logits"][0], forward["logits"][1], **TOL)
    torch.testing.assert_close(shuffled["logits"][1], forward["logits"][0], **TOL)


@pytest.mark.slow
def test_fork_survives_source_release(llm, tokenizer):
    """Forks are physical copies: releasing the source handle must not
    affect suffix forwards on the fork."""
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


@pytest.mark.slow
def test_prefix_api_validation(llm):
    with pytest.raises(ValueError, match="empty"):
        llm.prefill_prefix([])
    with pytest.raises(ValueError, match="nonempty"):
        llm.prefill_suffix_logits([], [[1]], [CANDIDATES])
    with pytest.raises(ValueError, match="equal length"):
        llm.prefill_suffix_logits([1], [[1], [2]], [CANDIDATES])
    with pytest.raises(ValueError, match="at least one token"):
        llm.prefill_suffix_logits([1], [[]], [CANDIDATES])
    with pytest.raises(ValueError, match="fork count"):
        llm.fork_prefix(1, 0)
    handle = llm.prefill_prefix([1, 2, 3])
    with pytest.raises(ValueError, match="unknown prefix handle"):
        llm.prefill_suffix_logits([999], [[1]], [CANDIDATES])
    # Double release raises on the second pop, but the first pop already
    # freed the handle (release is best-effort, not atomic).
    with pytest.raises(ValueError, match="unknown prefix handle"):
        llm.release_prefix(handle, handle)
    with pytest.raises(ValueError, match="unknown prefix handle"):
        llm.release_prefix(handle)
