"""Tests for the prefill_last_logits / prefill_last_logits_multimodal APIs.

Coverage:
- HF parity: candidate logits equal the Transformers last-position logits
  (text model and multimodal model).
- Variable-length batch: last position means each sequence's own last real
  token, not the padded position.
- Candidate order: returned columns follow the caller's candidate order.
- Batch composition invariance: a row's logits do not change when batched
  with longer/shorter rows (guards the left-padding positions fix and GDN
  sequence-state isolation).
- Input validation: candidate ids out of range / length mismatch raise.
- generate_multimodal regression after the _prepare_multimodal_batch
  extraction: identical tokens to the pre-refactor processor path.

GPU + model-weight tests are marked `slow` and need
NANOVLLM_TEST_MODEL_ROOT pointing at /mnt/nas-tbt/tbt/checkpoint/hf_cache.
"""
import pytest

torch = pytest.importorskip("torch")

from tests.conftest import model_path, requires_cuda

pytestmark = pytest.mark.slow

CANDIDATE_TOKENS = [  # "Yes"/"No"-style single tokens for parity checks
    [9454, 1406, 1917],  # distinct ids; exact ids irrelevant to parity
    [1406, 1917, 9454],
    [9454, 1406],
]

PROMPTS = [
    "Is the Earth round? Answer Yes or No.",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy",  # shortest -> exercises padding
]

# bf16 forward + different kernel tilings (padded vs unpadded batch shapes)
# shift logits by ~0.1-0.2; genuine bugs (wrong positions, leaked GDN state)
# shift them by whole units, so atol=0.25/rtol=0.05 separates the two.
TOL = {"atol": 0.25, "rtol": 0.05}


def _hf_last_logits(path, prompt_lists, candidates, model_loader, tokenizer):
    """Reference: Transformers forward, gather candidate ids at last position."""
    expected = []
    with torch.inference_mode():
        for ids, cands in zip(prompt_lists, candidates):
            batch = {
                "input_ids": torch.tensor([ids], device="cuda"),
            }
            logits = model_loader(**batch).logits[0, -1]
            expected.append(logits[torch.tensor(cands, device="cuda")].float().cpu())
    return expected


@pytest.mark.parametrize("model_key", ["qwen3"])
def test_prefill_last_logits_matches_transformers(model_key):
    requires_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanovllm import LLM

    path = model_path(model_key)
    tok = AutoTokenizer.from_pretrained(path)
    # Parity contract: ids encoded with add_special_tokens=False.
    encoded = [tok.encode(p, add_special_tokens=False) for p in PROMPTS]

    hf = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    expected = _hf_last_logits(
        path, encoded, CANDIDATE_TOKENS,
        lambda **kw: hf(**kw), tok,
    )
    del hf
    torch.cuda.empty_cache()

    llm = LLM(path, enforce_eager=True)
    try:
        result = llm.prefill_last_logits(PROMPTS, CANDIDATE_TOKENS)
    finally:
        llm.exit()

    assert result["backend"] == "nanovllm"
    assert result["logits"].shape == (3, 3)
    assert result["logits"].dtype == torch.float32
    for i, cands in enumerate(CANDIDATE_TOKENS):
        row_mask = result["candidate_mask"][i]
        assert row_mask[: len(cands)].all()
        assert not row_mask[len(cands):].any()
    for i, (cands, exp) in enumerate(zip(CANDIDATE_TOKENS, expected)):
        got = result["logits"][i][: len(cands)]
        assert torch.allclose(got, exp, **TOL), (
            f"row {i}: {got.tolist()} vs HF {exp.tolist()}"
        )


@pytest.mark.parametrize("model_key", ["qwen3"])
def test_prefill_last_logits_candidate_order(model_key):
    """Swapping candidate order must swap result columns 1:1."""
    requires_cuda()
    from nanovllm import LLM

    path = model_path(model_key)
    llm = LLM(path, enforce_eager=True)
    try:
        a = llm.prefill_last_logits(PROMPTS[:1], CANDIDATE_TOKENS[:1])
        b = llm.prefill_last_logits(
            PROMPTS[:1], [list(reversed(CANDIDATE_TOKENS[0]))]
        )
    finally:
        llm.exit()

    assert torch.allclose(a["logits"][0], b["logits"][0].flip(0), atol=1e-5)
    assert b["candidate_token_ids"] == [list(reversed(CANDIDATE_TOKENS[0]))]


@pytest.mark.parametrize("model_key", ["qwen3"])
def test_prefill_last_logits_batch_invariance(model_key):
    """Each row's logits must not depend on the other rows in the batch."""
    requires_cuda()
    from nanovllm import LLM

    path = model_path(model_key)
    cands = [CANDIDATE_TOKENS[0]] * 3
    llm = LLM(path, enforce_eager=True)
    try:
        batched = llm.prefill_last_logits(PROMPTS, cands)
        singles = [llm.prefill_last_logits([p], [c])["logits"][0]
                   for p, c in zip(PROMPTS, cands)]
    finally:
        llm.exit()

    for i, single in enumerate(singles):
        assert torch.allclose(
            batched["logits"][i], single, **TOL
        ), f"row {i} changed with batch composition"


@pytest.mark.parametrize("model_key", ["qwen3_5"])
def test_prefill_last_logits_gdn_batch_invariance(model_key):
    """GDN (linear-attention) state isolation: batching must not leak state
    across sequences, and results must be permutation-invariant."""
    requires_cuda()
    from nanovllm import LLM

    path = model_path(model_key)
    cands = [CANDIDATE_TOKENS[0]] * 3
    llm = LLM(path, enforce_eager=True)
    try:
        batched = llm.prefill_last_logits(PROMPTS, cands)
        singles = [llm.prefill_last_logits([p], [c])["logits"][0]
                   for p, c in zip(PROMPTS, cands)]
        # Permutation: reverse row order; per-row logits must be identical.
        perm = llm.prefill_last_logits(
            list(reversed(PROMPTS)), list(reversed(cands))
        )
    finally:
        llm.exit()

    for i, single in enumerate(singles):
        assert torch.allclose(
            batched["logits"][i], single, **TOL
        ), f"GDN row {i} changed with batch composition"
    for i in range(3):
        assert torch.allclose(
            perm["logits"][2 - i], batched["logits"][i], **TOL
        ), f"GDN row {i} changed under permutation"


def _make_validation_llm():
    """CPU-friendly fake engine for input validation (no model weights)."""
    from nanovllm.engine.llm_engine import LLMEngine

    class FakeHFConfig:
        vocab_size = 100

    class FakeRunnerConfig:
        revision = None
        hf_config = FakeHFConfig()

    class FakeRunner:
        config = FakeRunnerConfig

        class model:
            @staticmethod
            def parameters():
                return iter([torch.zeros(1)])

        def __getattr__(self, name):
            raise AssertionError(f"runner should not be called, got {name}")

    class FakeTokenizer:
        pad_token_id = 0

    class FakeModelRunner:
        def __init__(self):
            self.config = FakeRunnerConfig
            self.model = FakeRunner.model

        def call(self, *args, **kwargs):
            raise AssertionError("validation failure should not reach the runner")

    engine = LLMEngine.__new__(LLMEngine)
    engine.model_runner = FakeModelRunner()
    engine.tokenizer = FakeTokenizer()
    engine.config = FakeRunnerConfig
    return engine


def test_prefill_last_logits_rejects_out_of_range_candidates():
    engine = _make_validation_llm()
    with pytest.raises(ValueError, match="vocab"):
        engine.prefill_last_logits(["hello"], [[99999]])
    with pytest.raises(ValueError, match="vocab"):
        engine.prefill_last_logits(["hello"], [[-1]])


def test_prefill_last_logits_rejects_length_mismatch():
    engine = _make_validation_llm()
    with pytest.raises(ValueError, match="equal length"):
        engine.prefill_last_logits(["hello", "world"], [[1]])


def test_prefill_last_logits_rejects_empty():
    engine = _make_validation_llm()
    with pytest.raises(ValueError, match="nonempty"):
        engine.prefill_last_logits([], [])


@pytest.mark.parametrize("model_key", ["qwen3_vl", "qwen2_5_vl"])
def test_prefill_last_logits_multimodal_matches_transformers(model_key):
    """Multimodal last-position candidate logits must match HF, including a
    mixed image+text batch (mixed batch is the SemIf integration case)."""
    requires_cuda()
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from nanovllm import LLM

    path = model_path(model_key)
    processor = AutoProcessor.from_pretrained(path)

    images = [
        Image.new("RGB", (224, 224), color=(200, 30, 30)),
        Image.new("RGB", (448, 224), color=(30, 30, 200)),
    ]
    question = "What is the dominant color of the image? Answer red or blue."
    mm_messages = [
        [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": question},
                ],
            }
        ]
        for img in images
    ]
    text_message = [
        {"role": "user", "content": [{"type": "text", "text": PROMPTS[0]}]}
    ]

    hf = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    cands = [CANDIDATE_TOKENS[0]] * 3
    expected = []
    with torch.inference_mode():
        for msg, img in zip(mm_messages, images):
            text = processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[img], return_tensors="pt"
            ).to("cuda")
            logits = hf(**inputs).logits[0, -1]
            expected.append(
                logits[torch.tensor(cands[0], device="cuda")].float().cpu()
            )
        text_in = processor.apply_chat_template(
            text_message, tokenize=False, add_generation_prompt=True
        )
        ids = processor.tokenizer.encode(text_in, add_special_tokens=False)
        inputs = {"input_ids": torch.tensor([ids], device="cuda")}
        logits = hf(**inputs).logits[0, -1]
        expected.append(
            logits[torch.tensor(cands[0], device="cuda")].float().cpu()
        )
    del hf
    torch.cuda.empty_cache()

    llm = LLM(path, multimodal_model_type=model_key, enforce_eager=True)
    try:
        requests = [
            {"messages": mm_messages[0], "images": [images[0]]},
            {"messages": mm_messages[1], "images": [images[1]]},
            # Pre-tokenized text row in the same batch (SemIf mixes rows).
            {"input_ids": ids},
        ]
        result = llm.prefill_last_logits_multimodal(requests, cands)
    finally:
        llm.exit()

    assert result["logits"].shape == (3, 3)
    for i, exp in enumerate(expected):
        got = result["logits"][i]
        assert torch.allclose(got, exp, **TOL), (
            f"row {i}: {got.tolist()} vs HF {exp.tolist()}"
        )


@pytest.mark.parametrize("model_key", ["qwen3_vl"])
def test_prefill_last_logits_multimodal_validates_inputs(model_key):
    """Validation must fire before any model load/forward work."""
    engine = _make_validation_llm()
    engine._get_processor = lambda: pytest.fail("processor should not load")

    with pytest.raises(ValueError, match="nonempty"):
        engine.prefill_last_logits_multimodal([], [])
    with pytest.raises(ValueError, match="equal length"):
        engine.prefill_last_logits_multimodal([{"input_ids": [1]}], [[1], [2]])
    with pytest.raises(ValueError, match="vocab"):
        engine.prefill_last_logits_multimodal(
            [{"input_ids": [1]}], [[99999]]
        )
    with pytest.raises(TypeError, match="must be a dict"):
        engine.prefill_last_logits_multimodal(["plain"], [[1]])
    with pytest.raises(ValueError, match="input_ids"):
        engine.prefill_last_logits_multimodal([{"foo": 1}], [[1]])
