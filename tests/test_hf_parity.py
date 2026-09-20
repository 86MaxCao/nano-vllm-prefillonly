"""HF parity tests: compare nano-vllm-prefillonly against Transformers.

These are the regression net for weight loading, attention, RoPE, and pooling.
They need GPU + local model weights and are therefore marked `slow`.

Run with:
    pytest tests/test_hf_parity.py -m slow -v
"""
import pytest

torch = pytest.importorskip("torch")

from tests.conftest import model_path, requires_cuda

pytestmark = pytest.mark.slow


PROMPTS = [
    "Is the Earth round? Answer Yes or No.",
    "What is the capital of France?",
    "The quick brown fox jumps over the lazy",
]

TEXTS = [
    "What is deep learning?",
    "Explain the transformer architecture in one sentence.",
    "Retrieval augmented generation combines search with generation.",
]

PAIRS = [
    ("What is AI?", "Artificial intelligence simulates human intelligence."),
    ("What is Python?", "Python is a high-level programming language."),
    ("capital of France", "Bananas are a tropical fruit."),
]


def cosine(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a = a.float()
    b = b.float()
    return torch.nn.functional.cosine_similarity(a, b, dim=-1)


@pytest.mark.parametrize("model_key", ["qwen3"])
def test_generation_matches_transformers(model_key):
    """Single-token greedy generation must match HF token-for-token."""
    requires_cuda()
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from nanovllm import LLM, SamplingParams

    path = model_path(model_key)
    tok = AutoTokenizer.from_pretrained(path)

    hf = AutoModelForCausalLM.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    expected = []
    with torch.inference_mode():
        for prompt in PROMPTS:
            ids = tok(prompt, return_tensors="pt").to("cuda")
            logits = hf(**ids).logits[0, -1]
            expected.append(int(logits.argmax()))
    del hf
    torch.cuda.empty_cache()

    llm = LLM(path, enforce_eager=True)
    try:
        got = llm.generate_single_token(
            PROMPTS, SamplingParams(temperature=0.0, max_tokens=1)
        )
    finally:
        llm.exit()

    assert got == expected, f"token mismatch: {got} != {expected}"


@pytest.mark.parametrize(
    "model_key,embedding_type",
    [("qwen3_embedding", "qwen3"), ("gemma2_embedding", "gemma2")],
)
def test_embedding_matches_transformers(model_key, embedding_type):
    """Pooled + normalised embeddings must be near-identical to HF."""
    requires_cuda()
    from transformers import AutoModel, AutoTokenizer

    from nanovllm import LLM

    path = model_path(model_key)
    tok = AutoTokenizer.from_pretrained(path)
    tok.padding_side = "left"

    suffix = "<|endoftext|>" if embedding_type == "qwen3" else ""
    hf = AutoModel.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    with torch.inference_mode():
        batch = tok(
            [t + suffix for t in TEXTS], return_tensors="pt", padding=True
        ).to("cuda")
        out = hf(**batch).last_hidden_state
        expected = out[:, -1].float()
        expected = torch.nn.functional.normalize(expected, p=2, dim=-1)
    del hf
    torch.cuda.empty_cache()

    llm = LLM(path, is_embedding=True, embedding_type=embedding_type, enforce_eager=True)
    try:
        got = llm.embed_batch(TEXTS)
    finally:
        llm.exit()

    sims = cosine(got.cpu(), expected.cpu())
    assert sims.min() > 0.99, f"cosine similarity too low: {sims.tolist()}"


@pytest.mark.parametrize("model_key", ["qwen3_reranker"])
def test_reranker_ranks_consistently(model_key):
    """Relevant pairs must outrank the deliberately irrelevant one."""
    requires_cuda()
    from nanovllm import LLM

    path = model_path(model_key)
    llm = LLM(path, is_reranker=True, reranker_type="qwen3", enforce_eager=True)
    try:
        scores = llm.rerank_batch(PAIRS)
    finally:
        llm.exit()

    if isinstance(scores, tuple):
        scores = scores[0]
    scores = scores.float().cpu()
    assert scores.shape[0] == len(PAIRS)
    # The third pair is unrelated and must score lowest.
    assert scores[2] < scores[0] and scores[2] < scores[1], scores.tolist()


@pytest.mark.parametrize("model_key", ["qwen3_vl", "qwen2_5_vl"])
def test_multimodal_generation_matches_transformers(model_key):
    requires_cuda()
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from nanovllm import LLM, SamplingParams

    path = model_path(model_key)
    processor = AutoProcessor.from_pretrained(path)

    images = [
        Image.new("RGB", (224, 224), color=(200, 30, 30)),
        Image.new("RGB", (224, 224), color=(30, 30, 200)),
    ]
    question = "What is the dominant color? Answer in one word."
    messages = [
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

    hf = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    expected = []
    with torch.inference_mode():
        for msg in messages:
            text = processor.apply_chat_template(
                msg, tokenize=False, add_generation_prompt=True
            )
            inputs = processor(
                text=[text], images=[msg[0]["content"][0]["image"]], return_tensors="pt"
            ).to("cuda")
            logits = hf(**inputs).logits[0, -1]
            expected.append(int(logits.argmax()))
    del hf
    torch.cuda.empty_cache()

    llm = LLM(path, multimodal_model_type=model_key, enforce_eager=True)
    try:
        requests = [
            {"messages": msg, "images": [msg[0]["content"][0]["image"]]}
            for msg in messages
        ]
        results = llm.generate_multimodal(
            requests, SamplingParams(temperature=0.0, max_tokens=1), processor,
            use_tqdm=False,
        )
    finally:
        llm.exit()

    got = [r["token_ids"][0] for r in results]
    assert got == expected, f"token mismatch: {got} != {expected}"


@pytest.mark.parametrize("model_key", ["qwen3_vl"])
def test_mixed_image_and_text_generation_batch(model_key):
    """A text-only request in the same batch must not disturb the image request.

    Regression test: `prepare_prefill_only_inputs` used to drop pixel_values
    whenever any request in the batch had no image, so the image request was
    answered as if it were text-only. Purple requires vision to answer.
    """
    requires_cuda()
    from PIL import Image
    from transformers import AutoProcessor

    from nanovllm import LLM, SamplingParams

    path = model_path(model_key)
    processor = AutoProcessor.from_pretrained(path)

    img = Image.new("RGB", (224, 224), color=(128, 0, 128))
    question = "What is the dominant color? Answer in one word."

    def image_request():
        return {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
                        {"type": "text", "text": question},
                    ],
                }
            ],
            "images": [img],
        }

    text_request = {
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "Hello!"}]}
        ],
    }

    llm = LLM(path, multimodal_model_type=model_key, enforce_eager=True)
    try:
        params = SamplingParams(temperature=0.0, max_tokens=1)
        alone = llm.generate_multimodal(
            [image_request()], params, processor, use_tqdm=False
        )
        together = llm.generate_multimodal(
            [image_request(), text_request], params, processor, use_tqdm=False
        )
    finally:
        llm.exit()

    assert len(together) == 2
    assert (
        together[0]["token_ids"] == alone[0]["token_ids"]
    ), f"image request changed when batched with text: {together[0]} vs {alone[0]}"
    # The text-only request must still produce a valid token.
    assert len(together[1]["token_ids"]) == 1


@pytest.mark.parametrize("model_key", ["qwen3_vl", "qwen2_5_vl"])
def test_multi_image_single_request_matches_transformers(model_key):
    """Two images in one message must land in their own placeholder ranges."""
    requires_cuda()
    from PIL import Image
    from transformers import AutoModelForImageTextToText, AutoProcessor

    from nanovllm import LLM, SamplingParams

    path = model_path(model_key)
    processor = AutoProcessor.from_pretrained(path)

    img_red = Image.new("RGB", (224, 224), color=(200, 30, 30))
    img_blue = Image.new("RGB", (224, 224), color=(30, 30, 200))
    question = "What color is the second image? Answer in one word."
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img_red},
                {"type": "image", "image": img_blue},
                {"type": "text", "text": question},
            ],
        }
    ]

    hf = AutoModelForImageTextToText.from_pretrained(
        path, dtype=torch.bfloat16, attn_implementation="flash_attention_2"
    ).cuda().eval()
    with torch.inference_mode():
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text], images=[img_red, img_blue], return_tensors="pt"
        ).to("cuda")
        expected = int(hf(**inputs).logits[0, -1].argmax())
    del hf
    torch.cuda.empty_cache()

    llm = LLM(path, multimodal_model_type=model_key, enforce_eager=True)
    try:
        results = llm.generate_multimodal(
            [{"messages": messages, "images": [img_red, img_blue]}],
            SamplingParams(temperature=0.0, max_tokens=1),
            processor,
            use_tqdm=False,
        )
    finally:
        llm.exit()

    got = results[0]["token_ids"][0]
    assert got == expected, f"token mismatch: {got} != {expected}"


@pytest.mark.parametrize("model_key", ["qwen3_vl_embedding"])
def test_multimodal_embedding_is_discriminative(model_key):
    """Same-colour images must embed closer to each other than to a different one."""
    requires_cuda()
    from PIL import Image

    from nanovllm import LLM

    path = model_path(model_key)
    images = [
        Image.new("RGB", (224, 224), color=(220, 20, 20)),
        Image.new("RGB", (224, 224), color=(210, 30, 30)),
        Image.new("RGB", (224, 224), color=(20, 20, 220)),
    ]
    llm = LLM(
        path,
        multimodal_model_type="qwen3_vl",
        is_embedding=True,
        embedding_type="qwen3_vl",
        enforce_eager=True,
    )
    try:
        embeds = llm.embed_batch(["Describe the image."] * 3, images=images)
    finally:
        llm.exit()

    embeds = embeds.float().cpu()
    same = cosine(embeds[0], embeds[1])
    diff = cosine(embeds[0], embeds[2])
    assert same > diff, f"same={same:.4f} diff={diff:.4f}"


@pytest.mark.parametrize("model_key", ["qwen3_vl_embedding"])
def test_mixed_image_and_text_embedding_batch(model_key):
    """A mixed batch must not disturb the image request's embedding."""
    requires_cuda()
    from PIL import Image

    from nanovllm import LLM

    path = model_path(model_key)
    llm = LLM(
        path,
        multimodal_model_type="qwen3_vl",
        is_embedding=True,
        embedding_type="qwen3_vl",
        enforce_eager=True,
    )
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


@pytest.mark.parametrize("model_key", ["qwen3_vl_embedding"])
def test_none_image_entry_is_text_only(model_key):
    """images=[img, None] must embed the second entry as plain text."""
    requires_cuda()
    from PIL import Image

    from nanovllm import LLM

    path = model_path(model_key)
    llm = LLM(
        path,
        multimodal_model_type="qwen3_vl",
        is_embedding=True,
        embedding_type="qwen3_vl",
        enforce_eager=True,
    )
    try:
        img = Image.new("RGB", (224, 224), color=(30, 30, 200))
        texts = ["Describe the image.", "plain text query"]

        got = llm.embed_batch(texts, images=[img, None]).float().cpu()
        # Reference goes through the same multimodal branch (chat template
        # applied); embed_batch without images would not.
        ref_text = llm.embed_batch([texts[1]], images=[None]).float().cpu()
    finally:
        llm.exit()

    sim = cosine(got[1], ref_text[0]).item()
    assert sim > 0.999, f"None-image entry drifted: {sim:.6f}"


def test_batch_invariance_for_embeddings():
    """Varlen packing must not make results depend on batch composition."""
    requires_cuda()
    from nanovllm import LLM

    path = model_path("qwen3_embedding")
    llm = LLM(path, is_embedding=True, embedding_type="qwen3", enforce_eager=True)
    try:
        # Deliberately mixed lengths, which exercises cu_seqlens boundaries.
        mixed = ["short", "a somewhat longer sentence here", "x" * 200]
        together = llm.embed_batch(mixed).float().cpu()
        alone = torch.cat([llm.embed_batch([t]).float().cpu() for t in mixed])
    finally:
        llm.exit()

    sims = cosine(together, alone)
    assert sims.min() > 0.999, f"batching changed results: {sims.tolist()}"


def test_kv_cache_is_not_allocated_for_single_token_generation():
    """The headline memory claim: no KV cache for prefill-only workloads.

    prefill_only_mode is auto-enabled by max_tokens_hint=1; without the hint
    the engine allocates a KV cache and the no-cache assertion would fail.
    """
    requires_cuda()
    from nanovllm import LLM, SamplingParams

    path = model_path("qwen3")
    llm = LLM(path, enforce_eager=True, max_tokens_hint=1)
    try:
        assert llm.model_runner.kv_cache is None
        assert llm.model_runner.config.num_kvcache_blocks == 0
        out = llm.generate_single_token(
            PROMPTS, SamplingParams(temperature=0.0, max_tokens=1)
        )
        assert len(out) == len(PROMPTS)
    finally:
        llm.exit()


def test_kv_cache_is_allocated_without_single_token_hint():
    """Without max_tokens_hint=1 the engine must keep full decode capability."""
    requires_cuda()
    from nanovllm import LLM

    path = model_path("qwen3")
    llm = LLM(path, enforce_eager=True)
    try:
        assert llm.model_runner.kv_cache is not None
        assert llm.model_runner.config.num_kvcache_blocks > 0
    finally:
        llm.exit()


def test_multi_token_generation_is_correct_or_refused():
    """Decoding is out of scope, but it must never return silently wrong tokens.

    The paged-decode kernel in some flash-attn builds yields NaN logits, which
    would sample into plausible-looking garbage; the engine raises instead.
    """
    requires_cuda()
    from nanovllm import LLM, SamplingParams

    path = model_path("qwen3")
    llm = LLM(path, enforce_eager=True)
    try:
        assert llm.model_runner.kv_cache is not None
        try:
            outs = llm.generate(
                ["Count to three:"],
                SamplingParams(temperature=0.0, max_tokens=8),
                use_tqdm=False,
            )
        except RuntimeError as exc:
            assert "prefill-only" in str(exc)
            return
    finally:
        llm.exit()

    token_ids = outs[0]["token_ids"]
    assert len(token_ids) > 1
    # A working kernel must not degenerate into a run of repeated ids.
    assert len(set(token_ids)) > 1, f"suspicious decode output: {token_ids}"
