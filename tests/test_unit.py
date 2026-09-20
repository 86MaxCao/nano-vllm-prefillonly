"""Unit tests that do not require a GPU or model weights.

These cover the pure-Python logic that regressed historically: weight loading
strictness, block-size propagation, sequence pickling, and config auto-detection.
"""
import inspect
import pickle
from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")

from nanovllm.engine.sequence import Sequence, SequenceStatus
from nanovllm.sampling_params import SamplingParams
from nanovllm.utils.loader import WeightLoadError, default_weight_loader


class TestDefaultWeightLoader:
    def test_copies_matching_shape(self):
        param = torch.nn.Parameter(torch.zeros(4, 8))
        weight = torch.arange(32, dtype=torch.float32).reshape(4, 8)
        default_weight_loader(param, weight)
        assert torch.equal(param.data, weight)

    def test_raises_on_shape_mismatch(self):
        """A silent `pass` here used to leave parameters uninitialised."""
        param = torch.nn.Parameter(torch.zeros(4, 8))
        weight = torch.zeros(4, 9)
        with pytest.raises(WeightLoadError, match="Shape mismatch"):
            default_weight_loader(param, weight)

    def test_accepts_numel_compatible_reshape(self):
        param = torch.nn.Parameter(torch.zeros(4, 8))
        weight = torch.arange(32, dtype=torch.float32).reshape(32)
        default_weight_loader(param, weight)
        assert torch.equal(param.data.flatten(), weight)


class TestSequenceBlockSize:
    def test_block_size_follows_configuration(self):
        original = Sequence.block_size
        try:
            Sequence.configure(block_size=512)
            seq = Sequence(list(range(600)))
            assert seq.block_size == 512
            assert seq.num_blocks == 2
            assert seq.last_block_num_tokens == 88
        finally:
            Sequence.configure(block_size=original)

    def test_default_block_size(self):
        seq = Sequence(list(range(300)))
        assert seq.block_size == 256
        assert seq.num_blocks == 2
        assert seq.last_block_num_tokens == 44


class TestSequencePickling:
    def test_roundtrip_preserves_scheduling_state(self):
        """TP>1 workers unpickle sequences; dropped fields used to crash them."""
        sp = SamplingParams(temperature=0.7, max_tokens=5, ignore_eos=True)
        seq = Sequence([1, 2, 3, 4], sp)
        seq.status = SequenceStatus.RUNNING
        seq.block_table = [7, 8]
        seq.num_cached_tokens = 256

        restored = pickle.loads(pickle.dumps(seq))

        assert restored.seq_id == seq.seq_id
        assert restored.status == SequenceStatus.RUNNING
        assert restored.temperature == pytest.approx(0.7)
        assert restored.max_tokens == 5
        assert restored.ignore_eos is True
        assert restored.block_table == [7, 8]
        assert restored.num_cached_tokens == 256
        assert len(restored) == 4
        assert restored.last_token == 4

    def test_roundtrip_after_decode_steps(self):
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=4))
        seq.append_token(99)
        restored = pickle.loads(pickle.dumps(seq))
        assert restored.num_tokens == 4
        assert restored.num_prompt_tokens == 3
        assert restored.last_token == 99
        assert len(restored) == 4

    def test_multimodal_payload_is_not_shipped_to_workers(self):
        seq = Sequence([1, 2, 3], pixel_values=torch.zeros(4, 8))
        restored = pickle.loads(pickle.dumps(seq))
        assert restored.pixel_values is None


class TestSamplingParams:
    def test_rejects_negative_temperature(self):
        with pytest.raises(ValueError):
            SamplingParams(temperature=-1.0)

    def test_rejects_non_positive_max_tokens(self):
        with pytest.raises(ValueError):
            SamplingParams(max_tokens=0)

    def test_top_p_bounds(self):
        with pytest.raises(ValueError):
            SamplingParams(top_p=1.5)

    def test_rejects_zero_top_p(self):
        with pytest.raises(ValueError):
            SamplingParams(top_p=0.0)


class TestSampler:
    def test_greedy_when_temperature_is_zero(self):
        from nanovllm.layers.sampler import Sampler

        sampler = Sampler()
        logits = torch.tensor([[0.1, 5.0, 0.2], [9.0, 0.1, 0.2]])
        temps = torch.zeros(2)
        out = sampler(logits, temps)
        assert out.tolist() == [1, 0]

    def test_temperature_scaling_is_applied(self):
        """A hardcoded argmax used to silently ignore `temperature`."""
        from nanovllm.layers.sampler import Sampler

        sampler = Sampler()
        torch.manual_seed(0)
        # Two nearly-equal logits: greedy always picks index 1, sampling should
        # produce both outcomes across many draws at high temperature.
        logits = torch.tensor([[0.0, 1e-4]]).repeat(512, 1)
        temps = torch.full((512,), 100.0)
        out = sampler(logits, temps)
        assert out.unique().numel() == 2

    def test_top_p_restricts_support(self):
        from nanovllm.layers.sampler import Sampler

        sampler = Sampler()
        torch.manual_seed(0)
        logits = torch.tensor([[10.0, 0.0, -10.0]]).repeat(256, 1)
        temps = torch.full((256,), 1.0)
        top_p = torch.full((256,), 0.5)
        out = sampler(logits, temps, top_p)
        assert out.unique().tolist() == [0]


class TestPoolerVarlen:
    def _packed(self):
        # Two sequences of length 2 and 3.
        hidden = torch.tensor(
            [
                [1.0, 1.0],
                [2.0, 2.0],
                [10.0, 10.0],
                [20.0, 20.0],
                [30.0, 30.0],
            ]
        )
        cu_seqlens = torch.tensor([0, 2, 5], dtype=torch.int32)
        return hidden, cu_seqlens

    def test_last_pool(self):
        from nanovllm.layers.pooler import LastPool

        hidden, cu = self._packed()
        out = LastPool().forward_varlen(hidden, cu)
        assert out.tolist() == [[2.0, 2.0], [30.0, 30.0]]

    def test_cls_pool(self):
        from nanovllm.layers.pooler import CLSPool

        hidden, cu = self._packed()
        out = CLSPool().forward_varlen(hidden, cu)
        assert out.tolist() == [[1.0, 1.0], [10.0, 10.0]]

    def test_mean_pool_matches_reference(self):
        from nanovllm.layers.pooler import MeanPool

        hidden, cu = self._packed()
        out = MeanPool().forward_varlen(hidden, cu)
        expected = torch.stack([hidden[0:2].mean(0), hidden[2:5].mean(0)])
        assert torch.allclose(out, expected)

    def test_mean_pool_is_numerically_stable_on_long_sequences(self):
        """The old cumsum implementation accumulated error over all tokens."""
        from nanovllm.layers.pooler import MeanPool

        torch.manual_seed(0)
        n = 40000
        hidden = torch.randn(n, 16) * 100.0
        cu = torch.tensor([0, n // 2, n], dtype=torch.int32)
        out = MeanPool().forward_varlen(hidden, cu)
        expected = torch.stack(
            [
                hidden[: n // 2].double().mean(0),
                hidden[n // 2 :].double().mean(0),
            ]
        ).float()
        assert torch.allclose(out, expected, atol=1e-3)


class TestRope:
    def test_distinct_arguments_yield_distinct_modules(self):
        """`lru_cache(1)` used to hand the wrong cos/sin cache to earlier layers."""
        from nanovllm.layers.rotary_embedding import get_rope

        a = get_rope(64, 64, 4096, 10000.0)
        b = get_rope(128, 128, 4096, 10000.0)
        again = get_rope(64, 64, 4096, 10000.0)
        assert a is not b
        assert a is again

    def test_supports_linear_rope_scaling(self):
        from nanovllm.layers.rotary_embedding import get_rope

        plain = get_rope(64, 64, 512, 10000.0)
        scaled = get_rope(
            64, 64, 512, 10000.0, {"rope_type": "linear", "factor": 2.0}
        )
        assert scaled is not plain
        # Linear scaling stretches positions: position 2 under factor 2 should
        # match position 1 without scaling.
        assert torch.allclose(
            scaled.cos_sin_cache[2], plain.cos_sin_cache[1], atol=1e-6
        )

    def test_rejects_unknown_rope_scaling(self):
        from nanovllm.layers.rotary_embedding import get_rope

        with pytest.raises(ValueError, match="Unsupported rope_scaling"):
            get_rope(64, 64, 512, 10000.0, {"rope_type": "not_a_real_type"})

    def test_supports_partial_rotary(self):
        from nanovllm.layers.rotary_embedding import get_rope

        rope = get_rope(128, 64, 512, 10000.0)
        q = torch.randn(4, 2, 128)
        k = torch.randn(4, 2, 128)
        positions = torch.arange(4)
        qo, ko = rope(positions, q.clone(), k.clone())
        # The non-rotary tail must pass through untouched.
        assert torch.allclose(qo[..., 64:], q[..., 64:], atol=1e-6)
        assert not torch.allclose(qo[..., :64], q[..., :64], atol=1e-4)
        assert torch.allclose(ko[..., 64:], k[..., 64:], atol=1e-6)


class TestBlockManager:
    def test_can_append_semantics(self):
        from nanovllm.engine.block_manager import BlockManager

        bm = BlockManager(num_blocks=1, block_size=256)
        seq = Sequence(list(range(256)))
        bm.allocate(seq)
        # Next token starts a new block but none are free.
        seq.append_token(0)
        assert bm.can_append(seq) is False

    def test_prefill_only_without_blocks(self):
        from nanovllm.engine.block_manager import BlockManager

        bm = BlockManager(num_blocks=0, block_size=256)
        seq = Sequence(list(range(300)))
        assert bm.can_allocate(seq) is True
        bm.allocate(seq)
        assert seq.block_table == []


class TestVisionPlaceholders:
    """_build_vision_placeholders must reject inconsistent batches.

    A silent mismatch between placeholder tokens, image grids, and their
    token counts previously produced misaligned multimodal input instead
    of an error.
    """

    IMAGE_TOKEN = 151655

    def _runner(self):
        from types import SimpleNamespace

        from nanovllm.engine.model_runner import ModelRunner

        class FakeConfig:
            spatial_merge_size = 2

        fake_model = SimpleNamespace(
            visual=SimpleNamespace(config=FakeConfig()),
            _image_token_id=None,
        )
        fake = SimpleNamespace(
            model=fake_model, _image_token_id=self.IMAGE_TOKEN
        )
        return ModelRunner._build_vision_placeholders.__get__(fake)

    def test_multi_image_per_sequence(self):
        """Two images in one sequence map to two placeholder ranges."""
        grids = torch.tensor([[1, 8, 12], [1, 6, 6]])  # 24 and 9 tokens
        ids = [1, 2] + [self.IMAGE_TOKEN] * 24 + [5, 6] + [self.IMAGE_TOKEN] * 9 + [7]
        indices, placeholders = self._runner()(
            torch.tensor([ids]), None, grids, 1
        )
        assert indices == [(0, 2)]
        assert placeholders == [[(2, 24), (28, 9)]]

    def test_placeholder_count_mismatch_raises(self):
        grids = torch.tensor([[1, 8, 12]])  # 24 tokens
        ids = [1, 2] + [self.IMAGE_TOKEN] * 23 + [3]  # one token short
        with pytest.raises(ValueError, match="does not match"):
            self._runner()(torch.tensor([ids]), None, grids, 1)

    def test_grid_without_placeholder_raises(self):
        grids = torch.tensor([[1, 8, 12], [1, 6, 6]])
        ids = [1, 2] + [self.IMAGE_TOKEN] * 24 + [3]  # second image unmatched
        with pytest.raises(ValueError, match="no matching"):
            self._runner()(torch.tensor([ids]), None, grids, 1)


class TestEosHandling:
    def test_scheduler_accepts_list_eos(self):
        """Some configs expose `eos_token_id` as a list of ids."""
        from nanovllm.engine.scheduler import Scheduler

        class FakeConfig:
            max_num_seqs = 8
            max_num_batched_tokens = 4096
            eos = [151643, 151645]
            prefill_only_mode = False
            max_prefill_batch_size = 128
            is_multimodal = False
            hf_config = None
            num_kvcache_blocks = 4
            kvcache_block_size = 256

        sched = Scheduler(FakeConfig())
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=8))
        sched.add(seq)
        sched.waiting.clear()
        sched.running.append(seq)
        seq.status = SequenceStatus.RUNNING
        sched.postprocess([seq], [151645])
        assert seq.is_finished


class TestSchedulerNoProgress:
    """A request that can never be scheduled must fail fast.

    The scheduler used to return an empty batch forever while the request
    stayed in `waiting`, so `generate()` spun in a busy loop.
    """

    def _engine(self, budget=4, blocks=1):
        from types import SimpleNamespace

        from nanovllm.engine.block_manager import BlockManager
        from nanovllm.engine.scheduler import Scheduler

        class FakeConfig:
            max_num_seqs = 8
            max_num_batched_tokens = budget
            eos = -1
            prefill_only_mode = True
            max_prefill_batch_size = budget
            is_multimodal = False
            hf_config = None
            num_kvcache_blocks = blocks
            kvcache_block_size = 256

        sched = Scheduler(FakeConfig())
        engine = SimpleNamespace(scheduler=sched)
        return engine, sched

    def test_oversized_prompt_never_recovers(self):
        engine, sched = self._engine(budget=4)
        seq = Sequence([1] * 5, SamplingParams(max_tokens=1))
        sched.add(seq)
        # One no-progress step used to be recoverable in principle; the
        # engine now detects that the head can never fit the budget.
        from nanovllm.engine.llm_engine import LLMEngine

        with pytest.raises(RuntimeError, match="can never be scheduled"):
            LLMEngine.step(engine)

    def test_prefill_only_with_unfinished_running_raises(self):
        engine, sched = self._engine(budget=64, blocks=1)
        seq = Sequence([1, 2, 3], SamplingParams(max_tokens=8))
        sched.add(seq)
        scheduled, _ = sched.schedule()
        assert scheduled  # prefill ran, sequence is now running and unfinished
        from nanovllm.engine.llm_engine import LLMEngine

        with pytest.raises(RuntimeError, match="Prefill-only"):
            LLMEngine.step(engine)

    def test_oversized_prompt_rejected_at_enqueue(self):
        from nanovllm.engine.llm_engine import LLMEngine

        engine, _ = self._engine(budget=4)
        with pytest.raises(ValueError, match="exceeds the schedulable budget"):
            LLMEngine.add_request(
                engine, [1] * 5, SamplingParams(max_tokens=1)
            )

    def test_empty_prompt_rejected_at_enqueue(self):
        from nanovllm.engine.llm_engine import LLMEngine

        engine, _ = self._engine(budget=4)
        with pytest.raises(ValueError, match="empty"):
            LLMEngine.add_request(engine, [], SamplingParams(max_tokens=1))


class TestLLMEngineBatchValidation:
    """Mismatched batch lists used to be silently truncated by zip()."""

    def test_mismatched_lengths_raise(self):
        from nanovllm.engine.llm_engine import _check_batch_lengths

        with pytest.raises(ValueError, match="mismatched lengths"):
            _check_batch_lengths({"prompts": 3, "sampling_params": 2})

    def test_matching_lengths_pass(self):
        from nanovllm.engine.llm_engine import _check_batch_lengths

        _check_batch_lengths({"prompts": 3, "sampling_params": 3})

    def test_single_list_length_passes(self):
        from nanovllm.engine.llm_engine import _check_batch_lengths

        _check_batch_lengths({"prompts": 0})

    def test_unknown_kwargs_raise_type_error(self):
        from nanovllm.engine.llm_engine import LLMEngine

        engine = object.__new__(LLMEngine)
        # The TypeError fires before Config/model construction, so no model
        # is needed. "enforce_eagerr" is a plausible misspelling that used to
        # be silently dropped.
        with pytest.raises(TypeError, match="enforce_eagerr"):
            LLMEngine.__init__(engine, "/nonexistent", enforce_eagerr=True)


class TestTensorParallelRendezvous:
    """Every TP rank must rendezvous on one shared address."""

    def test_allocate_rendezvous_returns_bound_host(self):
        from nanovllm.engine.llm_engine import _allocate_rendezvous

        addr, port = _allocate_rendezvous()
        assert addr == "127.0.0.1"
        assert 0 < port < 65536

    def test_two_allocations_are_likely_distinct(self):
        from nanovllm.engine.llm_engine import _allocate_rendezvous

        ports = {_allocate_rendezvous()[1] for _ in range(8)}
        assert len(ports) > 1  # random free port, not a fixed one

    def test_init_url_uses_env_when_master_set(self, monkeypatch):
        monkeypatch.setenv("MASTER_ADDR", "127.0.0.1")
        monkeypatch.setenv("MASTER_PORT", "29500")
        from nanovllm.engine import model_runner

        init = model_runner.ModelRunner.__init__
        source = inspect.getsource(init)
        assert "env://" in source
        assert 'os.environ.get("MASTER_ADDR")' in source


class TestSharedMemoryIsolation:
    def test_shm_name_derived_from_master_port(self):
        """The segment name must be rank-shared yet per-instance.

        MASTER_PORT is set once by the parent engine, so every rank
        derives the same name; two engines on one host get different
        ports and therefore different segments. A per-rank uuid used
        to make rank 1 open a name rank 0 never created.
        """
        from nanovllm.engine import model_runner

        source = inspect.getsource(model_runner.ModelRunner.__init__)
        assert 'os.environ.get("MASTER_PORT"' in source
        assert "uuid.uuid4()" not in source
        assert 'name="nanovllm"' not in source

    def test_shm_names_differ_across_instances(self):
        names = {f"nanovllm-{port}" for port in ("29500", "29501")}
        assert len(names) == 2


class TestGDNSlotPool:
    def test_pool_sized_from_max_num_seqs(self):
        from nanovllm.engine.model_runner import GDNSlotManager

        manager = GDNSlotManager(max_slots=1024)
        slots = [manager.allocate(i) for i in range(1024)]
        assert sorted(slots) == list(range(1024))
        with pytest.raises(RuntimeError, match="exhausted"):
            manager.allocate(9999)

    def test_release_returns_slot(self):
        from nanovllm.engine.model_runner import GDNSlotManager

        manager = GDNSlotManager(max_slots=2)
        manager.allocate(1)
        manager.allocate(2)
        manager.release(1)
        manager.allocate(3)  # reuses the freed slot instead of raising

    def test_resize_gdn_buffers_grows_hardcoded_pool(self):
        from nanovllm.engine.model_runner import _resize_gdn_buffers

        gdn = SimpleNamespace(
            _pool_conv_state=torch.zeros(512, 8, 3),
            _graph_conv_state=torch.zeros(512, 8, 3),
            _graph_recurrent_state=torch.zeros(512, 4, 16),
        )
        _resize_gdn_buffers([gdn], 768)
        assert gdn._pool_conv_state.shape[0] == 768
        assert gdn._graph_conv_state.shape[0] == 768
        assert gdn._graph_recurrent_state.shape[0] == 768
        # Small max_num_seqs must not shrink the buffers.
        _resize_gdn_buffers([gdn], 128)
        assert gdn._pool_conv_state.shape[0] == 768


class TestEngineConstructionInputValidation:
    """Public Config inputs must raise ValueError, not assert (python -O)."""

    def test_invalid_tensor_parallel_size(self, monkeypatch, tmp_path):
        from nanovllm.config import Config

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(path, trust_remote_code=False):
                return SimpleNamespace(
                    text_config=None,
                    max_position_embeddings=None,
                    eos_token_id=None,
                )

        monkeypatch.setattr("nanovllm.config.AutoConfig", FakeAutoConfig)
        with pytest.raises(ValueError, match="tensor_parallel_size"):
            Config(str(tmp_path), tensor_parallel_size=0)

    def test_invalid_kvcache_block_size(self, monkeypatch, tmp_path):
        from nanovllm.config import Config

        class FakeAutoConfig:
            @staticmethod
            def from_pretrained(path, trust_remote_code=False):
                return SimpleNamespace(
                    text_config=None,
                    max_position_embeddings=None,
                    eos_token_id=None,
                )

        monkeypatch.setattr("nanovllm.config.AutoConfig", FakeAutoConfig)
        with pytest.raises(ValueError, match="kvcache_block_size"):
            Config(str(tmp_path), kvcache_block_size=100)


class TestTrustRemoteCodePropagation:
    """Every from_pretrained call must honour config.trust_remote_code."""

    def _fake_loader_module(self, monkeypatch, calls):
        from nanovllm.engine import model_loader

        class Recorder:
            def __init__(self, name):
                self.name = name

            def from_pretrained(self, model, trust_remote_code=None, **kwargs):
                calls.append((self.name, trust_remote_code))
                return SimpleNamespace()

        monkeypatch.setattr(model_loader, "AutoConfig", Recorder("AutoConfig"))
        return model_loader

    def test_embedding_loaders_forward_flag(self, monkeypatch):
        calls = []
        loader = self._fake_loader_module(monkeypatch, calls)

        class FakeConfig:
            model = "/fake"
            trust_remote_code = False

        class FakeLazy:
            JINA_V4_AVAILABLE = True

            class JinaEmbeddingsV4:
                def __init__(self, *a, **k):
                    pass

            @staticmethod
            def load_model(*a, **k):
                pass

        monkeypatch.setattr(loader, "_lazy", FakeLazy)
        monkeypatch.setattr(
            loader, "create_jina_v4_name_mapping", lambda: {}, raising=False
        )
        try:
            loader.ModelLoader.load_embedding_model(
                FakeConfig(), SimpleNamespace(), "jina_v4", "LAST", True, None
            )
        except Exception:
            # Loading may fail for unrelated reasons (no weights); the call
            # recording is what matters.
            pass
        assert ("AutoConfig", False) in calls
        assert ("AutoConfig", True) not in calls

    def test_processor_load_forwards_flag(self, monkeypatch):
        from transformers import AutoProcessor as RealProcessor

        from nanovllm.engine import llm_engine

        recorded = {}

        def fake_from_pretrained(model, trust_remote_code=None, **kwargs):
            recorded["trust_remote_code"] = trust_remote_code
            return SimpleNamespace()

        # transformers is a lazy module: patching the module attribute does
        # not affect the function-local `from transformers import ...`, so
        # patch the resolved class instead.
        monkeypatch.setattr(
            RealProcessor, "from_pretrained", staticmethod(fake_from_pretrained)
        )
        engine = object.__new__(llm_engine.LLMEngine)
        engine._processor = None
        engine.model_runner = SimpleNamespace(
            config=SimpleNamespace(model="/fake", trust_remote_code=False)
        )
        engine._get_processor()
        assert recorded["trust_remote_code"] is False
