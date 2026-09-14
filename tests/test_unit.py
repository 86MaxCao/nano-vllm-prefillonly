"""Unit tests that do not require a GPU or model weights.

These cover the pure-Python logic that regressed historically: weight loading
strictness, block-size propagation, sequence pickling, and config auto-detection.
"""
import pickle

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
        with pytest.raises(AssertionError):
            SamplingParams(temperature=-1.0)

    def test_rejects_non_positive_max_tokens(self):
        with pytest.raises(AssertionError):
            SamplingParams(max_tokens=0)

    def test_top_p_bounds(self):
        with pytest.raises(AssertionError):
            SamplingParams(top_p=1.5)


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
