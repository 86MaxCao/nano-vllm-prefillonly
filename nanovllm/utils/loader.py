"""Weight loading utilities.

Loading is deliberately strict: a checkpoint that does not fully populate the
model is a hard error. Silently tolerating mismatches leaves parameters holding
uninitialised memory, which surfaces later as unexplainable accuracy loss.
"""
import inspect
import logging
import os
from glob import glob

import torch
from safetensors import safe_open

logger = logging.getLogger(__name__)


class WeightLoadError(RuntimeError):
    """Raised when a checkpoint cannot be mapped onto the model."""


def default_weight_loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
    if param.shape != loaded_weight.shape:
        if param.numel() != loaded_weight.numel():
            raise WeightLoadError(
                f"Shape mismatch: parameter {tuple(param.shape)} vs "
                f"checkpoint {tuple(loaded_weight.shape)}"
            )
        loaded_weight = loaded_weight.view(param.shape)
    param.data.copy_(loaded_weight)


def sharded_weight_loader(shard_axis: int):
    """Shard `loaded_weight` along `shard_axis` by tensor-parallel rank.

    Used for parameters such as GatedDeltaNet's ``A_log`` and ``dt_bias`` that
    are stored unsharded in the checkpoint.
    """

    def loader(param: torch.nn.Parameter, loaded_weight: torch.Tensor):
        tp_rank = 0
        tp_size = 1
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank()
            tp_size = torch.distributed.get_world_size()
        shard_size = param.data.shape[shard_axis]
        full_size = loaded_weight.shape[shard_axis]
        if shard_size * tp_size != full_size:
            raise WeightLoadError(
                f"Cannot shard axis {shard_axis}: checkpoint size {full_size} is "
                f"not {tp_size} x parameter size {shard_size}"
            )
        loaded_shard = loaded_weight.narrow(shard_axis, tp_rank * shard_size, shard_size)
        param.data.copy_(loaded_shard)

    return loader


def _matches_component(weight_name: str, key: str) -> bool:
    """Match `key` against dot-delimited component boundaries of `weight_name`.

    Plain substring matching would let ``proj`` collide with ``q_proj``. Keys may
    span several components (``mlp.gate_proj``) and some models write them with
    leading dots to anchor a substring match, so normalise both sides to
    component lists before comparing.
    """
    parts = weight_name.split(".")
    key_parts = [p for p in key.split(".") if p]
    n = len(key_parts)
    if n == 0:
        return False
    return any(parts[i : i + n] == key_parts for i in range(len(parts) - n + 1))


def _resolve_packed(weight_name: str, target_name: str, packed_modules_mapping: dict):
    """Return ``(search_names, shard_id)`` for a possibly packed parameter."""
    for source_key, (target_key, shard_id) in packed_modules_mapping.items():
        if _matches_component(weight_name, source_key):
            return [target_name.replace(source_key, target_key)], shard_id
    return [target_name], None


def _tied_parameter_names(named_params: dict) -> dict:
    """Group parameter names by the storage they share.

    Weight tying (``lm_head`` reusing ``embed_tokens``) means one checkpoint
    tensor legitimately covers several named parameters.
    """
    by_storage: dict[int, list[str]] = {}
    for name, param in named_params.items():
        by_storage.setdefault(param.data_ptr(), []).append(name)
    return by_storage


def load_model(model: torch.nn.Module, path: str, name_mapping=None):
    """Load safetensors weights from `path` into `model`.

    Raises:
        WeightLoadError: if no checkpoint is found, a tensor cannot be copied,
            or any model parameter is left unpopulated.
    """
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})

    named_params = dict(model.named_parameters())
    model_keys = set(named_params.keys())
    loaded_names: set[str] = set()
    unmatched_checkpoint_keys: list[str] = []

    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise WeightLoadError(f"No .safetensors files found in {path}")

    for file in files:
        logger.info("Loading weights from %s", file)
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                target_name = weight_name
                if name_mapping is not None:
                    target_name = name_mapping(target_name)
                    if target_name is None:
                        continue

                is_vision = (
                    "visual" in weight_name.lower() or "vision" in weight_name.lower()
                )
                if is_vision:
                    search_names, shard_id = [target_name], None
                else:
                    search_names, shard_id = _resolve_packed(
                        weight_name, target_name, packed_modules_mapping
                    )

                found_param_name = None
                for name in search_names:
                    if name in model_keys:
                        found_param_name = name
                        break
                    if f"model.{name}" in model_keys:
                        found_param_name = f"model.{name}"
                        break

                if found_param_name is None:
                    unmatched_checkpoint_keys.append(weight_name)
                    continue

                param = named_params[found_param_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                tensor = f.get_tensor(weight_name)
                if tensor.dtype != param.dtype:
                    tensor = tensor.to(param.dtype)

                try:
                    if shard_id is not None:
                        weight_loader(param, tensor, shard_id)
                    else:
                        sig = inspect.signature(weight_loader)
                        takes_shard_id = (
                            len(sig.parameters) >= 3
                            and "loaded_shard_id" in sig.parameters
                        )
                        module = getattr(weight_loader, "__self__", None)
                        if takes_shard_id and module is not None and hasattr(
                            module, "output_sizes"
                        ):
                            # A single merged checkpoint tensor (gate_up_proj,
                            # in_proj_qkv): split it and load each shard.
                            offset = 0
                            for s_id, size in enumerate(module.output_sizes):
                                weight_loader(param, tensor.narrow(0, offset, size), s_id)
                                offset += size
                        elif takes_shard_id:
                            weight_loader(param, tensor, 0)
                        else:
                            weight_loader(param, tensor)
                except Exception as exc:
                    raise WeightLoadError(
                        f"Failed to load checkpoint tensor '{weight_name}' into "
                        f"parameter '{found_param_name}' "
                        f"(param {tuple(param.shape)}, tensor {tuple(tensor.shape)}): {exc}"
                    ) from exc

                loaded_names.add(found_param_name)

    _verify_coverage(named_params, model_keys, loaded_names)

    if unmatched_checkpoint_keys:
        preview = ", ".join(sorted(unmatched_checkpoint_keys)[:10])
        logger.warning(
            "%d checkpoint tensor(s) had no matching parameter and were ignored: %s%s",
            len(unmatched_checkpoint_keys),
            preview,
            "..." if len(unmatched_checkpoint_keys) > 10 else "",
        )


def _verify_coverage(named_params: dict, model_keys: set, loaded_names: set):
    """Fail if any parameter was never written to by the checkpoint."""
    missing = model_keys - loaded_names
    if not missing:
        return

    # Tied weights share storage with a parameter that was loaded.
    by_storage = _tied_parameter_names(named_params)
    still_missing = []
    for name in sorted(missing):
        siblings = by_storage.get(named_params[name].data_ptr(), [])
        if any(sibling in loaded_names for sibling in siblings):
            continue
        still_missing.append(name)

    if still_missing:
        preview = "\n  ".join(still_missing[:20])
        suffix = (
            f"\n  ... and {len(still_missing) - 20} more"
            if len(still_missing) > 20
            else ""
        )
        raise WeightLoadError(
            f"{len(still_missing)} parameter(s) were not initialised from the "
            f"checkpoint; the model would run on uninitialised memory:\n  "
            f"{preview}{suffix}"
        )
