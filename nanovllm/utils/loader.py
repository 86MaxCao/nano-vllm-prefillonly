import os
import re
from glob import glob
import torch
from safetensors import safe_open


def default_weight_loader(param, loaded_weight):
    try:
        if param.shape != loaded_weight.shape:
            raise ValueError(f"Shape mismatch: {param.shape} vs {loaded_weight.shape}")
        param.data.copy_(loaded_weight)
    except Exception as e:
        # 兼容一些特殊情况，有时候default loader可能接收多余参数，这里做个简单处理
        pass

    
def load_model(model, path, name_mapping=None):
    # 1. 获取模型的合并映射规则
    # 格式通常是: {'q_proj': ('qkv_proj', 'q'), 'up_proj': ('gate_up_proj', 'up'), ...}
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    
    model_params = dict(model.named_parameters())
    model_keys = set(model_params.keys())

    for file in glob(os.path.join(path, "*.safetensors")):
        print(f"Loading weights from {file}...")
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                target_name = weight_name
                if name_mapping is not None:
                    target_name = name_mapping(target_name)
                    if target_name is None: continue

                # 2. 初始化 shard_id
                shard_id = None
                
                # 视觉层通常不参与这种合并，先排除
                is_vision = "visual" in weight_name.lower() or "vision" in weight_name.lower()
                
                search_names = []

                if not is_vision:
                    # --- 核心修复逻辑开始 ---
                    # 检查当前权重名是否在 packed_modules_mapping 的 key 中
                    # 例如 weight_name 是 "model.layers.0.self_attn.q_proj.weight"
                    # 我们要找它是否包含 "q_proj"
                    
                    found_packing = False
                    for source_key, (target_key, s_id) in packed_modules_mapping.items():
                        # 使用简单的字符串包含判断，或者更严谨的正则
                        # 这里假设 source_key (如 'q_proj') 是 weight_name 的一部分
                        if source_key in weight_name:
                            # 构造合并后的名字：把 q_proj 换成 qkv_proj
                            packed_name = target_name.replace(source_key, target_key)
                            search_names.append(packed_name)
                            
                            # 【关键】记录 shard_id，比如 'q' 或 'up'
                            shard_id = s_id 
                            found_packing = True
                            # 找到了就跳出，防止匹配到多个
                            # (注意：如果 key 存在包含关系，如 'proj' 和 'q_proj'，需要小心顺序，通常 Qwen 的 key 区分度很高)
                            break
                    
                    if not found_packing:
                        # 没命中合并规则，尝试普通加载逻辑 (比如 output.weight)
                        search_names.append(target_name)
                    # --- 核心修复逻辑结束 ---
                else:
                    search_names.append(target_name)

                # 3. 寻找参数对象
                found_param_name = None
                for name in search_names:
                    if name in model_keys:
                        found_param_name = name
                        break
                    if f"model.{name}" in model_keys:
                        found_param_name = f"model.{name}"
                        break
                
                if found_param_name:
                    param = model.get_parameter(found_param_name)
                    weight_loader = getattr(param, "weight_loader", default_weight_loader)
                    
                    tensor = f.get_tensor(weight_name)
                    if tensor.dtype != param.dtype:
                        tensor = tensor.to(param.dtype)
                    
                    # 4. 调用 loader，传入 shard_id
                    try:
                        # 尝试传 shard_id (MergedColumnParallelLinear 需要这个)
                        if shard_id is not None:
                            weight_loader(param, tensor, shard_id)
                        else:
                            # 如果没有 shard_id，也尝试传一下，万一函数定义了默认值
                            # 或者有些 loader 不需要 shard_id
                            # 这里做一个鲁棒性处理：先试带参数的，不行试不带的
                            try:
                                weight_loader(param, tensor)
                            except TypeError:
                                # 如果报错说多了参数，说明是 default_loader，不需要 shard_id
                                # 如果报错说少了参数，那就是你需要 shard_id 但这里是 None (逻辑错误)
                                raise
                    except TypeError as e:
                        # 捕获具体的 "missing argument" 错误，重新抛出更清晰的信息
                        if "loaded_shard_id" in str(e):
                            raise TypeError(
                                f"Error loading {weight_name} into {found_param_name}: "
                                f"The layer requires a 'shard_id' (e.g. 'q', 'k', 'up'), but it was not found in packed_modules_mapping."
                            ) from e
                        else:
                            # 尝试带 shard_id 失败的情况 (比如 default_loader 不接受 shard_id)
                            # 如果前面 shard_id 不是 None，但 weight_loader 不接受它，这里兜底
                            if shard_id is not None:
                                weight_loader(param, tensor)
                            else:
                                raise e