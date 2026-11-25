# HandsOnWorld modified snapshot; see UPSTREAM.md and backend_manifest.json.
import torch
from typing import Dict, Union, List
from diffsynth.core import load_state_dict

def extend_patch_embedding_channels(
    state_dict: Dict[str, torch.Tensor],
    original_in_dim: int = 48,
    extended_in_dim: int = 64,
    patch_embedding_key: str = "patch_embedding.weight",
    verify_output: bool = True
) -> Dict[str, torch.Tensor]:
    """
    扩展 patch_embedding 的输入通道数，新增通道初始化为 0。
    """
    new_state_dict = state_dict.copy()
    
    if patch_embedding_key in state_dict:
        original_weight = state_dict[patch_embedding_key]
        
        out_channels = original_weight.shape[0]
        kernel_size = original_weight.shape[2:]
        
        new_weight = torch.zeros(
            out_channels, 
            extended_in_dim, 
            *kernel_size,
            dtype=original_weight.dtype,
            device=original_weight.device
        )
        
        new_weight[:, :original_in_dim, ...] = original_weight
        
        # 验证输出一致性
        if verify_output:
            print("\n🔍 Verifying output consistency...")
            # 创建测试输入 (batch=1, channels=original_in_dim, D, H, W)
            test_input = torch.randn(1, original_in_dim, 4, 8, 8, 
                                    dtype=original_weight.dtype, 
                                    device=original_weight.device)
            
            # 使用原始权重计算输出
            with torch.no_grad():
                original_output = torch.nn.functional.conv3d(
                    test_input, 
                    original_weight,
                    bias=state_dict.get(patch_embedding_key.replace(".weight", ".bias"))
                )
            
            # 扩展输入（新增通道补零）
            extended_input = torch.randn(1, extended_in_dim, 4, 8, 8,
                                        dtype=original_weight.dtype,
                                        device=original_weight.device)
            extended_input[:, :original_in_dim, ...] = test_input
            
            # 使用新权重计算输出
            with torch.no_grad():
                extended_output = torch.nn.functional.conv3d(
                    extended_input,
                    new_weight,
                    bias=state_dict.get(patch_embedding_key.replace(".weight", ".bias"))
                )
            
            # 比较输出
            max_diff = torch.abs(original_output - extended_output).max().item()
            mean_diff = torch.abs(original_output - extended_output).mean().item()
            
            print(f"  Max difference: {max_diff:.2e}")
            print(f"  Mean difference: {mean_diff:.2e}")
            
            if max_diff < 1e-6:
                print("  ✅ Verification passed: outputs are identical!")
            else:
                print(f"  ⚠️  Warning: outputs differ (max diff: {max_diff:.2e})")
        
        new_state_dict[patch_embedding_key] = new_weight
        
        print(f"\n✓ Extended {patch_embedding_key} from shape {original_weight.shape} to {new_weight.shape}")
        print(f"  - Original channels [0-{original_in_dim-1}]: copied from pretrained weights")
        print(f"  - New channels [{original_in_dim}-{extended_in_dim-1}]: initialized to zeros")
    else:
        print(f"⚠ Warning: {patch_embedding_key} not found in state_dict")
    
    bias_key = patch_embedding_key.replace(".weight", ".bias")
    if bias_key in state_dict:
        new_state_dict[bias_key] = state_dict[bias_key].clone()
        print(f"✓ Copied {bias_key} (shape: {state_dict[bias_key].shape})")
    
    return new_state_dict


def load_extended_model(
    model,
    original_checkpoint_path: Union[str, List[str]],
    original_in_dim: int = 48,
    extended_in_dim: int = 64
):
    """
    加载原始检查点并扩展到新模型。
    """
    # import safetensors.torch
    
    # 支持多个分片文件
    state_dict = load_state_dict(original_checkpoint_path)
    # if isinstance(original_checkpoint_path, list):
    #     state_dict = {}
    #     for path in original_checkpoint_path:
    #         print(f"Loading {path}...")
    #         if path.endswith('.safetensors'):
    #             shard = safetensors.torch.load_file(path)
    #         else:
    #             shard = torch.load(path, map_location='cpu')
    #             if 'state_dict' in shard:
    #                 shard = shard['state_dict']
    #         state_dict.update(shard)
    # else:
    #     if original_checkpoint_path.endswith('.safetensors'):
    #         state_dict = safetensors.torch.load_file(original_checkpoint_path)
    #     else:
    #         state_dict = torch.load(original_checkpoint_path, map_location='cpu')
    #         if 'state_dict' in state_dict:
    #             state_dict = state_dict['state_dict']
    
    # 扩展 patch_embedding 通道
    extended_state_dict = extend_patch_embedding_channels(
        state_dict, 
        original_in_dim=original_in_dim,
        extended_in_dim=extended_in_dim
    )
    
    # 加载到模型
    missing_keys, unexpected_keys = model.load_state_dict(extended_state_dict, strict=False)
    
    if missing_keys:
        print(f"⚠ Missing keys: {missing_keys}")
    if unexpected_keys:
        print(f"⚠ Unexpected keys: {unexpected_keys}")
    
    print("✓ Model loaded successfully with extended input dimensions")
    
    return model
