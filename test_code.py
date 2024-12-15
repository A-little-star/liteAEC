import torch
import json

def export_model_weights(checkpoint_path, output_path):
    # 加载 checkpoint 文件
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 假设 checkpoint 包含 model.state_dict() 或模型直接定义
    model_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 将参数转换为可序列化格式
    params = {}
    for key, value in model_state.items():
        params[key] = value.numpy().tolist()  # 转为 Python 的列表
    
    # 将参数保存为 JSON 文件
    with open(output_path, "w") as f:
        json.dump(params, f)

# 使用示例
export_model_weights("/home/node25_tmpdata/xcli/percepnet/c_aec/8.pt.tar", "model_weights.json")
