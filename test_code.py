import torch
import json

# 自定义 JSON 编码器，将长列表按每行 50 个元素格式化
class CustomJSONEncoder(json.JSONEncoder):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def encode(self, obj):
        if isinstance(obj, list):
            # 将列表分割成每行 50 个元素
            lines = [obj[i:i + 50] for i in range(0, len(obj), 50)]
            # 将每行转换为字符串并用换行符连接
            formatted_lines = ",\n".join("[" + ", ".join(map(str, line)) + "]" for line in lines)
            return formatted_lines
        return super().encode(obj)

def export_model_weights(checkpoint_path, output_path):
    # 加载 checkpoint 文件
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    
    # 假设 checkpoint 包含 model.state_dict() 或模型直接定义
    model_state = checkpoint['model_state_dict'] if 'model_state_dict' in checkpoint else checkpoint
    
    # 将参数转换为可序列化格式
    params = {}
    for key, value in model_state.items():
        params[key] = value.numpy().flatten().tolist()  # 转为 Python 的列表
    
    # 将参数保存为 JSON 文件
    with open(output_path, "w") as json_file:
        json.dump(params, json_file, cls=CustomJSONEncoder,indent=4)  # indent指定缩进，4表示每层缩进4个空格

# 使用示例
export_model_weights("/home/node25_tmpdata/xcli/percepnet/c_aec/8.pt.tar", "model_weights.json")
