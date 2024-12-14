import torch

def save_state_dict_to_txt(state_dict, output_file):
    """
    将模型的 state_dict 保存为文本文件。

    Args:
        state_dict (dict): 模型的 state_dict。
        output_file (str): 输出文件的路径。
    """
    with open(output_file, 'w') as f:
        for key, value in state_dict.items():
            # 如果 value 是 Tensor，将其展平为一维数组
            if isinstance(value, torch.Tensor):
                value = value.flatten().tolist()
            # 将 key 和展平后的 value 写入文件
            f.write(f"{key}: {value}\n")

# 示例用法
if __name__ == "__main__":
    # 加载模型的 checkpoint 文件
    checkpoint_path = "/home/node25_tmpdata/xcli/percepnet/c_aec/8.pt.tar"
    checkpoint = torch.load(checkpoint_path)
    
    # 获取 state_dict
    state_dict = checkpoint["model_state_dict"]  # 假设 checkpoint 中的 key 是 'model_state_dict'

    # 保存为文本文件
    output_file = "model_state_dict.txt"
    save_state_dict_to_txt(state_dict, output_file)

    print(f"State dict has been saved to {output_file}")
