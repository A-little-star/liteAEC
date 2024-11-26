import torch
import torch.nn as nn

input_data = [
    1, 2, 3, 4,  5, 6, 7, 8,  9, 10, 11, 12,  13, 14, 15, 16,  # 通道 1
    1, 1, 1, 1,  1, 1, 1, 1,  1, 1, 1, 1,   1, 1, 1, 1,        # 通道 2
    0, 1, 0, 1,  0, 1, 0, 1,  0, 1, 0, 1,   0, 1, 0, 1         # 通道 3
]

# 转换为 PyTorch 张量并调整形状为 [batch_size, channels, height, width]
input_tensor = torch.tensor(input_data, dtype=torch.float32).view(1, 3, 4, 4)

# 定义卷积层
conv_layer = nn.Conv2d(in_channels=3, out_channels=2, kernel_size=3, stride=1, padding=1, bias=False)

# 初始化卷积核的权重为1，偏置项默认为0（不设置）
with torch.no_grad():
    conv_layer.weight.fill_(1.0)  # 将权重全部设置为1

# 执行卷积操作
output_tensor = conv_layer(input_tensor)

# 打印输入和输出张量的形状
print("Input shape:", input_tensor.shape)   # 输入形状: [1, 3, 4, 4]
print("Output shape:", output_tensor.shape) # 输出形状: [1, 3, 4, 4]

# 打印结果
print("Output tensor:")
print(output_tensor)
