import torch
# 检查MPS是否可用
print(f"MPS可用: {torch.backends.mps.is_available()}")
# 检查PyTorch是否已构建MPS支持
print(f"MPS已构建: {torch.backends.mps.is_built()}")

device = torch.device("mps")  # 将device指定为MPS
print(f"正在使用设备: {device}")