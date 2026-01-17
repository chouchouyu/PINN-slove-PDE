"""
GPU加速的期权定价深度学习模型
支持CUDA、MPS和CPU多设备
优化了GPU内存管理和计算效率
修复了属性初始化错误
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
import time
from abc import ABC, abstractmethod
import os
import warnings
import sys
import math
import random
import gc

# 设置随机种子以保证可重复性


def set_seed(seed=42):
    """设置随机种子"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


# 抑制警告但不忽略错误
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


# 打印环境信息
print(f"Python版本: {sys.version}")
print(f"PyTorch版本: {torch.__version__}")


# 设备检测和设置 - GPU优化版
def setup_device():
    """自动检测并设置最佳计算设备（优化GPU兼容性）"""

    # 首先检查PyTorch版本
    torch_version = torch.__version__
    print(f"检测到PyTorch版本: {torch_version}")

    # 检测CUDA - 优先使用GPU
    cuda_available = False
    cuda_error = None

    try:
        if torch.cuda.is_available():
            cuda_available = True
            device_count = torch.cuda.device_count()
            print(f"检测到 {device_count} 个CUDA设备")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {device_name} ({device_memory:.2f} GB)")

            # 选择设备
            if device_count > 0:
                device = torch.device("cuda:0")
                print(f"✓ 使用CUDA设备: {device} - {torch.cuda.get_device_name(0)}")

                # CUDA特定配置
                torch.backends.cudnn.deterministic = True

                # 清空GPU缓存
                torch.cuda.empty_cache()

                return device, "cuda"
    except Exception as e:
        cuda_error = str(e)
        cuda_available = False

    if cuda_available and cuda_error:
        print(f"⚠ CUDA检测到但初始化失败: {cuda_error}")

    # 检测MPS (苹果芯片)
    mps_available = False
    mps_error = None

    try:
        if hasattr(torch.backends, "mps") and hasattr(
            torch.backends.mps, "is_available"
        ):
            mps_available = torch.backends.mps.is_available()
            if mps_available:
                print("MPS后端可用，尝试初始化...")

                device = torch.device("mps")
                print(f"✓ MPS设备初始化成功: {device}")

                # MPS特定配置
                # torch.set_default_dtype(torch.float32)

                # 设置MPS内存管理
                # if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                #     torch.mps.set_per_process_memory_fraction(0.9)
                torch.mps.empty_cache()
                return device, "mps"
    except Exception as e:
        mps_error = str(e)
        mps_available = False

    if mps_available and mps_error:
        print(f"⚠ MPS检测到但初始化失败: {mps_error}")

    # 默认CPU
    device = torch.device("cpu")
    print(f"✓ 使用CPU设备: {device}")
    return device, "cpu"


# 设置设备
# device, device_type = setup_device()
# print(f"当前使用设备类型: {device_type}")


def figsize(scale, nplots=1):
    """计算图表尺寸，使用黄金比例"""
    fig_width_pt = 438.17227
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = nplots * fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    return fig_size
