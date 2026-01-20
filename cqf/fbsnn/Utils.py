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


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


def setup_device():
    torch_version = torch.__version__
    print(f"Detected PyTorch version: {torch_version}")
    cuda_available = False
    cuda_error = None

    try:
        if torch.cuda.is_available():
            cuda_available = True
            device_count = torch.cuda.device_count()
            print(f"Detected {device_count} CUDA devices")

            for i in range(device_count):
                device_name = torch.cuda.get_device_name(i)
                device_memory = torch.cuda.get_device_properties(i).total_memory / 1e9
                print(f"  GPU {i}: {device_name} ({device_memory:.2f} GB)")

            if device_count > 0:
                device = torch.device("cuda:0")
                print(
                    f"✓ Using CUDA device: {device} - {torch.cuda.get_device_name(0)}"
                )
                torch.backends.cudnn.deterministic = True
                torch.cuda.empty_cache()

                return device, "cuda"
    except Exception as e:
        cuda_error = str(e)
        cuda_available = False

    if cuda_available and cuda_error:
        print(f"⚠ CUDA detected but initialization failed: {cuda_error}")

    mps_available = False
    mps_error = None

    try:
        if hasattr(torch.backends, "mps") and hasattr(
            torch.backends.mps, "is_available"
        ):
            mps_available = torch.backends.mps.is_available()
            if mps_available:
                print("MPS backend available, attempting initialization...")
                device = torch.device("mps")
                print(f"✓ MPS device initialization successful: {device}")
                torch.mps.empty_cache()
                return device, "mps"
    except Exception as e:
        mps_error = str(e)
        mps_available = False

    if mps_available and mps_error:
        print(f"⚠ MPS detected but initialization failed: {mps_error}")

    device = torch.device("cpu")
    print(f"✓ Using CPU device: {device}")
    return device, "cpu"


def figsize(scale, nplots=1):
    fig_width_pt = 438.17227
    inches_per_pt = 1.0 / 72.27
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0
    fig_width = fig_width_pt * inches_per_pt * scale
    fig_height = nplots * fig_width * golden_mean
    fig_size = [fig_width, fig_height]
    return fig_size
