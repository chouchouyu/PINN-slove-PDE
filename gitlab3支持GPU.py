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
    if hasattr(torch, 'mps') and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

set_seed(42)

# 抑制警告但不忽略错误
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', category=FutureWarning)

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
                torch.backends.cudnn.benchmark = True
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                
                # 清空GPU缓存
                torch.cuda.empty_cache()
                
                return device, 'cuda'
    except Exception as e:
        cuda_error = str(e)
        cuda_available = False
    
    if cuda_available and cuda_error:
        print(f"⚠ CUDA检测到但初始化失败: {cuda_error}")
    
    # 检测MPS (苹果芯片)
    mps_available = False
    mps_error = None
    
    try:
        if hasattr(torch.backends, 'mps') and hasattr(torch.backends.mps, 'is_available'):
            mps_available = torch.backends.mps.is_available()
            if mps_available:
                print("MPS后端可用，尝试初始化...")
                
                device = torch.device("mps")
                print(f"✓ MPS设备初始化成功: {device}")
                
                # MPS特定配置
                torch.set_default_dtype(torch.float32)
                
                # 设置MPS内存管理
                if hasattr(torch.mps, 'set_per_process_memory_fraction'):
                    torch.mps.set_per_process_memory_fraction(0.9)
                
                return device, 'mps'
    except Exception as e:
        mps_error = str(e)
        mps_available = False
    
    if mps_available and mps_error:
        print(f"⚠ MPS检测到但初始化失败: {mps_error}")
    
    # 默认CPU
    device = torch.device("cpu")
    print(f"✓ 使用CPU设备: {device}")
    return device, 'cpu'

# 设置设备
device, device_type = setup_device()
print(f"当前使用设备类型: {device_type}")

# # 简单的正态分布CDF函数，避免依赖scipy
# def norm_cdf(x):
#     """正态分布累积分布函数近似"""
#     return 0.5 * (1 + math.erf(x / math.sqrt(2)))

# # 向量化版本的正态分布CDF
# def norm_cdf_vec(x):
#     """向量化正态分布累积分布函数"""
#     if isinstance(x, (int, float)):
#         return norm_cdf(x)
#     elif isinstance(x, np.ndarray):
#         # 对数组中的每个元素应用norm_cdf
#         result = np.zeros_like(x, dtype=np.float64)
#         for i in range(x.size):
#             result.flat[i] = norm_cdf(x.flat[i])
#         return result
#     else:
#         raise TypeError(f"不支持的输入类型: {type(x)}")

# 优化版本的神经网络结构
class Sine(nn.Module):
    """正弦激活函数 - 优化GPU版本"""
    def __init__(self, w0=1.0):
        super(Sine, self).__init__()
        self.w0 = w0

    def forward(self, x):
        return torch.sin(self.w0 * x)

class OptimizedMLP(nn.Module):
    """优化的多层感知机 - GPU加速版本"""
    def __init__(self, layers, activation=None, device=None):
        super(OptimizedMLP, self).__init__()
        
        self.layers = layers
        self.device = device
        
        # 创建网络层
        self.net_layers = nn.ModuleList()
        
        # 构建网络
        for i in range(len(layers) - 1):
            self.net_layers.append(nn.Linear(layers[i], layers[i + 1]))
        
        self.activation = activation if activation is not None else nn.ReLU()
        
        # 初始化权重
        self._init_weights()
        
        # 移动到设备
        self.to(device)
    
    def _init_weights(self):
        """安全的权重初始化"""
        for layer in self.net_layers:
            if isinstance(layer, nn.Linear):
                # 使用Kaiming初始化
                nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    def forward(self, x):
        """前向传播 - 优化版本"""
        out = x
        
        for i, layer in enumerate(self.net_layers):
            out = layer(out)
            if i < len(self.net_layers) - 1:  # 除了最后一层都加激活函数
                out = self.activation(out)
        
        return out

# GPU优化的FBSNN基类
class GPUFBSNN(ABC):
    """GPU优化的Forward-Backward Stochastic Neural Network基类"""
    
    def __init__(self, Xi, T, M, N, D, layers, activation_name="Sine"):
        # 设备设置
        self.device = device
        self.device_type = device_type
        
        # 使用更稳定的张量创建方式
        self.Xi = torch.tensor(Xi, dtype=torch.float32, device=self.device, requires_grad=True)

        # 存储参数
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.strike = 1.0

        # 设置激活函数
        if activation_name == "Sine":
            self.activation = Sine(w0=1.0)
        elif activation_name == "ReLU":
            self.activation = nn.ReLU()
        elif activation_name == "Tanh":
            self.activation = nn.Tanh()
        else:
            self.activation = nn.ReLU()

        # 初始化神经网络
        self.model = OptimizedMLP(layers, activation=self.activation, device=self.device)

        # 训练记录
        self.training_loss = []
        self.iteration = []
        
        # 先初始化缓存清理属性
        self._last_cache_clear = 0
        
        # 应用设备优化
        self._apply_device_optimizations()
        
        print(f"模型初始化完成，使用{device_type}设备")
    
    def _apply_device_optimizations(self):
        """应用设备特定的优化"""
        if self.device_type == 'cuda':
            # CUDA优化
            # 启用混合精度训练
            self.use_amp = True
            self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)
        elif self.device_type == 'mps':
            # MPS优化
            torch.set_default_dtype(torch.float32)
            self.use_amp = False
        else:
            # CPU优化
            self.use_amp = False
    
    def _clear_device_cache_if_needed(self, iteration):
        """定期清理设备缓存"""
        if iteration - self._last_cache_clear >= 50:  # 每50次迭代清理一次
            if self.device_type == 'cuda':
                torch.cuda.empty_cache()
                gc.collect()
            elif self.device_type == 'mps':
                if hasattr(torch.mps, 'empty_cache'):
                    torch.mps.empty_cache()
            self._last_cache_clear = iteration
    
    def net_u(self, t, X):
        """神经网络前向传播 - GPU优化版本"""
        # 安全的张量连接
        input_tensor = torch.cat([t, X], dim=1)
        
        # 前向传播
        u = self.model(input_tensor)
        
        # 梯度计算
        if u.requires_grad and X.requires_grad:
            try:
                # 使用更稳定的梯度计算方式
                Du = torch.autograd.grad(
                    outputs=u,
                    inputs=X,
                    grad_outputs=torch.ones_like(u),
                    create_graph=True,
                    retain_graph=True
                )[0]
            except Exception as e:
                print(f"梯度计算警告: {e}")
                Du = torch.zeros_like(X, device=self.device)
        else:
            Du = torch.zeros_like(X, device=self.device)
        
        return u, Du
    
    @torch.no_grad()
    def fetch_minibatch(self):
        """生成小批量数据 - GPU优化版本"""
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        # 使用GPU优化的随机数生成
        dt = T / N
        
        # 生成时间步
        t = np.zeros((M, N + 1, 1), dtype=np.float32)
        for i in range(1, N + 1):
            t[:, i, :] = i * dt
        
        # 生成布朗运动增量
        dW = np.random.randn(M, N, D).astype(np.float32) * np.sqrt(dt)
        W = np.zeros((M, N + 1, D), dtype=np.float32)
        W[:, 1:, :] = np.cumsum(dW, axis=1)
        
        # 直接创建在目标设备上
        t_tensor = torch.from_numpy(t).float().to(self.device)
        W_tensor = torch.from_numpy(W).float().to(self.device)
        
        return t_tensor, W_tensor
    
    def loss_function(self, t, W, Xi, training=True):
        """计算损失函数 - GPU优化版本"""
        M = self.M
        D = self.D
        N = self.N
        
        if training:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=True)
        else:
            total_loss = torch.tensor(0.0, device=self.device, requires_grad=False)
        
        # 初始化
        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi.expand(M, D)
        
        Y0, Z0 = self.net_u(t0, X0)
        
        dt = self.T / N
        
        for n in range(N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            
            # 计算漂移和扩散
            mu = self.mu_tf(t0, X0, Y0, Z0)
            sigma = self.sigma_tf(t0, X0, Y0)
            
            # 布朗运动增量
            dW = W1 - W0
            
            # 更新X - 使用矩阵乘法优化
            sigma_dW = torch.bmm(sigma, dW.unsqueeze(-1)).squeeze(-1)
            X1 = X0 + mu * dt + sigma_dW
            
            # 更新Y的预测值
            phi = self.phi_tf(t0, X0, Y0, Z0)
            Y1_pred = Y0 + phi * dt + torch.sum(Z0 * sigma_dW, dim=1, keepdim=True)
            
            # 真实Y值
            Y1, Z1 = self.net_u(t1, X1)
            
            if training:
                # 路径损失
                path_loss = torch.mean((Y1 - Y1_pred) ** 2)
                total_loss = total_loss + path_loss
            else:
                path_loss = torch.mean((Y1 - Y1_pred) ** 2).detach()
                total_loss = total_loss + path_loss
            
            # 更新变量
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1
        
        # 终端条件损失
        g_X1 = self.g_tf(X1)
        Dg_X1 = self.Dg_tf(X1)
        
        if training:
            terminal_loss = torch.mean((Y1 - g_X1) ** 2)
            gradient_loss = torch.mean((Z1 - Dg_X1) ** 2)
            total_loss = total_loss + terminal_loss + gradient_loss
        else:
            terminal_loss = torch.mean((Y1 - g_X1) ** 2).detach()
            gradient_loss = torch.mean((Z1 - Dg_X1) ** 2).detach()
            total_loss = total_loss + terminal_loss + gradient_loss
        
        return total_loss, Y0, Y1
    
    def Dg_tf(self, X):
        """计算g函数的梯度"""
        X_clone = X.clone().detach().requires_grad_(True)
        g = self.g_tf(X_clone)
        
        if g.requires_grad:
            Dg = torch.autograd.grad(
                outputs=g,
                inputs=X_clone,
                grad_outputs=torch.ones_like(g),
                create_graph=True,
                retain_graph=True
            )[0]
        else:
            Dg = torch.zeros_like(X, device=self.device)
        
        return Dg
    
    def train(self, n_iter, learning_rate):
        """训练模型 - GPU优化版本"""
        self.model.train()
        
        # 优化器
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate
        )
        
        start_time = time.time()
        
        for it in range(n_iter):
            optimizer.zero_grad(set_to_none=True)  # 优化内存使用
            
            # 获取数据
            t_batch, W_batch = self.fetch_minibatch()
            
            # 混合精度训练
            if self.use_amp and self.device_type == 'cuda':
                with torch.cuda.amp.autocast():
                    loss, Y0_pred, Y1_pred = self.loss_function(t_batch, W_batch, self.Xi, training=True)
                
                # 缩放梯度并反向传播
                self.scaler.scale(loss).backward()
                
                # 梯度裁剪
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 优化器步骤
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # 计算损失
                loss, Y0_pred, Y1_pred = self.loss_function(t_batch, W_batch, self.Xi, training=True)
                
                # 反向传播
                loss.backward()
                
                # 梯度裁剪
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # 优化器步骤
                optimizer.step()
            
            # 定期清理设备缓存
            self._clear_device_cache_if_needed(it)
            
            # 打印进度
            if it % 50 == 0:
                elapsed = time.time() - start_time
                loss_value = loss.item()
                
                # 处理Y0_pred的形状
                if Y0_pred.dim() > 1 and Y0_pred.shape[1] == 1:
                    y0_value = Y0_pred[0, 0].item() if Y0_pred.shape[0] > 0 else 0.0
                elif Y0_pred.numel() > 1:
                    y0_value = Y0_pred.flatten()[0].item()
                else:
                    y0_value = Y0_pred.item()
                
                # GPU内存使用信息
                gpu_info = ""
                if self.device_type == 'cuda':
                    allocated = torch.cuda.memory_allocated() / 1e9
                    reserved = torch.cuda.memory_reserved() / 1e9
                    gpu_info = f", GPU内存: {allocated:.2f}/{reserved:.2f} GB"
                
                print(f'迭代: {it:5d}, 损失: {loss_value:.4e}, Y0: {y0_value:.4f}, '
                      f'时间: {elapsed:.1f}s{gpu_info}')
                
                start_time = time.time()
            
            if it % 50 == 0:
                self.training_loss.append(loss.item())
                self.iteration.append(it)
        
        print("训练完成!")
        return np.array([self.iteration, self.training_loss])
    
    def predict(self, Xi_star, t_star, W_star):
        """预测 - GPU优化版本"""
        self.model.eval()
        
        Xi_tensor = torch.tensor(Xi_star, dtype=torch.float32, device=self.device)
        
        with torch.no_grad():
            if self.use_amp and self.device_type == 'cuda':
                with torch.cuda.amp.autocast():
                    loss, Y0_pred, Y1_pred = self.loss_function(t_star, W_star, Xi_tensor, training=False)
            else:
                loss, Y0_pred, Y1_pred = self.loss_function(t_star, W_star, Xi_tensor, training=False)
        
        self.model.train()
        
        return Y0_pred, Y1_pred
    
    def save_model(self, filename):
        """保存模型"""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_loss': self.training_loss,
            'iteration': self.iteration,
            'device_type': self.device_type
        }, filename)
        print(f"模型已保存到: {filename}")
    
    def load_model(self, filename):
        """加载模型"""
        checkpoint = torch.load(filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_loss = checkpoint.get('training_loss', [])
        self.iteration = checkpoint.get('iteration', [])
        print(f"模型已从 {filename} 加载")
    
    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass
    
    @abstractmethod
    def g_tf(self, X):
        pass
    
    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        M = self.M
        D = self.D
        return torch.zeros([M, D], device=self.device)
    
    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = self.M
        D = self.D
        return torch.eye(D, device=self.device).unsqueeze(0).repeat(M, 1, 1)

# GPU优化的看涨期权实现
class GPUCallOption(GPUFBSNN):
    """GPU优化的看涨期权定价模型"""
    
    def __init__(self, Xi, T, M, N, D, layers, activation_name="Sine"):
        super().__init__(Xi, T, M, N, D, layers, activation_name)
        self.rate = 0.01
        self.volatility = 0.25
    
    def phi_tf(self, t, X, Y, Z):
        """漂移项"""
        return self.rate * Y
    
    def g_tf(self, X):
        """终端条件 - 看涨期权"""
        payoff = torch.sum(X, dim=1, keepdim=True) - self.strike
        return torch.maximum(payoff, torch.tensor(0.0, device=self.device))
    
    def mu_tf(self, t, X, Y, Z):
        """漂移系数"""
        return self.rate * X
    
    def sigma_tf(self, t, X, Y):
        """扩散系数"""
        M = self.M
        D = self.D
        # 创建对角矩阵
        sigma_matrix = torch.eye(D, device=self.device).unsqueeze(0).repeat(M, 1, 1)
        sigma_matrix = sigma_matrix * self.volatility
        return sigma_matrix

# 修复的Black-Scholes公式实现（支持数组输入）
def black_scholes_call_numpy(S, K, T, r, sigma):
    """使用numpy实现的Black-Scholes看涨期权定价，支持数组输入"""
    
    # 确保输入是numpy数组
    S = np.asarray(S, dtype=np.float64)
    T = np.asarray(T, dtype=np.float64)
    
    # 处理标量输入
    if S.ndim == 0 and T.ndim == 0:
        if T <= 0:
            return np.maximum(S - K, 0.0)
        
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        N_d1 = norm_cdf(d1)
        N_d2 = norm_cdf(d2)
        
        call_price = S * np.exp(-r * T) * N_d1 - K * np.exp(-r * T) * N_d2
        return call_price
    
    # 处理数组输入
    else:
        # 确保S和T形状相同
        if S.shape != T.shape:
            # 广播处理
            if S.ndim == 0:
                S = np.full_like(T, S)
            elif T.ndim == 0:
                T = np.full_like(S, T)
        
        # 初始化结果数组
        result = np.zeros_like(S, dtype=np.float64)
        
        # 计算每个元素
        for idx in np.ndindex(S.shape):
            s_val = S[idx]
            t_val = T[idx]
            
            if t_val <= 0:
                result[idx] = max(s_val - K, 0.0)
            else:
                d1 = (np.log(s_val / K) + (r + 0.5 * sigma ** 2) * t_val) / (sigma * np.sqrt(t_val))
                d2 = d1 - sigma * np.sqrt(t_val)
                
                N_d1 = norm_cdf(d1)
                N_d2 = norm_cdf(d2)
                
                result[idx] = s_val * np.exp(-r * t_val) * N_d1 - K * np.exp(-r * t_val) * N_d2
        
        return result

# 主测试函数
def test_option_pricing():
    """测试期权定价模型"""
    print("=" * 60)
    print("GPU加速期权定价模型测试")
    print("=" * 60)
    
    # 根据设备类型调整参数
    if device_type == 'cuda':
        M = 256  # GPU适合更大的批量
        N = 20
        layers = [2, 64, 128, 64, 1]  # 更深的网络
    elif device_type == 'mps':
        M = 128
        N = 10
        layers = [2, 32, 64, 32, 1]
    else:
        M = 64
        N = 10
        layers = [2, 32, 32, 1]
    
    D = 1
    T = 1.0
    
    Xi = np.array([[1.0]])
    
    print(f"设备类型: {device_type}")
    print(f"批量大小: {M}")
    print(f"时间步数: {N}")
    print(f"网络结构: {layers}")
    
    try:
        # 创建模型
        model = GPUCallOption(Xi, T, M, N, D, layers, "Sine")
        
        # 快速训练测试
        print("\n开始快速训练测试...")
        start_time = time.time()
        
        # 训练迭代
        result = model.train(n_iter=200, learning_rate=1e-3)
        
        train_time = time.time() - start_time
        print(f"训练完成，耗时: {train_time:.1f}秒")
        
        # 生成测试数据
        t_test, W_test = model.fetch_minibatch()
        
        # 预测
        Y0_pred, Y1_pred = model.predict(Xi, t_test, W_test)
        
        # 计算Black-Scholes基准
        S0 = 1.0
        K = 1.0
        r = 0.01
        sigma = 0.25
        
        bs_price = black_scholes_call_numpy(S0, K, T, r, sigma)
        
        # 处理Y0_pred的形状
        if Y0_pred.dim() > 1 and Y0_pred.shape[1] == 1:
            y0_value = Y0_pred[0, 0].item() if Y0_pred.shape[0] > 0 else 0.0
        elif Y0_pred.numel() > 1:
            y0_value = Y0_pred.flatten()[0].item()
        else:
            y0_value = Y0_pred.item()
        
        print(f"\n模型预测的初始价格: {y0_value:.6f}")
        print(f"Black-Scholes理论价格: {bs_price:.6f}")
        print(f"绝对误差: {abs(y0_value - bs_price):.6f}")
        
        # 绘制训练损失
        if len(model.training_loss) > 0:
            plt.figure(figsize=(10, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(model.iteration, model.training_loss)
            plt.xlabel('迭代次数')
            plt.ylabel('损失')
            plt.yscale('log')
            plt.title('训练损失')
            plt.grid(True, alpha=0.3)
            
            plt.subplot(1, 2, 2)
            # 绘制预测对比
            time_points = np.linspace(0, T, 20)
            bs_prices = black_scholes_call_numpy(S0, K, T - time_points, r, sigma)
            plt.plot(time_points, bs_prices, 'r-', label='Black-Scholes')
            plt.axhline(y=y0_value, color='b', linestyle='--', label='模型预测')
            plt.xlabel('时间')
            plt.ylabel('期权价格')
            plt.title(f'{device_type.upper()}设备价格对比')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(f'gpu_option_pricing_{device_type}.png', dpi=150, bbox_inches='tight')
            print(f"结果图表已保存为 'gpu_option_pricing_{device_type}.png'")
            plt.show()
        
        # 保存模型
        model.save_model(f'gpu_option_model_{device_type}.pth')
        
        return True
        
    except Exception as e:
        print(f"\n❌ 测试过程中发生错误: {e}")
        print("错误类型:", type(e).__name__)
        import traceback
        traceback.print_exc()
        
        # 提供调试建议
        print("\n调试建议:")
        print("1. 检查PyTorch版本: pip show torch")
        print("2. 确保PyTorch支持CUDA: torch.cuda.is_available()")
        print("3. 尝试使用CPU运行: device = torch.device('cpu')")
        print("4. 减少批量大小或网络复杂度")
        
        return False

# GPU内存检查函数
def check_gpu_memory():
    """检查GPU内存使用情况"""
    if device_type == 'cuda':
        print("\nGPU内存使用情况:")
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"  GPU {i}: {allocated:.2f} GB 已分配 / {reserved:.2f} GB 已保留 / {total:.2f} GB 总内存")
    elif device_type == 'mps':
        print("MPS设备 - 内存信息不可直接获取")
    else:
        print("CPU设备 - 无GPU内存信息")

# 主程序入口
if __name__ == "__main__":
    print("开始运行GPU加速期权定价模型测试...")
    
    # 检查GPU内存
    check_gpu_memory()
    
    # 运行测试
    success = test_option_pricing()
    
    if success:
        print("\n✅ 测试成功完成!")
    else:
        print("\n❌ 测试失败，请查看上面的错误信息")
    
    print("=" * 60)
    print("程序执行结束")
    print("=" * 60)
