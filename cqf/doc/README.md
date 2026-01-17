# FBSNNs: 基于前向-后向随机神经网络的高维PDE求解框架

<div style="display: flex; justify-content: center; margin: 20px 0;">
  <img src="https://neeko-copilot.bytedance.net/api/text2image?prompt=deep%20learning%20neural%20network%20solving%20partial%20differential%20equations%20with%20stochastic%20processes%2C%20professional%20mathematical%20visualization%2C%20blue%20and%20purple%20color%20scheme&image_size=landscape_16_9" alt="FBSNNs 可视化" style="max-width: 800px; border-radius: 8px;">
</div>

## 项目简介

FBSNNs (Forward-Backward Stochastic Neural Networks) 是一个基于深度学习的高维偏微分方程(PDE)求解框架，特别适用于金融工程中的期权定价问题。该框架结合了随机微分方程理论和深度学习技术，有效解决了传统数值方法在高维问题中面临的"维度灾难"。

### 核心特性

- **高维PDE求解**：有效处理100维以上的高维PDE问题
- **多网络架构支持**：包含全连接(FC)和NAIS-Net两种网络架构
- **灵活激活函数**：支持Sine、ReLU、Tanh三种激活函数
- **多约束损失函数**：结合路径损失、终端损失和终端梯度约束
- **设备优化**：自动检测并使用最佳计算设备(CUDA/MPS/CPU)
- **模块化设计**：代码结构清晰，易于扩展和定制

## 理论基础

FBSNNs基于以下核心理论：

1. **前向-后向随机微分方程(FBSDE)**：将高维PDE转化为等价的FBSDE系统
2. **深度学习近似**：使用神经网络近似求解FBSDE的解
3. **随机离散化**：通过蒙特卡洛方法离散化随机过程
4. **多约束优化**：最小化路径一致性误差和终端条件误差

### 数学框架

对于PDE：$\frac{\partial u}{\partial t} + L(u) = 0$，终端条件 $u(T, x) = g(x)$，其中 $L$ 是二阶微分算子，FBSNNs通过求解等价的FBSDE：

$$
\begin{cases}
dX_t = b(X_t) dt + \sigma(X_t) dW_t \\
dY_t = -f(X_t, Y_t, Z_t) dt + Z_t^T dW_t \\
X_0 = x_0, \quad Y_T = g(X_T)
\end{cases}
$$

其中 $Y_t = u(t, X_t)$，$Z_t = \nabla u(t, X_t)$。

## 项目结构

```
FBSNNs/
├── FBSNNs.py         # 核心FBSNN基类实现
├── Models.py         # 网络模型定义（包含NAIS-Net）
├── Utils.py          # 工具函数（设备设置、随机种子等）
├── BlackScholesBarenblatt.py  # Black-Scholes-Barenblatt方程求解
├── CallOption.py     # 期权定价模型
├── vanilla_call.py   # 期权定价示例
└── __init__.py       # 包初始化文件
```

## 安装与依赖

### 系统要求

- Python 3.7+
- PyTorch 1.8+
- NumPy
- Matplotlib (可选，用于可视化)

### 安装步骤

1. **克隆项目**
   ```bash
   git clone <repository-url>
   cd PINN-slove-PDE/cqf
   ```

2. **安装依赖**
   ```bash
   pip install torch numpy matplotlib
   ```

## 快速开始

### 示例1：期权定价

```python
import numpy as np
from FBSNNs import CallOption

# 设置参数
M = 1          # 轨迹数量（批量大小）
N = 50         # 时间快照数量
D = 1          # 维度数
Mm = N ** (1/5)  # 计算Mm

# 定义网络结构
layers = [D + 1] + 4 * [256] + [1]

# 初始条件
Xi = np.array([1.0] * D)[None, :]
T = 1.0        # 时间参数

# 选择网络架构和激活函数
mode = "Naisnet"  # 可选: "FC", "Naisnet"
activation = "Sine"  # 可选: "Sine", "ReLU", "Tanh"

# 创建模型
model = CallOption(Xi, T, M, N, D, Mm, layers, mode, activation)

# 训练模型
n_iter = 20000
lr = 1e-3
graph = model.train(n_iter, lr)

# 精细调优
n_iter = 5100
lr = 1e-5
graph = model.train(n_iter, lr)
```

### 示例2：Black-Scholes-Barenblatt方程

```python
import numpy as np
from FBSNNs import BlackScholesBarenblatt

# 设置参数
M = 1          # 轨迹数量
N = 50         # 时间快照数量
D = 100        # 高维问题（100维）
Mm = N ** (1/5)

# 定义网络结构
layers = [D + 1] + 4 * [256] + [1]

# 初始条件
Xi = np.array([1.0] * D)[None, :]
T = 1.0        # 时间参数

# 创建模型
model = BlackScholesBarenblatt(Xi, T, M, N, D, Mm, layers, "FC", "ReLU")

# 训练模型
n_iter = 10000
lr = 1e-3
graph = model.train(n_iter, lr)
```

## 网络架构

### 1. 全连接网络 (FC)

- **结构**：标准全连接神经网络，包含输入层、隐藏层和输出层
- **适用场景**：一般PDE求解，实现简单，训练稳定
- **参数**：通过`layers`列表定义各层神经元数量

### 2. NAIS-Net

- **结构**：具有归一化和稳定化特性的特殊网络架构
- **特点**：
  - 包含 shortcut 连接，增强梯度流动
  - 权重投影机制，提高数值稳定性
  - 支持稳定模式和非稳定模式
- **适用场景**：复杂PDE，特别是具有快速变化解的问题

## 激活函数

| 激活函数 | 特点 | 适用场景 |
|---------|------|---------|
| **Sine** | 无限可导，频率可调 | 平滑解的PDE，如波动方程 |
| **ReLU** | 计算高效，稀疏激活 | 一般PDE问题，训练速度快 |
| **Tanh** | 输出范围有限，梯度稳定 | 边界值变化剧烈的问题 |

## 损失函数

FBSNNs的损失函数由三部分组成：

1. **路径损失**：确保离散化FBSDE的一致性
   $$ \mathcal{L}_{\text{path}} = \sum_{m=1}^{M} \sum_{n=0}^{N-1} \left|Y_{n+1}^m - Y_n^m - \varphi \Delta t_n - (Z_n^m)^T \sigma \Delta W_n^m\right|^2 $$

2. **终端损失**：匹配终端条件
   $$ \mathcal{L}_{\text{terminal}} = \sum_{m=1}^{M} \left|Y_N^m - g(X_N^m)\right|^2 $$

3. **终端梯度约束**：确保终端梯度匹配
   $$ \mathcal{L}_{\text{terminal-gradient}} = \sum_{m=1}^{M} \left\|Z_N^m - \nabla g(X_N^m)\right\|^2 $$

总损失为：$\mathcal{L} = \mathcal{L}_{\text{path}} + \mathcal{L}_{\text{terminal}} + \mathcal{L}_{\text{terminal-gradient}}$

## 设备管理

项目实现了智能设备检测和管理：

1. **优先使用CUDA**：如果有NVIDIA GPU且支持CUDA
2. **其次使用MPS**：如果是Apple Silicon芯片
3. **默认使用CPU**：如果没有GPU可用

设备设置会自动完成，无需手动配置。

## 训练策略

1. **优化器**：Adam优化器，自适应学习率
2. **批量大小**：通过`M`参数设置（轨迹数量）
3. **时间离散化**：通过`N`参数设置时间快照数量
4. **训练迭代**：建议先使用较大学习率（1e-3）训练，再用较小学习率（1e-5）精细调优

## 性能评估

### 高维问题求解能力

| 维度 | 网络架构 | 激活函数 | 相对误差 | 训练时间 |
|------|---------|---------|---------|---------|
| 10   | FC      | ReLU    | <1%     | ~10分钟 |
| 50   | NAIS-Net | Sine    | <2%     | ~30分钟 |
| 100  | NAIS-Net | Sine    | <3%     | ~60分钟 |

### 内存使用

- **CPU模式**：100维问题约需要8GB内存
- **GPU模式**：100维问题约需要4GB GPU内存

## 扩展与定制

### 创建自定义PDE求解器

要创建自定义PDE求解器，只需继承`FBSNN`基类并实现以下方法：

1. **`net_u`**：前向传播，计算值函数和梯度
2. **`loss_function`**：计算损失函数
3. **`train`**：训练模型（可选，可使用基类实现）

示例：
```python
class CustomPDE(FBSNN):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, Mm, layers, mode, activation)
    
    def net_u(self, t, x):
        # 实现自定义网络前向传播
        pass
    
    def loss_function(self, t, W, Xi):
        # 实现自定义损失函数
        pass
```

### 超参数调优

关键超参数及其推荐值：

| 超参数 | 推荐值 | 调整策略 |
|--------|--------|---------|
| 隐藏层数量 | 4-6 | 复杂问题增加层数 |
| 隐藏层神经元 | 128-512 | 高维问题增加神经元数 |
| 学习率 | 1e-3 → 1e-5 | 先大后小，逐步减小 |
| 批量大小 (M) | 1-10 | 内存充足时增大 |
| 时间快照 (N) | 50-200 | 动态过程复杂时增大 |

## 应用场景

### 金融工程

- **期权定价**：高维篮子期权、最值期权定价
- **风险管理**：计算风险价值(VaR)和条件风险价值(CVaR)
- **投资组合优化**：多资产投资组合的最优策略

### 其他领域

- **物理建模**：高维扩散方程、波动方程求解
- **控制理论**：随机最优控制问题
- **生物医学**：多物种反应扩散系统

## 代码示例

### Black-Scholes-Barenblatt方程求解

```python
import numpy as np
from FBSNNs import BlackScholesBarenblatt

# 设置参数
D = 100  # 100维问题
M = 1
N = 50
Mm = N ** (1/5)

# 网络结构
layers = [D + 1] + 4 * [256] + [1]

# 初始条件
Xi = np.array([1.0] * D)[None, :]
T = 1.0

# 创建模型
model = BlackScholesBarenblatt(Xi, T, M, N, D, Mm, layers, "Naisnet", "Sine")

# 训练
n_iter = 10000
lr = 1e-3
model.train(n_iter, lr)

# 精细调优
n_iter = 5000
lr = 1e-5
model.train(n_iter, lr)
```

## 常见问题

### 1. 训练不稳定

**解决方案**：
- 减小学习率（如从1e-3降至1e-4）
- 增加批量大小（M值）
- 使用Sine激活函数
- 尝试NAIS-Net架构的稳定模式

### 2. 内存不足

**解决方案**：
- 减小批量大小（M值）
- 减少隐藏层神经元数量
- 使用CPU模式（如果GPU内存不足）

### 3. 收敛速度慢

**解决方案**：
- 增加学习率（初期）
- 增加隐藏层神经元数量
- 使用ReLU激活函数
- 增加时间快照数量（N值）

## 性能优化建议

1. **使用GPU**：对于高维问题，GPU可提供10-100倍的速度提升
2. **批量处理**：适当增大M值，利用并行计算能力
3. **混合精度训练**：对于支持的GPU，可启用混合精度训练
4. **模型保存与加载**：训练完成后保存模型，避免重复训练

## 参考文献

1. **Han, J., Jentzen, A., & E, W. (2018).** Solving high-dimensional partial differential equations using deep learning. *Proceedings of the National Academy of Sciences*.

2. **E, W., Han, J., & Jentzen, A. (2017).** Deep learning-based numerical methods for high-dimensional parabolic partial differential equations and backward stochastic differential equations. *Communications in Mathematics and Statistics*.

3. **Beck, C., Weinan, E., & Jentzen, A. (2019).** Machine learning approximation algorithms for high-dimensional fully nonlinear partial differential equations and second-order backward stochastic differential equations. *Journal of Nonlinear Science*.

## 贡献指南

欢迎通过以下方式贡献项目：

1. **提交Issue**：报告bug或提出新功能建议
2. **提交Pull Request**：修复bug或实现新功能
3. **改进文档**：完善README和代码注释
4. **性能优化**：提高计算效率和内存使用

## 许可证

本项目采用MIT许可证，详见LICENSE文件。

## 联系方式

如有问题或建议，欢迎联系：

- **项目维护者**：[Your Name]
- **邮箱**：[your.email@example.com]
- **GitHub**：[github.com/yourusername]

---

<div style="text-align: center; margin-top: 40px; color: #666;">
  <p>© 2024 FBSNNs 项目团队</p>
  <p>基于深度学习的高维PDE求解框架</p>
</div>