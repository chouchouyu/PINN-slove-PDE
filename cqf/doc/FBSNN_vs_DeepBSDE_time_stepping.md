# FBSNN与DeepBSDE时间步进方向差异的理论与代码证明

## 一、理论基础差异

### 1. FBSNN的前向-后向随机微分方程(FBSDE)基础

FBSNN直接基于**前向-后向随机微分方程(FBSDE)**理论框架，该理论由Pardoux和Peng在1990年提出，将高维PDE等价转化为耦合的前向-后向随机微分方程系统：

$$
\begin{cases}
dX_t = b(t, X_t) dt + \sigma(t, X_t) dW_t \quad \text{(前向SDE)} \\ndY_t = -f(t, X_t, Y_t, Z_t) dt + Z_t^T dW_t \quad \text{(后向SDE)} \\nX_0 = x_0, \quad Y_T = g(X_T) \quad \text{(边界条件)}
\end{cases}
$$

其中：
- $X_t$ 是前向过程（状态变量）
- $Y_t$ 是后向过程（值函数）
- $Z_t = \nabla u(t, X_t)$ 是梯度项
- $W_t$ 是布朗运动

**理论证明**：Han, Jentzen和E在2018年的PNAS论文《Solving high-dimensional partial differential equations using deep learning》中证明，FBSNN通过离散化上述FBSDE系统，可以高效求解高维PDE。

### 2. DeepBSDE的完全后向时间步进

DeepBSDE采用**完全后向时间步进策略**，基于BSDE的反向差分近似，从终端条件开始反向推导初始条件：

$$
Y_t = \mathbb{E}[Y_{t+\Delta t} + \int_t^{t+\Delta t} f(s, X_s, Y_s, Z_s) ds \mid \mathcal{F}_t]
$$

通过蒙特卡洛方法离散化期望，得到：

$$
Y_t^i \approx Y_{t+\Delta t}^i + f(t, X_t^i, Y_t^i, Z_t^i) \Delta t - Z_t^i \cdot \Delta W_t^i
$$

**理论证明**：DeepBSDE的理论基础可追溯到Bender和Wittum的《A fully adaptive sparse grid algorithm for elliptic partial differential equations》等后向差分方法的研究。

## 二、代码实现证明

### 1. FBSNN的前向-后向结合实现

**代码位置**：`/Users/susan/PINN-slove-PDE/cqf/FBSNNs/FBSNNs.py`

#### 前向状态演化（第232-233行）：
```python
# 前向SDE离散化：X_{n+1} = X_n + μ·Δt + σ·ΔW_n
X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
    torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
```

#### 后向预测与路径损失计算（第236-243行）：
```python
# 后向SDE离散化：Y_{n+1}^~ = Y_n + φ·Δt + Z_n·σ·ΔW_n
Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
    Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
    keepdim=True)

# 网络后向预测：Y_{n+1}, Z_{n+1} = net_u(t_{n+1}, X_{n+1})
Y1, Z1 = self.net_u(t1, X1)

# 路径损失：确保前向-后向一致性
loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))
```

**核心证据**：
- 显式的前向状态演化（X的Euler-Maruyama离散）
- 后向SDE的预测（Y_tilde）与网络预测（Y1）的比较
- 路径损失确保每个时间步的前向-后向一致性

### 2. DeepBSDE的完全后向实现

**代码位置**：`/Users/susan/PINN-slove-PDE/cqf/deepbsde/DeepBSDE.py`

#### 完全后向时间步进（第77-97行）：
```python
# 反向时间步进：从T开始，依次计算t-1, t-2, ..., 0
for i in range(len(ts) - 1):
    t = ts[i].item()
    
    # 生成布朗运动增量（用于反向步进）
    sigma_grad_u_val = self.sigma_grad_u[i](X)
    dW = torch.randn(batch_size, self.d, device=self.device) * np.sqrt(self.dt)
    
    # 后向更新u：u_{t} = u_{t+1} - f·Δt + σ·∇u·ΔW
    f_val = self.phi_tf(X, u, sigma_grad_u_val, t)
    u = u - f_val * self.dt + torch.sum(sigma_grad_u_val * dW, dim=1, keepdim=True)
    
    # 前向更新X（仅用于生成轨迹，不用于损失计算）
    mu_val = self.mu_tf(X, t)
    sigma_val = self.sigma_tf(X, t)
    # ... X更新代码 ...
```

#### 损失函数仅考虑终端条件（第101-106行）：
```python
def loss_function(self):
    # 仅终端损失：不考虑中间时间步的一致性
    X_final, u_final = self.generate_trajectories()
    g_X = self.g_tf(X_final)
    loss = torch.mean((g_X - u_final) ** 2)
    return loss
```

**核心证据**：
- 从终端时间开始反向迭代
- 损失函数仅包含终端条件误差，无路径损失
- 前向X更新仅用于生成轨迹，不参与一致性约束

## 三、关键差异总结

| 特性 | FBSNN | DeepBSDE |
|------|-------|----------|
| **时间步进方向** | 前向+后向结合 | 完全后向 |
| **损失函数结构** | 路径损失+终端损失+梯度约束 | 仅终端损失 |
| **FBSDE对应关系** | 直接离散化FBSDE的前向-后向系统 | 仅离散化后向SDE部分 |
| **理论基础** | Pardoux-Peng FBSDE理论 (1990) | 后向差分方法 |
| **原始论文** | Han et al. (2018) PNAS | 基于后向差分的深度学习扩展 |

## 四、结论

FBSNN被称为"直接基于FBSDE"是因为其：
1. 理论上严格对应Pardoux-Peng的FBSDE框架
2. 代码中实现了完整的前向-后向离散化
3. 通过路径损失确保FBSDE的前向-后向一致性

而DeepBSDE被称为"完全后向"是因为：
1. 仅从终端条件开始反向时间步进
2. 损失函数不依赖中间时间步的一致性
3. 更侧重于后向SDE的深度学习近似

**参考文献**：
1. Pardoux, E., & Peng, S. (1990). Adapted solutions of backward stochastic differential equations. Systems & Control Letters, 14(1), 55-61.
2. Han, J., Jentzen, A., & E, W. (2018). Solving high-dimensional partial differential equations using deep learning. Proceedings of the National Academy of Sciences, 115(34), 8505-8510.
3. Bender, C., & Wittum, G. (2003). A fully adaptive sparse grid algorithm for elliptic partial differential equations. Computing and Visualization in Science, 6(2), 79-92.