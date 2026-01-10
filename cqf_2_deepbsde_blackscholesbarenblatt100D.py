import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# 设置随机种子
np.random.seed(100)
torch.manual_seed(100)

# 相对L2误差计算
def rel_error_l2(u, uanal):
    if abs(uanal) >= 10 * np.finfo(type(uanal)).eps:
        return np.sqrt((u - uanal)**2 / u**2)
    else: # 防止溢出
        return abs(u - uanal)

# 解近似网络 u0
class U0Network(nn.Module):
    def __init__(self, d, hls):
        super(U0Network, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(d, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, 1)
        )

    def forward(self, x):
        return self.network(x)

# 空间梯度近似网络 sigmaT_gradu
class SigmaTGradUNetwork(nn.Module):
    def __init__(self, d, hls):
        super(SigmaTGradUNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(d, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, d)
        )

    def forward(self, x):
        return self.network(x)

# Black Scholes Barenblatt 100D方程测试
def test_black_scholes_barenblatt_100d():
    # 问题参数 - 与FBSNNs BlackScholesBarenblatt100D.py完全一致
    d = 100  # 维度（与FBSNNs的D=100相同）
    x0 = np.tile([1.0, 0.5], d // 2)  # 初始点（与FBSNNs的Xi相同）
    tspan = (0.0, 1.0)  # 时间区间（与FBSNNs的T=1.0相同）
    N_fbsnns = 50  # FBSNNs时间快照数量
    dt = (tspan[1] - tspan[0]) / (N_fbsnns - 1)  # 时间步长匹配FBSNNs
    time_steps = N_fbsnns - 1  # 时间步数（FBSNNs时间快照数-1）
    m = 100  # 轨迹数量（与FBSNNs的M=100相同）

    # 方程参数
    r = 0.05  # 利率（与FBSNNs一致）
    sigma = 0.4  # 波动率（与FBSNNs一致）

    # PDE定义
    def f(X, u, sigmaT_gradu, p, t):
        return r * (u - np.sum(X * sigmaT_gradu))

    def g(X):
        return np.sum(X**2)

    def mu_f(X, p, t):
        return np.zeros_like(X)

    def sigma_f(X, p, t):
        return np.diag(sigma * X)

    # 解析解
    def u_analytical(x, t):
        return np.exp((r + sigma**2) * (tspan[1] - t)) * np.sum(x**2)

    analytical_sol = u_analytical(x0, tspan[0])
    print(f"Analytical solution: {analytical_sol}")

    # 神经网络设置 - 与FBSNNs BlackScholesBarenblatt100D.py完全一致
    hls = 256  # 隐藏层大小（与FBSNNs的4*[256]相同）

    # 创建网络
    u0 = U0Network(d, hls)
    sigmaT_gradu = [SigmaTGradUNetwork(d, hls) for _ in range(time_steps)]

    # 第一阶段优化器
    opt = optim.Adam(
        list(u0.parameters()) + [param for net in sigmaT_gradu for param in net.parameters()],
        lr=0.001
    )

    # 转换为torch张量
    x0_tensor = torch.tensor(x0, dtype=torch.float32)

    # 训练过程 - 与FBSNNs BlackScholesBarenblatt100D.py完全一致
    print("第一阶段训练：20000次迭代，学习率1e-3")
    maxiters_stage1 = 20000
    losses = []

    for i in range(maxiters_stage1):
        opt.zero_grad()
        batch_loss = 0.0

        # 对每个轨迹进行模拟
        for _ in range(m):
            # 初始化
            X = x0_tensor.clone().unsqueeze(0)
            u = u0(X)

            # 模拟时间步
            for step in range(time_steps):
                t = tspan[0] + step * dt

                # 计算漂移项和扩散项
                mu_val = torch.tensor(mu_f(X.detach().numpy()[0], None, t), dtype=torch.float32)
                sigma_val = torch.tensor(sigma_f(X.detach().numpy()[0], None, t), dtype=torch.float32)

                # 计算sigmaT_gradu
                sigmaT_gradu_net = sigmaT_gradu[step]
                sigmaT_gradu_val = sigmaT_gradu_net(X)

                # 计算f值
                f_val = torch.tensor(f(X.detach().numpy()[0], u.detach().item(), 
                                      sigmaT_gradu_val.detach().numpy()[0], None, t), 
                                    dtype=torch.float32)

                # 生成布朗运动增量
                dW = torch.randn(1, d, dtype=torch.float32) * torch.sqrt(torch.tensor(dt))

                # 更新X和u
                X = X + mu_val * dt + dW @ sigma_val
                u = u - f_val * dt + dW @ sigmaT_gradu_val.T

            # 计算终端条件
            g_X = torch.tensor(g(X.detach().numpy()[0]), dtype=torch.float32)

            # 计算轨迹损失
            batch_loss += (g_X - u)**2

        # 平均损失
        batch_loss /= m
        losses.append(batch_loss.item())

        # 反向传播和优化
        batch_loss.backward()
        opt.step()

        # 打印训练进度
        if (i + 1) % 1000 == 0:
            print(f"迭代 {i+1}/{maxiters_stage1}, 损失: {batch_loss.item():.6f}")

    # 第二阶段训练 - 与FBSNNs BlackScholesBarenblatt100D.py完全一致
    print("\n第二阶段训练：5000次迭代，学习率1e-5")
    opt.param_groups[0]['lr'] = 0.00001  # 学习率降低到1e-5
    maxiters_stage2 = 5000

    for i in range(maxiters_stage2):
        opt.zero_grad()
        batch_loss = 0.0

        # 对每个轨迹进行模拟
        for _ in range(m):
            # 初始化
            X = x0_tensor.clone().unsqueeze(0)
            u = u0(X)

            # 模拟时间步
            for step in range(time_steps):
                t = tspan[0] + step * dt

                # 计算漂移项和扩散项
                mu_val = torch.tensor(mu_f(X.detach().numpy()[0], None, t), dtype=torch.float32)
                sigma_val = torch.tensor(sigma_f(X.detach().numpy()[0], None, t), dtype=torch.float32)

                # 计算sigmaT_gradu
                sigmaT_gradu_net = sigmaT_gradu[step]
                sigmaT_gradu_val = sigmaT_gradu_net(X)

                # 计算f值
                f_val = torch.tensor(f(X.detach().numpy()[0], u.detach().item(), 
                                      sigmaT_gradu_val.detach().numpy()[0], None, t), 
                                    dtype=torch.float32)

                # 生成布朗运动增量
                dW = torch.randn(1, d, dtype=torch.float32) * torch.sqrt(torch.tensor(dt))

                # 更新X和u
                X = X + mu_val * dt + dW @ sigma_val
                u = u - f_val * dt + dW @ sigmaT_gradu_val.T

            # 计算终端条件
            g_X = torch.tensor(g(X.detach().numpy()[0]), dtype=torch.float32)

            # 计算轨迹损失
            batch_loss += (g_X - u)**2

        # 平均损失
        batch_loss /= m
        losses.append(batch_loss.item())

        # 反向传播和优化
        batch_loss.backward()
        opt.step()

        # 打印训练进度
        if (i + 1) % 1000 == 0:
            print(f"迭代 {i+1}/{maxiters_stage2}, 损失: {batch_loss.item():.6f}")

    # 计算数值解
    u_pred = u0(x0_tensor.unsqueeze(0)).item()

    # 计算误差
    error_l2 = rel_error_l2(u_pred, analytical_sol)
    print(f"\n数值解: {u_pred}")
    print(f"解析解: {analytical_sol}")
    print(f"相对L2误差: {error_l2}")

    # 验证误差
    assert error_l2 < 1.0, f"误差太大: {error_l2}"
    print("测试通过!")

    return u_pred, analytical_sol, error_l2, losses

# 运行测试
if __name__ == "__main__":
    test_black_scholes_barenblatt_100d()