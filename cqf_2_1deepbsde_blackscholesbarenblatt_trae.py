import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

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

# Black Scholes Barenblatt方程测试
def test_black_scholes_barenblatt():
    # 问题参数
    d = 30  # 维度
    x0 = np.tile([1.0, 0.5], d // 2)  # 初始点
    tspan = (0.0, 1.0)  # 时间区间
    dt = 0.25  # 时间步长
    time_steps = int((tspan[1] - tspan[0]) / dt)  # 时间步数
    m = 30  # 轨迹数量（批大小）

    # 方程参数
    r = 0.05  # 利率
    sigma = 0.4  # 波动率

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
        return np.exp((r + sigma**2) * (tspan[1] - tspan[0])) * np.sum(x**2)

    analytical_sol = u_analytical(x0, tspan[0])
    print(f"Analytical solution: {analytical_sol}")

    # 神经网络设置
    hls = 10 + d  # 隐藏层大小

    # 创建网络
    u0 = U0Network(d, hls)
    sigmaT_gradu = [SigmaTGradUNetwork(d, hls) for _ in range(time_steps)]

    # 优化器
    opt = optim.Adam(
        list(u0.parameters()) + [param for net in sigmaT_gradu for param in net.parameters()],
        lr=0.001
    )

    # 转换为torch张量
    x0_tensor = torch.tensor(x0, dtype=torch.float32)

    # 训练过程
    maxiters = 150
    losses = []
    iters = []  # 存储训练过程中的u0值

    print("开始训练...")
    for i in range(maxiters):
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

        # 计算当前u0值
        current_u0 = u0(x0_tensor.unsqueeze(0))[0, 0].item()
        iters.append(current_u0)

        # 反向传播和优化
        batch_loss.backward()
        opt.step()

        # 打印训练进度
        if (i % 10 == 0 or i == maxiters - 1):
            print(f'Epoch {i}, Loss: {batch_loss.item():.6f}, u0: {current_u0:.6f}')

    # 计算数值解
    u_pred = u0(x0_tensor.unsqueeze(0)).item()

    # 计算误差
    error_l2 = rel_error_l2(u_pred, analytical_sol)
    print(f"\nPredicted u0: {u_pred:.6f}")
    print(f"Analytical u0: {analytical_sol:.6f}")
    print(f"Relative L2 Error: {error_l2:.6f}")

    # 验证误差
    if error_l2 < 1.0:
        print("✓ 测试通过：相对误差 < 1.0")
    else:
        print("✗ 测试未通过：相对误差 >= 1.0")
        assert error_l2 < 1.0, f"误差太大: {error_l2}"

    # 绘制训练损失曲线和u0收敛曲线
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(losses)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(iters)
    plt.xlabel('Epoch')
    plt.ylabel('u0 estimate')
    plt.title('u0 Convergence')
    
    plt.tight_layout()
    plt.show()

    return u_pred, analytical_sol, error_l2

# 运行测试
if __name__ == "__main__":
    test_black_scholes_barenblatt()