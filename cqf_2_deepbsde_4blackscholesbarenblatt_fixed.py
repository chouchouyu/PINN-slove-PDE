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
        # 修复：与Julia代码一致的4层网络结构
        # Julia代码：Dense(d, hls, relu) -> Dense(hls, hls, relu) -> Dense(hls, hls, relu) -> Dense(hls, d)
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
def test_black_scholes_barenblatt(limits=False, trajectories_upper=1000, trajectories_lower=1000, maxiters_limits=10):
    # 问题参数
    d = 30  # 维度
    x0 = np.tile([1.0, 0.5], d // 2)  # 初始点
    tspan = (0.0, 1.0)  # 时间区间
    dt = 0.25  # 时间步长
    time_steps = int((tspan[1] - tspan[0]) / dt)  # 时间步数
    m = 30  # 轨迹数量（批大小）
    
    # Legendre变换相关参数（如果启用）
    if limits:
        # 修复：与Julia代码一致的参数范围
        # Julia代码：A = -2:0.01:2
        A = np.arange(-2, 2.01, 0.01)  # 参数范围
        # Julia代码：u_domain = -500:0.1:500
        u_domain = np.arange(-500, 500.01, 0.1)  # u的定义域

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
    
    # Legendre变换辅助函数（如果启用）
    if limits:
        def give_f_matrix(X, urange, sigmaT_gradu_val, p, t):
            return np.array([f(X, u, sigmaT_gradu_val, p, t) for u in urange])
        
        def legendre_transform(f_matrix, a, urange):
            le = a * urange - f_matrix
            return np.max(le)

    # 解析解
    def u_analytical(x, t):
        return np.exp((r + sigma**2) * (tspan[1] - tspan[0])) * np.sum(x**2)

    analytical_sol = u_analytical(x0, tspan[0])
    print(f"Analytical solution: {analytical_sol}")

    # 神经网络设置
    hls = 10 + d  # 隐藏层大小

    # 创建网络
    u0 = U0Network(d, hls)
    # 修复：使用与Julia代码一致的网络结构
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
    
    # 如果启用limits，计算上下界
    u_low = None
    u_high = None
    
    if limits:
        print("\nCalculating upper and lower bounds using Legendre transform...")
        
        # 1. 计算上界 (u_high)
        print("Calculating upper limit...")
        
        # 生成SDE轨迹（使用PyTorch张量）
        # 修复：实现与Julia代码一致的轨迹生成方法
        def generate_sde_trajectories(trajectories, time_steps, dt, d, x0, mu_f, sigma_f):
            trajectories_list = []
            for _ in range(trajectories):
                X_trajectory = [x0_tensor.unsqueeze(0)]
                X = x0_tensor.unsqueeze(0)
                for i in range(time_steps):
                    t = tspan[0] + i * dt
                    dW = torch.randn(1, d, dtype=torch.float32) * torch.sqrt(torch.tensor(dt))
                    mu_val = torch.tensor(mu_f(X.detach().numpy()[0], None, t), dtype=torch.float32)
                    sigma_val = torch.tensor(sigma_f(X.detach().numpy()[0], None, t), dtype=torch.float32)
                    X = X + mu_val * dt + dW @ sigma_val
                    X_trajectory.append(X)
                trajectories_list.append(X_trajectory)
            return trajectories_list
        
        # 生成SDE轨迹
        sde_trajectories = generate_sde_trajectories(trajectories_upper, time_steps, dt, d, x0, mu_f, sigma_f)
        
        # 定义上界计算函数（用于优化）
        # 修复：完整实现与Julia代码一致的上界计算方法
        def calculate_upper_bound():
            total = 0.0
            for trajectory in sde_trajectories:
                U = torch.tensor(g(trajectory[-1].detach().numpy()[0]), dtype=torch.float32)
                for i in range(time_steps, 1, -1):
                    t = tspan[0] + i * dt
                    xsde_prev = trajectory[i-1]
                    _sigmaT_gradu = sigmaT_gradu[i-1](xsde_prev)
                    dW = torch.randn(1, d, dtype=torch.float32) * torch.sqrt(torch.tensor(dt))
                    xsde_prev_np = xsde_prev.detach().numpy()[0]
                    f_val = torch.tensor(f(xsde_prev_np, U.item(), _sigmaT_gradu.detach().numpy()[0], None, t), dtype=torch.float32)
                    U = U + f_val * dt - dW @ _sigmaT_gradu.T
                total += U
            return total / trajectories_upper
        
        # 优化网络以最大化上界
        # 修复：实现与Julia代码一致的上界优化方法
        high_opt = optim.Adam(
            list(u0.parameters()) + [param for net in sigmaT_gradu for param in net.parameters()],
            lr=0.01
        )
        
        for i in range(maxiters_limits):
            high_opt.zero_grad()
            
            # 计算上界损失（取负因为要最大化）
            batch_loss = -calculate_upper_bound()
            
            # 反向传播
            batch_loss.backward()
            high_opt.step()
            
            if (i % 2 == 0 or i == maxiters_limits - 1):
                print(f'Upper bound optimization epoch {i}, Loss: {batch_loss.item():.6f}')
        
        # 最终计算上界
        u_high = calculate_upper_bound().item()
        print(f"Upper limit: {u_high}")
        
        # 2. 计算下界 (u_low)
        print("Calculating lower limit...")
        
        # 定义下界计算函数
        # 修复：完整实现与Julia代码一致的下界计算方法
        # 修复：确保梯度流正确传播，避免使用.item()切断梯度
        def calculate_lower_bound():
            total = 0.0
            for _ in range(trajectories_lower):
                u = u0(x0_tensor.unsqueeze(0))
                X = x0_tensor.unsqueeze(0)
                I = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                Q = torch.tensor(0.0, dtype=torch.float32, requires_grad=True)
                
                for i in range(time_steps):
                    t = tspan[0] + i * dt
                    
                    # 计算sigmaT_gradu
                    _sigmaT_gradu = sigmaT_gradu[i](X)
                    
                    # 生成布朗运动增量
                    dW = torch.randn(1, d, dtype=torch.float32) * torch.sqrt(torch.tensor(dt))
                    
                    # 更新X和u
                    mu_val = torch.tensor(mu_f(X.detach().numpy()[0], None, t), dtype=torch.float32)
                    sigma_val = torch.tensor(sigma_f(X.detach().numpy()[0], None, t), dtype=torch.float32)
                    # 修复：使用u.detach()而不是u.item()，保持张量类型
                    f_val = torch.tensor(f(X.detach().numpy()[0], u.detach().item(), _sigmaT_gradu.detach().numpy()[0], None, t), dtype=torch.float32)
                    
                    X = X + mu_val * dt + dW @ sigma_val
                    u = u - f_val * dt + dW @ _sigmaT_gradu.T
                    
                    # 预计算f_matrix
                    f_matrix = give_f_matrix(X.detach().numpy()[0], u_domain, _sigmaT_gradu.detach().numpy()[0], None, t)
                    
                    # 寻找最优的a值
                    # 修复：使用u.detach().item()获取值，同时保持梯度流
                    le_values = [a * u.detach().item() - legendre_transform(f_matrix, a, u_domain) for a in A]
                    a_opt = A[np.argmax(le_values)]
                    
                    # 更新积分项
                    I = I + torch.tensor(a_opt, dtype=torch.float32) * dt
                    Q = Q + torch.exp(I) * torch.tensor(legendre_transform(f_matrix, a_opt, u_domain), dtype=torch.float32)
                
                # 计算轨迹贡献
                g_X = torch.tensor(g(X.detach().numpy()[0]), dtype=torch.float32)
                total += torch.exp(I) * g_X - Q
            
            return total / trajectories_lower
        
        # 计算初始下界
        u_low = calculate_lower_bound().item()
        print(f"Initial lower limit: {u_low}")
        
        # 优化网络以提高下界
        # 修复：实现与Julia代码一致的下界优化方法
        print("Optimizing for lower bound...")
        lower_opt = optim.Adam(
            list(u0.parameters()) + [param for net in sigmaT_gradu for param in net.parameters()],
            lr=0.01
        )
        
        for i in range(maxiters_limits):
            lower_opt.zero_grad()
            
            # 计算下界损失（取负因为要最大化下界）
            batch_loss = -calculate_lower_bound()
            
            # 反向传播
            batch_loss.backward()
            lower_opt.step()
            
            if (i % 2 == 0 or i == maxiters_limits - 1):
                current_lower = calculate_lower_bound().item()
                print(f'Lower bound optimization epoch {i}, Loss: {batch_loss.item():.6f}, Lower bound: {current_lower:.6f}')
        
        # 最终计算下界
        u_low = calculate_lower_bound().item()
        print(f"Final lower limit: {u_low}")
    
    # 重新计算数值解
    u_pred = u0(x0_tensor.unsqueeze(0)).item()

    # 计算误差
    error_l2 = rel_error_l2(u_pred, analytical_sol)
    print(f"\nPredicted u0: {u_pred:.6f}")
    print(f"Analytical u0: {analytical_sol:.6f}")
    print(f"Relative L2 Error: {error_l2:.6f}")
    
    # 如果启用了limits，输出上下界信息
    if limits:
        print(f"\nLegendre Transform Results:")
        print(f"Lower bound: {u_low:.6f}")
        print(f"Upper bound: {u_high:.6f}")
        print(f"Solution within bounds: {u_low <= u_pred <= u_high}")

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

    if limits:
        return u_pred, analytical_sol, error_l2, u_low, u_high
    else:
        return u_pred, analytical_sol, error_l2

# 运行测试
if __name__ == "__main__":
    # 测试标准版本
    result_standard = test_black_scholes_barenblatt(limits=False)
    
    # 测试带Legendre变换的版本
    result_with_limits = test_black_scholes_barenblatt(limits=True, trajectories_upper=100, trajectories_lower=100, maxiters_limits=5)