import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt

# 设置随机种子
np.random.seed(100)
torch.manual_seed(100)



 

def rel_error_l2(u, uanal):
    if abs(uanal) >= 10 * np.finfo(type(uanal)).eps:
        return np.sqrt((u - uanal)**2 / uanal**2)  # 修正分母
    else:
        return abs(u - uanal)

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

class SigmaTGradUNetwork(nn.Module):
    def __init__(self, d, hls):
        super(SigmaTGradUNetwork, self).__init__()
        # 修正：与Julia一致的4层网络结构
        self.network = nn.Sequential(
            nn.Linear(d, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),
            nn.ReLU(),
            nn.Linear(hls, hls),  # 额外隐藏层
            nn.ReLU(),
            nn.Linear(hls, hls),  # 与Julia一致的4层
            nn.ReLU(),
            nn.Linear(hls, d)
        )

    def forward(self, x):
        return self.network(x)

def test_black_scholes_barenblatt(limits=False, trajectories_upper=100, trajectories_lower=100, maxiters_limits=5):
    d = 30
    x0 = np.tile([1.0, 0.5], d // 2)
    tspan = (0.0, 1.0)
    dt = 0.25
    time_steps = int((tspan[1] - tspan[0]) / dt)
    m = 30
    
    # 修正：与Julia一致的参数范围
    if limits:
        A = np.arange(-5.0, 5.01, 0.1)  # 合理的控制变量范围
        u_domain = np.arange(-10.0, 10.01, 0.5)  # 合理的状态变量范围

    r = 0.05
    sigma = 0.4

    def f(X, u, sigmaT_gradu, p, t):
        return r * (u - np.sum(X * sigmaT_gradu))

    def g(X):
        return np.sum(X**2)

    def mu_f(X, p, t):
        return np.zeros_like(X)

    def sigma_f(X, p, t):
        return np.diag(sigma * X)
    
    # PyTorch版本的函数
    def mu_f_torch(X, t):
        return torch.zeros_like(X)
    
    def sigma_f_torch(X, t):
        batch_size = X.shape[0] if len(X.shape) > 1 else 1
        d = X.shape[-1]
        if batch_size == 1:
            return torch.diag(sigma * X.squeeze())
        else:
            return torch.diag_embed(sigma * X)
    
    if limits:
        def give_f_matrix(X, urange, sigmaT_gradu_val, p, t):
            return np.array([f(X, u, sigmaT_gradu_val, p, t) for u in urange])
        
        def legendre_transform(f_matrix, a, urange):
            le = a * urange - f_matrix
            return np.max(le)

    def u_analytical(x, t):
        return np.exp((r + sigma**2) * (tspan[1] - tspan[0])) * np.sum(x**2)

    analytical_sol = u_analytical(x0, tspan[0])
    print(f"Analytical solution: {analytical_sol}")

    hls = 10 + d

    u0 = U0Network(d, hls)
    sigmaT_gradu = [SigmaTGradUNetwork(d, hls) for _ in range(time_steps)]

    opt = optim.Adam(
        list(u0.parameters()) + [param for net in sigmaT_gradu for param in net.parameters()],
        lr=0.001
    )

    x0_tensor = torch.tensor(x0, dtype=torch.float32)

    maxiters = 150
    losses = []
    iters = []

    print("开始训练...")
    for i in range(maxiters):
        opt.zero_grad()
        batch_loss = 0.0

        for _ in range(m):
            X = x0_tensor.clone().unsqueeze(0)
            u = u0(X)

            for step in range(time_steps):
                t = tspan[0] + step * dt
                
                # 使用PyTorch版本函数，减少类型转换
                mu_val = mu_f_torch(X, t)
                sigma_val = sigma_f_torch(X, t)
                
                sigmaT_gradu_net = sigmaT_gradu[step]
                sigmaT_gradu_val = sigmaT_gradu_net(X)
                
                # 计算f值（尽量减少numpy转换）
                X_np = X.detach().numpy()[0]
                sigmaT_gradu_val_np = sigmaT_gradu_val.detach().numpy()[0]
                f_val = f(X_np, u.item(), sigmaT_gradu_val_np, None, t)
                f_val_tensor = torch.tensor(f_val, dtype=torch.float32)
                
                dW = torch.randn(1, d, dtype=torch.float32) * np.sqrt(dt)
                
                # 修正：正确的矩阵乘法
                if len(sigma_val.shape) == 2:
                    X = X + mu_val * dt + torch.mm(dW, sigma_val)
                else:
                    X = X + mu_val * dt + torch.bmm(dW.unsqueeze(1), sigma_val.unsqueeze(-1)).squeeze()
                
                u = u - f_val_tensor * dt + torch.mm(dW, sigmaT_gradu_val.transpose(1, 0))

            g_X = torch.tensor(g(X.detach().numpy()[0]), dtype=torch.float32)
            batch_loss += (g_X - u)**2

        batch_loss /= m
        losses.append(batch_loss.item())
        current_u0 = u0(x0_tensor.unsqueeze(0))[0, 0].item()
        iters.append(current_u0)

        batch_loss.backward()
        opt.step()

        if (i % 10 == 0 or i == maxiters - 1):
            print(f'Epoch {i}, Loss: {batch_loss.item():.6f}, u0: {current_u0:.6f}')

    u_pred = u0(x0_tensor.unsqueeze(0)).item()
    u_low, u_high = None, None
    
    if limits:
        print("\nCalculating bounds using Legendre transform...")
        
        # 为上界计算创建独立的网络副本
        u0_high = U0Network(d, hls)
        u0_high.load_state_dict(u0.state_dict())
        sigmaT_gradu_high = [SigmaTGradUNetwork(d, hls) for _ in range(time_steps)]
        for i, net in enumerate(sigmaT_gradu_high):
            net.load_state_dict(sigmaT_gradu[i].state_dict())
        
        def calculate_upper_bound(trajectories=trajectories_upper):
            total = 0.0
            for _ in range(trajectories):
                X = x0_tensor.clone().unsqueeze(0)
                U = torch.tensor(g(X.detach().numpy()[0]), dtype=torch.float32)
                
                # 存储轨迹
                X_trajectory = [X]
                for i in range(time_steps):
                    t = tspan[0] + i * dt
                    dW = torch.randn(1, d, dtype=torch.float32) * np.sqrt(dt)
                    mu_val = mu_f_torch(X, t)
                    sigma_val = sigma_f_torch(X, t)
                    X = X + mu_val * dt + torch.mm(dW, sigma_val) if len(sigma_val.shape) == 2 else X + mu_val * dt + torch.bmm(dW.unsqueeze(1), sigma_val.unsqueeze(-1)).squeeze()
                    X_trajectory.append(X)
                
                # 反向计算
                for i in range(len(X_trajectory)-2, 0, -1):
                    t = tspan[0] + i * dt
                    X_prev = X_trajectory[i]
                    sigmaT_gradu_val = sigmaT_gradu_high[i](X_prev)
                    dW = torch.randn(1, d, dtype=torch.float32) * np.sqrt(dt)
                    
                    X_prev_np = X_prev.detach().numpy()[0]
                    sigmaT_gradu_val_np = sigmaT_gradu_val.detach().numpy()[0]
                    f_val = f(X_prev_np, U.item(), sigmaT_gradu_val_np, None, t)
                    f_val_tensor = torch.tensor(f_val, dtype=torch.float32)
                    
                    U = U + f_val_tensor * dt - torch.mm(dW, sigmaT_gradu_val.transpose(1, 0))
                
                total += U.item()
            return total / trajectories
        
        # 简化上界计算（完整实现需要更多工作）
        u_high = calculate_upper_bound(10)  # 使用较少的轨迹进行演示
        print(f"Upper bound: {u_high:.6f}")
        
        # 下界计算（简化版）
        def calculate_lower_bound(trajectories=trajectories_lower):
            total = 0.0
            for _ in range(trajectories):
                X = x0_tensor.clone().unsqueeze(0)
                u = u0(X)
                I = 0.0
                Q = 0.0
                
                for i in range(time_steps):
                    t = tspan[0] + i * dt
                    sigmaT_gradu_val = sigmaT_gradu[i](X)
                    dW = torch.randn(1, d, dtype=torch.float32) * np.sqrt(dt)
                    
                    mu_val = mu_f_torch(X, t)
                    sigma_val = sigma_f_torch(X, t)
                    X_np = X.detach().numpy()[0]
                    sigmaT_gradu_val_np = sigmaT_gradu_val.detach().numpy()[0]
                    f_val = f(X_np, u.item(), sigmaT_gradu_val_np, None, t)
                    
                    X = X + mu_val * dt + torch.mm(dW, sigma_val) if len(sigma_val.shape) == 2 else X + mu_val * dt + torch.bmm(dW.unsqueeze(1), sigma_val.unsqueeze(-1)).squeeze()
                    u_val = u.item() - f_val * dt + torch.mm(dW, sigmaT_gradu_val.transpose(1, 0)).item()
                    u = torch.tensor(u_val, dtype=torch.float32).unsqueeze(0)
                    
                    if limits:
                        f_matrix = give_f_matrix(X_np, u_domain, sigmaT_gradu_val_np, None, t)
                        le_values = [a * u_val - legendre_transform(f_matrix, a, u_domain) for a in A]
                        a_opt = A[np.argmax(le_values)]
                        I += a_opt * dt
                        Q += np.exp(I) * legendre_transform(f_matrix, a_opt, u_domain)
                
                g_X = g(X.detach().numpy()[0])
                total += np.exp(I) * g_X - Q
            
            return total / trajectories
        
        u_low = calculate_lower_bound(10)  # 使用较少的轨迹进行演示
        print(f"Lower bound: {u_low:.6f}")
    
    u_pred = u0(x0_tensor.unsqueeze(0)).item()
    error_l2 = rel_error_l2(u_pred, analytical_sol)
    
    print(f"\nPredicted u0: {u_pred:.6f}")
    print(f"Analytical u0: {analytical_sol:.6f}")
    print(f"Relative L2 Error: {error_l2:.6f}")
    
    if limits:
        print(f"\nBounds:")
        print(f"Lower: {u_low:.6f}")
        print(f"Upper: {u_high:.6f}")
        print(f"Within bounds: {u_low <= u_pred <= u_high if u_low is not None and u_high is not None else 'N/A'}")

    if error_l2 < 1.0:
        print("✓ Test passed: Relative error < 1.0")
    else:
        print("✗ Test failed: Relative error >= 1.0")

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

if __name__ == "__main__":
    # 测试标准版本
    result_standard = test_black_scholes_barenblatt(limits=False)
    
    # 测试带Legendre变换的版本
    result_with_limits = test_black_scholes_barenblatt(limits=True, trajectories_upper=50, trajectories_lower=50, maxiters_limits=3)