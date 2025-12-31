# "把图片的内容识别出来打印出来"点击查看元宝的回答
# https://yb.tencent.com/s/tZGohBnDWOs7

import numpy as np
from numpy import exp, sqrt

def MC_3Asset(S0, Smin, sigma, C, r, q, T, N):
    dt = T / N  # 添加缺失的时间步长定义
    
    mu = np.zeros(3)
    for i in range(3):
        mu[i] = r - q[i] - 0.5 * sigma[i] ** 2  # 几何布朗运动的漂移项[1](@ref)
    
    U = np.linalg.cholesky(C)  # Cholesky分解处理资产相关性[6](@ref)
    
    psv = np.zeros((3, N))
    bb = np.random.randn(3, N)  # 生成随机数用于蒙特卡洛模拟[1](@ref)
    
    for i in range(N):
        psv[0, i] = U[0, 0] * bb[0, i]
        psv[1, i] = U[1, 0] * bb[0, i] + U[1, 1] * bb[1, i]
        psv[2, i] = U[2, 0] * bb[0, i] + U[2, 1] * bb[1, i] + U[2, 2] * bb[2, i]
    
    S = np.zeros((3, N + 1))
    S[:, 0] = S0  # 修正：直接使用S0而不是S0[:,]
    
    for i in range(N):
        S[:, i + 1] = S[:, i] * exp(mu[:] * dt + psv[:, i] * sigma[:] * sqrt(dt))
        # 使用几何布朗运动模拟三条标的资产价格路径[1](@ref)
        
    Sm11 = max(max(S[0, :]), max(S[1, :]), max(S[2, :]))
    Sm22 = min(min(S[0, :]), min(S[1, :]), min(S[2, :]), Smin)  # 修正：直接使用Smin
    a = max(Sm11 - Sm22, 0)  # 计算期权收益[6](@ref)
    
    MC_noCV = exp(-r * T) * a  # 用无风险利率贴现[1](@ref)
    return MC_noCV

def MC_3Asset_sum(S0, Smin, sigma, C, r, q, T, N, no_samples):
    a = 0
    for i in range(no_samples):
        a = a + MC_3Asset(S0, Smin, sigma, C, r, q, T, N)
    v = a / no_samples  # 多次模拟取平均值[1](@ref)
    return v

if __name__ == "__main__":
    # ----1 veriosn------
    # # 设置期权参数
    # S0 = np.array([100.0, 100.0, 100.0])  # 三种资产的初始价格
    # Smin = 80.0  # 最低价格阈值
    # sigma = np.array([0.2, 0.2, 0.2])  # 三种资产的波动率
    # C = np.array([[1.0, 0.5, 0.3],  # 资产相关性矩阵[6](@ref)
    #               [0.5, 1.0, 0.4],
    #               [0.3, 0.4, 1.0]])
    # r = 0.05  # 无风险利率[1](@ref)
    # q = np.array([0.01, 0.01, 0.01])  # 股息率
    # T = 1.0  # 到期时间（年）
    # N = 252  # 价格路径节点数（假设一年有252个交易日）
    # no_samples = 10000  # 蒙特卡洛模拟次数
    
    # # 调用函数计算期权价格
    # price = MC_3Asset_sum(S0, Smin, sigma, C, r, q, T, N, no_samples)
    # print(f"三资产期权价格: {price:.4f}") #三资产期权价格: 50.0732


    # ----2 version------