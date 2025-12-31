import numpy as np
import time
from typing import Tuple, List

def adaptive_mlmc_100d_basket_call(
    S0: float, 
    K: float, 
    r: float, 
    sigma: float, 
    T: float, 
    d: int = 100,  # 维度，默认为100
    M: int = 2, 
    Lmin: int = 2, 
    Lmax: int = 7, 
    eps_target: float = 0.01,
    rho: float = 0.0  # 资产间相关系数，0表示独立
) -> Tuple[float, float, List[int], List[float]]:
    """
    自适应多级蒙特卡洛方法定价100维篮子欧式看涨期权
    
    参数:
    S0: 每个资产的初始价格
    K: 行权价
    r: 无风险利率
    sigma: 每个资产的波动率（假设相同）
    T: 到期时间
    d: 资产数量（维度）
    M: 层级精度倍增因子
    Lmin: 最粗糙层级
    Lmax: 最精细层级限制
    eps_target: 目标均方根误差
    rho: 资产间的相关系数
    
    返回:
    price: 期权价格估计
    computational_cost: 总计算成本
    Nl_list: 各层级最终模拟次数
    Vl_list: 各层级最终方差估计
    """
    
    # 初始化数组
    Vl = np.zeros(Lmax)  # 存储各层级方差估计
    Cl = np.zeros(Lmax)  # 存储各层级单次模拟成本
    Nl = np.zeros(Lmax, dtype=int)  # 存储各层级模拟次数
    # sums存储: [差值之和, 精细层级之和, 粗糙层级之和]
    sums = np.zeros((Lmax, 3))
    
    computational_cost = 0.0
    
    # 初始运行少量模拟，以估计各级别方差和成本
    print('=' * 60)
    print(f'自适应MLMC 100维篮子期权定价')
    print(f'资产数量: {d}, 目标误差: {eps_target:.4f}')
    print('=' * 60)
    print('初步估计各级别方差和成本...')
    
    for l in range(Lmin-1, Lmax):
        Np = 500  # 初始用于估计的模拟次数（适当减少以加快速度）
        cost_temp, variance_temp, Pfine, Pcoarse = mlmc_level_100d(
            l+1, M, Np, S0, K, r, sigma, T, d, rho
        )
        computational_cost += cost_temp
        Vl[l] = variance_temp
        Cl[l] = cost_temp / Np
        
        if l == Lmin-1:  # 最粗糙层级
            sums[l, 0] = np.sum(Pfine)  # 差值就是Pfine本身
            sums[l, 1] = np.sum(Pfine)
        else:
            sums[l, 0] = np.sum(Pfine - Pcoarse)  # 估计差值 E[Pl - Pl-1]
            sums[l, 1] = np.sum(Pfine)
            sums[l, 2] = np.sum(Pcoarse)
        
        print(f'层级 l={l+1}, 步长={T/(M**(l+1)):.4f}, 估计方差={Vl[l]:.2e}, '
              f'单次成本={Cl[l]:.2e}')
    
    # 自适应确定最优模拟次数 Nl
    print('\n根据目标误差自适应分配模拟次数...')
    theta = 0.95  # 分配给粗糙层级的误差权重
    max_iter = 15  # 最大迭代次数
    initial_runs = 500  # 初始运行次数
    
    for iter_num in range(1, max_iter+1):
        # 计算总方差（根据MLMC最优分配公式）
        total_variance = 0.0
        for l in range(Lmin-1, Lmax):
            if Vl[l] > 0 and Cl[l] > 0:
                total_variance += np.sqrt(Vl[l] * Cl[l])
        
        # 计算各层级最优模拟次数
        for l in range(Lmin-1, Lmax):
            if Vl[l] > 0 and Cl[l] > 0:
                Nl[l] = int(np.ceil(
                    (1.0 / (eps_target**2)) * 
                    np.sqrt(Vl[l] / Cl[l]) * 
                    total_variance / theta
                ))
            else:
                Nl[l] = 0
        
        # 运行新增的模拟
        for l in range(Lmin-1, Lmax):
            if Nl[l] > 0:
                # 确保至少运行一些模拟
                runs_to_do = max(100, Nl[l] - (initial_runs if iter_num == 1 else 0))
                
                cost_temp, variance_temp, Pfine, Pcoarse = mlmc_level_100d(
                    l+1, M, runs_to_do, S0, K, r, sigma, T, d, rho
                )
                computational_cost += cost_temp
                
                # 更新方差估计（使用指数加权平均）
                Vl[l] = 0.7 * Vl[l] + 0.3 * variance_temp
                
                # 更新累加和
                if l == Lmin-1:
                    sums[l, 0] += np.sum(Pfine)
                    sums[l, 1] += np.sum(Pfine)
                else:
                    sums[l, 0] += np.sum(Pfine - Pcoarse)
                    sums[l, 1] += np.sum(Pfine)
                    sums[l, 2] += np.sum(Pcoarse)
        
        # 计算当前估计误差
        current_eps = 0.0
        total_samples_used = 0
        
        for l in range(Lmin-1, Lmax):
            if Nl[l] > 0:
                # 包括初始运行
                total_samples = Nl[l] + (initial_runs if l >= Lmin-1 else 0)
                if total_samples > 0:
                    current_eps += Vl[l] / total_samples
                    total_samples_used += total_samples
        
        if total_samples_used > 0:
            current_eps = np.sqrt(current_eps)
        
        print(f'迭代 {iter_num}: 当前估计误差 = {current_eps:.4e}, '
              f'总成本 = {computational_cost:.2e}')
        
        if current_eps < eps_target:
            print(f'已达到目标精度 {eps_target:.4f}。')
            break
    
    # 计算最终的期权价格估计
    price = 0.0
    for l in range(Lmin-1, Lmax):
        if l == Lmin-1:
            total_runs_l = Nl[l] + initial_runs
        else:
            total_runs_l = Nl[l] + initial_runs
            
        if total_runs_l > 0:
            price += sums[l, 0] / total_runs_l
    
    # 折现
    price = np.exp(-r * T) * price
    
    print('=' * 60)
    print(f'最终期权价格估计: {price:.6f}')
    print(f'总计算成本: {computational_cost:.2e}')
    print(f'各层级模拟次数: {[int(Nl[l] + (initial_runs if l >= Lmin-1 else 0)) for l in range(Lmin-1, Lmax)]}')
    print('=' * 60)
    
    return price, computational_cost, Nl[Lmin-1:Lmax].tolist(), Vl[Lmin-1:Lmax].tolist()


def mlmc_level_100d(
    l: int, 
    M: int, 
    Nl: int, 
    S0: float, 
    K: float, 
    r: float, 
    sigma: float, 
    T: float, 
    d: int = 100,
    rho: float = 0.0
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    模拟特定层级 l 的100维资产路径和篮子期权收益
    
    参数:
    l: 当前层级
    M: 精度倍增因子
    Nl: 模拟次数
    S0: 每个资产的初始价格
    K: 行权价
    r: 无风险利率
    sigma: 每个资产的波动率
    T: 到期时间
    d: 资产数量
    rho: 资产间相关系数
    
    返回:
    cost: 总计算成本
    variance: 本层级差值的方差
    Pfine: 精细层级收益
    Pcoarse: 粗糙层级收益
    """
    
    Nfine = M ** l  # 精细层级的时间步数
    dt_fine = T / Nfine
    
    # 初始化收益数组
    Pfine = np.zeros(Nl)
    Pcoarse = np.zeros(Nl)
    
    # 生成相关矩阵（简化处理，假设所有资产间相关系数相同）
    if rho == 0.0:
        # 资产独立的情况
        corr_matrix = np.eye(d)
    else:
        # 资产相关的情况
        corr_matrix = np.full((d, d), rho)
        np.fill_diagonal(corr_matrix, 1.0)
    
    # 对相关矩阵进行Cholesky分解
    try:
        L = np.linalg.cholesky(corr_matrix)
    except np.linalg.LinAlgError:
        # 如果Cholesky分解失败，使用特征值分解
        eigenvalues, eigenvectors = np.linalg.eigh(corr_matrix)
        eigenvalues[eigenvalues < 1e-10] = 1e-10  # 避免负特征值
        L = eigenvectors @ np.diag(np.sqrt(eigenvalues))
    
    for i in range(Nl):
        # 生成独立的随机数
        Z_fine = np.random.standard_normal((d, Nfine))
        
        # 应用相关性
        if rho != 0.0:
            Z_fine = L @ Z_fine
        
        # 生成布朗运动增量
        dW_fine = np.sqrt(dt_fine) * Z_fine
        
        # 计算布朗路径（累积和）
        W_fine = np.zeros((d, Nfine + 1))
        W_fine[:, 1:] = np.cumsum(dW_fine, axis=1)
        
        # 时间点
        t_values = np.linspace(0, T, Nfine + 1)
        
        # 计算每个资产的价格路径（向量化计算）
        drift = (r - 0.5 * sigma**2) * t_values
        diffusion = sigma * W_fine
        
        # 为所有资产同时计算价格
        S_fine = S0 * np.exp(drift + diffusion)
        
        # 计算篮子价格（等权重平均值）
        basket_price_fine = np.mean(S_fine[:, -1])
        
        # 计算篮子期权收益
        Pfine[i] = max(basket_price_fine - K, 0.0)
        
        if l > 1:  # 如果存在更粗糙的层级
            Ncoarse = M ** (l-1)
            dt_coarse = T / Ncoarse
            
            # 从精细路径中抽取对应粗糙时间点的路径（实现耦合）
            indices = np.arange(0, Nfine + 1, M, dtype=int)
            W_coarse = W_fine[:, indices]
            
            # 粗糙层级的时间点
            t_coarse = np.linspace(0, T, Ncoarse + 1)
            
            # 计算粗糙层级的资产价格
            drift_coarse = (r - 0.5 * sigma**2) * t_coarse
            
            # 为所有资产同时计算粗糙层级价格
            diffusion_coarse = sigma * W_coarse
            S_coarse = S0 * np.exp(drift_coarse + diffusion_coarse)
            
            # 计算粗糙层级的篮子价格
            basket_price_coarse = np.mean(S_coarse[:, -1])
            
            # 计算粗糙层级的篮子期权收益
            Pcoarse[i] = max(basket_price_coarse - K, 0.0)
        else:
            Pcoarse[i] = 0.0  # 最粗糙层级没有更粗的一级
    
    # 计算本层级差值的方差
    if l == 1:
        Y = Pfine  # 最粗糙层级的"差值"就是其本身
    else:
        Y = Pfine - Pcoarse
    
    variance = np.var(Y) if len(Y) > 1 else 0.0
    
    # 计算成本：假设成本与资产数量d和时间步数Nfine成正比
    cost = Nl * Nfine * d
    
    return cost, variance, Pfine, Pcoarse


def test_basket_option():
    """
    测试100维篮子期权定价
    """
    print("测试1: 独立资产篮子期权定价")
    print("-" * 40)
    
    # 测试参数
    S0 = 100.0
    K = 100.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    d = 100
    M = 2
    Lmin = 2
    Lmax = 6
    eps_target = 0.005
    
    start_time = time.time()
    
    price, cost, Nl_list, Vl_list = adaptive_mlmc_100d_basket_call(
        S0=S0, K=K, r=r, sigma=sigma, T=T, d=d,
        M=M, Lmin=Lmin, Lmax=Lmax, eps_target=eps_target,
        rho=0.0  # 独立资产
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"计算时间: {elapsed_time:.2f} 秒")
    print(f"各层级模拟次数: {Nl_list}")
    print(f"各层级方差估计: {[f'{v:.2e}' for v in Vl_list]}")
    
    # 使用标准蒙特卡洛作为基准比较
    print("\n" + "=" * 60)
    print("基准: 标准蒙特卡洛 (10^5次模拟)")
    print("=" * 60)
    
    N_mc = 100000
    Z = np.random.standard_normal((d, N_mc))
    ST = S0 * np.exp((r - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * Z)
    basket_prices = np.mean(ST, axis=0)
    payoffs = np.maximum(basket_prices - K, 0)
    mc_price = np.exp(-r * T) * np.mean(payoffs)
    mc_std = np.exp(-r * T) * np.std(payoffs) / np.sqrt(N_mc)
    
    print(f"标准蒙特卡洛价格: {mc_price:.6f} ± {1.96 * mc_std:.6f} (95% 置信区间)")
    print(f"MLMC价格: {price:.6f}")
    print(f"差异: {abs(price - mc_price):.6f} ({abs(price - mc_price)/mc_std:.2f} 倍标准误差)")
    
    return price, mc_price, mc_std


if __name__ == "__main__":
    # 运行测试
    test_basket_option()
