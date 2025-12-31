import numpy as np
import matplotlib.pyplot as plt

def adaptive_mlmc_european_call(S0, K, r, sigma, T, M, Lmin, Lmax, eps_target):
    """
    自适应多级蒙特卡洛方法定价欧式看涨期权

    参数:
    S0: 初始股价
    K: 行权价
    r: 无风险利率
    sigma: 波动率
    T: 到期时间
    M: 层级精度倍增因子（例如，M=2表示每一级时间步翻倍）
    Lmin: 最粗糙层级
    Lmax: 最精细层级限制
    eps_target: 目标均方根误差 (RMSE)

    返回:
    price: 期权价格估计
    computational_cost: 总计算成本
    """
    # 初始化数组 (注意Python索引从0开始)
    Vl = np.zeros(Lmax) # 存储各层级方差估计
    Cl = np.zeros(Lmax) # 存储各层级单次模拟成本
    Nl = np.zeros(Lmax, dtype=int) # 存储各层级模拟次数
    # 用于统计: [P0, Pl, Pl-1] (Python中索引调整)
    sums = np.zeros((Lmax, 3))

    computational_cost = 0

    # 初始运行少量模拟，以估计各级别方差和成本
    print('初步估计各级别方差和成本...')
    for l in range(Lmin-1, Lmax): # 注意Python循环范围 (l从Lmin到Lmax-1)
        Np = 1000 # 初始用于估计的模拟次数
        cost_temp, variance_temp, Pfine, Pcoarse = mlmc_level_l(l+1, M, Np, S0, K, r, sigma, T) # l+1因索引差异
        computational_cost += cost_temp
        Vl[l] = variance_temp
        Cl[l] = cost_temp / Np
        if l == Lmin-1: # 最粗糙层级
            sums[l, 0] = sums[l, 0] + np.sum(Pfine)
            sums[l, 1] = sums[l, 1] + np.sum(Pfine)
        else:
            sums[l, 0] = sums[l, 0] + np.sum(Pfine - Pcoarse) # 估计差值 E[Pl - Pl-1]
            sums[l, 1] = sums[l, 1] + np.sum(Pfine)
            sums[l, 2] = sums[l, 2] + np.sum(Pcoarse)
        print(f'层级 l={l+1}, 估计方差 Vl={Vl[l]:.4e}, 单次成本 Cl={Cl[l]:.4e}')

    # 自适应确定最优模拟次数 Nl
    print('\n根据目标误差自适应分配模拟次数...')
    theta = 0.95 # 分配给粗糙层级的误差权重（例如95%）
    max_iter = 10 # 迭代次数限制，防止无限循环

    for iter in range(1, max_iter+1):
        total_variance = 0.0
        for l in range(Lmin-1, Lmax):
            total_variance += np.sqrt(Vl[l] * Cl[l])
        
        for l in range(Lmin-1, Lmax):
            # 根据方差和成本，使用MLMC最优分配公式计算Nl
            Nl[l] = int(np.ceil((1.0 / (eps_target**2)) * np.sqrt(Vl[l] / Cl[l]) * total_variance / theta))

        # 运行新增的模拟次数
        for l in range(Lmin-1, Lmax):
            if Nl[l] > 0:
                cost_temp, variance_temp, Pfine, Pcoarse = mlmc_level_l(l+1, M, Nl[l], S0, K, r, sigma, T)
                computational_cost += cost_temp
                # 更新方差估计 (简化处理，实际中可能需要更复杂的合并)
                Vl[l] = (Vl[l] + variance_temp) / 2.0
                if l == Lmin-1:
                    sums[l, 0] = sums[l, 0] + np.sum(Pfine)
                    sums[l, 1] = sums[l, 1] + np.sum(Pfine)
                else:
                    sums[l, 0] = sums[l, 0] + np.sum(Pfine - Pcoarse)
                    sums[l, 1] = sums[l, 1] + np.sum(Pfine)
                    sums[l, 2] = sums[l, 2] + np.sum(Pcoarse)

        # 检查是否达到目标精度 (简化版，实际应计算置信区间)
        # 注意: 这里对Vl和Nl的索引处理需谨慎，示例中计算了所有层级的总体误差
        total_variance_estimate = 0.0
        total_samples = 0
        for l in range(Lmin-1, Lmax):
            if Nl[l] > 0:
                total_variance_estimate += Vl[l] / Nl[l]
                total_samples += Nl[l]
        if total_samples > 0:
            current_eps = np.sqrt(total_variance_estimate)
        else:
            current_eps = eps_target + 1 # 确保继续迭代

        print(f'迭代 {iter}: 当前估计误差 = {current_eps:.4e}')
        if current_eps < eps_target:
            print('已达到目标精度。')
            break

    # 计算最终的期权价格估计
    price = 0.0
    for l in range(Lmin-1, Lmax):
        if Nl[l] > 0: # 避免除以零
            if l == Lmin-1:
                price += sums[l, 0] / (Nl[l] + 1000) # 加上初始的1000次
            else:
                price += sums[l, 0] / (Nl[l] + 1000)
    price = np.exp(-r * T) * price # 折现

    print(f'\n最终期权价格估计: {price:.4f}')
    print(f'总计算成本 (任意单位): {computational_cost:.2f}')
    
    return price, computational_cost


def mlmc_level_l(l, M, Nl, S0, K, r, sigma, T):
    """
    模拟特定层级 l 的路径和收益

    参数:
    l: 当前层级 (从1开始计数，与MATLAB一致)
    M: 精度倍增因子
    Nl: 模拟次数
    S0: 初始股价
    K: 行权价
    r: 无风险利率
    sigma: 波动率
    T: 到期时间

    返回:
    cost: 总计算成本
    variance: 本层级差值的方差
    Pfine: 精细层级收益
    Pcoarse: 粗糙层级收益 (对于l=1，Pcoarse=0)
    """
    Nfine = M ** l # 精细层级的时间步数
    dt_fine = T / Nfine

    Pfine = np.zeros(Nl)
    Pcoarse = np.zeros(Nl)

    # 使用相同的随机数种子实现路径耦合 (简化方式)
    np.random.seed(42) # 固定种子用于演示，实际应用中可能需要更精细的控制

    for i in range(Nl):
        # 生成布朗运动路径增量 (使用正态分布)
        Z_fine = np.random.standard_normal(Nfine)
        # 构建布朗路径 (使用累积和)
        W_fine = np.zeros(Nfine + 1)
        W_fine[1:] = np.cumsum(np.sqrt(dt_fine) * Z_fine) # W[0] = 0

        # 计算几何布朗运动路径 (向量化计算)
        t_values = np.linspace(0, T, Nfine + 1)
        S_fine = S0 * np.exp((r - 0.5 * sigma**2) * t_values + sigma * W_fine)
        Pfine[i] = max(S_fine[-1] - K, 0.0)

        if l > 1: # 如果存在更粗糙的层级
            Ncoarse = M ** (l-1)
            dt_coarse = T / Ncoarse
            
            # 从精细路径中抽取对应粗糙时间点的路径（实现耦合）
            # 注意: 这里假设M为整数，且精细路径点数 = M * 粗糙路径点数
            indices = np.arange(0, Nfine + 1, M, dtype=int) # 抽取的索引
            W_coarse = W_fine[indices] # 耦合的粗糙路径
            
            t_coarse = np.linspace(0, T, Ncoarse + 1)
            S_coarse = S0 * np.exp((r - 0.5 * sigma**2) * t_coarse + sigma * W_coarse)
            Pcoarse[i] = max(S_coarse[-1] - K, 0.0)
        else:
            Pcoarse[i] = 0.0 # 最粗糙层级没有更粗的一级

    # 计算本层级差值的方差
    if l == 1:
        Y = Pfine # 最粗糙层级的"差值"就是其本身
    else:
        Y = Pfine - Pcoarse
    variance = np.var(Y)

    # 计算成本：假设成本与时间步数（即模拟路径的精细程度）成正比
    cost = Nl * Nfine

    return cost, variance, Pfine, Pcoarse


# 示例调用
if __name__ == "__main__":
    S0 = 100.0
    K = 110.0
    r = 0.05
    sigma = 0.2
    T = 1.0
    M = 2
    Lmin = 1
    Lmax = 5
    eps_target = 0.01

    price, cost = adaptive_mlmc_european_call(S0, K, r, sigma, T, M, Lmin, Lmax, eps_target)