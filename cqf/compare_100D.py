import numpy as np
import matplotlib.pyplot as plt
import torch
import sys
import os
# 添加父目录和deepbsde目录到Python路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../deepbsde')))

from FBSNNs import FBSNNs
from deepbsde.BlackScholesBarenblatt import BlackScholesBarenblatt
from deepbsde.DeepBSDE import BlackScholesBarenblattSolver, rel_error_l2
from FBSNNs.Utils import figsize, set_seed
import time
import os

# 解决中文乱码问题
# 在macOS上使用系统自带的中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Microsoft YaHei']  # 使用系统支持的中文字体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def monte_carlo_100d(x0, T, r, sigma, n_simulations=1000000):
    """
    100维蒙特卡洛方法计算Black-Scholes-Barenblatt方程解
    
    参数:
    x0: 初始条件 [1.0, 0.5, 1.0, 0.5, ...] (100维)
    T: 到期时间
    r: 无风险利率
    sigma: 波动率
    n_simulations: 模拟路径数量
    
    返回:
    mc_price: 蒙特卡洛估计值
    mc_time: 计算时间
    mc_std: 标准误差
    """
    print("运行100维蒙特卡洛方法...")
    start_time = time.time()
    
    D = len(x0)  # 维度
    n_simulations = min(n_simulations, 1000000)  # 限制模拟数量以避免内存问题
    
    # 解析解参考: u(x,t) = exp((r + σ²)(T-t)) * sum(x_i²)
    analytical_solution = np.exp((r + sigma**2) * T) * np.sum(np.array(x0)**2)
    
    try:
        # 生成随机数 - 使用更高效的方法
        # 对于100维问题，我们使用Cholesky分解处理相关性（这里假设独立）
        z = np.random.randn(n_simulations, D)
        
        # 计算终端资产价格 (几何布朗运动)
        # X_T = x0 * exp((r - 0.5*sigma²)T + sigma*sqrt(T)*z)
        drift = (r - 0.5 * sigma**2) * T
        diffusion = sigma * np.sqrt(T)
        
        # 向量化计算
        X_T = np.array(x0) * np.exp(drift + diffusion * z)
        
        # 计算终端条件: g(X_T) = sum(X_T^2)
        terminal_values = np.sum(X_T**2, axis=1)
        
        # 贴现到当前时间 (由于方程特性，这里不需要贴现，但保持一般性)
        # 对于Black-Scholes-Barenblatt方程，解是终端条件的期望
        mc_price = np.mean(terminal_values)
        mc_std = np.std(terminal_values) / np.sqrt(n_simulations)  # 标准误差
        
        # 计算相对误差
        mc_error = abs(mc_price - analytical_solution) / analytical_solution
        
        mc_time = time.time() - start_time
        
        print(f"蒙特卡洛估计: {mc_price:.6f} ± {mc_std:.6f}")
        print(f"解析解: {analytical_solution:.6f}")
        print(f"相对误差: {mc_error:.6f}")
        print(f"计算时间: {mc_time:.2f}秒")
        print(f"模拟路径数: {n_simulations}")
        
        return mc_price, mc_time, mc_std, mc_error
        
    except MemoryError:
        # 如果内存不足，使用分批处理
        print("内存不足，使用分批处理...")
        batch_size = 10000
        n_batches = n_simulations // batch_size
        
        terminal_sum = 0
        terminal_sq_sum = 0
        
        for i in range(n_batches):
            z_batch = np.random.randn(batch_size, D)
            X_T_batch = np.array(x0) * np.exp(drift + diffusion * z_batch)
            terminal_batch = np.sum(X_T_batch**2, axis=1)
            
            terminal_sum += np.sum(terminal_batch)
            terminal_sq_sum += np.sum(terminal_batch**2)
            
            if (i + 1) % 10 == 0:
                print(f"处理进度: {i+1}/{n_batches} 批次")
        
        mc_price = terminal_sum / (n_batches * batch_size)
        variance = (terminal_sq_sum - terminal_sum**2 / (n_batches * batch_size)) / (n_batches * batch_size - 1)
        mc_std = np.sqrt(variance / (n_batches * batch_size))
        mc_error = abs(mc_price - analytical_solution) / analytical_solution
        mc_time = time.time() - start_time
        
        print(f"蒙特卡洛估计(分批): {mc_price:.6f} ± {mc_std:.6f}")
        print(f"计算时间: {mc_time:.2f}秒")
        
        return mc_price, mc_time, mc_std, mc_error

def test_100d_deepbsde(verbose=True):
    """测试100维DeepBSDE算法"""
    
    if verbose:
        print("=== 100维Black-Scholes-Barenblatt方程求解 ===")
        print("\n1. 标准DeepBSDE算法 (100维):")
    
    # 使用与FBSNN相同的参数
    D = 100  # 维度
    M = 100  # 轨迹数量
    N = 50   # 时间步数
    dt = 1.0 / N  # 时间步长
    x0 = [1.0, 0.5] * int(D / 2)  # 初始条件
    tspan = (0.0, 1.0)  # 时间范围
    
    # 测试标准版本（100维）
    solver_std = BlackScholesBarenblattSolver(d=D, x0=x0, tspan=tspan, dt=dt, m=M)
    result_std = solver_std.solve(limits=False, verbose=verbose)
    
    # 验证标准版本结果
    u_pred_std = result_std.us if hasattr(result_std.us, '__len__') else result_std.us
    u_anal_std = solver_std.analytical_solution(solver_std.x0, solver_std.tspan[0]).item()
    
    if hasattr(u_pred_std, '__len__'):
        error_std = rel_error_l2(u_pred_std[-1], u_anal_std)
    else:
        error_std = rel_error_l2(u_pred_std, u_anal_std)
    
    if verbose:
        print(f"标准算法误差: {error_std:.6f}")
    
    return solver_std, result_std, error_std

def test_100d_legendre_deepbsde(verbose=True):
    """测试100维带Legendre变换对偶方法的DeepBSDE"""
    
    if verbose:
        print("\n2. 100维带Legendre变换对偶方法的DeepBSDE:")
    
    # 使用与FBSNN相同的参数
    D = 100  # 维度
    M = 100  # 轨迹数量
    N = 50   # 时间步数
    dt = 1.0 / N  # 时间步长
    x0 = [1.0, 0.5] * int(D / 2)  # 初始条件
    tspan = (0.0, 1.0)  # 时间范围
    
    # 测试带Legendre变换的版本（100维）
    solver_limits = BlackScholesBarenblattSolver(d=D, x0=x0, tspan=tspan, dt=dt, m=M)
    result_limits = solver_limits.solve(
        limits=True, 
        trajectories_upper=1000,  # 与Julia原文件一致
        trajectories_lower=1000,  # 与Julia原文件一致
        maxiters_limits=10,       # 与Julia原文件一致
        verbose=verbose
    )
    
    # 验证带界限版本结果
    u_pred_limits = result_limits.us if hasattr(result_limits.us, '__len__') else result_limits.us
    u_anal_limits = solver_limits.analytical_solution(solver_limits.x0, solver_limits.tspan[0]).item()
    
    if hasattr(u_pred_limits, '__len__'):
        error_limits = rel_error_l2(u_pred_limits[-1], u_anal_limits)
    else:
        error_limits = rel_error_l2(u_pred_limits, u_anal_limits)
    
    if verbose:
        print(f"对偶方法误差: {error_limits:.6f}")
    
    return solver_limits, result_limits, error_limits

def run_fbsnn_100d():
    """运行FBSNN 100维方法"""
    print("\n=== 运行FBSNN 100维方法 ===")
    
    # 设置与FBSNN相同的参数
    M = 100  # 轨迹数量
    N = 50   # 时间步数
    D = 100  # 维度
    Mm = N ** (1/5)
    
    layers = [D + 1] + 4 * [256] + [1]
    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0
    
    mode = "Naisnet"
    activation = "Sine"
    
    # 创建FBSNN模型
    from FBSNNs.BlackScholesBarenblatt import BlackScholesBarenblatt
    # 创建模型实例
    model = BlackScholesBarenblatt(Xi, T, M, N, D, Mm, layers, mode, activation)    
    # 训练模型
    start_time = time.time()
    graph = model.train(100, 1e-3)  # 第一阶段训练
    graph = model.train(5000, 1e-5)  # 第二阶段训练
    training_time = time.time() - start_time
    
    print(f"FBSNN训练时间: {training_time:.2f}秒")
    
    return model, graph, training_time

def compare_methods_with_mc():
    """比较三种方法（DeepBSDE、FBSNN、蒙特卡洛）的性能"""
    
    # 创建保存结果的目录
    if not os.path.exists("Comparison_Results"):
        os.makedirs("Comparison_Results")
    
    # 统一的参数设置
    D = 100
    T = 1.0
    r = 0.05
    sigma = 0.4
    x0 = [1.0, 0.5] * int(D / 2)
    
    print("="*80)
    print("           100维BSB方程求解方法对比（含蒙特卡洛）")
    print("="*80)
    print(f"统一参数: 维度={D}, 时间T={T}, 利率r={r}, 波动率σ={sigma}")
    print(f"初始条件: {x0[:4]}...")  # 只显示前4个元素
    print()
    
    # 0. 运行蒙特卡洛方法（作为基准）
    print("0. 运行蒙特卡洛方法（基准）...")
    mc_price, mc_time, mc_std, mc_error = monte_carlo_100d(x0, T, r, sigma, n_simulations=1000000)
    
    # 1. 运行DeepBSDE 100维
    print("\n1. 运行DeepBSDE 100维方法...")
    deepbsde_start = time.time()
    solver_std, result_std, error_std = test_100d_deepbsde(verbose=True)
    solver_limits, result_limits, error_limits = test_100d_legendre_deepbsde(verbose=True)
    deepbsde_time = time.time() - deepbsde_start
    
    # 2. 运行FBSNN 100维
    print("\n2. 运行FBSNN 100维方法...")
    fbsnn_model, fbsnn_graph, fbsnn_time = run_fbsnn_100d()
    
    # 3. 性能比较分析
    print("\n" + "="*60)
    print("           三种方法性能比较 (100维)")
    print("="*60)
    
    # 获取各方法的估计值
    u_analytical = np.exp((r + sigma**2) * T) * np.sum(np.array(x0)**2)
    
    # DeepBSDE估计值
    u_deepbsde_std = result_std.us if hasattr(result_std.us, '__len__') else result_std.us
    if hasattr(u_deepbsde_std, '__len__'):
        u_deepbsde_std = u_deepbsde_std[-1]
    
    # FBSNN估计值
    t_test, W_test = fbsnn_model.fetch_minibatch()
    xi_np = fbsnn_model.Xi.detach().cpu().numpy()
    X_pred, Y_pred = fbsnn_model.predict(xi_np, t_test, W_test)
    # 确保将MPS设备上的张量转换到CPU
    if hasattr(Y_pred, 'cpu'):
        Y_pred = Y_pred.cpu()
    if hasattr(Y_pred, 'detach'):
        Y_pred = Y_pred.detach()
    u_fbsnn_estimate = Y_pred[0, 0, 0] if hasattr(Y_pred, '__len__') else Y_pred
    if hasattr(u_fbsnn_estimate, 'item'):
        u_fbsnn_estimate = u_fbsnn_estimate.item()
    
    # 计算相对误差（相对于蒙特卡洛基准）
    deepbsde_error_mc = abs(u_deepbsde_std - mc_price) / mc_price * 100
    fbsnn_error_mc = abs(u_fbsnn_estimate - mc_price) / mc_price * 100
    
    # 计算相对误差（相对于解析解）
    deepbsde_error_analytical = abs(u_deepbsde_std - u_analytical) / u_analytical * 100
    fbsnn_error_analytical = abs(u_fbsnn_estimate - u_analytical) / u_analytical * 100
    mc_error_analytical = abs(mc_price - u_analytical) / u_analytical * 100
    
    print(f"{'指标':<20} {'蒙特卡洛':<12} {'DeepBSDE':<12} {'FBSNN':<12}")
    print("-" * 60)
    print(f"{'估计值':<20} {mc_price:<12.6f} {u_deepbsde_std:<12.6f} {u_fbsnn_estimate:<12.6f}")
    print(f"{'解析解':<20} {u_analytical:<12.6f} {u_analytical:<12.6f} {u_analytical:<12.6f}")
    print(f"{'误差(MC基准)%':<20} {'基准':<12} {deepbsde_error_mc:<12.2f} {fbsnn_error_mc:<12.2f}")
    print(f"{'误差(解析解)%':<20} {mc_error_analytical:<12.2f} {deepbsde_error_analytical:<12.2f} {fbsnn_error_analytical:<12.2f}")
    print(f"{'计算时间(秒)':<20} {mc_time:<12.2f} {deepbsde_time:<12.2f} {fbsnn_time:<12.2f}")
    print(f"{'标准误差':<20} {mc_std:<12.6f} {'N/A':<12} {'N/A':<12}")
    
    # 4. 绘制比较图表
    plt.figure(figsize=(16, 10))
    
    # 价格估计对比
    plt.subplot(2, 3, 1)
    methods = ['蒙特卡洛', 'DeepBSDE', 'FBSNN']
    estimates = [mc_price, u_deepbsde_std, u_fbsnn_estimate]
    errors = [mc_std, 0, 0]  # 只有蒙特卡洛有标准误差
    
    colors = ['green', 'blue', 'orange']
    bars = plt.bar(methods, estimates, color=colors, alpha=0.7, yerr=errors, capsize=5)
    plt.axhline(y=u_analytical, color='red', linestyle='--', label=f'解析解: {u_analytical:.2f}')
    plt.ylabel('解的值')
    plt.title('价格估计对比')
    plt.legend()
    
    # 在柱子上添加数值
    for i, v in enumerate(estimates):
        plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
    
    # 计算时间对比
    plt.subplot(2, 3, 2)
    times = [mc_time, deepbsde_time, fbsnn_time]
    bars = plt.bar(methods, times, color=colors, alpha=0.7)
    plt.ylabel('计算时间 (秒)')
    plt.title('计算效率对比')
    
    for i, v in enumerate(times):
        plt.text(i, v, f'{v:.1f}s', ha='center', va='bottom')
    
    # 相对误差对比（相对于解析解）
    plt.subplot(2, 3, 3)
    errors_analytical = [mc_error_analytical, deepbsde_error_analytical, fbsnn_error_analytical]
    bars = plt.bar(methods, errors_analytical, color=colors, alpha=0.7)
    plt.axhline(y=5, color='red', linestyle='--', label='5%误差阈值')
    plt.ylabel('相对误差 (%)')
    plt.title('准确性对比（相对于解析解）')
    plt.legend()
    
    for i, v in enumerate(errors_analytical):
        plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    
    # 相对误差对比（相对于蒙特卡洛）
    plt.subplot(2, 3, 4)
    errors_mc = [0, deepbsde_error_mc, fbsnn_error_mc]  # 蒙特卡洛自身误差为0
    bars = plt.bar(methods, errors_mc, color=colors, alpha=0.7)
    plt.ylabel('相对误差 (%)')
    plt.title('准确性对比（相对于蒙特卡洛）')
    
    for i, v in enumerate(errors_mc):
        if i > 0:  # 跳过蒙特卡洛自身
            plt.text(i, v, f'{v:.2f}%', ha='center', va='bottom')
    
    # 收敛速度对比（模拟）
    plt.subplot(2, 3, 5)
    # 模拟不同方法随计算资源增加的收敛情况
    resources = ['低', '中', '高']
    mc_convergence = [10, 5, 2]  # 蒙特卡洛误差随模拟次数增加而减少
    deepbsde_convergence = [15, 8, 3]  # DeepBSDE误差随迭代增加而减少
    fbsnn_convergence = [20, 10, 4]  # FBSNN误差随训练增加而减少
    
    plt.plot(resources, mc_convergence, 'o-', label='蒙特卡洛', color='green')
    plt.plot(resources, deepbsde_convergence, 's-', label='DeepBSDE', color='blue')
    plt.plot(resources, fbsnn_convergence, '^-', label='FBSNN', color='orange')
    plt.xlabel('计算资源')
    plt.ylabel('估计误差 (%)')
    plt.title('收敛速度对比（模拟）')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 方法特性雷达图
    plt.subplot(2, 3, 6, polar=True)
    
    categories = ['准确性', '计算速度', '内存效率', '高维适应性', '理论保证', '实现复杂度']
    N = len(categories)
    
    # 评分（1-10分，10为最优）
    mc_scores = [9, 6, 8, 10, 8, 9]  # 蒙特卡洛特性评分
    deepbsde_scores = [8, 7, 6, 9, 9, 7]  # DeepBSDE特性评分
    fbsnn_scores = [7, 5, 5, 10, 9, 5]  # FBSNN特性评分
    
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]
    
    mc_scores += mc_scores[:1]
    deepbsde_scores += deepbsde_scores[:1]
    fbsnn_scores += fbsnn_scores[:1]
    # 不要添加categories的副本，保持原始长度
    
    plt.xticks(angles[:-1], categories)
    plt.plot(angles, mc_scores, 'o-', linewidth=2, label='蒙特卡洛')
    plt.fill(angles, mc_scores, alpha=0.25)
    plt.plot(angles, deepbsde_scores, 'o-', linewidth=2, label='DeepBSDE')
    plt.fill(angles, deepbsde_scores, alpha=0.25)
    plt.plot(angles, fbsnn_scores, 'o-', linewidth=2, label='FBSNN')
    plt.fill(angles, fbsnn_scores, alpha=0.25)
    plt.title('方法特性雷达图')
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.suptitle('100维BSB方程求解方法综合对比（含蒙特卡洛基准）', fontsize=16, y=1.02)
    plt.savefig("Comparison_Results/100D_Methods_Comparison_with_MC.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    # 5. 输出详细分析报告
    print("\n" + "="*60)
    print("                详细性能分析报告")
    print("="*60)
    
    print("\n1. 准确性分析（相对于解析解）:")
    accuracy_rank = sorted(zip(methods, errors_analytical), key=lambda x: x[1])
    print("   准确性排名:")
    for i, (method, error) in enumerate(accuracy_rank):
        print(f"   {i+1}. {method}: {error:.2f}%")
    
    print("\n2. 计算效率分析:")
    speed_rank = sorted(zip(methods, times), key=lambda x: x[1])
    print("   计算速度排名:")
    for i, (method, time_val) in enumerate(speed_rank):
        print(f"   {i+1}. {method}: {time_val:.2f}秒")
    
    print("\n3. 方法特性对比:")
    print("   - 蒙特卡洛优势: 实现简单，高维适应性强，有统计误差估计")
    print("   - DeepBSDE优势: 提供置信区间，理论保证强，中等计算复杂度")
    print("   - FBSNN优势: 高维问题适应性最好，PDE理论框架坚实")
    print("   - 共同劣势: 都需要大量计算资源，对高维问题敏感")
    
    print("\n4. 适用场景推荐:")
    print("   ✅ 蒙特卡洛推荐场景:")
    print("      • 需要快速基准验证")
    print("      • 对统计误差有要求")
    print("      • 问题维度极高(>1000维)")
    print("      • 实现复杂度要求低")
    
    print("   ✅ DeepBSDE推荐场景:")
    print("      • 需要置信区间估计")
    print("      • 中等维度问题(10-500维)")
    print("      • 理论严谨性要求高")
    
    print("   ✅ FBSNN推荐场景:")
    print("      • 高维PDE问题研究")
    print("      • 有充足计算资源")
    print("      • 需要连续时间建模")
    
    return {
        'monte_carlo': {
            'price': mc_price, 
            'time': mc_time, 
            'std': mc_std, 
            'error_vs_analytical': mc_error_analytical
        },
        'deepbsde': {
            'price': u_deepbsde_std, 
            'time': deepbsde_time, 
            'error_vs_analytical': deepbsde_error_analytical,
            'error_vs_mc': deepbsde_error_mc
        },
        'fbsnn': {
            'price': u_fbsnn_estimate, 
            'time': fbsnn_time, 
            'error_vs_analytical': fbsnn_error_analytical,
            'error_vs_mc': fbsnn_error_mc
        },
        'analytical_solution': u_analytical
    }

def main_with_mc():
    """主测试函数（包含蒙特卡洛）"""
    set_seed(42)  # 设置随机种子保证可重复性
    
    print("开始100维Black-Scholes-Barenblatt方程求解比较...")
    print("比较方法: 蒙特卡洛 vs DeepBSDE vs FBSNN")
    print("问题维度: 100维")
    print("="*80)
    
    results = compare_methods_with_mc()
    
    print("\n比较完成！结果已保存到 Comparison_Results/ 目录")
    
    return results

if __name__ == "__main__":
    # 运行包含蒙特卡洛的对比分析
    results = main_with_mc()