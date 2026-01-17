import numpy as np
import matplotlib.pyplot as plt
from cqf.deepbsde.BlackScholesBarenblatt import BlackScholesBarenblatt
from cqf.deepbsde.DeepBSDE import BlackScholesBarenblattSolver, rel_error_l2



def test_standard_deepbsde(d=30, verbose=True):
    """测试标准DeepBSDE算法
    
    参数:
    d: 问题维度，默认为30
    verbose: 是否打印详细信息，默认为True
    
    返回:
    solver_std: 标准算法求解器
    result_std: 标准算法求解结果
    error_std: 标准算法误差
    """
    if verbose:
        print("=== 30维Black-Scholes-Barenblatt方程求解 ===")
        print("\n1. 标准DeepBSDE算法:")
    
    # 测试标准版本（limits=false）
    solver_std = BlackScholesBarenblattSolver(d=d)
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


def test_legendre_deepbsde(d=30, verbose=True):
    """测试带Legendre变换对偶方法的DeepBSDE
    
    参数:
    d: 问题维度，默认为30
    verbose: 是否打印详细信息，默认为True
    
    返回:
    solver_limits: 带Legendre变换的求解器
    result_limits: 带Legendre变换的求解结果
    error_limits: 带Legendre变换的求解误差
    """
    if verbose:
        print("\n2. 带Legendre变换对偶方法的DeepBSDE:")
    
    # 测试带Legendre变换的版本（limits=true）
    solver_limits = BlackScholesBarenblattSolver(d=d)
    result_limits = solver_limits.solve(
        limits=True, 
        trajectories_upper=1000,  # 与Julia原文件一致 (之前: 100)
        trajectories_lower=1000,  # 与Julia原文件一致 (之前: 100)
        maxiters_limits=10,       # 与Julia原文件一致 (之前: 5)
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




if __name__ == "__main__":
    # 调用两个测试方法
    solver_std, result_std, error_std = test_standard_deepbsde(verbose=True)
    solver_limits, result_limits, error_limits = test_legendre_deepbsde(verbose=True)
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.semilogy(solver_std.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (Standard)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 2)
    plt.semilogy(solver_limits.losses)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss (With Limits)')
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 3, 3)
    if hasattr(result_limits, 'limits') and result_limits.limits is not None:
        u_low, u_high = result_limits.limits
        if hasattr(result_limits.us, '__len__'):
            u_point = result_limits.us[-1]
        else:
            u_point = result_limits.us
        u_anal = solver_limits.analytical_solution(solver_limits.x0, solver_limits.tspan[0]).item()
        
        plt.axhline(y=u_anal, color='green', linestyle='--', label='Analytical', alpha=0.7)
        plt.axhline(y=u_point, color='blue', linestyle='-', label='Point Estimate', alpha=0.7)
        plt.axhspan(u_low, u_high, alpha=0.3, color='red', label='Confidence Interval')
        plt.ylabel('Solution Value')
        plt.title('Solution with Bounds')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # 创建Figures目录（如果不存在）
    import os
    figures_dir = "Figures"
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)
    
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "DeepBSDE_standard_vs_Legendre.png"))
    plt.close()
    
    # 验证结果
    print(f"\n验证结果:")
    print(f"标准算法误差: {error_std:.6f} {'✓ < 1.0' if error_std < 1.0 else '✗ >= 1.0'}")
    print(f"对偶方法误差: {error_limits:.6f} {'✓ < 1.0' if error_limits < 1.0 else '✗ >= 1.0'}")
    
    if hasattr(result_limits, 'limits') and result_limits.limits is not None:
        u_low, u_high = result_limits.limits
        if hasattr(result_limits.us, '__len__'):
            u_point = result_limits.us[-1]
        else:
            u_point = result_limits.us
        if u_low <= u_point <= u_high:
            print("✓ 点估计在上下界范围内")
        else:
            print("✗ 点估计超出上下界范围")
