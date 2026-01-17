import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from BlackScholesBarenblatt import BlackScholesBarenblatt
from DeepBSDE import BlackScholesBarenblattSolver, rel_error_l2
import time
import pandas as pd
from scipy import stats
from matplotlib.patches import Patch

# 解决中文乱码问题
# 在macOS上使用系统自带的中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Microsoft YaHei']  # 使用系统支持的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


# 从test.py中导入的函数
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


# 从cqf_2_deepbsde_6yaobaoTrue2.py中导入的辅助函数
def calculate_proper_error_bars(intervals, point_estimates):
    """计算合适的误差条 - 修复版本"""
    errors_lower = []
    errors_upper = []
    violations = 0
    
    for (low, high), u0 in zip(intervals, point_estimates):
        if u0 < low:
            # 点估计低于下界，调整显示
            errors_lower.append(low - u0)
            errors_upper.append(high - u0)
            violations += 1
        elif u0 > high:
            # 点估计高于上界
            errors_lower.append(u0 - low)
            errors_upper.append(u0 - high)
            violations += 1
        else:
            # 正常情况
            errors_lower.append(u0 - low)
            errors_upper.append(high - u0)
    
    if violations > 0:
        print(f"警告: {violations}个点估计超出置信区间")
    
    return errors_lower, errors_upper, violations


def plot_improved_confidence_intervals(metrics, analytical_value, ax):
    """改进的置信区间可视化 - 修复版本"""
    if not metrics.get('limits_intervals'):
        ax.text(0.5, 0.5, '无置信区间数据', 
                transform=ax.transAxes, ha='center', va='center')
        ax.set_title('置信区间可视化')
        return ax
    
    # 计算误差条
    errors_lower, errors_upper, violations = calculate_proper_error_bars(
        metrics['limits_intervals'], metrics['limits_u0']
    )
    
    # 确保误差条非负
    errors_lower = np.abs(errors_lower)
    errors_upper = np.abs(errors_upper)
    
    # 绘制误差条
    x_positions = range(len(metrics['limits_intervals']))
    
    # 使用不同的颜色标记违规点
    colors = ['red' if (u0 < low or u0 > high) else 'blue' 
             for (low, high), u0 in zip(metrics['limits_intervals'], metrics['limits_u0'])]
    
    # 分别绘制正常点和违规点
    normal_x = []
    normal_y = []
    normal_lower = []
    normal_upper = []
    
    violation_x = []
    violation_y = []
    violation_lower = []
    violation_upper = []
    
    for i, ((low, high), u0, color) in enumerate(zip(metrics['limits_intervals'], metrics['limits_u0'], colors)):
        if color == 'blue':
            normal_x.append(i)
            normal_y.append(u0)
            normal_lower.append(errors_lower[i])
            normal_upper.append(errors_upper[i])
        else:
            violation_x.append(i)
            violation_y.append(u0)
            violation_lower.append(errors_lower[i])
            violation_upper.append(errors_upper[i])
    
    # 绘制正常点
    if normal_x:
        ax.errorbar(
            normal_x, normal_y, 
            yerr=[normal_lower, normal_upper],
            fmt='o', capsize=5, color='blue', alpha=0.7,
            label='正常点'
        )
    
    # 绘制违规点
    if violation_x:
        ax.errorbar(
            violation_x, violation_y, 
            yerr=[violation_lower, violation_upper],
            fmt='o', capsize=5, color='red', alpha=0.7,
            label='违规点'
        )
    
    # 添加解析解参考线
    ax.axhline(y=analytical_value, color='green', linestyle='--', 
              label=f'解析解: {analytical_value:.2f}')
    
    # 添加置信区间背景
    for i, (low, high) in enumerate(metrics['limits_intervals']):
        ax.axhspan(low, high, alpha=0.1, color='gray')
    
    ax.set_xlabel('试验次数')
    ax.set_ylabel('u0估计值')
    ax.set_title('置信区间可视化（改进版）')
    
    # 添加统计信息
    coverage = sum(1 for (low, high), u0 in zip(metrics['limits_intervals'], metrics['limits_u0'])
                   if low <= u0 <= high) / len(metrics['limits_intervals']) * 100
    
    stats_text = f'覆盖率: {coverage:.1f}%\n违规点: {violations}个'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # 添加图例
    legend_elements = [
        Patch(facecolor='blue', alpha=0.7, label='正常点'),
        Patch(facecolor='red', alpha=0.7, label='违规点'),
        Patch(facecolor='gray', alpha=0.1, label='置信区间')
    ]
    ax.legend(handles=legend_elements)
    
    ax.grid(True, alpha=0.3)
    return ax


def plot_comprehensive_comparison(metrics, results_std, results_limits, d):
    """绘制综合对比图表 - 集成方案3改进"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # 1. 误差分布对比
    axes[0,0].boxplot([metrics['std_errors'], metrics['limits_errors']], 
                      labels=['标准方法', '对偶方法'])
    axes[0,0].set_ylabel('相对误差')
    axes[0,0].set_title('误差分布对比')
    axes[0,0].grid(True, alpha=0.3)
    
    # 2. 训练时间对比
    axes[0,1].boxplot([metrics['std_times'], metrics['limits_times']], 
                     labels=['标准方法', '对偶方法'])
    axes[0,1].set_ylabel('训练时间 (秒)')
    axes[0,1].set_title('计算效率对比')
    axes[0,1].grid(True, alpha=0.3)
    
    # 3. 解值稳定性
    axes[0,2].plot(metrics['std_u0'], 'bo-', label='标准方法', alpha=0.7)
    axes[0,2].plot(metrics['limits_u0'], 'ro-', label='对偶方法', alpha=0.7)
    analytical_value = results_std[0][0].analytical_solution(
        results_std[0][0].x0, results_std[0][0].tspan[0]).item()
    axes[0,2].axhline(y=analytical_value, color='green', linestyle='--', 
                     label=f'解析解: {analytical_value:.2f}')
    axes[0,2].set_xlabel('试验次数')
    axes[0,2].set_ylabel('u0估计值')
    axes[0,2].set_title('解值稳定性对比')
    axes[0,2].legend()
    axes[0,2].grid(True, alpha=0.3)
    
    # 4. 训练损失曲线对比（最后一次试验）
    if results_std and hasattr(results_std[-1][0], 'losses'):
        axes[1,0].semilogy(results_std[-1][0].losses, label='标准方法')
    if results_limits and hasattr(results_limits[-1][0], 'losses'):
        axes[1,0].semilogy(results_limits[-1][0].losses, label='对偶方法')
    axes[1,0].set_xlabel('迭代次数')
    axes[1,0].set_ylabel('损失值')
    axes[1,0].set_title('训练过程对比')
    axes[1,0].legend()
    axes[1,0].grid(True, alpha=0.3)
    
    # 5. 置信区间可视化 - 使用方案3改进方法
    if metrics.get('limits_intervals'):
        plot_improved_confidence_intervals(metrics, analytical_value, axes[1,1])
    else:
        axes[1,1].text(0.5, 0.5, '无置信区间数据', 
                      transform=axes[1,1].transAxes, ha='center', va='center')
        axes[1,1].set_title('置信区间可视化')
    
    # 6. 方法特性雷达图 - 修复：使用正确的极坐标设置
    # 首先创建一个极坐标轴
    fig.delaxes(axes[1,2])  # 删除原来的轴
    ax_radar = fig.add_subplot(2, 3, 6, projection='polar')  # 创建极坐标轴
    
    characteristics = ['准确性', '效率', '稳定性', '理论保证', '易用性', '适用性']
    std_scores = [8, 9, 7, 7, 8, 9]
    limits_scores = [7, 6, 8, 9, 6, 8]
    
    angles = np.linspace(0, 2*np.pi, len(characteristics), endpoint=False).tolist()
    angles += angles[:1]
    std_scores += std_scores[:1]
    limits_scores += limits_scores[:1]
    characteristics += characteristics[:1]
    
    ax_radar.plot(angles, std_scores, 'o-', linewidth=2, label='标准方法')
    ax_radar.fill(angles, std_scores, alpha=0.25)
    ax_radar.plot(angles, limits_scores, 'o-', linewidth=2, label='对偶方法')
    ax_radar.fill(angles, limits_scores, alpha=0.25)
    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(characteristics[:-1])
    ax_radar.set_title('方法特性雷达图')
    ax_radar.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    
    plt.tight_layout()
    plt.suptitle(f'{d}维Black-Scholes-Barenblatt方程求解方法对比', fontsize=16, y=1.02)
    plt.show()


def comprehensive_comparison(d=30, num_trials=3):
    """全面的方法对比分析 - 集成方案3改进"""
    print("=== DeepBSDE方法全面对比分析 ===")
    print(f"维度: {d}维, 试验次数: {num_trials}")
    print("=" * 60)
    
    results_std = []
    results_limits = []
    
    performance_metrics = {
        'std_errors': [], 'std_times': [], 'std_u0': [],
        'limits_errors': [], 'limits_times': [], 'limits_u0': [],
        'limits_lower': [], 'limits_upper': [], 'limits_intervals': []
    }
    
    for trial in range(num_trials):
        print(f"\n--- 试验 {trial + 1}/{num_trials} ---")
        
        # 测试标准方法
        start_time = time.time()
        solver_std = BlackScholesBarenblattSolver(d=d)
        result_std = solver_std.solve(limits=False, verbose=False)
        std_time = time.time() - start_time
        
        u_pred_std = result_std.us if hasattr(result_std.us, '__len__') else result_std.us
        u_anal_std = solver_std.analytical_solution(solver_std.x0, solver_std.tspan[0]).item()
        error_std = rel_error_l2(u_pred_std, u_anal_std)
        
        # 测试带Legendre变换方法
        start_time = time.time()
        solver_limits = BlackScholesBarenblattSolver(d=d)
        result_limits = solver_limits.solve(
            limits=True, 
            trajectories_upper=200,
            trajectories_lower=200,
            maxiters_limits=5,
            verbose=False
        )
        limits_time = time.time() - start_time
        
        u_pred_limits = result_limits.us if hasattr(result_limits.us, '__len__') else result_limits.us
        u_anal_limits = solver_limits.analytical_solution(solver_limits.x0, solver_limits.tspan[0]).item()
        error_limits = rel_error_l2(u_pred_limits, u_anal_limits)
        
        # 存储结果
        results_std.append((solver_std, result_std, error_std, std_time))
        results_limits.append((solver_limits, result_limits, error_limits, limits_time))
        
        # 存储性能指标
        performance_metrics['std_errors'].append(error_std)
        performance_metrics['std_times'].append(std_time)
        performance_metrics['std_u0'].append(u_pred_std)
        
        performance_metrics['limits_errors'].append(error_limits)
        performance_metrics['limits_times'].append(limits_time)
        performance_metrics['limits_u0'].append(u_pred_limits)
        
        if hasattr(result_limits, 'limits') and result_limits.limits is not None:
            u_low, u_high = result_limits.limits
            performance_metrics['limits_lower'].append(u_low)
            performance_metrics['limits_upper'].append(u_high)
            performance_metrics['limits_intervals'].append((u_low, u_high))
        
        print(f"标准方法 - 误差: {error_std:.6f}, 时间: {std_time:.2f}s")
        print(f"对偶方法 - 误差: {error_limits:.6f}, 时间: {limits_time:.2f}s")
        if hasattr(result_limits, 'limits') and result_limits.limits is not None:
            print(f"置信区间: [{u_low:.4f}, {u_high:.4f}]")
    
    # 性能统计分析
    print("\n" + "="*60)
    print("                性能对比分析结果")
    print("="*60)
    
    # 创建对比表格
    comparison_data = []
    
    # 准确性对比
    std_error_mean = np.mean(performance_metrics['std_errors'])
    std_error_std = np.std(performance_metrics['std_errors'])
    limits_error_mean = np.mean(performance_metrics['limits_errors'])
    limits_error_std = np.std(performance_metrics['limits_errors'])
    
    # 计算时间对比
    std_time_mean = np.mean(performance_metrics['std_times'])
    std_time_std = np.std(performance_metrics['std_times'])
    limits_time_mean = np.mean(performance_metrics['limits_times'])
    limits_time_std = np.std(performance_metrics['limits_times'])
    
    # 解值稳定性对比
    std_u0_std = np.std(performance_metrics['std_u0'])
    limits_u0_std = np.std(performance_metrics['limits_u0'])
    
    # 置信区间分析
    coverage = 0
    if performance_metrics['limits_intervals']:
        analytical_value = results_limits[0][0].analytical_solution(
            results_limits[0][0].x0, results_limits[0][0].tspan[0]).item()
        for interval in performance_metrics['limits_intervals']:
            if interval[0] <= analytical_value <= interval[1]:
                coverage += 1
        coverage_rate = coverage / len(performance_metrics['limits_intervals'])
        interval_widths = [interval[1] - interval[0] for interval in performance_metrics['limits_intervals']]
        avg_interval_width = np.mean(interval_widths) if interval_widths else 0
    else:
        coverage_rate = 0
        avg_interval_width = 0
    
    # 统计显著性检验
    if len(performance_metrics['std_errors']) > 1 and len(performance_metrics['limits_errors']) > 1:
        t_stat, p_value = stats.ttest_ind(performance_metrics['std_errors'], 
                                         performance_metrics['limits_errors'])
    else:
        t_stat, p_value = 0, 1.0
    
    # 输出详细对比表格
    print(f"\n{'指标':<25} {'标准方法':<15} {'对偶方法':<15} {'优劣分析':<20}")
    print("-" * 80)
    
    comparison_data.append([
        "平均相对误差", 
        f"{std_error_mean:.6f} ± {std_error_std:.6f}", 
        f"{limits_error_mean:.6f} ± {limits_error_std:.6f}",
        "✓ 标准方法更优" if std_error_mean < limits_error_mean else "✓ 对偶方法更优"
    ])
    
    comparison_data.append([
        "平均训练时间(s)", 
        f"{std_time_mean:.2f} ± {std_time_std:.2f}", 
        f"{limits_time_mean:.2f} ± {limits_time_std:.2f}",
        "✓ 标准方法更快" if std_time_mean < limits_time_mean else "✓ 对偶方法更快"
    ])
    
    comparison_data.append([
        "解值稳定性(标准差)", 
        f"{std_u0_std:.6f}", 
        f"{limits_u0_std:.6f}",
        "✓ 标准方法更稳定" if std_u0_std < limits_u0_std else "✓ 对偶方法更稳定"
    ])
    
    if performance_metrics['limits_intervals']:
        comparison_data.append([
            "置信区间覆盖率", 
            "N/A", 
            f"{coverage_rate*100:.1f}%",
            "✓ 良好" if coverage_rate >= 0.9 else "○ 一般" if coverage_rate >= 0.7 else "✗ 较差"
        ])
        
        comparison_data.append([
            "平均区间宽度", 
            "N/A", 
            f"{avg_interval_width:.4f}",
            "✓ 较窄" if avg_interval_width < 1.0 else "○ 适中" if avg_interval_width < 3.0 else "✗ 较宽"
        ])
    
    comparison_data.append([
        "统计显著性(p值)", 
        "N/A", 
        f"{p_value:.6f}",
        "✓ 显著差异" if p_value < 0.05 else "○ 无显著差异"
    ])
    
    for row in comparison_data:
        print(f"{row[0]:<25} {row[1]:<15} {row[2]:<15} {row[3]:<20}")
    
    # 绘制综合对比图表（包含方案3改进的置信区间可视化）
    plot_comprehensive_comparison(performance_metrics, results_std, results_limits, d)
    
    # 方法特性评分
    print("\n" + "="*60)
    print("                方法特性综合评分")
    print("="*60)
    
    characteristics = {
        '准确性': [max(0, 10 - std_error_mean*100), max(0, 10 - limits_error_mean*100)],
        '计算效率': [max(0, 10 - std_time_mean/10), max(0, 10 - limits_time_mean/10)],
        '数值稳定性': [max(0, 10 - std_u0_std*10), max(0, 10 - limits_u0_std*10)],
        '理论保证': [7, 9],
        '实现复杂度': [8, 6],
        '适用性': [9, 8]
    }
    
    if performance_metrics['limits_intervals']:
        characteristics['不确定性量化'] = [5, 8]
    
    methods = ['标准方法', '对偶方法']
    print(f"{'特性':<15} {'标准方法':<10} {'对偶方法':<10} {'推荐':<10}")
    print("-" * 50)
    
    for char, scores in characteristics.items():
        std_score, limits_score = scores
        recommendation = "标准方法" if std_score > limits_score else "对偶方法" if limits_score > std_score else "相当"
        print(f"{char:<15} {std_score:<10.1f} {limits_score:<10.1f} {recommendation:<10}")
    
    # 总体推荐
    total_std = sum([score[0] for score in characteristics.values()])
    total_limits = sum([score[1] for score in characteristics.values()])
    
    print("-" * 50)
    print(f"{'总分':<15} {total_std:<10.1f} {total_limits:<10.1f} ", end="")
    if total_std > total_limits:
        print("✓ 推荐标准方法")
    elif total_limits > total_std:
        print("✓ 推荐对偶方法")
    else:
        print("○ 两种方法相当")
    
    return results_std, results_limits, performance_metrics


def main():
    """更新后的主函数 - 调用全面对比分析"""
    print("=== DeepBSDE方法对比分析 ===")
    
    # 运行全面对比分析
    results_std, results_limits, metrics = comprehensive_comparison(d=30, num_trials=3)
    
    # 输出最终建议
    print("\n" + "="*60)
    print("                最终使用建议")
    print("="*60)
    
    print("\n📊 基于对比分析，建议如下：")
    print("\n✅ 推荐使用标准DeepBSDE方法的情况：")
    print("   • 需要快速得到点估计")
    print("   • 计算资源有限")
    print("   • 问题相对简单，不需要不确定性量化")
    print("   • 实现复杂度要求低")
    
    print("\n✅ 推荐使用带Legendre变换对偶方法的情况：")
    print("   • 需要置信区间估计")
    print("   • 对解的可靠性要求高")
    print("   • 有充足的计算资源")
    print("   • 需要进行严格的理论分析")
    
    print("\n🔍 关键发现：")
    if metrics['std_errors'] and metrics['limits_errors']:
        if np.mean(metrics['std_errors']) < np.mean(metrics['limits_errors']):
            print("   • 标准方法在准确性上略优")
        else:
            print("   • 对偶方法在准确性上略优")
        
        if np.mean(metrics['std_times']) < np.mean(metrics['limits_times']):
            print("   • 标准方法在计算效率上明显更优")
        else:
            print("   • 对偶方法在计算效率上更优")
    
    if metrics.get('limits_intervals'):
        analytical_value = results_limits[0][0].analytical_solution(
            results_limits[0][0].x0, results_limits[0][0].tspan[0]).item()
        coverage = sum(1 for interval in metrics['limits_intervals'] 
                      if interval[0] <= analytical_value <= interval[1])
        print(f"   • 对偶方法提供置信区间，覆盖率为{coverage/len(metrics['limits_intervals'])*100:.1f}%")
    
    return results_std, results_limits, metrics


if __name__ == "__main__":
    results_std, results_limits, metrics = main()
