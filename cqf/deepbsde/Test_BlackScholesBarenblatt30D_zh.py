import numpy as np
import matplotlib.pyplot as plt
import time
import os
import torch
from cqf.deepbsde.BlackScholesBarenblatt import BlackScholesBarenblatt
from cqf.deepbsde.DeepBSDE import BlackScholesBarenblattSolver, rel_error_l2
from cqf.fbsnn.Utils import figsize





# 模型参数设置 - 与 BlackScholesBarenblatt100D.py 相同结构
D = 30  # 问题维度（30维Black-Scholes-Barenblatt方程）
M = 100  # 每批次训练的轨迹数量
N = 50  # 时间轴上的离散点数量

# 创建Figures目录
figures_dir = "Figures"
if not os.path.exists(figures_dir):
    os.makedirs(figures_dir)


print("=== 30维Black-Scholes-Barenblatt方程求解 ===")
print("使用DeepBSDE方法")

# 创建求解器实例
start_time = time.time()
solver = BlackScholesBarenblattSolver(d=D)

# 训练求解器 - 使用标准DeepBSDE算法
result = solver.solve(limits=False, verbose=True)

total_time = time.time() - start_time
print(f"总计算时间: {total_time:.2f} 秒")


# 获取训练历史
iterations = list(range(len(solver.losses)))
training_loss = solver.losses


# 1. 绘制训练损失曲线
plt.figure(figsize=figsize(1))
plt.plot(iterations, training_loss, 'b')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.yscale("log")
plt.title('Evolution of the training loss')
plt.savefig(f"{figures_dir}/DeepBSDE_BlackScholesBarenblattSolver30D_Loss.png")
plt.close()
print("已保存训练损失曲线")



# 计算DeepBSDE估计值与解析解的相对误差
x0 = solver.x0
u_deepbsde = result.us
u_analytical = solver.analytical_solution(x0, 0).item()
relative_error = rel_error_l2(u_deepbsde, u_analytical)

# 输出结果总结
print("\n=== 结果总结 ===")
print(f"DeepBSDE估计值 (u0): {u_deepbsde:.6f}")
print(f"解析解 (u0): {u_analytical:.6f}")
print(f"相对误差: {relative_error:.6f} ({relative_error*100:.4f}%)")
print(f"\n生成结果图片已保存到 {os.path.abspath(figures_dir)}/DeepBSDE_BlackScholesBarenblattSolver30D_Loss.png")
