import numpy as np
import torch
from cqf.deepbsde.DeepBSDE import BlackScholesBarenblattSolver

# 测试1: 使用默认参数初始化
print("测试1: 使用默认参数初始化...")
try:
    solver_default = BlackScholesBarenblattSolver()
    print(f"默认参数: d={solver_default.d}, m={solver_default.m}, dt={solver_default.dt}, tspan={solver_default.tspan}")
    print("测试1通过")
except Exception as e:
    print(f"测试1失败: {e}")

# 测试2: 使用与FBSNNs一致的参数初始化
print("\n测试2: 使用与FBSNNs一致的参数初始化...")
try:
    # 与FBSNNs文件中的参数保持一致
    D = 100  # 维度
    M = 100  # 训练轨迹数 (batch size)
    N = 50   # 时间快照数
    dt = 1.0 / N  # 时间步长，与FBSNNs的时间快照数匹配
    
    # 初始化参数
    x0 = [1.0, 0.5] * int(D / 2)  # 初始条件与FBSNNs一致
    tspan = (0.0, 1.0)
    
    solver_fbsnn = BlackScholesBarenblattSolver(
        d=D,
        x0=x0,
        tspan=tspan,
        dt=dt,
        m=M
    )
    
    print(f"FBSNNs参数: d={D}, m={M}, N={N}, dt={dt}")
    print(f"Solver参数: d={solver_fbsnn.d}, m={solver_fbsnn.m}, dt={solver_fbsnn.dt}, time_steps={solver_fbsnn.time_steps}")
    print("测试2通过")
except Exception as e:
    print(f"测试2失败: {e}")

# 测试3: 验证参数传递是否正确
print("\n测试3: 验证参数传递是否正确...")
try:
    custom_params = {
        "d": 50,
        "x0": [2.0, 1.0] * 25,  # 自定义初始条件
        "tspan": (0.0, 2.0),      # 自定义时间范围
        "dt": 0.1,               # 自定义时间步长
        "m": 200                 # 自定义轨迹数
    }
    
    solver_custom = BlackScholesBarenblattSolver(**custom_params)
    
    print(f"自定义参数: {custom_params}")
    print(f"Solver参数: d={solver_custom.d}, x0.shape={solver_custom.x0.shape}, tspan={solver_custom.tspan}, dt={solver_custom.dt}, m={solver_custom.m}, time_steps={solver_custom.time_steps}")
    print("测试3通过")
except Exception as e:
    print(f"测试3失败: {e}")

# 测试4: 验证x0支持numpy数组
print("\n测试4: 验证x0支持numpy数组...")
try:
    d = 10
    x0_np = np.array([1.5, 0.75] * int(d / 2))
    solver_np = BlackScholesBarenblattSolver(d=d, x0=x0_np)
    
    print(f"numpy x0: {x0_np}")
    print(f"Solver x0: {solver_np.x0}")
    print("测试4通过")
except Exception as e:
    print(f"测试4失败: {e}")

print("\n所有测试完成！")
