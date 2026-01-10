#!/usr/bin/env python3
"""
比较 American.py 和 cqf_mc_American.py 的 _get_discounted_cashflow_at_t0 方法
输入数据: /Users/susan/PINN-slove-PDE/cash_flow.csv
"""

import numpy as np
import pandas as pd

def american_get_discounted_cashflow_at_t0(cashflow_matrix, ir, dt):
    """
    American.py 版本的实现
    """
    future_cashflows = cashflow_matrix[:, 1:]
    
    first_nonzero_positions = np.argmax(future_cashflows != 0, axis=1)
    
    has_cashflow = np.any(future_cashflows != 0, axis=1)
    
    time_indices = first_nonzero_positions + 1
    
    discount_factors = np.exp(-ir * time_indices * dt)
    
    discounted_values = future_cashflows[np.arange(len(cashflow_matrix)), first_nonzero_positions] * discount_factors
    
    return discounted_values[has_cashflow].mean()

def cqf_get_discounted_cashflow_at_t0(cashflow_matrix, ir, dt):
    """
    cqf_mc_American.py 版本的实现
    """
    future_cashflows = cashflow_matrix[:, 1:]
    
    first_nonzero_positions = np.argmax(future_cashflows != 0, axis=1)
    
    has_cashflow = np.any(future_cashflows != 0, axis=1)
    
    time_indices = first_nonzero_positions + 1
    
    discount_factors = np.exp(-ir * time_indices * dt)
    
    discounted_values = future_cashflows[np.arange(len(cashflow_matrix)), first_nonzero_positions] * discount_factors
    
    return discounted_values[has_cashflow].mean()

def main():
    csv_path = "/Users/susan/PINN-slove-PDE/cash_flow.csv"
    cashflow_matrix = pd.read_csv(csv_path, header=None).values
    
    print("现金流矩阵形状:", cashflow_matrix.shape)
    print("前5行前10列数据:")
    print(cashflow_matrix[:5, :10])
    
    ir = 0.05
    dt = 1.0 / 50
    
    american_result = american_get_discounted_cashflow_at_t0(cashflow_matrix, ir, dt)
    cqf_result = cqf_get_discounted_cashflow_at_t0(cashflow_matrix, ir, dt)
    
    print(f"\nAmerican.py 结果: {american_result}")
    print(f"cqf_mc_American.py 结果: {cqf_result}")
    print(f"差异: {abs(american_result - cqf_result)}")
    
    if abs(american_result - cqf_result) < 1e-10:
        print("\n✓ 两种实现结果相等！")
    else:
        print("\n✗ 两种实现结果不相等，需要修改 cqf_mc_American.py")
        
        future_cashflows = cashflow_matrix[:, 1:]
        american_indices = np.argmax(future_cashflows != 0, axis=1)
        print(f"\nAmerican.py 索引统计:")
        print(f"  唯一索引值: {np.unique(american_indices)}")
        print(f"  每个索引出现次数: {np.bincount(american_indices)}")
        
        nonzero_mask = cashflow_matrix != 0
        cqf_indices = np.where(nonzero_mask.any(axis=1),
                              nonzero_mask.argmax(axis=1),
                              cashflow_matrix.shape[1])
        print(f"\ncqf_mc_American.py 索引统计:")
        print(f"  唯一索引值: {np.unique(cqf_indices)}")
        
        diff_mask = american_indices != cqf_indices
        if np.any(diff_mask):
            diff_indices = np.where(diff_mask)[0]
            print(f"\n存在差异的行索引数量: {len(diff_indices)}")
            print(f"前10个差异行索引: {diff_indices[:10]}")

if __name__ == "__main__":
    main()
