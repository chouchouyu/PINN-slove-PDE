# Regression类分析：逻辑与数学原理

## 1. 概述

Regression类是一个用于美式期权定价的最小二乘回归工具，基于**单项式基函数**构建，支持多维度输入和灵活的回归次数调整。它是Least Square Monte Carlo (LSMC)方法中的核心组件，用于估计期权的**存续价值**（continuation value）。

## 2. 类结构与依赖关系

```
┌─────────────────┐
│   Monomial      │
└────────┬────────┘
         │
┌────────▼────────┐
│ Monomial_Basis  │
└────────┬────────┘
         │
┌────────▼────────┐
│   Regression    │
└─────────────────┘
```

- **Monomial**：表示单个单项式项（如 x_1^2x_2^3）
- **Monomial_Basis**：生成完整的单项式基函数集合
- **Regression**：执行最小二乘回归计算

## 3. 数学原理

### 3.1 单项式基函数

对于维度为 d 的输入向量 x = (x_1, x_2, dots, x_d) 和次数为 chi 的回归，单项式基函数集合定义为：

 B =  x_1^{a_1}x_2^{a_2}dots x_d^{a_d} id a_1 + a_2 + dots + a_d eq chi  

其中 a_i 是非负整数。

**示例**：
- 1D, chi=2:  1, x, x^2 
- 2D, chi=2:  1, x_1, x_2, x_1^2, x_1x_2, x_2^2 
- 3D, chi=1:  1, x_1, x_2, x_3 

### 3.2 最小二乘回归

对于数据集 (X_i, Y_i)，其中 X_i 是输入向量，Y_i 是目标值，回归模型为：

 Y = um_{b n B} eta_b b(X) + psilon 

其中 beta_b 是回归系数，psilon 是误差项。

通过最小化残差平方和来估计系数：

 in_{eta} um_{i=1}^n eft( Y_i - um_{b n B} eta_b b(X_i) 
ight)^2 

解为：

 eta = (A^TA)^{-1}A^TY 

其中 A 是设计矩阵，A_{ij} = b_j(X_i)。

## 4. 代码实现分析

### 4.1 Monomial类

```python
class Monomial:
    def __init__(self, a_vec):
        self.a_vec = np.array(a_vec)  # 指数向量，如 [2, 3] 表示 x1^2x2^3
        self.dimension = len(a_vec)
    
    def evaluate(self, input_vec):
        assert self.dimension == len(input_vec), "维度不匹配"
        return np.prod(np.power(input_vec, self.a_vec))  # 计算单项式的值
```

**功能**：表示单个单项式并计算其值。

**数学原理**：对于输入向量 x = (x_1, x_2) 和指数向量 a = (a_1, a_2)，计算 x_1^{a_1}x_2^{a_2}。

### 4.2 Monomial_Basis类

```python
class Monomial_Basis:
    def __init__(self, chi, dimension):
        permutations = Monomial_Basis._get_all_permutations(chi, dimension)
        self.monomials = [Monomial(x) for x in permutations]
    
    @staticmethod
    def _get_all_permutations(chi, dimension):
        # 递归生成所有满足 a1+a2+...+ad <= chi 的非负整数组合
        if chi == 0:
            return [[0]*dimension]
        elif dimension == 1:
            return [[i] for i in range(chi+1)]
        else:
            results = []
            for i in range(chi+1):
                results += [[i] + x for x in Monomial_Basis._get_all_permutations(chi - i, dimension-1)]
            return results
    
    def evaluate(self, X):
        return np.array([m.evaluate(X) for m in self.monomials])  # 计算所有基函数的值
```

**功能**：生成单项式基函数集合。

**关键算法**：递归生成指数组合。
- 基本情况1：chi=0时，只有常数项 [0,0,...,0]
- 基本情况2：dimension=1时，生成 [0], [1], ..., [chi]
- 递归情况：对每个维度分配0到chi的指数，然后递归生成剩余维度的组合

### 4.3 Regression类

```python
class Regression:
    def __init__(self, X_mat, Y, chi=2, payoff_func=lambda x: np.max(np.sum(x)-100, 0)):
        assert len(X_mat.shape) == 2, "X必须是2D矩阵"
        self.dimension = len(X_mat[0])
        self.basis = Monomial_Basis(chi, self.dimension)
        
        # 筛选实值期权路径（payoff > 0）
        self.index = np.array([i for i in range(len(X_mat)) if payoff_func(X_mat[i]) > 0])
        
        self.has_intrinsic_value = False if len(self.index) == 0 else True
        if not self.has_intrinsic_value: return
        
        target_X, target_Y = X_mat[self.index], Y[self.index]
        
        # 构建设计矩阵A
        target_matrix_A = np.array([self.basis.evaluate(x) for x in target_X])
        
        # 最小二乘求解
        self.coefficients = np.linalg.lstsq(target_matrix_A, target_Y, rcond=None)[0]
    
    def evaluate(self, X):
        if not self.has_intrinsic_value: raise RuntimeError("无有效数据")
        assert len(X) == self.dimension, "输入维度不匹配"
        monomial_terms = self.basis.evaluate(X)
        return np.sum(np.multiply(self.coefficients, monomial_terms))  # 计算回归值
```

**功能**：执行最小二乘回归并评估回归模型。

**核心步骤**：
1. **初始化**：生成基函数集合
2. **实值筛选**：只处理期权处于实值状态的路径
3. **设计矩阵构建**：计算每个样本的基函数值
4. **最小二乘求解**：使用`np.linalg.lstsq`计算回归系数
5. **模型评估**：使用回归系数和基函数计算预测值

## 5. 多维度支持

Regression类天然支持多维度输入，这是通过单项式基函数的交叉项实现的。

### 示例：2D输入，chi=2

对于输入 x = (x_1, x_2)，生成的基函数为：

 B = [1, x_1, x_2, x_1^2, x_1x_2, x_2^2] 

设计矩阵的一行示例：

 [1, x_{i1}, x_{i2}, x_{i1}^2, x_{i1}x_{i2}, x_{i2}^2] 

## 6. 与传统回归方法的比较

| 特性 | Regression类 | np.polyfit |
|------|--------------|------------|
| 基函数 | 单项式（支持交叉项） | 简单多项式 |
| 多维支持 | 原生支持 | 需手动构造特征 |
| 实值筛选 | 自动 | 需手动 |
| 灵活性 | 高（可调整chi和维度） | 低（主要1D） |
| 性能 | 中（基函数生成有开销） | 高 |

## 7. 应用场景

Regression类主要用于**美式期权定价**的LSMC方法中，其核心作用是：

1. **存续价值估计**：通过回归预测期权继续持有到下一时刻的价值
2. **行权决策**：比较内在价值（intrinsic value）与存续价值，决定是否提前行权
3. **多资产支持**：处理复杂的多资产期权（如篮子期权、彩虹期权等）

## 8. 代码示例

### 8.1 1D看跌期权示例

```python
import numpy as np
from blackscholes.utils.Regression import Regression

# 生成模拟数据
X = np.random.normal(100, 20, (100, 1))  # 100个样本，1维输入
Y = np.maximum(100 - X.flatten(), 0) + np.random.normal(0, 5, 100)  # 带噪声的看跌期权收益

# 定义收益函数
def put_payoff(*l):
    return max(100 - np.sum(l), 0)

# 创建回归对象，chi=2（二次回归）
r = Regression(X, Y, chi=2, payoff_func=put_payoff)

# 评估新样本
new_x = np.array([85])
predicted_y = r.evaluate(new_x)
print(f"输入: {new_x}, 预测值: {predicted_y}")
```

### 8.2 2D看涨期权示例

```python
# 200个样本，2维输入
X = np.random.normal(100, 20, (200, 2))
# 取最大值的看涨期权收益
Y = np.maximum(np.max(X, axis=1) - 100, 0) + np.random.normal(0, 5, 200)

# 定义收益函数
def call_payoff(*l):
    return max(np.max(l) - 100, 0)

# 创建回归对象，chi=2
r = Regression(X, Y, chi=2, payoff_func=call_payoff)

# 评估新样本
new_x = np.array([110, 95])
predicted_y = r.evaluate(new_x)
print(f"输入: {new_x}, 预测值: {predicted_y}")
```

## 9. 关键性能优化点

1. **实值筛选**：只处理有正收益的路径，减少计算量
2. **NumPy向量化**：使用NumPy函数提高计算效率
3. **递归优化**：基函数生成采用递归，但对于小chi和d效率可接受
4. **条件检查**：通过`has_intrinsic_value`避免无效计算

## 10. 局限性

1. **维度诅咒**：当维度d和次数chi较大时，基函数数量呈指数增长
2. **过拟合风险**：较高的chi值可能导致过拟合
3. **计算复杂度**：基函数生成和矩阵乘法的计算量较大

## 11. 总结

Regression类是一个设计精良的最小二乘回归工具，专门针对美式期权定价的需求优化。它通过灵活的单项式基函数生成、自动的实值筛选和原生的多维度支持，为LSMC方法提供了强大的回归能力。理解其数学原理和实现细节，对于掌握美式期权的数值定价方法至关重要。