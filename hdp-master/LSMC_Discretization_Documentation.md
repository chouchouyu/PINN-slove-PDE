# 最小二乘蒙特卡洛模拟定价方法详解

## 1. 概述

最小二乘蒙特卡洛（Least Square Monte Carlo, LSMC）是解决美式期权定价问题的一种重要数值方法。该方法由Longstaff和Schwartz在2001年提出，通过蒙特卡洛模拟生成资产价格路径，并使用最小二乘回归估计继续持有期权的价值，从而确定最优行权策略。

在 `/Users/susan/Downloads/hdp-master/src/blackscholes/mc/American.py` 文件中，实现了基于LSMC的美式期权定价方法。

## 2. LSMC方法核心原理

LSMC方法的基本步骤：
1. **生成路径**：模拟大量资产价格路径
2. **反向递归**：从到期日开始，反向遍历每个时间点
3. **回归估计**：使用最小二乘回归估计继续持有期权的价值
4. **行权决策**：比较立即行权价值和继续持有价值，决定最优行权策略
5. **折现计算**：将所有现金流折现到初始时刻，取平均值作为期权价格

## 3. 路径生成的离散化方法

### 3.1 核心代码分析

在 `American.py` 的 `price` 方法中，路径生成通过以下代码实现：

```python
def price(self, path_num=1000):
    """Least Square Monte Carlo method"""
    self.simulation_result = self.random_walk.simulateV2(path_num)
    # ... 后续处理代码 ...
```

关键在于 `self.random_walk.simulateV2(path_num)` 方法，它调用了 `GBM` 类的 `simulateV2` 方法来生成资产价格路径。

### 3.2 离散化方法：对数正态分布

`simulateV2` 方法使用的是**对数正态分布**的精确离散化方法，而非欧拉方法。

#### 数学推导

几何布朗运动的随机微分方程为：
$$ dS_t = S_t (\mu dt + \sigma dW_t) $$

其解析解为：
$$ S_t = S_0 \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)t + \sigma W_t \right) $$

在离散时间步长 $\Delta t$ 下，价格演化可表示为：
$$ S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} \varepsilon \right) $$

#### 代码实现

```python
def simulateV2(self, M):
    simulations = []
    drift_vec = self.ir - self.dividend_vec
    L = np.linalg.cholesky(self.corr_mat)
    for _ in range(M):
        sim = np.zeros([self.asset_num, self.N+1])
        sim[:, 0] = self.init_price_vec
        for i in range(1, self.N+1):
            dW = L.dot(np.random.normal(size=self.asset_num))*sqrt(self.dt)
            rand_term = np.multiply(self.vol_vec, dW)
            sim[:, i] = np.multiply(sim[:, i-1], np.exp((drift_vec-self.vol_vec**2/2)*self.dt + rand_term))
        simulations.append(sim)
    return np.array(simulations)
```

**核心更新公式**：
```python
sim[:, i] = np.multiply(sim[:, i-1], np.exp((drift_vec-self.vol_vec**2/2)*self.dt + rand_term))
```

## 4. 对数正态离散化 vs 欧拉离散化

### 4.1 欧拉离散化

欧拉方法是一种数值积分方法，用于近似求解随机微分方程：

$$ S_{t+\Delta t} = S_t + S_t \mu \Delta t + S_t \sigma \sqrt{\Delta t} \varepsilon $$

或者写成：

$$ S_{t+\Delta t} = S_t (1 + \mu \Delta t) + S_t \sigma \sqrt{\Delta t} \varepsilon $$

**优点**：计算简单，效率略高
**缺点**：
- 存在数值误差，时间步长越小误差越小
- 可能产生负的资产价格（需要额外处理）
- 不精确满足几何布朗运动的理论分布

### 4.2 对数正态离散化

对数正态方法是几何布朗运动的精确离散化：

$$ S_{t+\Delta t} = S_t \exp\left( \left(\mu - \frac{\sigma^2}{2}\right)\Delta t + \sigma \sqrt{\Delta t} \varepsilon \right) $$

**优点**：
- 精确符合几何布朗运动的解析解
- 保证资产价格非负（指数函数特性）
- 数值稳定性好，无截断误差

**缺点**：
- 计算效率略低（多一次指数运算）

### 4.3 两种方法的比较

| 特性 | 欧拉离散化 | 对数正态离散化 |
|------|------------|----------------|
| 数学基础 | 数值积分近似 | 解析解精确实现 |
| 价格非负性 | 不保证，可能出现负值 | 保证非负 |
| 计算效率 | 略高 | 略低 |
| 数值精度 | 较低，依赖时间步长 | 高，与时间步长无关 |
| 理论一致性 | 近似 | 精确符合GBM理论 |

## 5. LSMC中的路径生成流程

### 5.1 完整流程

1. **初始化**：
   - 设置资产数量、初始价格、波动率、无风险利率等参数
   - 创建GBM实例

2. **生成随机数**：
   - 生成独立的标准正态随机变量
   - 通过Cholesky分解生成相关的随机变量
   - 缩放为布朗运动增量

3. **价格演化**：
   - 使用对数正态公式更新每个时间步的价格
   - 存储完整的价格路径

4. **路径返回**：
   - 返回三维数组，包含所有路径的完整时间序列

### 5.2 数据结构

`simulation_result` 是一个三维数组：
- 第一维：路径数量（path_num）
- 第二维：资产数量（asset_num）
- 第三维：时间步长（N+1，包括初始时刻）

## 6. LSMC方法的完整实现

### 6.1 反向递归计算

```python
cashflow_matrix = np.zeros([path_num, self.random_walk.N+1])
cur_price = np.array([x[:, -1] for x in self.simulation_result])
cur_payoff = np.array(list(map(self.payoff_func, cur_price)))
cashflow_matrix[:, self.random_walk.N] = cur_payoff
for t in range(self.random_walk.N-1, 0, -1):
    discounted_cashflow = self._get_discounted_cashflow(t, cashflow_matrix, path_num)
    r = Regression(self.simulation_result[:, :, t], discounted_cashflow, payoff_func=self.payoff_func)
    if not r.has_intrinsic_value: continue
    cur_price = np.array([x[:, t] for x in self.simulation_result])
    cur_payoff = np.array(list(map(self.payoff_func, cur_price[r.index])))
    continuation = np.array([r.evaluate(X) for X in cur_price[r.index]])
    exercise_index = r.index[cur_payoff >= continuation]
    cashflow_matrix[exercise_index] = np.zeros(cashflow_matrix[exercise_index].shape)
    cashflow_matrix[exercise_index, t] = np.array(list(map(self.payoff_func, cur_price)))[exercise_index]
```

### 6.2 回归分析

使用多项式回归估计继续持有期权的价值：

```python
r = Regression(self.simulation_result[:, :, t], discounted_cashflow, payoff_func=self.payoff_func)
```

### 6.3 行权决策

比较立即行权价值和继续持有价值：

```python
exercise_index = r.index[cur_payoff >= continuation]
```

## 7. 为什么选择对数正态离散化？

在LSMC方法中，选择对数正态离散化的主要原因：

1. **理论一致性**：精确符合几何布朗运动的理论模型，保证模拟结果的正确性
2. **数值稳定性**：避免了欧拉方法可能产生的负值价格问题
3. **定价准确性**：提高了期权定价的精度，特别是对于长期期权
4. **收敛性**：收敛速度快，与时间步长无关

## 8. 应用场景

LSMC方法使用对数正态离散化的路径生成，适用于：

1. **美式期权定价**：需要考虑提前行权的期权
2. **多资产期权**：篮子期权、彩虹期权等
3. **复杂支付结构**：路径依赖期权、障碍期权等
4. **风险管理**：VaR计算、情景分析等

## 9. 总结

1. **最小二乘蒙特卡洛方法**：
   - 用于解决美式期权定价问题
   - 核心是使用最小二乘回归估计继续持有价值

2. **路径生成的离散化方法**：
   - 使用对数正态分布的精确离散化
   - 不是欧拉方法
   - 精确符合几何布朗运动的解析解

3. **关键优势**：
   - 保证资产价格非负
   - 数值稳定性好
   - 定价精度高
   - 理论一致性强

在 `American.py` 文件中，`price` 方法通过调用 `self.random_walk.simulateV2(path_num)` 生成价格路径，其中 `simulateV2` 方法使用对数正态离散化方法，确保了LSMC方法的准确性和可靠性。