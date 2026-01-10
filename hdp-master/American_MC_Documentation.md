# American 期权蒙特卡洛定价实现文档

## 1. 文件概述

`/Users/susan/Downloads/hdp-master/src/blackscholes/mc/American.py` 实现了**多维美式期权**的定价，采用了**最小二乘蒙特卡洛 (Least Square Monte Carlo, LSMC)** 方法。该实现允许为具有提前行权特性的期权进行定价，支持多资产维度。

## 2. 核心概念

### 2.1 美式期权特性
与欧式期权不同，美式期权允许持有者在到期日之前的任何时间点行权。这一特性使得美式期权的定价更加复杂，因为需要考虑提前行权的最优时机。

### 2.2 最小二乘蒙特卡洛方法
LSMC方法由Longstaff和Schwartz在2001年提出，是解决美式期权定价的一种重要数值方法。其核心思想是：
1. 模拟大量资产价格路径
2. 从到期日开始，反向递归计算每个时间点的期权价值
3. 使用回归分析估计继续持有期权的价值（continuation value）
4. 比较继续持有价值与立即行权价值，决定最优行权策略

## 3. 代码结构

### 3.1 类定义

```python
class American:
    """
    Multi-Dimensional American Option. Priced by the Least Square Monte Carlo method.
    """
```

### 3.2 初始化函数

```python
def __init__(self, payoff_func, random_walk):
    """
    payoff: A function that takes ${asset_num} variables as input, returns the a scalar payoff
    random_walk: A random walk generator, e.g. GBM (geometric brownian motion)
    """
    self.payoff_func = payoff_func
    self.random_walk = random_walk
```

**参数说明**：
- `payoff_func`: 支付函数，接收资产价格变量，返回标量支付值
- `random_walk`: 随机游走生成器，如几何布朗运动（GBM），用于模拟资产价格路径

## 4. 核心算法实现

### 4.1 价格计算主函数

```python
def price(self, path_num=1000):
    """Least Square Monte Carlo method"""
    self.simulation_result = self.random_walk.simulateV2(path_num)
    cashflow_matrix = np.zeros([path_num, self.random_walk.N+1])
    cur_price = np.array([x[:, -1] for x in self.simulation_result])
    cur_payoff = np.array(list(map(self.payoff_func, cur_price)))
    cashflow_matrix[:, self.random_walk.N] = cur_payoff
    for t in range(self.random_walk.N-1, 0, -1):
        discounted_cashflow = self._get_discounted_cashflow(t, cashflow_matrix, path_num)
        # Compute the discounted payoff
        r = Regression(self.simulation_result[:, :, t], discounted_cashflow, payoff_func=self.payoff_func)
        if not r.has_intrinsic_value: continue # Intrinsic value = 0
        cur_price = np.array([x[:, t] for x in self.simulation_result])
        cur_payoff = np.array(list(map(self.payoff_func, cur_price[r.index])))
        continuation = np.array([r.evaluate(X) for X in cur_price[r.index]])
        exercise_index = r.index[cur_payoff >= continuation]
        cashflow_matrix[exercise_index] = np.zeros(cashflow_matrix[exercise_index].shape)
        cashflow_matrix[exercise_index, t] = np.array(list(map(self.payoff_func, cur_price)))[exercise_index]
    return self._get_discounted_cashflow_at_t0(cashflow_matrix)
```

**算法步骤**：
1. **模拟价格路径**：调用`random_walk.simulateV2()`生成指定数量的资产价格路径
2. **初始化现金流矩阵**：创建一个大小为`[path_num, N+1]`的零矩阵，用于记录每条路径在每个时间点的现金流
3. **计算到期日支付**：在到期日，所有路径的期权价值等于其内在价值（支付函数值）
4. **反向递归计算**：从到期日前一个时间点开始，反向遍历每个时间点：
   - 计算该时间点的折现现金流
   - 使用回归分析估计继续持有期权的价值
   - 对于具有正内在价值的路径，比较立即行权价值和继续持有价值
   - 如果立即行权价值大于或等于继续持有价值，则该路径在该时间点行权
5. **计算初始时刻价值**：将所有路径的最优现金流折现到初始时刻，并取平均值

### 4.2 辅助函数

#### 4.2.1 计算折现现金流

```python
def _get_discounted_cashflow(self, t, cashflow_matrix, path_num):
    discounted_cashflow = np.zeros(path_num)
    for i in range(len(cashflow_matrix)):
        cashflow = cashflow_matrix[i]
        for j in range(self.random_walk.N, t, -1):
            if cashflow[j] != 0:
                discounted_cashflow[i] = cashflow[j] * np.exp((t-j)*self.random_walk.dt*self.random_walk.ir)
                break
    return discounted_cashflow
```

该函数计算每条路径在时间`t`的折现现金流：
- 对于每条路径，找到其最后一个非零现金流（即实际行权时间点）
- 将该现金流折现到当前时间`t`
- 返回所有路径的折现现金流数组

#### 4.2.2 计算初始时刻折现现金流

```python
def _get_discounted_cashflow_at_t0(self, cashflow_matrix):
    summation = 0
    for cashflow in cashflow_matrix:
        for i in range(1, len(cashflow)):
            if cashflow[i] != 0:
                summation += cashflow[i]*np.exp(-self.random_walk.ir*i*self.random_walk.dt)
                break
    return summation / len(cashflow_matrix)
```

该函数计算所有路径的现金流在初始时刻（t=0）的折现值，并取平均值作为期权价格：
- 对于每条路径，找到其实际行权时间点
- 将该时间点的现金流折现到初始时刻
- 对所有路径的折现值求和并取平均值

## 5. 回归分析的作用

代码中使用了`Regression`类进行回归分析，这是LSMC方法的核心：

```python
r = Regression(self.simulation_result[:, :, t], discounted_cashflow, payoff_func=self.payoff_func)
```

回归分析的主要作用是：
1. 识别具有正内在价值的路径（`r.has_intrinsic_value`）
2. 对于这些路径，使用多项式回归估计继续持有期权的价值（`r.evaluate(X)`）
3. 通过比较继续持有价值和立即行权价值，决定是否行权

## 6. 数据结构

### 6.1 模拟结果

`self.simulation_result`是随机游走模拟生成的价格路径数据，维度为：
- 第一维：路径数量
- 第二维：资产数量（多维期权）
- 第三维：时间步长

### 6.2 现金流矩阵

`cashflow_matrix`记录每条路径在每个时间点的现金流：
- 行：路径编号
- 列：时间步长
- 非零值表示在该时间点行权的现金流

## 7. 算法复杂度

- **时间复杂度**：O(M*N*P)，其中M是路径数量，N是时间步长，P是多项式回归的阶数
- **空间复杂度**：O(M*N)，主要用于存储价格路径和现金流矩阵

## 8. 与欧式期权的比较

| 特性 | 欧式期权 | 美式期权 |
|------|----------|----------|
| 行权时机 | 仅到期日 | 到期日前任意时间 |
| 定价方法 | 蒙特卡洛、Black-Scholes公式 | 最小二乘蒙特卡洛、有限差分 |
| 复杂度 | 较低 | 较高，需要考虑提前行权 |
| 价值 | 通常低于或等于美式期权 | 通常高于或等于欧式期权 |

## 9. 使用示例

```python
# 1. 定义支付函数（例如，看跌期权）
def put_payoff(prices):
    strike = 100
    return max(strike - prices[0], 0)

# 2. 创建随机游走生成器（例如，GBM）
from blackscholes.mc.GBM import GBM
import numpy as np

spot_price = np.array([100])  # 当前价格
volatility = np.array([0.2])  # 波动率
interest_rate = 0.05  # 无风险利率
dividend_yield = np.array([0])  # 股息率
time_to_maturity = 1.0  # 到期时间
N = 100  # 时间步长

# 创建GBM实例
gbm = GBM(spot_price, volatility, interest_rate, dividend_yield, time_to_maturity, N)

# 3. 创建American期权实例并定价
from blackscholes.mc.American import American

american_option = American(put_payoff, gbm)
price = american_option.price(path_num=10000)
print(f"American Put Option Price: {price}")
```

## 10. 扩展与优化

该实现可以从以下几个方面进行扩展和优化：

1. **支持更多回归函数**：当前实现使用多项式回归，可以扩展支持其他基函数（如指数函数、对数函数等）
2. **并行计算**：蒙特卡洛模拟和路径计算可以并行化，提高计算效率
3. **方差减少技术**：可以实现控制变量法、对偶变量法等方差减少技术，提高模拟精度
4. **自适应时间步长**：根据资产价格的变化动态调整时间步长，提高计算效率
5. **更复杂的支付函数**：扩展支持障碍期权、亚式期权等复杂期权的支付函数

## 11. 结论

该实现提供了一个灵活、高效的美式期权定价框架，采用最小二乘蒙特卡洛方法解决了提前行权的最优时机问题。通过支持多维资产和自定义支付函数，该实现可以应用于各种复杂的美式期权定价场景。