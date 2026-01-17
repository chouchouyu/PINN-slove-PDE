# 欧式期权蒙特卡洛定价类 (`Euro.py`)

## 文件概述

`Euro.py` 是一个用于**多维欧式期权定价**的Python类实现，基于蒙特卡洛模拟方法。该类支持多种模拟技术和方差缩减策略，适用于复杂的金融衍生品定价场景。

## 核心功能

- **多维资产支持**：可同时为多只相关股票（通过相关性矩阵定义）的欧式期权定价
- **多种模拟技术**：实现了标准蒙特卡洛、准蒙特卡洛（Sobol序列）等
- **方差缩减策略**：包含对偶变量法、控制变量法、重要性抽样等高级技术
- **灵活的支付函数**：支持自定义期权支付函数

## 类定义与初始化

```python
class Euro:
    """Vanilla Multi-Dimensional European Option"""
    
    def __init__(self, payoff_func, random_walk):
        """
        参数:
        payoff_func: 支付函数，接收asset_num个变量，返回标量支付值
        random_walk: 随机游走生成器，如GBM（几何布朗运动）
        """
        self.payoff_func = payoff_func
        self.random_walk = random_walk
```

## 定价方法详解

### 1. 基础定价方法

#### price() - 标准蒙特卡洛
```python
def price(self, path_num=1000, ci=False):
    """标准蒙特卡洛定价"""
    self.simulation_result = self.random_walk.simulate(path_num)
    last_price = [x[:, -1] for x in self.simulation_result]
    payoff = self.payoff_func(last_price) * np.exp(-self.random_walk.ir * self.random_walk.T)
    value = np.mean(payoff) 
    # 可选返回置信区间
```

#### priceV2() - 解析解近似
```python
def priceV2(self, path_num=10000, ci=False):
    """使用SDE解析解近似股票价格"""
    self.simulation_result = self.random_walk.simulateV2_T(path_num)
    payoff = self.payoff_func(self.simulation_result) * np.exp(-self.random_walk.ir * self.random_walk.T)
    value = np.mean(payoff) 
```

#### priceV3() - 全时间步解析解
```python
def priceV3(self, path_num=10000, ci=False):
    """返回每个时间步的解析解，但仅使用到期价格"""
    # ...
```

### 2. 准蒙特卡洛方法

#### priceV4() - Sobol序列
```python
def priceV4(self, path_num=10000, ci=False):
    """使用Sobol低差异序列的蒙特卡洛"""
    self.simulation_result = self.random_walk.simulateV4_T(path_num)
    # ...
```

### 3. 方差缩减技术

#### priceV5() - 对偶变量法
```python
def priceV5(self, path_num=10000, ci=False):
    """使用对偶变量的蒙特卡洛（方差缩减）"""
    self.simulation_result = self.random_walk.simulateV2_T_antithetic(int(path_num/2))
    # ...
```

#### priceV6() - Sobol+对偶变量
```python
def priceV6(self, path_num=10000, ci=False):
    """Sobol序列 + 对偶变量（组合方差缩减）"""
    self.simulation_result = self.random_walk.simulateV4_T_antithetic(int(path_num/2))
    # ...
```

#### priceV7() - 控制变量法
```python
def priceV7(self, path_num=10000):
    """使用控制变量的蒙特卡洛"""
    self.simulation_result = self.random_walk.simulateV2_T(path_num)
    expectation = np.exp((self.random_walk.ir-self.random_walk.dividend_vec)*self.random_walk.T) \
        * self.random_walk.init_price_vec
    # 计算协方差和最优控制变量系数
    cov = np.cov(self.simulation_result.T, payoff)
    Sx = cov[:-1, :-1]
    Sxy = cov[-1, :-1]
    b_hat = np.linalg.inv(Sx) @ Sxy
    return np.mean(payoff) - b_hat.dot(np.mean(self.simulation_result, axis=0) - expectation)
```

#### price_importance_sampling() - 重要性抽样
```python
def price_importance_sampling(self, path_num):
    """使用重要性抽样的蒙特卡洛"""
    assert hasattr(self.payoff_func, 'strike'), '缺少支付函数的执行价信息'
    strike = self.payoff_func.strike
    self.simulation_result, Zs = self.random_walk.importance_sampling_simulate_T(path_num, strike)
    # 计算密度比和调整后的期望
    density_ratio = norm.pdf(Zs, loc=norm_mean_old, scale=scale)/norm.pdf(Zs, loc=norm_mean_new, scale=scale)
    payoff = self.payoff_func(self.simulation_result)
    return np.mean(np.multiply(payoff, density_ratio.T)) * np.exp(-self.random_walk.ir*self.random_walk.T)
```

## 辅助定价方法

### 一维控制变量法
```python
def price1d_control_variates(self, path_num=1000):
    """专为一维问题优化的控制变量法"""
    # ...
```

### 对偶变量法（兼容旧接口）
```python
def price_antithetic_variates(self, path_num=1000):
    """使用对偶变量的蒙特卡洛（旧接口）"""
    # ...
```

## 技术特点

### 1. 多维资产处理
- 支持任意数量的相关资产
- 通过相关性矩阵定义资产间关系
- 自动处理向量和矩阵运算

### 2. 方差缩减技术
| 技术 | 实现方法 | 预期效果 |
|------|----------|----------|
| 对偶变量法 | 同时模拟正负随机路径 | 方差降低30-50% |
| 控制变量法 | 利用已知期望的控制变量 | 方差降低40-60% |
| 重要性抽样 | 调整随机变量分布 | 极端事件模拟效率提升 |
| 准蒙特卡洛 | 使用Sobol低差异序列 | 收敛速度提高 |

### 3. 性能优化
- 向量化运算减少循环开销
- 可选的置信区间计算
- 支持大规模路径模拟

## 使用示例

```python
if __name__ == "__main__":
    import sys, os
    sys.path.append(os.path.dirname(os.path.abspath(__file__))+'/../../')
    from blackscholes.utils.GBM import GBM

    # 初始化参数
    init_price_vec = np.ones(5)  # 5只股票，初始价格均为1
    vol_vec = 0.2*np.ones(5)     # 波动率20%
    ir = 0.00                    # 无风险利率0%
    dividend_vec = np.zeros(5)   # 无股息
    corr_mat = np.eye(5)         # 不相关
    
    # 创建GBM随机游走生成器
    random_walk = GBM(3, 100, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    
    # 定义支付函数（篮子看涨期权）
    def test_payoff(*l):
        return max(np.sum(l) - 5, 0)  # 执行价为5的篮子看涨期权
    
    # 定价
    a = Euro(test_payoff, random_walk).price(10000)
    print(a)
```

## 代码优化建议

1. **类型提示**：添加类型注解提高代码可读性和IDE支持
2. **并行计算**：引入多进程或多线程加速大规模模拟
3. **错误处理**：增强异常处理机制，提高鲁棒性
4. **文档字符串**：完善部分方法的文档说明

## 总结

`Euro.py` 是一个功能强大的欧式期权定价类，通过多种蒙特卡洛技术和方差缩减策略，为复杂的多维金融衍生品提供高效、准确的定价解决方案。其灵活的设计使其适用于学术研究和实际金融应用场景。