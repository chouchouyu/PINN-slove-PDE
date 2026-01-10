# MCAmericanTest.py 文档说明

## 1. 文件概述

`MCAmericanTest.py` 是用于测试美式期权蒙特卡洛定价实现的单元测试文件。该文件验证了 `/src/blackscholes/mc/American.py` 中 `American` 类的核心功能，包括1D和2D美式期权定价、折现现金流计算等关键方法的正确性。

## 2. 测试环境设置

### 2.1 导入依赖

```python
import sys, os
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/../src")
from blackscholes.utils.GBM import GBM
from blackscholes.mc.American import American
import unittest
import numpy as np
```

**依赖说明**：
- `GBM`：几何布朗运动类，用于生成资产价格路径
- `American`：待测试的美式期权定价类
- `unittest`：Python标准单元测试框架
- `numpy`：数值计算库，用于处理数组和数学运算

### 2.2 测试类定义

```python
class Test(unittest.TestCase):
```

使用Python标准的`unittest.TestCase`类定义测试类，支持丰富的断言方法和测试生命周期管理。

## 3. 测试用例设计

### 3.1 测试初始化（setUp）

```python
def setUp(self):
    strike = 1
    asset_num = 1
    init_price_vec = 0.99*np.ones(asset_num)
    vol_vec = 0.2*np.ones(asset_num)
    ir = 0.03
    dividend_vec = np.zeros(asset_num)
    corr_mat = np.eye(asset_num)
    random_walk = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    def test_payoff(*l):
        return max(strike - np.sum(l), 0)
    self.opt1 = American(test_payoff, random_walk)
```

**初始化设置**：
- 创建1D美式看跌期权测试实例
- 参数配置：
  - 执行价（strike）：1.0
  - 初始价格：0.99
  - 波动率：0.2
  - 无风险利率：0.03
  - 股息率：0
  - 时间期限：1年
  - 时间步长：300

### 3.2 1D美式期权定价测试

```python
def test_price1d(self):
    np.random.seed(444)
    price = self.opt1.price(3000)
    assert abs(price - 0.07187167189125372) < 1e-10
```

**测试内容**：
- 测试1维美式期权的定价功能
- 设置随机种子确保测试可重复性
- 使用3000条路径进行蒙特卡洛模拟
- 验证计算结果与预期值的误差小于1e-10

### 3.3 2D美式期权定价测试

```python
def test_price2d(self):
    np.random.seed(555)
    strike = 100
    asset_num = 2
    init_price_vec = 100*np.ones(asset_num)
    vol_vec = 0.2*np.ones(asset_num)
    ir = 0.05
    dividend_vec = 0.1*np.ones(asset_num)
    corr_mat = np.eye(asset_num)
    corr_mat[0, 1] = 0.3
    corr_mat[1, 0] = 0.3
    random_walk = GBM(1, 300, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
    def test_payoff(*l):
        return max(np.max(l) - strike, 0)
    opt = American(test_payoff, random_walk)
    put = opt.price(3000)
    real_put = 9.6333
    assert abs(put - 9.557936820537265) < 0.00000000000001
    assert abs(put - real_put)/real_put < 0.00783
    # when init = 110, price is 18.021487449289822/18.15771299285956, real is 17.3487
    # when init = 100, price is 10.072509537503821/9.992812015410516, real is 9.6333
```

**测试内容**：
- 测试2维美式期权的定价功能
- 参数配置：
  - 执行价：100
  - 初始价格：[100, 100]
  - 波动率：[0.2, 0.2]
  - 无风险利率：0.05
  - 股息率：[0.1, 0.1]
  - 资产间相关系数：0.3
- 支付函数：最大值看涨期权（max-call option）
- 验证逻辑：
  1. 与预期数值结果的误差小于1e-14
  2. 与理论值的相对误差小于0.783%

### 3.4 折现现金流计算测试

```python
def test_get_discounted_cashflow(self):
    random_walk = GBM(3, 3, np.ones(1), 0.03, np.ones(1), np.zeros(1), np.eye(1))
    def test_payoff(*l):
        return max(3 - np.sum(l), 0)
    opt = American(test_payoff, random_walk)
    cashflow_matrix = np.array([[0, 0, 0, 3], [0, 0, 0, 0], [0, 0, 0, 2]])
    discounted = opt._get_discounted_cashflow(2, cashflow_matrix, 3)
    assert sum(abs(discounted - np.array([2.9113366, 0, 1.94089107]))) < 0.00000001

    cashflow_matrix2 = np.array([[0, 0, 3, 0], [0, 0, 0, 0], [0, 0, 0, 2]])
    discounted2 = opt._get_discounted_cashflow(0, cashflow_matrix2, 3)
    assert sum(abs(discounted2 - np.array([2.8252936, 0, 1.82786237]))) < 0.00000001
```

**测试内容**：
- 测试`_get_discounted_cashflow`方法的正确性
- 测试场景1：
  - 现金流矩阵：3条路径，到期日（t=3）行权
  - 折现到时间t=2
  - 预期结果：[2.9113366, 0, 1.94089107]
- 测试场景2：
  - 现金流矩阵：路径1在t=2行权，路径3在t=3行权
  - 折现到时间t=0
  - 预期结果：[2.8252936, 0, 1.82786237]

### 3.5 初始时刻折现现金流计算测试

```python
def test_get_discounted_cashflow_at_t0(self):
    random_walk = GBM(3, 3, np.ones(1), 0.03, np.ones(1), np.zeros(1), np.eye(1))
    def test_payoff(*l):
        return max(3 - np.sum(l), 0)
    opt = American(test_payoff, random_walk)
    discount = opt._get_discounted_cashflow_at_t0(np.array([[0, 0, 0, 0], [0, 0, 1, 0], [0, 2, 0, 0]]))
    assert discount == (0+np.exp(-2*0.03)+2*np.exp(-1*0.03))/3
```

**测试内容**：
- 测试`_get_discounted_cashflow_at_t0`方法的正确性
- 现金流矩阵：
  - 路径1：不行权（现金流为0）
  - 路径2：在t=2行权，现金流为1
  - 路径3：在t=1行权，现金流为2
- 验证计算结果是否等于理论值：(0 + e^(-2*0.03) + 2*e^(-1*0.03))/3

## 4. 测试执行

```python
if __name__ == '__main__':
    unittest.main()
```

使用标准的unittest框架运行所有测试用例。

## 5. 测试与被测试代码的对应关系

| 测试方法 | 被测试代码 | 测试内容 |
|---------|-----------|---------|
| `test_price1d` | `American.price()` | 1D美式期权定价 |
| `test_price2d` | `American.price()` | 2D美式期权定价 |
| `test_get_discounted_cashflow` | `American._get_discounted_cashflow()` | 任意时间点折现现金流计算 |
| `test_get_discounted_cashflow_at_t0` | `American._get_discounted_cashflow_at_t0()` | 初始时刻折现现金流计算 |

## 6. 测试策略分析

### 6.1 测试覆盖范围

该测试文件覆盖了：
- 核心定价功能（1D和2D期权）
- 关键辅助方法（折现现金流计算）
- 不同维度的期权（1D和2D）
- 不同的支付函数（看跌期权和最大值看涨期权）

### 6.2 测试设计原则

1. **可重复性**：通过设置固定的随机种子（`np.random.seed()`）确保蒙特卡洛模拟结果可重现
2. **边界条件**：测试包含了不行权（现金流为0）的路径
3. **数值精度**：使用严格的数值比较（如`abs(price - expected) < 1e-10`）验证计算准确性
4. **相对误差验证**：在2D期权测试中，同时验证绝对误差和相对误差

### 6.3 测试参数选择

| 参数 | 选择理由 |
|------|---------|
| 路径数量（3000） | 平衡计算精度和测试执行时间 |
| 时间步长（300） | 足够的离散化精度，同时保持计算效率 |
| 随机种子（444, 555） | 固定种子确保测试结果可重现 |
| 相关系数（0.3） | 测试非独立资产的情况，更接近实际市场 |

## 7. 测试结果分析

### 7.1 预期测试结果

| 测试方法 | 预期结果 |
|---------|---------|
| `test_price1d` | 通过，价格约为0.07187167189125372 |
| `test_price2d` | 通过，价格约为9.557936820537265 |
| `test_get_discounted_cashflow` | 通过，折现结果与预期一致 |
| `test_get_discounted_cashflow_at_t0` | 通过，初始时刻折现价格计算正确 |

### 7.2 可能的测试失败原因

1. **随机种子问题**：如果修改了随机种子，蒙特卡洛模拟结果会变化
2. **数值精度问题**：不同的计算环境可能导致微小的数值差异
3. **参数配置问题**：修改GBM或American类的参数可能影响测试结果
4. **算法实现变更**：如果修改了American类的核心算法，测试断言需要相应更新

## 8. 测试执行指南

### 8.1 直接执行

```bash
cd /Users/susan/Downloads/hdp-master/tests
python MCAmericanTest.py
```

### 8.2 通过TestAll.py执行

```bash
cd /Users/susan/Downloads/hdp-master
bash -c "source activate fbsde_env && python tests/TestAll.py"
```

### 8.3 执行单个测试用例

```bash
python -m unittest tests.MCAmericanTest.Test.test_price1d
```

## 9. 扩展与维护建议

### 9.1 扩展测试用例

1. **增加更多维度测试**：测试3D及以上的美式期权定价
2. **测试不同支付函数**：如最小值期权、障碍期权等复杂支付结构
3. **测试不同随机过程**：除GBM外，测试其他随机过程（如跳扩散过程）
4. **性能测试**：测试不同路径数量和时间步长对计算效率的影响

### 9.2 维护建议

1. **定期更新预期值**：如果算法或参数有变更，及时更新测试断言中的预期值
2. **保持测试独立性**：确保每个测试用例独立执行，不依赖其他测试的结果
3. **添加注释说明**：对复杂的测试场景和参数配置添加详细注释
4. **使用参数化测试**：考虑使用`unittest.parameterized`简化多个参数组合的测试

## 10. 结论

`MCAmericanTest.py` 提供了全面的单元测试，验证了美式期权蒙特卡洛定价实现的正确性和稳定性。通过覆盖不同维度、不同支付函数和不同的测试场景，确保了`American`类在各种情况下都能正确工作。这些测试为代码的开发、维护和重构提供了重要的质量保障。