# PDE 求解器文档

本文档详细介绍了 `src/blackscholes/pde` 文件夹中的PDE（偏微分方程）求解器实现，该求解器主要用于期权定价。

## 文件夹结构

```
src/blackscholes/pde/
├── Parabolic.py    # 抛物线型PDE求解器核心实现
├── Euro.py         # 欧式期权PDE求解
└── American.py     # 美式期权PDE求解
```

## 1. Parabolic.py - 核心PDE求解器

`Parabolic.py` 实现了抛物线型PDE的数值求解器，主要用于求解时间相关的二阶PDE（如Black-Scholes方程）。

### 1.1 数学基础

抛物线型PDE的一般形式为：

```
u_t = p(x, t)u_xx + q(x, t)u_x + r(x, t)u + f(x, t)
```

其中：
- `u_t`：函数对时间的一阶偏导数
- `u_xx`：函数对空间的二阶偏导数
- `u_x`：函数对空间的一阶偏导数
- `p, q, r, f`：系数函数

### 1.2 核心类

#### Solver1d - 一维PDE求解器

```python
class Solver1d:
    def __init__(self, p, q, r, f, domain):
        # 初始化系数函数和求解域
        # p: u_xx的系数
        # q: u_x的系数
        # r: u的系数
        # f: 源项
        # domain: 求解域对象，包含边界条件和初始条件
```

##### 求解方法 - solve

```python
def solve(self, nx, nt):
    # nx: 空间离散点数
    # nt: 时间离散点数
    # 使用Crank-Nicolson方法求解PDE
    # 返回数值解
```

##### 主要实现步骤：
1. 计算空间和时间步长
2. 初始化时间和空间网格
3. 设置初始条件
4. 对于每个时间步：
   - 构建有限差分矩阵A和右侧向量B
   - 应用边界条件
   - 求解线性方程组
   - 更新解
5. 返回完整的数值解

##### 评估方法

- `evaluate(self, X, t, interp="cubic")`: 在指定时间和空间点评估解
- `evaluateV2(self, X, t, interp="linear")`: 高效评估大量点的解

#### Solver1d_penalty - 带惩罚方法的一维求解器

```python
class Solver1d_penalty(Solver1d):
    # Solver1d的子类，用于处理带不等式约束的PDE
    # 主要用于美式期权定价（期权价值≥立即行权价值）
```

##### 惩罚方法实现

```python
@staticmethod
def penalty_iter(A, b, domain_linspace, ic, err_tol, init_guess):
    # A: 有限差分矩阵
    # b: 右侧向量
    # domain_linspace: 空间离散点
    # ic: 初始条件（期权的行权价值）
    # err_tol: 误差容限
    # init_guess: 初始猜测解
    #
    # 实现惩罚迭代：
    # 1. 对于期权价值<行权价值的点，添加大的惩罚系数
    # 2. 求解修改后的线性方程组
    # 3. 重复直到收敛
```

#### Coef2d - 二维PDE系数表示

```python
class Coef2d:
    # 表示二维PDE的系数
    # 系数形式：sum_{i=0}^n f1(x, t)f2(y, t)
```

#### Solver2d - 二维PDE求解器

```python
class Solver2d:
    # 求解二维抛物线型PDE
    # u_t = coef_a*u_xx + coef_b*u_xy + coef_c*u_yy + coef_d*u_x + coef_e*u_y + coef_f*u + g(x, y, t)
```

## 2. Euro.py - 欧式期权PDE求解

`Euro.py` 实现了一维欧式期权的PDE求解器。

### 2.1 欧式期权的Black-Scholes方程

对于欧式期权，Black-Scholes方程为：

```
V_t + 0.5σ²S²V_SS + (r - δ)SV_S - rV = 0
```

其中：
- `V`：期权价值
- `S`：标的资产价格
- `t`：时间
- `σ`：波动率
- `r`：无风险利率
- `δ`：股息率

### 2.2 Euro1d类实现

```python
class Euro1d(Solver1d):
    def __init__(self, domain, vol, ir, dividend, strike, cp_type):
        # domain: 求解域
        # vol: 波动率
        # ir: 无风险利率
        # dividend: 股息率
        # strike: 行权价格
        # cp_type: 期权类型（1:看涨期权, -1:看跌期权）
        
        # 设置Black-Scholes方程的系数
        p = lambda S, t: vol**2*S**2/2  # V_SS的系数
        q = lambda S, t: (ir-dividend)*S  # V_S的系数
        r = lambda S, t: -ir*np.ones(len(S))  # V的系数
        f = lambda S, t: 0  # 源项
        
        # 设置初始条件（到期日的期权价值）
        domain.ic = lambda S, t: np.maximum(cp_type*(S - strike), 0)
        
        # 设置边界条件
        domain.bc = lambda S, t: strike*np.exp(-ir*t) if abs(S) < 7/3-4/3-1 else 0
        
        # 调用父类构造函数
        super().__init__(p, q, r, f, domain)
```

## 3. American.py - 美式期权PDE求解

`American.py` 实现了一维美式期权的PDE求解器。

### 3.1 美式期权的Black-Scholes不等式

对于美式期权，由于可以提前行权，期权价值满足不等式：

```
V_t + 0.5σ²S²V_SS + (r - δ)SV_S - rV ≤ 0
V ≥ 立即行权价值
```

### 3.2 Amer1d类实现

```python
class Amer1d(Solver1d_penalty):
    def __init__(self, domain, vol, ir, dividend, strike, cp_type):
        # 与Euro1d类的初始化参数相同
        
        # 设置Black-Scholes方程的系数（与欧式期权相同）
        p = lambda S, t: vol**2*S**2/2
        q = lambda S, t: (ir-dividend)*S
        r = lambda S, t: -ir*np.ones(len(S))
        f = lambda S, t: 0
        
        # 设置初始条件（同时也是立即行权价值）
        domain.ic = lambda S, t: np.maximum(cp_type*(S - strike), 0)
        
        # 设置边界条件
        domain.bc = lambda S, t: strike*np.exp(-ir*t) if abs(S) < 7/3-4/3-1 else 0
        
        # 调用父类构造函数（Solver1d_penalty）
        super().__init__(p, q, r, f, domain)
```

### 3.3 提前行权约束处理

美式期权的提前行权约束通过 `Solver1d_penalty` 类中的惩罚方法处理：
1. 在每个时间步，检查期权价值是否小于立即行权价值
2. 对于违反约束的点，添加大的惩罚系数到有限差分矩阵
3. 求解修改后的线性方程组
4. 重复直到所有点都满足约束

## 4. 使用示例

### 4.1 欧式期权定价示例

```python
import numpy as np
from blackscholes.pde.Euro import Euro1d

# 定义求解域
class Domain:
    def __init__(self, a, b, T):
        self.a = a  # 资产价格下限
        self.b = b  # 资产价格上限
        self.T = T  # 到期时间
    
    def get_discretization_size(self, nx, nt):
        hx = (self.b - self.a) / nx
        ht = self.T / nt
        return hx, ht

# 参数设置
domain = Domain(a=0, b=200, T=1)  # 资产价格范围：0-200，到期时间：1年
vol = 0.2  # 波动率：20%
ir = 0.05  # 无风险利率：5%
dividend = 0  # 股息率：0%
strike = 100  # 行权价格：100
cp_type = 1  # 1: 看涨期权

# 创建欧式期权求解器
euro_solver = Euro1d(domain, vol, ir, dividend, strike, cp_type)

# 求解（空间离散点：100，时间离散点：100）
euro_solver.solve(nx=100, nt=100)

# 评估在特定资产价格和时间的期权价值
t = 0.5  # 0.5年后
S = 110  # 资产价格：110
option_value = euro_solver.evaluate([S], t)[0]
print(f"欧式看涨期权价值：{option_value:.4f}")
```

### 4.2 美式期权定价示例

```python
import numpy as np
from blackscholes.pde.American import Amer1d

# 定义求解域（与欧式期权相同）
class Domain:
    def __init__(self, a, b, T):
        self.a = a
        self.b = b
        self.T = T
    
    def get_discretization_size(self, nx, nt):
        hx = (self.b - self.a) / nx
        ht = self.T / nt
        return hx, ht

# 参数设置（与欧式期权相同）
domain = Domain(a=0, b=200, T=1)
vol = 0.2
ir = 0.05
dividend = 0
strike = 100
cp_type = -1  # -1: 看跌期权

# 创建美式期权求解器
amer_solver = Amer1d(domain, vol, ir, dividend, strike, cp_type)

# 求解
amer_solver.solve(nx=100, nt=100)

# 评估期权价值
t = 0.5
S = 90
option_value = amer_solver.evaluate([S], t)[0]
print(f"美式看跌期权价值：{option_value:.4f}")

# 计算立即行权价值
immediate_exercise_value = max(cp_type*(S - strike), 0)
print(f"立即行权价值：{immediate_exercise_value:.4f}")
print(f"期权价值≥立即行权价值：{option_value >= immediate_exercise_value:.4f}")
```

## 5. 关键技术点

### 5.1 Crank-Nicolson方法

Parabolic.py中使用Crank-Nicolson方法求解PDE，这是一种隐式方法：
- 结合了显式方法的简单性和隐式方法的稳定性
- 时间上具有二阶精度，空间上具有二阶精度
- 对于Black-Scholes方程等抛物线型PDE，无条件稳定

### 5.2 惩罚方法

Solver1d_penalty类使用惩罚方法处理美式期权的提前行权约束：
- 对于违反约束（期权价值<行权价值）的点，添加大的惩罚系数
- 通过迭代调整惩罚系数，直到所有点都满足约束
- 优点：实现简单，容易扩展到高维问题
- 缺点：需要调整惩罚参数，可能影响收敛速度

### 5.3 有限差分离散化

对于Black-Scholes方程：
- 空间导数使用中心差分：`u_x = (u_{i+1} - u_{i-1})/(2hx)`，`u_xx = (u_{i+1} - 2u_i + u_{i-1})/hx²`
- 时间导数使用Crank-Nicolson近似：`u_t = (u^{n+1}_i - u^n_i)/ht`

## 6. 代码优化建议

1. **边界条件改进**：
   ```python
   # 当前边界条件
   domain.bc = lambda S, t: strike*np.exp(-ir*t) if abs(S) < 7/3-4/3-1 else 0
   
   # 改进建议：更清晰的边界条件
   domain.bc = lambda S, t: strike*np.exp(-ir*t) if S < 2*strike else 0  # 对于看跌期权
   # 或
   domain.bc = lambda S, t: (S - strike*np.exp(-ir*t)) if S > 2*strike else 0  # 对于看涨期权
   ```

2. **并行计算**：
   - 对于大规模问题，可以考虑使用并行计算加速有限差分矩阵的构建和求解
   - 可以使用NumPy或SciPy的并行功能，或考虑使用GPU加速

3. **自适应网格**：
   - 在期权价值变化剧烈的区域（如行权价格附近）使用更细的网格
   - 可以提高精度并减少计算量

## 7. 总结

pde文件夹实现了一套完整的PDE求解器，主要用于期权定价：
- `Parabolic.py` 提供了核心的PDE求解功能，支持一维和二维问题
- `Euro.py` 和 `American.py` 分别实现了欧式和美式期权的PDE求解
- 美式期权的提前行权约束通过惩罚方法处理
- 求解器使用Crank-Nicolson方法，具有良好的稳定性和精度

这套PDE求解器可以用于：
- 期权定价（欧式和美式）
- 其他金融衍生品的定价
- 科学和工程中的PDE求解问题

通过结合有限差分方法和惩罚方法，该求解器能够高效、准确地求解带约束的PDE问题，特别是金融衍生品定价中的Black-Scholes方程。