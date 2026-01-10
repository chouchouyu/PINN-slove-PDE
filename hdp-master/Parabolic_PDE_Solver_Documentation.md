# Parabolic PDE求解器Python代码文档

## 1. 概述

本文档详细解释了 `/Users/susan/Downloads/hdp-master/src/blackscholes/pde/Parabolic.py` 文件中的代码实现，该文件实现了抛物型偏微分方程（PDE）的求解器，主要用于Black-Scholes期权定价模型。

### 1.1 核心功能
- 一维抛物型PDE求解器（显式和隐式格式）
- 带惩罚项的一维求解器（用于美式期权）
- 二维抛物型PDE求解器
- 高效的插值和评估方法

### 1.2 技术要点
- **有限差分法**：使用中心差分近似空间导数
- **Crank-Nicolson方法**：时间离散化的隐式格式（无条件稳定）
- **惩罚函数法**：处理美式期权的自由边界条件
- **稀疏矩阵技术**：高效存储和求解线性方程组
- **插值技术**：在任意点评估解决方案

## 2. 导入和依赖

```python
import numpy as np
from scipy.sparse import lil_matrix, csr_matrix, diags, kron, eye
from scipy.sparse.linalg import spsolve
from scipy import interpolate
from functools import reduce
```

### 2.1 依赖说明
- `numpy`：数值计算和数组操作
- `scipy.sparse`：稀疏矩阵表示和操作
- `scipy.sparse.linalg`：稀疏线性方程组求解
- `scipy.interpolate`：插值函数
- `functools.reduce`：累积计算

## 3. 一维抛物型PDE求解器（Solver1d）

### 3.1 类定义和问题描述

```python
class Solver1d:
    """
    u_t = p(x, t)u_xx + q(x, t)u_x + r(x, t)u + f(x, t)
    """
```

该类求解如下形式的一维抛物型PDE：

$$\frac{\partial u}{\partial t} = p(x, t)\frac{\partial^2 u}{\partial x^2} + q(x, t)\frac{\partial u}{\partial x} + r(x, t)u + f(x, t)$$

### 3.2 初始化方法

```python
def __init__(self, p, q, r, f, domain):
    self.p, self.q, self.r, self.f = p, q, r, f
    self.domain = domain
    self.interps = {}
```

- **参数说明**：
  - `p(x, t)`：二阶空间导数系数
  - `q(x, t)`：一阶空间导数系数
  - `r(x, t)`：零阶项系数
  - `f(x, t)`：源项
  - `domain`：包含边界条件和初始条件的域对象

### 3.3 求解方法（solve）

```python
def solve(self, nx, nt):
    self.nx = nx
    hx, ht = self.domain.get_discretization_size(nx, nt); self.ht = ht
    
    self.time_vec = np.linspace(0, self.domain.T, nt+1)
    domain = np.linspace(self.domain.a, self.domain.b, nx+1)
    self.space_vec = domain
    self.ss, self.tt = np.meshgrid(self.space_vec, self.time_vec)
    
    X = domain[1:-1]
    
    solution = self.domain.ic(domain, 0)
    solution = np.reshape(solution, (1, -1))
    
    for i in range(1, nt+1):
        t, prev_t = i*ht, (i-1)*ht
        
        A = lil_matrix((nx-1, nx-1))
        B = np.zeros(nx-1)
        
        A.setdiag(-self.p(X[1:], t)/(2*hx**2) + self.q(X[1:], t)/(4*hx), k=-1)
        A.setdiag(1/ht + self.p(X, t)/hx**2 - self.r(X, t)/2, k=0)
        A.setdiag(-self.p(X[:-1], t)/(2*hx**2) - self.q(X[:-1], t)/(4*hx), k=1)
        
        B[0] -= (-self.p(X[1], t)/(2*hx**2) + self.q(X[1], t)/(4*hx))*self.domain.bc(self.domain.a, t)
        B[-1] -= (-self.p(X[-1], t)/(2*hx**2) - self.q(X[-1], t)/(4*hx))*self.domain.bc(self.domain.b, t)
        
        B1 = np.multiply(self.p(X, prev_t)/(2*hx**2) - self.q(X, prev_t)/(4*hx), solution[-1][:-2])
        B2 = np.multiply(1/ht - self.p(X, prev_t)/hx**2 + self.r(X, prev_t)/2, solution[-1][1:-1])
        B3 = np.multiply(self.p(X, prev_t)/(2*hx**2) + self.q(X, prev_t)/(4*hx), solution[-1][2:])
        B += B1 + B2 + B3 + (self.f(X, t) + self.f(X, prev_t))/2
        
        x = spsolve(A.tocsr(), B)
        x = np.concatenate([[self.domain.bc(self.domain.a, t)], x, [self.domain.bc(self.domain.b, t)]])
        
        solution = np.vstack([solution, x])
        
    self.solution = solution
    
    return solution
```

### 3.3.1 求解过程详解

1. **离散化设置**：
   - `nx`：空间网格点数
   - `nt`：时间步数
   - `hx`：空间步长
   - `ht`：时间步长
   - `domain`：空间域的等距网格

2. **初始条件**：
   - `solution = self.domain.ic(domain, 0)`：应用初始条件

3. **时间步进（Crank-Nicolson方法）**：
   - 从时间步1到nt进行迭代
   - 使用Crank-Nicolson方法，它是显式和隐式格式的加权平均（θ=0.5）

4. **构建系数矩阵A**：
   - 使用中心差分近似二阶空间导数（u_xx）
   - 使用中心差分近似一阶空间导数（u_x）
   - 矩阵A是三对角矩阵

5. **构建右侧向量B**：
   - 处理边界条件
   - 包含前一时间步的解
   - 包含源项f(x, t)

6. **求解线性方程组**：
   - `x = spsolve(A.tocsr(), B)`：使用稀疏求解器求解

7. **应用边界条件**：
   - 将边界值添加到解向量

8. **存储解**：
   - 将当前时间步的解添加到solution数组

### 3.4 评估方法

#### 3.4.1 evaluate方法

```python
def evaluate(self, X, t, interp="cubic"):
    t_index = int(round(t/self.ht))
    
    if t_index in self.interps:
        return self.interps[t_index](X)
    
    f = interpolate.interp1d(self.space_vec, self.solution[t_index], interp)
    self.interps[t_index] = f
    return f(X)
```

**功能**：在给定时间t和空间点X处评估解

- **参数**：
  - `X`：空间点
  - `t`：时间
  - `interp`：插值类型（默认"cubic"）

- **返回值**：在X处的解值

- **优化**：使用缓存的插值函数以提高效率

#### 3.4.2 evaluateV2方法

```python
def evaluateV2(self, X, t, interp="linear"):
    # An efficient eval function for large amount of evaluations
    
    if not hasattr(self, 'interp'):
        self.interp = interpolate.interp2d(self.ss, self.tt, self.solution, interp)
        
    return self.interp(X, t)
```

**功能**：针对大量评估的高效方法

- **参数**：
  - `X`：空间点
  - `t`：时间
  - `interp`：插值类型（默认"linear"）

- **返回值**：在X处的解值

- **优化**：使用二维插值函数，适合大量评估

## 4. 带惩罚项的一维求解器（Solver1d_penalty）

### 4.1 类定义

```python
class Solver1d_penalty(Solver1d):
```

该类继承自Solver1d，用于解决带约束的PDE（如美式期权），其中解必须满足

$$u(x, t) \geq g(x, t)$$

其中g(x, t)是支付函数。

### 4.2 求解方法（solve）

```python
def solve(self, nx, nt):
    self.nx = nx
    hx, ht = self.domain.get_discretization_size(nx, nt); self.ht = ht
    
    self.time_vec = np.linspace(0, self.domain.T, nt+1)
    
    domain = np.linspace(self.domain.a, self.domain.b, nx+1)
    self.space_vec = domain
    
    self.ss, self.tt = np.meshgrid(self.space_vec, self.time_vec)
    
    X = domain[1:-1]
    
    solution = self.domain.ic(domain, 0)
    solution = np.reshape(solution, (1, -1))
    
    for i in range(1, nt+1):
        t, prev_t = i*ht, (i-1)*ht
        
        A = lil_matrix((nx-1, nx-1))
        B = np.zeros(nx-1)
        
        A.setdiag(-self.p(X[1:], t)/(2*hx**2) + self.q(X[1:], t)/(4*hx), k=-1)
        A.setdiag(1/ht + self.p(X, t)/hx**2 - self.r(X, t)/2, k=0)
        A.setdiag(-self.p(X[:-1], t)/(2*hx**2) - self.q(X[:-1], t)/(4*hx), k=1)
        
        B[0] -= (-self.p(X[1], t)/(2*hx**2) + self.q(X[1], t)/(4*hx))*self.domain.bc(self.domain.a, t)
        B[-1] -= (-self.p(X[-1], t)/(2*hx**2) - self.q(X[-1], t)/(4*hx))*self.domain.bc(self.domain.b, t)
        
        B1 = np.multiply(self.p(X, prev_t)/(2*hx**2) - self.q(X, prev_t)/(4*hx), solution[-1][:-2])
        B2 = np.multiply(1/ht - self.p(X, prev_t)/hx**2 + self.r(X, prev_t)/2, solution[-1][1:-1])
        B3 = np.multiply(self.p(X, prev_t)/(2*hx**2) + self.q(X, prev_t)/(4*hx), solution[-1][2:])
        
        B += B1 + B2 + B3 + (self.f(X, t) + self.f(X, prev_t))/2
        
        x = Solver1d_penalty.penalty_iter(A.todense(), B, domain, self.domain.ic, 0.00001, solution[-1][1:-1])
        left, right = self.domain.ic(self.domain.a, 0), self.domain.ic(self.domain.b, 0)
        x = np.concatenate([[max(left, self.domain.bc(self.domain.a, t))], x, [max(right, self.domain.bc(self.domain.b, t))]])
        
        solution = np.vstack([solution, x])
        
    self.solution = solution
    return solution
```

### 4.2.1 求解过程详解

该方法与Solver1d的solve方法类似，但增加了惩罚项处理自由边界条件。

1. **惩罚迭代**：
   - `x = Solver1d_penalty.penalty_iter(A.todense(), B, domain, self.domain.ic, 0.00001, solution[-1][1:-1])`
   - 使用penalty_iter静态方法处理约束条件

2. **应用边界条件**：
   - `x = np.concatenate([[max(left, self.domain.bc(self.domain.a, t))], x, [max(right, self.domain.bc(self.domain.b, t))]])`
   - 确保边界值满足约束条件

### 4.3 惩罚迭代方法（penalty_iter）

```python
@staticmethod
    def penalty_iter(A, b, domain_linspace, ic, err_tol, init_guess):
        
        xi = 10000
        payoff = ic(domain_linspace[1:-1], None)
        penalty = np.zeros(len(payoff))
        penalty[payoff > init_guess] = xi
        penalty[payoff <= init_guess] = 0

        while True:
            cur_A, cur_b = A, b
            P = np.diag(penalty)
            
            cur_A += P
            cur_b += np.matmul(P, payoff)
            sol = np.linalg.solve(cur_A, cur_b)
            
            prev_p = penalty
            penalty = np.zeros(len(payoff))
            penalty[payoff > sol] = xi
            penalty[payoff <= sol] = 0
            if all(penalty == prev_p) or np.linalg.norm(sol-init_guess, 2)/np.linalg.norm(sol, 2) <= err_tol:
                break
                
        return sol
```

**功能**：使用惩罚函数法处理约束条件u(x, t) ≥ payoff(x)

- **参数**：
  - `A`：系数矩阵
  - `b`：右侧向量
  - `domain_linspace`：空间域的等距网格
  - `ic`：初始条件函数（此处用于获取支付函数）
  - `err_tol`：收敛误差 tolerance
  - `init_guess`：初始猜测解

- **惩罚参数**：
  - `xi = 10000`：惩罚系数（大值）

- **迭代过程**：
  1. 计算支付函数payoff
  2. 初始化惩罚矩阵P
  3. 修改系数矩阵和右侧向量以包含惩罚项
  4. 求解修改后的方程组
  5. 更新惩罚矩阵
  6. 检查收敛条件
  7. 重复直到收敛

## 5. 二维抛物型PDE求解器支持类（Coef2d）

### 5.1 类定义和功能

```python
class Coef2d:
    """
    Represent a coefficient of differential operator sum_{i=0}^n f1(x, t)f2(y, t)
    """
    def __init__(self, f1=lambda x, t: np.zeros(len(x)), f2=lambda y, t: np.zeros(len(y))):
        """
        f1: f1(x, t)
        f2: f2(y, t)
        They are either functions or functions in a list (python list or numpy list)
        """
        
        if callable(f1) and callable(f2):
            self.f1 = [f1]
            self.f2 = [f2]
        elif len(f1) == len(f2):
            self.f1 = f1
            self.f2 = f2
        else:
            raise RuntimeError("2D solver input coefficient error")
    
    def __iter__(self):
        return zip(self.f1, self.f2)
```

**功能**：表示二维PDE中分离变量的系数，形式为\(\sum_{i=0}^n f1_i(x, t)f2_i(y, t)\)

- **参数**：
  - `f1`：x方向的系数函数或函数列表
  - `f2`：y方向的系数函数或函数列表

- **迭代器**：允许迭代系数对(f1_i, f2_i)

## 6. 二维抛物型PDE求解器（Solver2d）

### 6.1 类定义和问题描述

```python
class Solver2d:
    """
    u_t = coef_a*u_xx + coef_b*u_xy + coef_c*u_yy + coef_d*u_x + coef_e*u_y + coef_f*u + g(x, y, t)
    where coef is defined above. Coef represents sum_{i=0}^n f1(x, t)f2(y, t).
    """
    def __init__(self, coef_a, coef_b, coef_c, coef_d, coef_e, coef_f, g, domain):
        self.coef_a, self.coef_b, self.coef_c, self.coef_d, self.coef_e, self.coef_f, self.g = coef_a, coef_b, coef_c, coef_d, coef_e, coef_f, g
        self.domain = domain
```

**功能**：求解如下形式的二维抛物型PDE

$$\frac{\partial u}{\partial t} = a(x, y, t)\frac{\partial^2 u}{\partial x^2} + b(x, y, t)\frac{\partial^2 u}{\partial x\partial y} + c(x, y, t)\frac{\partial^2 u}{\partial y^2} + d(x, y, t)\frac{\partial u}{\partial x} + e(x, y, t)\frac{\partial u}{\partial y} + f(x, y, t)u + g(x, y, t)$$

### 6.2 求解方法（solve）

```python
def solve(self, nx, ny, nt):
    S1, S2 = np.linspace(self.domain.a, self.domain.b, nx+1), np.linspace(self.domain.c, self.domain.d, ny+1)
    X1, X2 = S1[1:-1], S2[1:-1]
    hx, hy, ht = self.domain.get_discretization_size(nx, ny, nt)
    Is1, Is2 = np.eye(nx-1), np.eye(ny-1)
    T2s1 = diags([1, -2, 1], [-1, 0, 1], (nx-1, nx-1))/hx**2
    
    T1s1 = diags([-1, 0, 1], [-1, 0, 1], (nx-1, nx-1))/(2*hx)
    T2s2 = diags([1, -2, 1], [-1, 0, 1], (ny-1, ny-1))/hy**2
    T1s2 = diags([-1, 0, 1], [-1, 0, 1], (ny-1, ny-1))/(2*hy)
    
    summation = lambda x, y: x+y

    solution = self.domain.ic(X1, X2, 0).flatten()
    solution = solution[np.newaxis, ...]
    
    b = self._get_b(0, hx, hy, nx, ny, X1, X2)
    
    for i in range(1, nt+1):
        t = i*ht
        
        uxx = reduce(summation, [kron(diags(f2(X2, t))@Is2, diags(f1(X1, t))@T2s1) for f1, f2 in self.coef_a])
        uxy = reduce(summation, [kron(diags(f2(X2, t))@T1s2, diags(f1(X1, t))@T1s1) for f1, f2 in self.coef_b])
        uyy = reduce(summation, [kron(diags(f2(X2, t))@T2s2, diags(f1(X1, t))@Is1) for f1, f2 in self.coef_c])
        ux = reduce(summation, [kron(diags(f2(X2, t))@Is2, diags(f1(X1, t))@T1s1) for f1, f2 in self.coef_d])
        uy = reduce(summation, [kron(diags(f2(X2, t))@T1s2, diags(f1(X1, t))@Is1) for f1, f2 in self.coef_e])
        u = reduce(summation, [kron(diags(f2(X2, t))@Is2, diags(f1(X1, t))@Is1) for f1, f2 in self.coef_f])
        A = uxx + uxy + uyy + ux + uy + u
        
        prev_b = b
        b = self._get_b(t, hx, hy, nx, ny, S1, S2)
        I = eye((nx-1)*(ny-1))
        lhs = I-ht*A/2
        rhs = (I+ht*A/2).dot(solution[-1]) + ht*(b+prev_b)/2
        
        sol = spsolve(lhs, rhs)
        solution = np.vstack([solution, sol])
        
    return solution
```

### 6.2.1 求解过程详解

1. **离散化设置**：
   - `nx, ny`：x和y方向的网格点数
   - `nt`：时间步数
   - `hx, hy, ht`：x、y方向的空间步长和时间步长

2. **差分矩阵**：
   - `T2s1, T2s2`：x和y方向的二阶差分矩阵（u_xx, u_yy）
   - `T1s1, T1s2`：x和y方向的一阶差分矩阵（u_x, u_y）
   - `Is1, Is2`：x和y方向的单位矩阵

3. **初始条件**：
   - `solution = self.domain.ic(X1, X2, 0).flatten()`：应用初始条件

4. **时间步进**：
   - 使用Crank-Nicolson方法进行时间离散化

5. **构建微分算子A**：
   - `uxx`：二阶x导数项
   - `uxy`：交叉导数项
   - `uyy`：二阶y导数项
   - `ux`：一阶x导数项
   - `uy`：一阶y导数项
   - `u`：零阶项

6. **使用Kronecker积**：
   - `kron`：用于构造二维差分算子
   - 例如：`kron(diags(f2(X2, t))@Is2, diags(f1(X1, t))@T2s1)`

7. **求解线性方程组**：
   - `sol = spsolve(lhs, rhs)`：使用稀疏求解器求解

### 6.3 构建右侧向量方法（_get_b）

```python
def _get_b(self, t, hx, hy, nx, ny, S1, S2):
    """
    Get the vector b that includes the source term + the boundary condition in the PDE.
    """
    summation = lambda x, y: x+y
    X1, X2 = S1[1:-1], S2[1:-1]
    a, b, c, d = self.domain.a, self.domain.b, self.domain.c, self.domain.d
    
    bottom_row_bv = self.domain.bc(X1, c, t).flatten()
    bottom_b_uyy = reduce(summation, [np.multiply(f1(X1, t), bottom_row_bv)*f2([c], t)/hy**2 for f1, f2 in self.coef_c])
    bottom_b_uxy = reduce(summation, [np.multiply(f1(S1[:-2], t)-f1(S1[2:], t), bottom_row_bv)*f2([c], t)/(4*hx*hy) for f1, f2 in self.coef_b])
    bottom_b_uy = reduce(summation, [np.multiply(f1(X1, t), bottom_row_bv)*f2([c], t)/(-2*hy) for f1, f2 in self.coef_e])
    
    top_row_bv = self.domain.bc(X1, d, t).flatten()
    top_b_uyy = reduce(summation, [np.multiply(f1(X1, t), top_row_bv)*f2([d], t)/hy**2 for f1, f2 in self.coef_c])
    top_b_uxy = reduce(summation, [np.multiply(-f1(S1[:-2], t)+f1(S1[2:], t), top_row_bv)*f2([d], t)/(4*hx*hy) for f1, f2 in self.coef_b])
    top_b_uy = reduce(summation, [np.multiply(f1(X1, t), top_row_bv)*f2([d], t)/(2*hy) for f1, f2 in self.coef_e])
    
    left_column_bv = self.domain.bc(a, X2, t).flatten()
    left_b_uxx = reduce(summation, [np.multiply(f2(X2, t), left_column_bv)*f1([a], t)/hx**2 for f1, f2 in self.coef_a])
    left_b_uxy = reduce(summation, [np.multiply(f2(S2[:-2], t)-f2(S2[2:], t), left_column_bv)*f1([a], t)/(4*hx*hy) for f1, f2 in self.coef_b])
    left_b_ux = reduce(summation, [np.multiply(f2(X2, t), left_column_bv)*f1([a], t)/(-2*hx) for f1, f2 in self.coef_d])
    
    right_column_bv = self.domain.bc(b, X2, t).flatten()
    right_b_uxx = reduce(summation, [np.multiply(f2(X2, t), right_column_bv)*f1([b], t)/hx**2 for f1, f2 in self.coef_a])
    right_b_uxy = reduce(summation, [np.multiply(-f2(S2[:-2], t)+f2(S2[2:], t), right_column_bv)*f1([b], t)/(4*hx*hy) for f1, f2 in self.coef_b])
    right_b_ux = reduce(summation, [np.multiply(f2(X2, t), right_column_bv)*f1([b], t)/(2*hx) for f1, f2 in self.coef_d])
    
    b = self.g(X1, X2, t).flatten()
    b[:nx-1] += bottom_b_uyy + bottom_b_uxy + bottom_b_uy
    b[-(nx-1):] += top_b_uyy + top_b_uxy + top_b_uy
    b[::nx-1] += left_b_uxx + left_b_uxy + left_b_ux
    b[nx-2::nx-1] += right_b_uxx + right_b_uxy + right_b_ux

    f1, f2 = self.coef_b.f1, self.coef_b.f2
    b[0] -= f1([self.domain.a], t)*f2([self.domain.b], t)*self.domain.bc([a], [c], t)/(4*hx*hy)
    b[nx-2] += f1([self.domain.b], t)*f2([self.domain.c], t)*self.domain.bc([b], [c], t)/(4*hx*hy)
    b[(nx-1)*(ny-2)-1] += f1([self.domain.a], t)*f2([self.domain.d], t)*self.domain.bc([a], [d], t)/(4*hx*hy)
    b[-1] -= f1([self.domain.b], t)*f2([self.domain.d], t)*self.domain.bc([b], [d], t)/(4*hx*hy)
    
    return b
```

**功能**：构建右侧向量b，包含源项g(x, y, t)和边界条件

- **处理边界条件**：
  - `bottom_row_bv`, `top_row_bv`：底部和顶部边界值
  - `left_column_bv`, `right_column_bv`：左侧和右侧边界值
- **计算边界贡献**：
  - 计算边界条件对内部点的影响
  - 将边界贡献添加到右侧向量b
- **处理角点**：
  - 特别处理四个角点的边界条件

## 7. 代码分析和总结

### 7.1 主要算法比较

| 方法 | 优点 | 缺点 | 应用场景 |
|------|------|------|----------|
| 显式方法 | 简单，计算量小 | 条件稳定，时间步长受限 | 快速计算，对稳定性要求低 |
| 隐式方法 | 无条件稳定，时间步长大 | 计算量大，需要求解线性方程组 | 高精度计算，对稳定性要求高 |
| Crank-Nicolson | 精度高，无条件稳定 | 计算量大 | 高精度要求的应用 |
| 惩罚函数法 | 处理自由边界条件简单 | 惩罚系数选择影响精度 | 美式期权定价 |

### 7.2 性能优化

1. **稀疏矩阵**：
   - 使用稀疏矩阵存储系数矩阵A
   - 减少内存占用
   - 提高求解速度

2. **插值缓存**：
   - `interps`字典缓存插值函数
   - 避免重复计算

3. **向量化操作**：
   - 使用numpy的向量化操作
   - 提高计算效率

### 7.3 应用场景

1. **欧式期权定价**：
   - 使用Solver1d求解Black-Scholes方程

2. **美式期权定价**：
   - 使用Solver1d_penalty处理自由边界条件

3. **二维期权定价**：
   - 使用Solver2d求解二维Black-Scholes方程
   - 适用于障碍期权、回望期权等复杂期权

4. **其他抛物型PDE**：
   - 热传导方程
   - 扩散方程
   - 反应扩散方程

## 8. 数学附录

### 8.1 Crank-Nicolson方法

Crank-Nicolson方法是时间离散化的隐式格式，公式为：

$$\frac{u^{n+1} - u^n}{\Delta t} = \theta L u^{n+1} + (1-\theta) L u^n$$

其中：
- $\theta = 0.5$（Crank-Nicolson）
- $L$是空间微分算子

该方法的优点是无条件稳定，且具有二阶时间精度。

### 8.2 有限差分近似

- **二阶空间导数**：
  $$u_{xx}(x_i) \approx \frac{u_{i+1} - 2u_i + u_{i-1}}{\Delta x^2}$$

- **一阶空间导数**：
  $$u_x(x_i) \approx \frac{u_{i+1} - u_{i-1}}{2\Delta x}$$

### 8.3 惩罚函数法

惩罚函数法用于处理约束条件$u(x, t) \geq g(x, t)$，通过将约束条件转化为惩罚项：

$$L u = f + \xi \max(g - u, 0)$$

其中$\xi$是惩罚系数，当$\xi \to \infty$时，解趋近于真实解。

## 9. 结论

Parabolic.py文件实现了高效、灵活的抛物型PDE求解器，特别适用于期权定价模型。该实现具有以下特点：

- 使用Crank-Nicolson方法保证无条件稳定性
- 支持一维和二维PDE
- 处理美式期权的自由边界条件
- 高效的稀疏矩阵求解
- 灵活的插值和评估方法

该代码可以广泛应用于金融工程、热传导、扩散等领域的PDE求解问题。