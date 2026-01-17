# 使用深度学习求解PDE和金融期权定价

该仓库包含了各种数值方法的实现，用于求解偏微分方程（PDE）和定价金融期权，特别关注高维问题。

## 目录

- [介绍](#介绍)
- [项目结构](#项目结构)
- [方法](#方法)
  - [蒙特卡洛方法](#蒙特卡洛方法)
  - [前向-后向随机神经网络（FBSNNs）](#前向-后向随机神经网络-fbsnns)
  - [深度后向随机微分方程（DeepBSDE）](#深度后向随机微分方程-deepbsde)
- [应用](#应用)
  - [Black-Scholes-Barenblatt方程](#black-scholes-barenblatt方程)
  - [美式期权定价](#美式期权定价)
  - [香草看涨期权定价](#香草看涨期权定价)
- [快速开始](#快速开始)
  - [前置条件](#前置条件)
  - [安装](#安装)
  - [运行示例](#运行示例)
- [方法比较](#方法比较)
- [许可证](#许可证)

## 介绍

本项目探索了传统数值方法和深度学习技术在求解PDE和定价金融衍生品中的应用。重点关注高维问题，传统方法在这类问题上往往因维度诅咒而难以处理。

## 项目结构

```
cqf/
├── mc/                  # 用于期权定价的蒙特卡洛方法
│   ├── American.py      # 使用Longstaff-Schwartz算法的美式期权定价
│   ├── Paths_generater.py  # 几何布朗运动路径生成器
│   ├── Regression.py    # 用于期权定价的回归方法
│   └── Test.py          # 蒙特卡洛方法的测试用例
├── fbsnn/               # 前向-后向随机神经网络
│   ├── CallOption.py    # 香草看涨期权实现
│   ├── FBSNNs.py        # FBSNN框架的主要实现
│   ├── Models.py        # 神经网络模型
│   ├── Utils.py         # 工具函数
│   └── Test_*.py        # 各种测试用例
├── deepbsde/            # 深度BSDE实现
│   ├── BlackScholesBarenblatt.py  # Black-Scholes-Barenblatt方程
│   ├── DeepBSDE.py      # DeepBSDE求解器的主要实现
│   ├── Models.py        # DeepBSDE使用的神经网络模型
│   └── Test_*.py        # 各种测试用例
├── compare_100D.py      # 100维问题上的方法比较
└── requirements.txt     # 所需依赖
```

## 方法

### 蒙特卡洛方法

`mc/`目录包含期权定价的蒙特卡洛方法实现：

- **American.py**：实现了美式期权定价的Longstaff-Schwartz算法，该算法使用回归来近似延续价值。
- **Paths_generater.py**：为蒙特卡洛模拟生成几何布朗运动（GBM）路径。
- **Regression.py**：提供Longstaff-Schwartz算法中使用的多项式回归方法。

### 前向-后向随机神经网络（FBSNNs）

`fbsnn/`目录实现了用于求解PDE的FBSNN方法：

- **FBSNNs.py**：FBSNN框架的主要实现，使用神经网络求解前向-后向随机微分方程。
- **Models.py**：定义FBSNN中使用的神经网络架构。
- **Utils.py**：用于设备设置、随机种子设置和其他辅助函数的工具函数。

### 深度后向随机微分方程（DeepBSDE）

`deepbsde/`目录实现了DeepBSDE方法：

- **DeepBSDE.py**：DeepBSDE求解器的主要实现。
- **Models.py**：DeepBSDE中使用的神经网络模型。
- **BoundsCalculator.py**：使用勒让德变换计算解的边界。

## 应用

### Black-Scholes-Barenblatt方程

这是Black-Scholes方程的非线性扩展，用于模拟波动率不确定的期权定价：

- **BlackScholesBarenblatt.py**：方程和解析解的实现。
- **Test_BlackScholesBarenblatt100D_Optuna.py**：100维问题的超参数优化。
- **Test_BlackScholesBarenblatt100D_benchmark.py**：100维问题的基准测试。

### 美式期权定价

实现了Longstaff-Schwartz算法用于定价美式期权：

- **American.py**：算法的核心实现。
- **Test.py**：美式期权定价的测试用例。

### 香草看涨期权定价

使用FBSNN实现香草看涨期权定价：

- **CallOption.py**：香草看涨期权的实现。
- **Test_vanilla_call.py**：香草看涨期权定价的测试用例。

## 快速开始

### 前置条件

项目需要以下依赖：

- numpy>=1.21.0
- pytorch>=1.10.0
- matplotlib>=3.4.0
- optuna>=3.0.0
- joblib>=1.1.0
- plotly>=5.0.0
- scipy>=1.7.0

### 安装

1. 克隆仓库：

```bash
git clone https://github.com/chouchouyu/PINN-slove-PDE.git
cd PINN-slove-PDE/cqf
```

2. 安装所需依赖：

```bash
pip install -r requirements.txt
```

### 运行示例

1. **蒙特卡洛美式期权定价**：

```bash
python mc/Test.py
```

2. **使用FBSNN的香草看涨期权**：

```bash
python fbsnn/Test_vanilla_call.py
```

3. **使用DeepBSDE的100维Black-Scholes-Barenblatt方程**：

```bash
python compare_100D.py
```

## 方法比较

`compare_100D.py`脚本比较了不同方法在求解100维Black-Scholes-Barenblatt方程时的性能：

- **DeepBSDE**：带和不带勒让德变换的深度BSDE求解器
- **FBSNN**：前向-后向随机神经网络

比较包括精度、运行时间和收敛行为等指标。

 