# PDE and Financial Option Pricing with Deep Learning

This repository contains implementations of various numerical methods for solving partial differential equations (PDEs) and pricing financial options, with a focus on high-dimensional problems.

## Table of Contents

- [Introduction](#introduction)
- [Project Structure](#project-structure)
- [Methods](#methods)
  - [Monte Carlo Methods](#monte-carlo-methods)
  - [Forward-Backward Stochastic Neural Networks (FBSNNs)](#forward-backward-stochastic-neural-networks-fbsnns)
  - [Deep Backward Stochastic Differential Equations (DeepBSDE)](#deep-backward-stochastic-differential-equations-deepbsde)
- [Applications](#applications)
  - [Black-Scholes-Barenblatt Equation](#black-scholes-barenblatt-equation)
  - [American Option Pricing](#american-option-pricing)
  - [Vanilla Call Option Pricing](#vanilla-call-option-pricing)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running Examples](#running-examples)
- [Comparison](#comparison)
- [License](#license)

## Introduction

This project explores the use of both traditional numerical methods and deep learning techniques for solving PDEs and pricing financial derivatives. The focus is on high-dimensional problems, where traditional methods often struggle due to the curse of dimensionality.

## Project Structure

```
cqf/
├── mc/                  # Monte Carlo methods for option pricing
│   ├── American.py      # American option pricing using Longstaff-Schwartz algorithm
│   ├── Paths_generater.py  # Geometric Brownian Motion path generation
│   ├── Regression.py    # Regression methods for option pricing
│   └── Test.py          # Tests for Monte Carlo methods
├── fbsnn/               # Forward-Backward Stochastic Neural Networks
│   ├── CallOption.py    # Vanilla call option implementation
│   ├── FBSNNs.py        # Main FBSNN implementation
│   ├── Models.py        # Neural network models
│   ├── Utils.py         # Utility functions
│   └── Test_*.py        # Various test cases
├── deepbsde/            # Deep BSDE implementation
│   ├── BlackScholesBarenblatt.py  # Black-Scholes-Barenblatt equation
│   ├── DeepBSDE.py      # Main DeepBSDE solver
│   ├── Models.py        # Neural network models
│   └── Test_*.py        # Various test cases
├── compare_100D.py      # Comparison of methods on 100D problem
└── requirements.txt     # Required dependencies
```

## Methods

### Monte Carlo Methods

The `mc/` directory contains implementations of Monte Carlo methods for option pricing:

- **American.py**: Implements the Longstaff-Schwartz algorithm for American option pricing, which uses regression to approximate the continuation value.
- **Paths_generater.py**: Generates Geometric Brownian Motion (GBM) paths for Monte Carlo simulations.
- **Regression.py**: Provides polynomial regression methods used in the Longstaff-Schwartz algorithm.

### Forward-Backward Stochastic Neural Networks (FBSNNs)

The `fbsnn/` directory implements the FBSNN approach for solving PDEs:

- **FBSNNs.py**: Main implementation of the FBSNN framework, which uses neural networks to solve forward-backward stochastic differential equations.
- **Models.py**: Defines neural network architectures used in FBSNNs.
- **Utils.py**: Utility functions for device setup, random seed setting, and other helper functions.

### Deep Backward Stochastic Differential Equations (DeepBSDE)

The `deepbsde/` directory implements the DeepBSDE approach:

- **DeepBSDE.py**: Main DeepBSDE solver implementation.
- **Models.py**: Neural network models used in DeepBSDE.
- **BoundsCalculator.py**: Calculates bounds for the solution using Legendre transform.

## Applications

### Black-Scholes-Barenblatt Equation

This is a nonlinear extension of the Black-Scholes equation, used to model option pricing with uncertain volatility:

- **BlackScholesBarenblatt.py**: Implementation of the equation and analytical solution.
- **Test_BlackScholesBarenblatt100D_Optuna.py**: Hyperparameter optimization for 100D problem.
- **Test_BlackScholesBarenblatt100D_benchmark.py**: Benchmark tests for 100D problem.

### American Option Pricing

The Longstaff-Schwartz algorithm is implemented for pricing American options:

- **American.py**: Core implementation of the algorithm.
- **Test.py**: Test cases for American option pricing.

### Vanilla Call Option Pricing

Implementation of vanilla call option pricing using FBSNNs:

- **CallOption.py**: Vanilla call option implementation.
- **Test_vanilla_call.py**: Test cases for vanilla call option pricing.

## Getting Started

### Prerequisites

The project requires the following dependencies:

- numpy>=1.21.0
- pytorch>=1.10.0
- matplotlib>=3.4.0
- optuna>=3.0.0
- joblib>=1.1.0
- plotly>=5.0.0
- scipy>=1.7.0

### Installation

1. Clone the repository:

```bash
git clone https://github.com/chouchouyu/PINN-slove-PDE.git
cd PINN-slove-PDE/cqf
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Running Examples

1. **Monte Carlo American Option Pricing**:

```bash
python mc/Test.py
```

2. **Vanilla Call Option with FBSNNs**:

```bash
python fbsnn/Test_vanilla_call.py
```

3. **Black-Scholes-Barenblatt Equation (100D) with DeepBSDE**:

```bash
python compare_100D.py
```

## Comparison

The `compare_100D.py` script compares the performance of different methods on solving the 100-dimensional Black-Scholes-Barenblatt equation:

- **DeepBSDE**: Deep BSDE solver with and without Legendre transform
- **FBSNN**: Forward-Backward Stochastic Neural Network

The comparison includes metrics such as accuracy, runtime, and convergence behavior.

 
