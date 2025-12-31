# FBSNN Model Architecture Analysis

## Table of Contents
1. [Overview](#overview)
2. [Design Principles](#design-principles)
3. [Layer Functions and Structure](#layer-functions-and-structure)
4. [Data Flow Architecture](#data-flow-architecture)
5. [Forward Propagation Mechanism](#forward-propagation-mechanism)
6. [Backward Propagation and Optimization](#backward-propagation-and-optimization)
7. [1D vs 100D Dimensional Comparison](#1d-vs-100d-dimensional-comparison)
8. [Implementation Considerations](#implementation-considerations)

---

## Overview

The Forward-Backward Stochastic Neural Network (FBSNN) is a specialized neural network architecture designed to solve Forward-Backward Stochastic Differential Equations (FBSDEs) and high-dimensional partial differential equations (PDEs) using physics-informed neural networks (PINNs).

### Key Characteristics:
- **Purpose**: Solves FBSDEs with arbitrary dimension support (1D to 100D+)
- **Approach**: Combines neural networks with stochastic differential equations
- **Flexibility**: Handles both low and high-dimensional problems
- **Physics-Informed**: Embeds differential equations directly into the loss function

---

## Design Principles

### 1. **Physics-Informed Learning**
The network is built on the principle that neural network parameters should be optimized to satisfy the differential equations that govern the physical system, not just to fit training data.

```math
L_total = L_data + λ_physics * L_physics
```

Where:
- `L_data`: Data fitting loss (if training data is available)
- `L_physics`: Physics constraint loss (residual from PDE)
- `λ_physics`: Weight balancing physics compliance

### 2. **Dimension-Agnostic Design**
The architecture is designed to handle both low-dimensional (1D) and high-dimensional (100D+) problems through:
- Scalable fully-connected layers
- Adaptive batch processing
- Efficient memory management

### 3. **Stochastic Differential Equation Compatibility**
The network implements forward and backward equations simultaneously:
- **Forward SDE**: Evolution of the state variable X(t)
- **Backward SDE**: Evolution of the adjoint/sensitivity variable Y(t)

### 4. **Modular Architecture**
Components are designed for flexibility:
- Independent encoder/decoder networks
- Customizable activation functions
- Pluggable loss functions
- Configurable optimization strategies

---

## Layer Functions and Structure

### Network Architecture Components

```
┌─────────────────────────────────────────────────────────────┐
│                    FBSNN Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Input Layer (Dimension d)                                  │
│         │                                                   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │   Hidden Layer 1 (Dense + Activation)   │               │
│  │   Units: h₁ (typically 128-512)        │               │
│  └─────────────────────────────────────────┘               │
│         │                                                   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │   Hidden Layer 2 (Dense + Activation)   │               │
│  │   Units: h₂                             │               │
│  └─────────────────────────────────────────┘               │
│         │                                                   │
│         ↓                                                   │
│  ┌─────────────────────────────────────────┐               │
│  │   Hidden Layer N (Dense + Activation)   │               │
│  │   Units: hₙ                             │               │
│  └─────────────────────────────────────────┘               │
│         │                                                   │
│         ↓                                                   │
│  Output Layer (Dense, No Activation)                        │
│  Output Dimension: 1 (for scalar Y) or d (for vector Z)   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 1. **Input Layer**
- **Dimension**: Matches problem dimensionality (d)
- **Input Type**: Time-space coordinates (t, X) or sampled SDE states
- **Normalization**: Typically normalized to [-1, 1] for numerical stability

### 2. **Hidden Layers**

#### Dense (Fully Connected) Layers
```
z = W @ x + b
```
Where:
- `W`: Weight matrix (h_out × h_in)
- `b`: Bias vector (h_out)
- `x`: Input vector (h_in)
- `z`: Pre-activation output

**Key Properties:**
- All neurons connected to all neurons in previous layer
- Weight sharing across time steps
- Typically 3-8 hidden layers for PDE problems

#### Activation Functions
Common choices:
- **ReLU**: Fast, sparse, but non-smooth
  ```
  f(z) = max(0, z)
  ```
- **Tanh**: Smooth, bounded, works well with PINNs
  ```
  f(z) = (e^z - e^(-z)) / (e^z + e^(-z))
  ```
- **Sine (Sinusoidal)**: Effective for periodic/smooth solutions
  ```
  f(z) = sin(z)
  ```
- **GELU**: Modern smooth activation
  ```
  f(z) = z * Φ(z) where Φ is cumulative Gaussian
  ```

**Selection Criteria:**
- Use smooth activations (Tanh, Sine, GELU) for PDE problems
- ReLU suits classification but may cause issues with derivatives
- Layer count typically: 4-6 hidden layers with 64-256 units each

### 3. **Output Layer**
- **Linear activation** (no non-linearity)
- **Outputs**:
  - **Y_network**: Approximates Y(t, X) - the solution
  - **Z_network**: Approximates Z(t, X) - the gradient/control

---

## Data Flow Architecture

### Training Data Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    Data Generation                            │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  1. Sample Initial Conditions (X₀)                           │
│     - Monte Carlo sampling from initial distribution         │
│     - Sample size: N (typically 100-10000)                  │
│                                                               │
│  2. Generate Time Grid                                       │
│     - Points: 0 = t₀ < t₁ < ... < tₙ = T                   │
│     - Uniform or adaptive spacing                           │
│                                                               │
│  3. Simulate Forward SDE                                     │
│     dX = μ(t, X)dt + σ(t, X)dW                             │
│     - Euler-Maruyama or Milstein scheme                     │
│     - Generate paths X^n_{i,j} for i steps, j paths        │
│                                                               │
│  4. Create Training Batch                                    │
│     Input: (t, X_{t_i}, B_t_i) - time, state, Brownian motion
│     Target: Y(T, X_T) = g(X_T) - terminal condition        │
│                                                               │
└───────────────────────────────────────────────────��──────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│                 Forward Data Flow                             │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  Input: (t_i, X_{t_i}, ...)                                 │
│    │                                                          │
│    ├─→ Concatenate or Stack                                 │
│    │   Result: [t_i, X_{t_i}] ∈ ℝ^(d+1)                   │
│    │                                                          │
│    ├─→ Y-Network Forward Pass                               │
│    │   Y_{net}(t_i, X_{t_i}) ≈ Y(t_i, X_{t_i})            │
│    │                                                          │
│    ├─→ Compute Gradients                                     │
│    │   ∇Y = ∂Y_{net}/∂X (via autograd)                    │
│    │                                                          │
│    ├─→ Z-Network Forward Pass                               │
│    │   Z_{net}(t_i, X_{t_i}) ≈ σ(t_i, X_{t_i})^T ∇Y      │
│    │                                                          │
│    └─→ Loss Computation (see section 6)                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
                          │
                          ↓
┌──────────────────────────────────────────────────────────────┐
│            Loss Aggregation and Optimization                  │
├──────────────────────────────────────────────────────────────┤
│                                                               │
│  L_total = L_FBSDE + λ_terminal * L_terminal                │
│                                                               │
│  where:                                                      │
│  - L_FBSDE: Backward equation residual                      │
│  - L_terminal: Terminal condition error                     │
│                                                               │
└──────────────────────────────────────────────────────────────┘
```

### Specific Data Structures

**Input Tensor Format**:
```
Single sample:  [t, x₁, x₂, ..., x_d] ∈ ℝ^(d+1)
Batch:          (N, d+1) where N is batch size
```

**Network Outputs**:
```
Y_output: (N, 1)     - scalar solution at each sample
Z_output: (N, d)     - gradient vector at each sample
```

---

## Forward Propagation Mechanism

### Step-by-Step Forward Pass

#### 1. **Input Preparation**
```python
# Given: time t, state X ∈ ℝ^d
input_tensor = torch.cat([t.unsqueeze(-1), X], dim=-1)  # Shape: (N, d+1)
```

#### 2. **Layer-wise Computation**

For each hidden layer l:
```
h_l = σ(W_l @ h_{l-1} + b_l)

Where:
- h_0 = input_tensor
- h_l: output of layer l
- σ: activation function
- W_l: weight matrix of layer l
- b_l: bias vector of layer l
```

**Detailed Layer Computation**:
```
Layer 0:    h_0 = input (d+1 → d+1)
Layer 1:    h_1 = σ(W_1 h_0 + b_1)  (d+1 → h_1)
Layer 2:    h_2 = σ(W_2 h_1 + b_2)  (h_1 → h_2)
...
Layer n:    h_n = σ(W_n h_{n-1} + b_n)  (h_{n-1} → h_n)
Output:     y = W_{out} h_n + b_{out}  (h_n → 1)
```

#### 3. **Y-Network Output**

```python
def Y_network_forward(t, X):
    """
    Outputs Y(t, X) - the solution to the FBSDE
    
    Args:
        t: Time tensor (N,)
        X: State tensor (N, d)
    
    Returns:
        Y: Solution (N, 1)
    """
    # Concatenate time and state
    x = torch.cat([t.reshape(-1, 1), X], dim=1)
    
    # Forward through hidden layers
    for layer in self.hidden_layers:
        x = layer(x)
        x = torch.tanh(x)  # or other activation
    
    # Final output layer
    Y = self.output_layer(x)
    
    return Y
```

#### 4. **Gradient Computation (Critical for Z)**

```python
def compute_gradient(Y, X):
    """
    Compute ∇_X Y(t, X)
    
    Uses automatic differentiation
    """
    Y.sum().backward(create_graph=True)
    grad_Y = X.grad  # Shape: (N, d)
    return grad_Y
```

**Mathematical Expression**:
```
∇_X Y = ∂Y/∂x₁, ∂Y/∂x₂, ..., ∂Y/∂x_d ∈ ℝ^(d×1)
```

#### 5. **Z-Network Output**

```python
def Z_network_forward(t, X, grad_Y):
    """
    Outputs Z(t, X) = σ(t, X)^T ∇Y(t, X)
    
    Args:
        t: Time
        X: State
        grad_Y: Gradient of Y w.r.t X
    
    Returns:
        Z: Diffusion coefficient approximation (N, d)
    """
    x = torch.cat([t.reshape(-1, 1), X], dim=1)
    
    # Forward through network
    for layer in self.z_hidden_layers:
        x = layer(x)
        x = torch.tanh(x)
    
    Z = self.z_output_layer(x)
    
    return Z
```

### Key Computations in Forward Pass

| Operation | Complexity | Notes |
|-----------|-----------|-------|
| Dense Layer | O(h_in × h_out) | Matrix multiplication |
| Activation | O(h) | Element-wise operation |
| Gradient Computation | O(h² × d) | Autograd overhead, scales with d |
| Full Forward Pass | O(L × h × d) | L layers, h units, d dimensions |

---

## Backward Propagation and Optimization

### 1. **Loss Function Formulation**

The FBSNN loss combines multiple components:

```
L_total = α·L_FBSDE + β·L_terminal + γ·L_regularization
```

#### Component A: FBSDE Loss (Backward Equation)
```
L_FBSDE = E[|Y(t_{i+1}, X_{t_{i+1}}) - (Y(t_i, X_{t_i}) - f(t_i, X_{t_i}, Y, Z)Δt - Z(t_i, X_{t_i})·ΔW_i)|²]

Where:
- Y: Neural network approximation of solution
- Z: Neural network approximation of gradient term
- f: Driver function (problem-dependent)
- ΔW: Brownian increment
- Δt: Time step
```

**Interpretation**:
- Enforces the discrete backward equation
- Couples Y and Z networks
- Acts as the main physics constraint

#### Component B: Terminal Condition Loss
```
L_terminal = E[|Y(T, X_T) - g(X_T)|²]

Where:
- g(X_T): Terminal condition (boundary condition)
```

**Importance**:
- Ensures solution satisfies boundary condition
- Typically weighted higher (β > α)
- Critical for well-posedness

#### Component C: Regularization Loss (Optional)
```
L_reg = λ₁·||W||_F² + λ₂·||b||²

Where:
- ||·||_F: Frobenius norm (weight matrices)
- λ₁, λ₂: Regularization coefficients
```

**Purpose**:
- Prevents overfitting
- Improves generalization
- Stabilizes training

### 2. **Gradient Computation Through Loss**

```
Computational Graph:

Input (t, X) 
    ↓
Y_network → Y(t, X)
    ↓         ↓
    ├─ Gradient computation → ∇Y
    ↓                           ↓
    └─────────────────────────  Z_network → Z(t, X)
                                    ↓
                            Physics constraint computation
                                    ↓
                              Loss L_FBSDE
                                    ↓
                        Backpropagation through loss
                                    ↓
                    ∇L/∂W_Y, ∇L/∂b_Y (for Y-network)
                    ∇L/∂W_Z, ∇L/∂b_Z (for Z-network)
```

### 3. **Backward Pass Algorithm**

```python
def training_step(t_batch, X_batch, dW_batch, X_terminal):
    """
    Single training iteration
    """
    # 1. Forward pass for Y
    Y_pred = Y_network(t_batch, X_batch)
    
    # 2. Compute gradient of Y (requires autograd)
    Y_pred.sum().backward(create_graph=True)
    grad_Y = X_batch.grad
    X_batch.grad = None  # Clear gradient
    
    # 3. Forward pass for Z
    Z_pred = Z_network(t_batch, X_batch)
    
    # 4. Compute FBSDE residual
    # Y_{t+Δt} should equal Y_t - f(...)*Δt - Z_t*ΔW
    Y_next = Y_network(t_batch + dt, X_batch + f(...)*dt + Z_pred*dW_batch)
    
    fbsde_residual = Y_pred - Y_next  # Backward equation
    L_fbsde = (fbsde_residual ** 2).mean()
    
    # 5. Terminal condition loss
    Y_terminal = Y_network(T, X_terminal)
    L_term = ((Y_terminal - g(X_terminal)) ** 2).mean()
    
    # 6. Total loss
    L_total = L_fbsde + λ_terminal * L_term
    
    # 7. Backward pass (compute gradients)
    L_total.backward()
    
    # 8. Parameter update (done by optimizer)
    optimizer.step()
    optimizer.zero_grad()
    
    return L_total.item()
```

### 4. **Gradient Flow Analysis**

```
Loss L
  │
  └─→ ∂L/∂Z_pred
       │
       └─→ ∂Z_pred/∂[weights_Z, X_batch]
            │
            ├─→ Gradient w.r.t weights_Z (used to update Z-network)
            └─→ ∂Z_pred/∂X (implicit constraint from physics)
       
  └─→ ∂L/∂Y_pred
       │
       └─→ ∂Y_pred/∂[weights_Y, X_batch]
            │
            ├─→ Gradient w.r.t weights_Y (used to update Y-network)
            └─→ ∂Y_pred/∂X → ∂²Y_pred/∂X² (second derivatives)
```

### 5. **Optimization Strategy**

**Common Optimizers**:
- **Adam**: Adaptive learning rates, handles sparse gradients well
  ```
  m_t = β₁·m_{t-1} + (1-β₁)·∇L
  v_t = β₂·v_{t-1} + (1-β₂)·(∇L)²
  θ_t = θ_{t-1} - α·m_t/(√v_t + ε)
  ```
  
- **SGD with Momentum**: Faster convergence for well-scaled problems
  ```
  v_t = γ·v_{t-1} + α·∇L
  θ_t = θ_{t-1} - v_t
  ```

**Learning Rate Schedule**:
```
lr(epoch) = initial_lr × decay_factor^(epoch/decay_steps)
```

Common settings:
- Initial LR: 1e-3 to 1e-4
- Decay factor: 0.9-0.99
- Decay steps: 1000-5000

### 6. **Convergence Criteria**

Training continues until one of:
1. Maximum epochs reached
2. Loss converges: |L_t - L_{t-k}| < tolerance
3. Gradient norm < threshold: ||∇L|| < ε
4. No improvement for N consecutive epochs (early stopping)

---

## 1D vs 100D Dimensional Comparison

### 1D Problem Characteristics

**Problem Setup**:
```
dX_t = μ(t, X_t)dt + σ(t, X_t)dW_t         (Forward SDE)
dY_t = -f(t, X_t, Y_t, Z_t)dt + Z_t dW_t  (Backward SDE)

Where: X_t, Y_t, Z_t ∈ ℝ (scalars)
       W_t ∈ ℝ (1D Brownian motion)
```

**Network Architecture**:
```
Input:  (t, x) ∈ ℝ²
Hidden: 3-4 layers × 64-128 units
Output: (Y, Z) ∈ ℝ²

Total parameters: ~2,000-5,000
```

**Computational Complexity**:
| Operation | Complexity | Time |
|-----------|-----------|------|
| Forward pass | O(h²) | ~1 ms |
| Gradient computation | O(h²) | ~2 ms |
| Backward pass | O(h²) | ~3 ms |
| Total per iteration | O(h²) | ~6 ms |

**Advantages**:
- Fast convergence (100-500 epochs)
- Easy debugging and visualization
- Stable training with standard optimizers
- Can solve on CPU efficiently

**Training Example**:
```
Epochs: 300
Batch size: 256
Time steps: 100
Total samples: 256 × 100 × 300 = 7.68M operations
Total training time: ~1-2 seconds
```

---

### 100D Problem Characteristics

**Problem Setup**:
```
dX_t = μ(t, X_t)dt + σ(t, X_t)dW_t         (Forward SDE)
dY_t = -f(t, X_t, Y_t, Z_t)dt + Z_t dW_t  (Backward SDE)

Where: X_t ∈ ℝ¹⁰⁰, Y_t, Z_t ∈ ℝ¹⁰⁰ (vectors)
       W_t ∈ ℝ¹⁰⁰ (100D Brownian motion)
```

**Network Architecture**:
```
Input:  (t, x₁, ..., x₁₀₀) ∈ ℝ¹⁰¹
Hidden: 4-6 layers × 256-512 units
Output: (Y₁, ..., Y₁₀₀, Z₁, ..., Z₁₀₀) ∈ ℝ²⁰⁰

Total parameters: ~200,000-500,000
```

**Computational Complexity**:
| Operation | Complexity | Time |
|-----------|-----------|------|
| Forward pass | O(100 × 512²) | ~25 ms |
| Gradient computation | O(100² × 512²) | ~250 ms |
| Backward pass | O(100 × 512²) | ~50 ms |
| Total per iteration | O(100² × 512²) | ~325 ms |

**Challenges**:
1. **Curse of Dimensionality**
   - Sample complexity: O(d^p) where p is problem degree
   - Need more samples to cover space effectively
   
2. **Computational Cost**
   - Gradient computation dominates (250 ms for gradient vs 25 ms forward)
   - Memory usage: O(d × h²) for storing activations
   - GPU required for practical training
   
3. **Convergence Issues**
   - More prone to local minima
   - Requires careful learning rate scheduling
   - May need 5,000-20,000 epochs
   
4. **Numerical Instability**
   - Gradient explosion/vanishing possible
   - Batch normalization or layer normalization helpful
   - Need gradient clipping

**Training Example**:
```
Epochs: 10,000
Batch size: 128
Time steps: 100
Total samples: 128 × 100 × 10,000 = 128M operations
Total training time: ~5-10 hours (on single GPU)
```

---

### Comparative Analysis Table

| Aspect | 1D | 100D |
|--------|---|------|
| **Input Dimension** | 2 | 101 |
| **Output Dimension** | 2 | 200 |
| **Network Size** | Small (2-5K params) | Large (200-500K params) |
| **Samples Needed** | 100-1K | 10K-100K |
| **Training Time** | 1-5 sec | 1-10 hours |
| **GPU Required** | No | Yes |
| **Convergence** | Fast (100-500 epochs) | Slow (5K-20K epochs) |
| **Overfitting Risk** | Low | High |
| **Numerical Stability** | High | Moderate |
| **Interpretability** | Easy | Difficult |
| **Gradient Cost** | ~20% of total | ~80% of total |
| **Memory Usage** | <100 MB | 1-10 GB |
| **Parallelization** | Limited benefit | High benefit |

---

### Scaling Challenges and Solutions

#### Challenge 1: Curse of Dimensionality

**Problem**:
- Number of samples needed grows exponentially with dimension
- 1D: 100 samples sufficient
- 100D: 100^d ≈ 10^200 samples would be ideal (infeasible)

**Solution - FBSNN Approach**:
```
Use neural network as function approximator instead of grid
- Network learns smooth interpolation
- Generalizes from limited samples
- Particularly effective for smooth problems
```

#### Challenge 2: Gradient Computation

**Problem**:
- Computing ∇Y for each sample is expensive in 100D
- Autograd creates computational graph of size O(d × network_size)

**Solution**:
```python
# Use gradient scaling
gradient = autograd(Y)  # O(100 × 512²) = expensive
# For 100D, this dominates computation

# Optimization: batch multiple gradients
Y_batch = network(X_batch)  # Compute many at once
grad_batch = autograd(Y_batch)  # Vectorized gradient
```

#### Challenge 3: Learning Rate Sensitivity

**1D**: Can use constant learning rate
**100D**: Requires adaptive learning rates
```python
# 1D: works fine
optimizer = SGD(lr=0.01)

# 100D: better with adaptive methods
optimizer = Adam(lr=1e-3, betas=(0.9, 0.999))
# or
optimizer = AdamW(lr=1e-4)  # with weight decay
```

#### Challenge 4: Convergence

**1D Problem**:
```
Epoch 10:   Loss = 0.5
Epoch 50:   Loss = 0.01
Epoch 100:  Loss = 0.001  ✓ Converged
```

**100D Problem**:
```
Epoch 100:   Loss = 0.3 (still high)
Epoch 1000:  Loss = 0.1
Epoch 5000:  Loss = 0.01
Epoch 10000: Loss = 0.001 ✓ Converged
```

**Strategies**:
- Batch normalization between layers
- Layer normalization (better for high-D)
- Careful weight initialization (He/Xavier)
- Learning rate warmup (gradual increase)

---

## Implementation Considerations

### 1. **Numerical Stability**

#### Activation Function Choice
```python
# For 1D: Any smooth activation works
hidden = torch.tanh(x)  # Bounded, smooth

# For 100D: Prefer bounded activations to prevent explosion
# Good choices:
hidden = torch.tanh(x)        # Bounded [-1, 1]
hidden = torch.sigmoid(x)     # Bounded [0, 1], but prone to saturation
hidden = F.gelu(x)            # Unbounded but smooth

# Avoid:
hidden = F.relu(x)            # Can cause dead neurons in high-D
hidden = F.elu(x)             # Better than ReLU but still issues
```

#### Gradient Clipping
```python
# Important for high-dimensional problems
def training_step(...):
    loss.backward()
    
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(
        model.parameters(), 
        max_norm=1.0  # Clip to max gradient norm of 1.0
    )
    
    optimizer.step()
```

#### Weight Initialization
```python
# 1D: Standard initialization works
nn.init.xavier_uniform_(layer.weight)

# 100D: Need careful scaling
def init_weights_he(module):
    if isinstance(module, nn.Linear):
        nn.init.kaiming_normal_(
            module.weight, 
            a=0, 
            mode='fan_in',  # Account for input dimension
            nonlinearity='relu'
        )
```

### 2. **Memory Management**

#### 1D Training
```python
# Can store all data in memory
X_all = torch.randn(10000, 1)  # 80 KB
Y_all = torch.randn(10000, 1)  # 80 KB
# Total: < 1 MB
```

#### 100D Training
```python
# Must use mini-batches
batch_size = 128
X_batch = torch.randn(128, 100)  # 1 MB
Y_batch = torch.randn(128, 100)  # 1 MB

# Gradient tensors add more memory
grad_tensors ≈ network_size × batch_size
# With 256-unit layers: 256 × 128 × float32 ≈ 128 KB per layer
# 6 layers ≈ 768 KB just for activations

# Store strategories:
# - Use checkpointing: trade computation for memory
# - Reduce batch size if memory limited
# - Use mixed precision (float16) to halve memory
```

### 3. **Validation Strategy**

#### For 1D Problems
```python
# Can validate on dense grid
t_test = torch.linspace(0, 1, 100)
x_test = torch.linspace(-5, 5, 100)
X_test, T_test = torch.meshgrid(x_test, t_test, indexing='ij')

Y_pred = model(T_test.flatten(), X_test.flatten())
# Compare with analytical solution or reference

# Visualize easily
plt.plot(x_test, Y_pred)
```

#### For 100D Problems
```python
# Cannot visualize, validate numerically instead
def validate(model, X_test):
    Y_pred = model(X_test)
    
    # Metrics:
    # 1. Terminal condition error
    term_error = (Y_pred[-1] - g(X_test[-1])).norm()
    
    # 2. Physics residual (FBSDE equation)
    fbsde_residual = compute_fbsde_residual(Y_pred, Z_pred)
    
    # 3. Gradient magnitude check (should be reasonable)
    grad_magnitude = Z_pred.norm(dim=1).mean()
    
    return {
        'terminal_error': term_error,
        'fbsde_residual': fbsde_residual,
        'grad_magnitude': grad_magnitude
    }
```

### 4. **Hyperparameter Tuning**

#### For 1D
```
Network size:      64-128 units, 3-4 layers
Learning rate:     1e-3 to 1e-4
Optimizer:         SGD or Adam (both work)
Epochs:            100-500
Batch size:        32-256
Activation:        tanh or sine
```

#### For 100D
```
Network size:      256-512 units, 4-6 layers
Learning rate:     1e-4 to 1e-5 (smaller!)
Optimizer:         Adam (required)
Epochs:            5,000-20,000
Batch size:        128-512 (must be large)
Activation:        tanh or gelu (smooth)
Regularization:    L2 regularization 1e-5 to 1e-4
Learning rate decay: 0.9-0.99 per 1000 epochs
Gradient clipping:  max_norm = 1.0
```

### 5. **Debugging Strategy**

#### 1D Debugging Checklist
```python
# 1. Check input shapes
assert X.shape == (N, 1)
assert Y.shape == (N, 1)

# 2. Monitor loss components separately
print(f"L_FBSDE: {L_fbsde:.6f}")
print(f"L_terminal: {L_terminal:.6f}")
print(f"L_total: {L_total:.6f}")

# 3. Visualize predictions vs true solution
plot_1d_comparison()

# 4. Check gradient magnitude
grad_norm = compute_grad_norm(model)
print(f"Gradient norm: {grad_norm:.6f}")
```

#### 100D Debugging Checklist
```python
# 1. Verify dimension compatibility
assert X.shape == (N, 100)
assert Z.shape == (N, 100)

# 2. Check for NaN/Inf
assert not torch.isnan(loss).any()
assert not torch.isinf(loss).any()

# 3. Monitor loss statistics
loss_history.append(loss.item())
print(f"Loss: {loss.item():.6f}, "
      f"Mean: {np.mean(loss_history[-100:]):.6f}")

# 4. Validate dimensions at each step
Y_pred = model.Y_net(t, X)
assert Y_pred.shape == (N, 1)

grad_Y = compute_gradient(Y_pred, X)
assert grad_Y.shape == (N, 100)

# 5. Check if network is learning
print(f"Epoch 100 loss: {loss_history[100]:.6f}")
print(f"Epoch 1000 loss: {loss_history[1000]:.6f}")
if loss_history[1000] >= loss_history[100]:
    print("WARNING: Not converging!")
```

---

## Summary

### Key Takeaways

1. **Architecture Design**:
   - Modular with separate Y and Z networks
   - Uses fully-connected layers with smooth activations
   - Flexible for any dimension from 1D to 100D+

2. **Forward Pass**:
   - Concatenates time and state as input
   - Computes Y(t,X) directly
   - Differentiates to get ∇Y, then computes Z
   - Output networks are linear (no final activation)

3. **Backward Propagation**:
   - Loss couples FBSDE residual and terminal condition
   - Gradient flow through both networks simultaneously
   - Requires second-order derivatives for ∇Y computation

4. **Dimensional Scaling**:
   - 1D: Simple, fast, all components manageable
   - 100D: Complex, slow gradient computation (80% of time), needs GPU
   - Main bottleneck: Computing gradients in high dimensions

5. **Practical Strategies**:
   - Use smooth activations (tanh, gelu) for PDEs
   - Apply gradient clipping for stability
   - Use adaptive optimizers (Adam) for high-D
   - Implement layer normalization if needed
   - Monitor loss components separately

### When to Use FBSNN

**Good for**:
- Solving FBSDEs and PDEs
- Problems with 1-100 dimensions
- When you have terminal conditions
- High-dimensional financial problems
- Path-dependent options pricing

**Less suitable for**:
- Problems > 100 dimensions (exponential cost)
- Real-time inference with GPU unavailable
- When analytical solutions exist
- Very stiff ODEs/SDEs (may need specialized schemes)

