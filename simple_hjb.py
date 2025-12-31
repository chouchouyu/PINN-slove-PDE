import torch
from FBSNNs import FBSNN
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch.nn as nn

class LQRProblem(FBSNN):
    def __init__(self, Xi, T, M, N, D, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, layers, mode, activation)
        self.A = torch.eye(D).to(self.device)
        self.B = torch.eye(D).to(self.device)
        self.Q = torch.eye(D).to(self.device)
        self.R = torch.eye(D).to(self.device)
        self.G = torch.eye(D).to(self.device)

    def phi_tf(self, t, X, Y, Z):
        Rinv = torch.inverse(self.R)
        a = -torch.bmm(Z.unsqueeze(1), Rinv.unsqueeze(0).expand(Z.size(0), -1, -1)).squeeze(1)
        term1 = torch.sum((X @ self.Q) * X, dim=1, keepdim=True)
        term2 = 0.5 * torch.sum(a * (a @ self.R), dim=1, keepdim=True)
        return term1 + term2
    
    def g_tf(self, X):
        return torch.sum((X @ self.G) * X, dim=1, keepdim=True)

    def mu_tf(self, t, X, Y, Z):
        Rinv = torch.inverse(self.R)
        a = -torch.bmm(Z.unsqueeze(1), Rinv.unsqueeze(0).expand(Z.size(0), -1, -1)).squeeze(1)
        return (X @ self.A.T) + (a @ self.B.T)

    def sigma_tf(self, t, X, Y):
        return torch.eye(self.D).unsqueeze(0).repeat(X.shape[0], 1, 1).to(self.device) * 0.1

def riccati_ode(t, P_flat, A, B, Q, Rinv):
    D = A.shape[0]
    P = P_flat.reshape(D, D)
    dPdt = -A.T @ P - P @ A + P @ B @ Rinv @ B.T @ P - Q
    return dPdt.flatten()

def riccati_solution(T, t_tensor, A, B, Q, R, G):
    """
    Returns P(t) evaluated at each t in t_tensor.
    Assumes A, B, Q, R, G are numpy arrays (D x D)
    t_tensor: [M, N+1, 1] torch tensor of time points (on GPU)
    """

    Rinv = np.linalg.inv(R)
    D = A.shape[0]

    # Time grid for integration
    t_eval = np.linspace(0, T, t_tensor.shape[1])
    sol = solve_ivp(
        fun=lambda t, y: riccati_ode(t, y, A, B, Q, Rinv),
        t_span=[T, 0],  # integrate BACKWARD
        y0=G.flatten(),
        t_eval=t_eval[::-1],  # reverse time grid to match backward solve
        method='RK45'
    )

    # Result is shape (D*D, N+1)
    P_seq = np.copy(sol.y.T[::-1])  # [N+1, D*D] with safe strides
    P_seq = P_seq.reshape(-1, D, D)  # [N+1, D, D]

    # Broadcast to batch: [M, N+1, D, D]
    M = t_tensor.shape[0]
    P_batched = torch.tensor(P_seq, dtype=torch.float32).unsqueeze(0).expand(M, -1, -1, -1).to(t_tensor.device)
    return P_batched

def u_exact(t, X, P_t):
    """
    Compute u(t, x) = P(t) * x^2 for 1D, or xᵀ P(t) x for multi-D
    """
    if P_t.ndim == 3 and P_t.shape[-1] == 1:
        # Scalar (1D) case
        return P_t * X**2  # Elementwise: [M, N+1, 1]
    
    # Multi-D logic for P_t: [M, N+1, D, D]
    M, N_plus_1, D = X.shape
    if P_t.ndim == 2:  # [D, D]
        P_t = P_t.unsqueeze(0).unsqueeze(0).expand(M, N_plus_1, D, D)
    elif P_t.ndim == 4 and P_t.shape[:2] != (M, N_plus_1):
        raise ValueError(f"Expected P_t shape [M,N+1,D,D], got {P_t.shape}")

    X_unsq = X.unsqueeze(-2)
    P_X = torch.matmul(X_unsq, P_t)
    X_P_X = torch.matmul(P_X, X_unsq.transpose(-1, -2))
    return X_P_X.squeeze(-1).squeeze(-1).unsqueeze(-1)

def a_exact(X, P_t, Rinv, B):
    """
    Compute a*(t, x) = -R⁻¹ Bᵀ ∇u(t, x)
    Works for both 1D and multi-D.
    
    X:      [M, N+1, D]
    P_t:    [M, N+1, 1] (1D) or [D,D] or [M,N+1,D,D]
    Rinv:   [D, D]
    B:      [D, D]
    Returns: [M, N+1, D]
    """
    M, N_plus_1, D = X.shape

    # --- Case 1: Scalar 1D case (D = 1, P_t is [M, N+1, 1]) ---
    if D == 1 and P_t.ndim == 3 and P_t.shape[-1] == 1:
        grad_u = 2 * P_t * X  # [M, N+1, 1]
        return -grad_u        # since R = B = identity in 1D

    # --- Case 2: Constant matrix P_t ---
    if P_t.ndim == 2:
        P_t = P_t.unsqueeze(0).unsqueeze(0).expand(M, N_plus_1, D, D)

    # --- Case 3: Check P_t shape ---
    if P_t.ndim == 4 and P_t.shape[:2] != (M, N_plus_1):
        raise ValueError(f"P_t shape must be [D,D] or [M,N+1,D,D], got {P_t.shape}")

    # --- General multi-D case ---
    X_unsq = X.unsqueeze(-2)                         # [M, N+1, 1, D]
    grad_u = 2 * torch.matmul(X_unsq, P_t).squeeze(-2)  # [M, N+1, D]
    BTRinv = torch.matmul(B.T, Rinv)                 # [D, D]
    a = -torch.matmul(grad_u, BTRinv.T)              # [M, N+1, D]
    return a



def generate_analytical_data(T=1.0, D=1, n_samples=10000):
    t_vals = np.random.uniform(0, T, size=(n_samples, 1))
    x_vals = np.random.normal(loc=0.0, scale=1.0, size=(n_samples, D))

    # Convert to torch
    t_tensor = torch.tensor(t_vals, dtype=torch.float32)
    x_tensor = torch.tensor(x_vals, dtype=torch.float32)

    # Compute P(t) from Riccati
    t_grid = torch.linspace(0, T, steps=100).view(1, -1, 1)
    P_t_all = riccati_solution(
        T=T,
        t_tensor=t_grid.expand(n_samples, -1, -1),
        A=np.eye(D),
        B=np.eye(D),
        Q=np.eye(D),
        R=np.eye(D),
        G=np.eye(D),
    )
    
    # Interpolate P at the sample t values
    idx = np.clip((t_vals * 99).astype(int), 0, 99)
    P_t = P_t_all[0, idx.squeeze(), :, :]  # [n_samples, D, D]

    # Compute u(t,x)
    X_unsq = x_tensor.unsqueeze(-2)
    P_X = torch.bmm(X_unsq, P_t)
    X_P_X = torch.bmm(P_X, X_unsq.transpose(1, 2)).squeeze()
    u_vals = X_P_X.unsqueeze(-1)  # [n_samples, 1]

    return t_tensor, x_tensor, u_vals

def pretrain_on_analytical(model, t_tensor, x_tensor, u_vals, epochs=10000, lr=1e-3):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    
    for epoch in range(epochs):
        optimizer.zero_grad()
        inputs = torch.cat([t_tensor, x_tensor], dim=1).to(t_tensor.device)
        targets = u_vals.to(t_tensor.device)
        preds = model(inputs)

        loss = loss_fn(preds, targets)
        loss.backward()
        optimizer.step()

        if epoch % 100 == 0:
            print(f"Pretraining Epoch {epoch}, Loss = {loss.item():.6e}")


if __name__ == "__main__":

    M, N, D = 100, 30, 1
    layers = [D + 1] + 4 * [64] + [1]
    Xi = np.ones((1, D))
    T = 1.0
    mode, activation = "NAIS-Net", "Sine"
    model = LQRProblem(Xi, T, M, N, D, layers, mode, activation)
    try:
        # model_path = f"best_model_{mode}_{activation}.pt"
        model_path = "model_NAIS-Net_sine_9000.pt"
        model.model.load_state_dict(torch.load("equations/" + model_path, map_location=model.device))
        model.model.eval()
        print("Pre-trained model loaded.")
    except FileNotFoundError:
        print("No pre-trained model found. Training from scratch.")

    epochs = 10000

    t_tensor, x_tensor, u_vals = generate_analytical_data(T=T, D=D)
    pretrain_on_analytical(
        model.model, 
        t_tensor.to(model.device), 
        x_tensor.to(model.device), 
        u_vals.to(model.device)
    )
    graph = model.train(epochs, 1e-3)

    t, W = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t, W)

    t_np = t.cpu().numpy()
    X_np = X_pred.detach().cpu().numpy()
    Y_np = Y_pred.detach().cpu().numpy()

    # exact solution
    P_t = riccati_solution(
        T=T,
        t_tensor=t,  # shape [M, N+1, 1]
        A=model.A.cpu().numpy(),
        B=model.B.cpu().numpy(),
        Q=model.Q.cpu().numpy(),
        R=model.R.cpu().numpy(),
        G=model.G.cpu().numpy(),
    )
    u_ex = u_exact(t, X_pred, P_t).detach().cpu().numpy()
    a_ex = a_exact(X_pred, P_t, torch.inverse(model.R), model.B).detach().cpu().numpy()

    # predicted control
    _, grad_u = model.net_u(t[:, 0, :], X_pred[:, 0, :])

    a_pred_list = []
    for n in range(N + 1):
        _, grad_u = model.net_u(t[:, n, :], X_pred[:, n, :])  # [M, D]
        a_n = -torch.bmm(grad_u.unsqueeze(1), torch.inverse(model.R).unsqueeze(0).expand(M, -1, -1)).squeeze(1)  # [M, D]
        a_pred_list.append(a_n.unsqueeze(1))  # shape [M, 1, D]

    a_pred = torch.cat(a_pred_list, dim=1)  # shape [M, N+1, D]

    plt.figure()
    plt.plot(graph[0], graph[1])
    plt.yscale("log")
    plt.title("Training Loss")

    plt.figure()
    plt.plot(t_np[0, :, 0], Y_np[0, :, 0], label="NN $u$")
    plt.plot(t_np[0, :, 0], u_ex[0, :, 0], '--', label="Exact $u$")
    plt.legend()
    plt.title("Value Function over Time")

    plt.figure()
    plt.plot(t_np[0, :, 0], a_pred[0, :, 0].detach().cpu().numpy(), label="NN $a$")
    plt.plot(t_np[0, :, 0], a_ex[0, :, 0], '--', label="Exact $a$")
    plt.legend()
    plt.title("Optimal Control over Time")

    plt.show()
