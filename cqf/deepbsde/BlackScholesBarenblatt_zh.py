import torch

class BlackScholesBarenblatt:
    """Black-Scholes-Barenblatt方程的基类"""
    
    def __init__(self, d=30):
        self.d = d
        self.r = 0.05
        self.sigma = 0.4
        
    def g_tf(self, X):
        """终端条件：g(X) = sum(X^2)"""
        return torch.sum(X**2, dim=1, keepdim=True)
    
    def phi_tf(self, X, u, sigma_grad_u, t):
        """非线性项：phi(X, u, σᵀ∇u, p, t) = r * (u - sum(X * σᵀ∇u))"""
        return self.r * (u - torch.sum(X * sigma_grad_u, dim=1, keepdim=True))
    
    def mu_tf(self, X, t):
        """漂移项：μ(X, p, t) = 0"""
        return torch.zeros_like(X)
    
    def sigma_tf(self, X, t):
        """扩散项：σ(X, p, t) = Diagonal(sigma * X)"""
        if len(X.shape) == 1:
            # 一维情况：返回2D对角矩阵
            return torch.diag(self.sigma * X)
        else:
            # 二维情况：返回3D对角矩阵批次
            batch_size = X.shape[0]
            return torch.diag_embed(self.sigma * X)
    
    def analytical_solution(self, x, t, T=1.0):
        """解析解"""
        exponent = (self.r + self.sigma**2) * (T - t)
        return torch.exp(torch.tensor(exponent, device=x.device)) * torch.sum(x**2)
