import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
import time
import sys
import os
from abc import ABC, abstractmethod
from scipy.stats import multivariate_normal as normal


class Sine(nn.Module):
    """This class defines the sine activation function as a nn.Module"""
    def __init__(self):
        super(Sine, self).__init__()

    def forward(self, x):
        return torch.sin(x)


class Naisnet(nn.Module):
    def __init__(self, layers, stable, activation):
        super(Naisnet, self).__init__()

        self.layers = layers
        self.layer1 = nn.Linear(in_features=layers[0], out_features=layers[1])
        self.layer2 = nn.Linear(in_features=layers[1], out_features=layers[2])
        self.layer2_input = nn.Linear(in_features=layers[0], out_features=layers[2])
        self.layer3 = nn.Linear(in_features=layers[2], out_features=layers[3])
        if len(layers) == 5:
            self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
            self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
        elif len(layers) == 6:
            self.layer3_input = nn.Linear(in_features=layers[0], out_features=layers[3])
            self.layer4 = nn.Linear(in_features=layers[3], out_features=layers[4])
            self.layer4_input = nn.Linear(in_features=layers[0], out_features=layers[4])
            self.layer5 = nn.Linear(in_features=layers[4], out_features=layers[5])

        self.activation = activation
        self.epsilon = 0.01
        self.stable = stable

    def project(self, layer, out):
        weights = layer.weight
        delta = 1 - 2 * self.epsilon
        RtR = torch.matmul(weights.t(), weights)
        norm = torch.norm(RtR)
        if norm > delta:
            RtR = delta ** (1 / 2) * RtR / (norm ** (1 / 2))
        A = RtR + torch.eye(RtR.shape[0]) * self.epsilon
        return F.linear(out, -A, layer.bias)

    def forward(self, x):
        u = x
        out = self.layer1(x)
        out = self.activation(out)
        shortcut = out
        
        if self.stable:
            out = self.project(self.layer2, out)
            out = out + self.layer2_input(u)
        else:
            out = self.layer2(out)
        out = self.activation(out)
        out = out + shortcut

        if len(self.layers) == 4:
            out = self.layer3(out)
            return out

        if len(self.layers) == 5:
            shortcut = out
            if self.stable:
                out = self.project(self.layer3, out)
                out = out + self.layer3_input(u)
            else:
                out = self.layer3(out)
            out = self.activation(out)
            out = out + shortcut
            out = self.layer4(out)
            return out
        
        if len(self.layers) == 6:
            shortcut = out
            if self.stable:
                out = self.project(self.layer3, out)
                out = out + self.layer3_input(u)
            else:
                out = self.layer3(out)
            out = self.activation(out)
            out = out + shortcut

            shortcut = out
            if self.stable:
                out = self.project(self.layer4, out)
                out = out + self.layer4_input(u)
            else:
                out = self.layer4(out)
            out = self.activation(out)
            out = out + shortcut
            out = self.layer5(out)
            return out

        return out


class FBSNN(ABC):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
        device_idx = 0
        if torch.cuda.is_available():
            self.device = torch.device("cuda:" + str(device_idx) if torch.cuda.is_available() else "cpu")
            torch.backends.cudnn.deterministic = True
        else:
            self.device = torch.device("cpu")

        self.Xi = torch.from_numpy(Xi).float().to(self.device)
        self.Xi.requires_grad = True
        self.T = T
        self.M = M
        self.N = N
        self.D = D
        self.Mm = Mm
        self.strike = 1.0 * self.D
        self.mode = mode
        self.activation = activation
        
        if activation == "Sine":
            self.activation_function = Sine()
        elif activation == "ReLU":
            self.activation_function = nn.ReLU()
        elif activation == "Tanh":
            self.activation_function = nn.Tanh()

        if self.mode == "FC":
            self.layers = []
            for i in range(len(layers) - 2):
                self.layers.append(nn.Linear(in_features=layers[i], out_features=layers[i + 1]))
                self.layers.append(self.activation_function)
            self.layers.append(nn.Linear(in_features=layers[-2], out_features=layers[-1]))
            self.model = nn.Sequential(*self.layers).to(self.device)
        elif self.mode == "Naisnet":
            self.model = Naisnet(layers, stable=True, activation=self.activation_function).to(self.device)

        self.model.apply(self.weights_init)
        self.training_loss = []
        self.iteration = []

    def weights_init(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)

    def net_u(self, t, X):
        input = torch.cat((t, X), 1)
        u = self.model(input)
        Du = torch.autograd.grad(outputs=[u], inputs=[X], grad_outputs=torch.ones_like(u), 
                                allow_unused=True, retain_graph=True, create_graph=True)[0]
        return u, Du

    def Dg_tf(self, X):
        g = self.g_tf(X)
        Dg = torch.autograd.grad(outputs=[g], inputs=[X], grad_outputs=torch.ones_like(g), 
                                allow_unused=True, retain_graph=True, create_graph=True)[0]
        return Dg

    def loss_function(self, t, W, Xi):
        loss = 0
        X_list = []
        Y_list = []

        t0 = t[:, 0, :]
        W0 = W[:, 0, :]
        X0 = Xi.repeat(self.M, 1).view(self.M, self.D)
        Y0, Z0 = self.net_u(t0, X0)

        X_list.append(X0)
        Y_list.append(Y0)

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]
            X1 = X0 + self.mu_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.squeeze(
                torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1)), dim=-1)
            
            Y1_tilde = Y0 + self.phi_tf(t0, X0, Y0, Z0) * (t1 - t0) + torch.sum(
                Z0 * torch.squeeze(torch.matmul(self.sigma_tf(t0, X0, Y0), (W1 - W0).unsqueeze(-1))), dim=1,
                keepdim=True)
            
            Y1, Z1 = self.net_u(t1, X1)
            loss += torch.sum(torch.pow(Y1 - Y1_tilde, 2))
            t0, W0, X0, Y0, Z0 = t1, W1, X1, Y1, Z1
            X_list.append(X0)
            Y_list.append(Y0)

        loss += torch.sum(torch.pow(Y1 - self.g_tf(X1), 2))
        loss += torch.sum(torch.pow(Z1 - self.Dg_tf(X1), 2))

        X = torch.stack(X_list, dim=1)
        Y = torch.stack(Y_list, dim=1)
        return loss, X, Y, Y[0, 0, 0]

    def fetch_minibatch(self):
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1))
        DW = np.zeros((M, N + 1, D))
        dt = T / N
        Dt[:, 1:, :] = dt
        DW_uncorrelated = np.sqrt(dt) * np.random.normal(size=(M, N, D))
        DW[:, 1:, :] = DW_uncorrelated

        t = np.cumsum(Dt, axis=1)
        W = np.cumsum(DW, axis=1)
        t = torch.from_numpy(t).float().to(self.device)
        W = torch.from_numpy(W).float().to(self.device)
        return t, W

    def train(self, N_Iter, learning_rate):
        loss_temp = np.array([])
        previous_it = 0
        if self.iteration != []:
            previous_it = self.iteration[-1]

        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        start_time = time.time()
        
        for it in range(previous_it, previous_it + N_Iter):
            if it >= 400 and it < 2000:
                self.N = int(np.ceil(self.Mm ** (int(it / 400) + 1)))
            elif it < 400:
                self.N = int(np.ceil(self.Mm))

            self.optimizer.zero_grad()
            t_batch, W_batch = self.fetch_minibatch()
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)
            
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            loss_temp = np.append(loss_temp, loss.cpu().detach().numpy())
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Loss: %.3e, Y0: %.3f, Time: %.2f, Learning Rate: %.3e' %
                    (it, loss, Y0_pred, elapsed, learning_rate))
                start_time = time.time()

            if it % 10 == 0:
                self.training_loss.append(loss_temp.mean())
                loss_temp = np.array([])
                self.iteration.append(it)
                
        graph = np.stack((self.iteration, self.training_loss))
        return graph

    def predict(self, Xi_star, t_star, W_star):
        Xi_star = torch.from_numpy(Xi_star).float().to(self.device)
        Xi_star.requires_grad = True
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)
        return X_star, Y_star

    def save_model(self, file_name):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'training_loss': self.training_loss,
            'iteration': self.iteration
        }, file_name)
    
    def load_model(self, file_name):
        checkpoint = torch.load(file_name, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.training_loss = checkpoint['training_loss']
        self.iteration = checkpoint['iteration']

    @abstractmethod
    def phi_tf(self, t, X, Y, Z):
        pass

    @abstractmethod
    def g_tf(self, X):
        pass

    @abstractmethod
    def mu_tf(self, t, X, Y, Z):
        M = self.M
        D = self.D
        return torch.zeros([M, D]).to(self.device)

    @abstractmethod
    def sigma_tf(self, t, X, Y):
        M = self.M
        D = self.D
        return torch.diag_embed(torch.ones([M, D])).to(self.device)


class CallOption(FBSNN):
    def __init__(self, Xi, T, M, N, D, Mm, layers, mode, activation):
        super().__init__(Xi, T, M, N, D, Mm, layers, mode, activation)

    def phi_tf(self, t, X, Y, Z):
        rate = 0.01
        return rate * (Y)

    def g_tf(self, X):
        temp = torch.sum(X, dim=1, keepdim=True)
        return torch.maximum(temp - self.strike, torch.tensor(0.0))

    def mu_tf(self, t, X, Y, Z):
        rate = 0.01
        return rate * X

    def sigma_tf(self, t, X, Y):
        sigma = 0.25
        return sigma * torch.diag_embed(X)


if __name__ == "__main__":
    print("Starting test run...")
    
    # Use smaller parameters for quick testing
    M = 1
    N = 10
    D = 1
    Mm = N ** (1/5)
    layers = [D + 1] + 2 * [32] + [1]  # Smaller network
    Xi = np.array([1.0] * D)[None, :]
    T = 1.0

    mode = "Naisnet"
    activation = "Sine"
    
    print(f"Creating model with D={D}, layers={layers}, mode={mode}, activation={activation}")
    model = CallOption(Xi, T, M, N, D, Mm, layers, mode, activation)
    print(f"Model created successfully on device: {model.device}")

    # Run only a few iterations for testing
    n_iter = 50
    lr = 1e-3
    
    print(f"Starting training with {n_iter} iterations...")
    tot = time.time()
    graph = model.train(n_iter, lr)
    print(f"Training completed in {time.time() - tot:.2f} seconds")

    # Test prediction
    print("Testing prediction...")
    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)
    print("Prediction completed successfully")

    print("Test run completed!")
