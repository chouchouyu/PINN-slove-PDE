import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import datetime
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'deepbsde')))

from FBSNNs.BlackScholesBarenblatt import BlackScholesBarenblatt, u_exact
from FBSNNs.Utils import figsize, set_seed
 
 
 

 

def run_model(model, N_Iter1, learning_rate1, N_Iter2, learning_rate2):
    # 第一阶段训练
    tot = time.time()
    samples = 5
    print(model.device)
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    graph = model.train(N_Iter1, learning_rate1)
    print("第一阶段训练完成，总时间:", time.time() - tot, "s")
    
    # 第二阶段训练
    tot = time.time()
    print(model.device)
    graph = model.train(N_Iter2, learning_rate2)
    print("第二阶段训练完成，总时间:", time.time() - tot, "s")


    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != 'numpy':
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != 'numpy':
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != 'numpy':
        Y_pred = Y_pred.cpu().detach().numpy()

    Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D]), T),
                        [M, -1, 1])

    # 创建输出目录
    save_dir = "results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    plt.figure(figsize=figsize(1))
    graph = model.iteration, model.training_loss
    plt.plot(graph[0], graph[1])
    plt.xlabel('Iterations')
    plt.ylabel('Value')
    plt.yscale("log")
    plt.title('Evolution of the training loss')
    plt.savefig(f"{save_dir}/BSB_{model.D}D_{model.mode}_{model.activation}_{timestamp}_Loss.png", dpi=300, bbox_inches='tight')

    plt.figure(figsize=figsize(1))
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label='Learned $u(t,X_t)$')
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label='Exact $u(t,X_t)$')
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label='$Y_T = u(T,X_T)$')

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

    plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

    plt.xlabel('$t$')
    plt.ylabel('$Y_t = u(t,X_t)$')
    plt.title('D=' + str(D) + ' Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig(f"{save_dir}/BSB_{model.D}D_{model.mode}_{model.activation}_{timestamp}.png", dpi=300, bbox_inches='tight')

    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure(figsize=figsize(1))
    plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    plt.xlabel('$t$')
    plt.ylabel('relative error')
    plt.title('D=' + str(D) + ' Black-Scholes-Barenblatt, ' + model.mode + "-" + model.activation)
    plt.legend()
    plt.savefig(f"{save_dir}/BSB_{model.D}D_{model.mode}_{model.activation}_{timestamp}_Errors.png", dpi=300, bbox_inches='tight')
    
    # 保存最终模型
    model_save_dir = "optuna_outcomes/models"
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    
    model_filename = f"BSB_raw_model_{model.mode}_{model.activation}_{timestamp}.pth"
    model_path = os.path.join(model_save_dir, model_filename)
    
    save_dict = {
        'model_state_dict': model.model.state_dict(),
        'mode': model.mode,
        'activation': model.activation,
        'Xi': Xi,
        'T': T,
        'D': D
    }
    
    torch.save(save_dict, model_path)
    print(f"最终模型已保存到: {model_path}")
    
    plt.show()


if __name__ == "__main__":
    tot = time.time()
    M = 100  # number of trajectories (batch size)
    N = 50  # number of time snapshots
    D = 100  # number of dimensions
    Mm = N **(1/5)
    
    layers = [D + 1] + 4 * [256] + [1]

    Xi = np.array([1.0, 0.5] * int(D / 2))[None, :]
    T = 1.0

    "Available architectures"
    mode = "Naisnet"  # FC, Resnet and Naisnet are available
    activation = "Sine"  # Sine and ReLU are available
    model = BlackScholesBarenblatt(Xi, T,
                                   M, N, D, Mm,
                                   layers, mode, activation)
    set_seed(42)
 
    # 打印run_model调用前的入参
    print("\n" + "=" * 60)
    print("run_model 入参:")
    print("=" * 60)
    N_Iter1 = 2 * 10 ** 4
    learning_rate1 = 1e-3
    N_Iter2 = 5000
    learning_rate2 = 1e-5
    print(f"N_Iter1: {N_Iter1}")
    print(f"learning_rate1: {learning_rate1}")
    print(f"N_Iter2: {N_Iter2}")
    print(f"learning_rate2: {learning_rate2}")
    print(f"Xi shape: {Xi.shape}, Xi first few values: {Xi[0, :3] if Xi.size > 0 else 'empty'}")
    print(f"T: {T}")
    print(f"D: {D}")
    print(f"M: {M}")
    print(f"model.mode: {model.mode}")
    print(f"model.activation: {model.activation}")
    print(f"model.D: {model.D}")
    print(f"model.M: {model.M}")
    print(f"model.N: {model.N}")
    print("=" * 60)
    
    # 2 * 10 ** 4
    # 运行两阶段训练：第一阶段100次迭代，学习率1e-3；第二阶段5000次迭代，学习率1e-5
    run_model(model, N_Iter1, learning_rate1, N_Iter2, learning_rate2)
     