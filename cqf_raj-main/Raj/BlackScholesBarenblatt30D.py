import time
import matplotlib.pyplot as plt
import matplotlib
import torch
import numpy as np
from BlackScholesBarenblatt import *
from FBSNNs import *
import sys
import os

# 解决中文乱码问题
# 在macOS上使用系统自带的中文字体
matplotlib.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'Heiti TC', 'Microsoft YaHei']  # 使用系统支持的中文字体
matplotlib.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 设置参数，与deepbsde实现保持一致
d = 30  # 维度
D = d
x0 = np.tile([1.0, 0.5], d // 2)  # 初始点
Xi = x0[None, :]  # FBSNNs需要的形状
T = 1.0  # 时间区间(0, 1)

# 计算时间步数，与deepbsde保持一致
dt_deepbsde = 0.25  # deepbsde中的时间步长
time_steps_deepbsde = int(T / dt_deepbsde)  # deepbsde中的时间步数
N = time_steps_deepbsde + 1  # FBSNNs中的时间快照数量(N = 时间步数 + 1)

# 轨迹数量(批大小)与deepbsde保持一致
M = 30  # deepbsde中的m=30
Mm = N  # SDE离散化点数量

# 神经网络结构
# deepbsde中使用了隐藏层大小hls = 10 + d
# FBSNNs中使用了全连接网络，我们使用类似的结构
layers = [D + 1] + 4 * [10 + D] + [1]

# 网络配置
mode = "FC"  # FC和Naisnet可用
activation = "ReLU"  # Sine, ReLU和Tanh可用

# 创建模型
model = BlackScholesBarenblatt(Xi, T, M, N, D, Mm, layers, mode, activation)

print(f"使用参数:")
print(f"维度: D = {D}")
print(f"初始条件: Xi = {Xi}")
print(f"终端时间: T = {T}")
print(f"时间快照数量: N = {N}")
print(f"轨迹数量(批大小): M = {M}")
print(f"网络结构: {layers}")
print(f"网络模式: {mode}")
print(f"激活函数: {activation}")
print(f"设备: {model.device}")

# 训练模型
# 与deepbsde保持一致的训练迭代次数
tot = time.time()
n_iter = 150  # deepbsde中使用了150次迭代
lr = 1e-3  # deepbsde中使用了0.001的学习率
print(f"\n开始训练...")
graph = model.train(n_iter, lr)
print(f"总训练时间: {time.time() - tot} s")

# 评估模型
print(f"\n评估模型...")
# 生成测试数据
np.random.seed(100)  # 与deepbsde保持相同的随机种子
t_test, W_test = model.fetch_minibatch()
X_pred, Y_pred = model.predict(Xi, t_test, W_test)

# 转换为numpy数组
if type(t_test).__module__ != 'numpy':
    t_test = t_test.cpu().numpy()
if type(X_pred).__module__ != 'numpy':
    X_pred = X_pred.cpu().detach().numpy()
if type(Y_pred).__module__ != 'numpy':
    Y_pred = Y_pred.cpu().detach().numpy()

# 计算精确解
Y_test = np.reshape(u_exact(T, np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
                   [M, -1, 1])

# 计算数值解和解析解
numerical_sol = Y_pred[0, 0, 0]  # 初始点的数值解
analytical_sol = u_exact(T, np.array([[0.0]]), Xi)[0, 0]  # 初始点的精确解

# 计算相对L2误差
def rel_error_l2(u, uanal):
    if abs(uanal) >= 10 * np.finfo(type(uanal)).eps:
        return np.sqrt((u - uanal)**2 / u**2)
    else:
        return abs(u - uanal)

error_l2 = rel_error_l2(numerical_sol, analytical_sol)

print(f"数值解: {numerical_sol}")
print(f"解析解: {analytical_sol}")
print(f"相对L2误差: {error_l2}")

# 创建Figures目录（如果不存在）
save_dir = "Figures"
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# 绘制损失函数图像
plt.figure(figsize=(10, 6))
plt.plot(model.iteration, model.training_loss, 'b')
plt.xlabel('迭代次数')
plt.ylabel('损失值')
plt.yscale("log")
plt.title('训练损失演化 (D=' + str(D) + ', ' + model.mode + "-" + model.activation + ')')
plt.savefig(os.path.join(save_dir, f"BlackScholesBarenblatt30D_FC_ReLU_Loss.png"))

# 绘制学习的解与精确解对比
samples = 5
plt.figure(figsize=(10, 6))
plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T,
         'b', label='学习的解 $u(t,X_t)$')
plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T,
         'r--', label='精确解 $u(t,X_t)$')
plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0],
         'ko', label='$Y_T = u(T,X_T)$')

plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')

plt.plot([0], Y_test[0, 0, 0], 'ks', label='$Y_0 = u(0,X_0)$')

plt.xlabel('$t$')
plt.ylabel('$Y_t = u(t,X_t)$')
plt.title('D=' + str(D) + ' Black-Scholes-Barenblatt, ' +
          model.mode + "-" + model.activation)
plt.legend()
plt.savefig(os.path.join(save_dir, f"BlackScholesBarenblatt30D_FC_ReLU_.png"))

# 绘制相对误差
errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
mean_errors = np.mean(errors, 0)
std_errors = np.std(errors, 0)
plt.figure(figsize=(10, 6))
plt.plot(t_test[0, :, 0], mean_errors, 'b', label='均值')
plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors,
         'r--', label='均值 + 两倍标准差')
plt.xlabel('$t$')
plt.ylabel('相对误差')
plt.title('D=' + str(D) + ' Black-Scholes-Barenblatt, ' +
          model.mode + "-" + model.activation)
plt.legend()
plt.savefig(os.path.join(save_dir, f"BlackScholesBarenblatt30D_FC_ReLU_Errors.png"))

plt.show()

# 保存模型
# model.save_model("models/BlackScholesBarenblatt30D.pth")
