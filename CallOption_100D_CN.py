# -*- coding: utf-8 -*-
"""
100维欧式看涨期权Black-Scholes模型PINN实现
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers
import matplotlib.pyplot as plt
from datetime import datetime

# ============================================================================
# 配置参数
# ============================================================================

class BlackScholesConfig:
    """Black-Scholes模型配置类"""
    
    def __init__(self):
        # 问题维度
        self.dimension = 100
        
        # 物理参数
        self.r = 0.05  # 无风险利率
        self.sigma = 0.2  # 波动率（对所有资产相同）
        
        # 时间范围
        self.T = 1.0  # 期权到期时间（年）
        self.t_min = 0.0  # 最小时间
        self.t_max = self.T  # 最大时间
        
        # 空间范围（对数价格）
        self.S_min = 0.5  # 最小对数价格
        self.S_max = 2.0  # 最大对数价格
        
        # 行权价格
        self.K = 1.0  # 执行价格（取对数空间中为1.0）
        
        # 训练参数
        self.num_samples = 10000  # 训练点数量
        self.num_boundary_samples = 5000  # 边界点数量
        self.num_initial_samples = 5000  # 初始条件点数量
        
        # 神经网络参数
        self.nn_layers = [self.dimension + 1, 128, 128, 128, 128, 1]  # 网络层大小
        self.learning_rate = 0.001
        self.epochs = 5000
        self.batch_size = 256


# ============================================================================
# 物理约束实现
# ============================================================================

class BlackScholesEquation:
    """
    Black-Scholes偏微分方程定义
    在100维空间中的推广形式为：
    
    ∂u/∂t + (1/2)Σ(σ²∂²u/∂S_i²) + r·Σ(S_i∂u/∂S_i) - r·u = 0
    
    其中：
    - u(t, S_1, ..., S_100) 是期权价格
    - σ 是波动率
    - r 是无风险利率
    - S_i 是第i个资产的对数价格
    """
    
    def __init__(self, config):
        self.config = config
        self.r = config.r
        self.sigma = config.sigma
        
    def pde_residual(self, model, t, S):
        """
        计算Black-Scholes偏微分方程的残差
        
        参数：
            model: 神经网络模型
            t: 时间张量，形状为[N, 1]
            S: 空间坐标张量，形状为[N, dimension]
        
        返回：
            pde_loss: PDE残差的平方和
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch(t)
            tape.watch(S)
            
            # 拼接输入为[t, S_1, ..., S_100]
            inputs = tf.concat([t, S], axis=1)
            u = model(inputs)
            
        # 计算∂u/∂t
        u_t = tape.gradient(u, t)
        
        # 计算关于所有空间维度的一阶偏导数
        u_S = tape.gradient(u, S)
        
        # 计算二阶偏导数（对每个空间维度）
        u_SS = []
        for i in range(self.config.dimension):
            u_S_i = u_S[:, i:i+1]
            u_S_S_i = tape.gradient(tf.reduce_sum(u_S_i), S)[:, i:i+1]
            u_SS.append(u_S_S_i)
        
        u_SS = tf.concat(u_SS, axis=1)  # 形状：[N, dimension]
        
        del tape
        
        # 构建Black-Scholes PDE
        # 扩散项：(1/2)σ² Σ ∂²u/∂S_i²
        diffusion = 0.5 * (self.sigma ** 2) * tf.reduce_sum(u_SS, axis=1, keepdims=True)
        
        # 漂移项：r Σ S_i ∂u/∂S_i
        drift = self.r * tf.reduce_sum(S * u_S, axis=1, keepdims=True)
        
        # 时间衰减项：-r·u
        decay = -self.r * u
        
        # PDE：∂u/∂t + diffusion + drift + decay = 0
        pde = u_t + diffusion + drift + decay
        
        # 返回残差（平方和）
        return tf.reduce_mean(tf.square(pde))
    
    def boundary_condition(self, model, S):
        """
        计算边界条件约束
        
        边界条件：当任何资产价格接近边界时
        - 当S_i → -∞：u → 0（价格接近0）
        - 当S_i → +∞：u → S_avg（深度实值期权）
        
        参数：
            model: 神经网络模型
            S: 边界处的空间坐标，形状为[N, dimension]
        
        返回：
            bc_loss: 边界条件的违反程度
        """
        # 需要添加时间维度进行模型输入
        # 这里使用到期时间的多个时间点
        num_samples = tf.shape(S)[0]
        
        # 对多个时间点评估边界条件
        bc_loss = 0.0
        time_points = tf.linspace(self.config.t_min, self.config.t_max, 5)
        
        for t_val in time_points:
            t = tf.ones([num_samples, 1]) * t_val
            inputs = tf.concat([t, S], axis=1)
            u = model(inputs)
            
            # 对于边界处的资产，约束为0或线性增长
            # 简化约束：边界处u应该相对较小（因为价格接近边界）
            bc_loss += tf.reduce_mean(tf.square(u))
        
        return bc_loss / 5.0
    
    def initial_condition(self, model, S):
        """
        计算初始条件约束（在t=T时）
        
        初始条件：u(T, S_1, ..., S_100) = max(mean(S_i) - K, 0)
        （篮子期权的到期价值）
        
        参数：
            model: 神经网络模型
            S: 初始时刻的空间坐标，形状为[N, dimension]
        
        返回：
            ic_loss: 初始条件的违反程度
        """
        # 在到期时刻t=T
        t = tf.ones([tf.shape(S)[0], 1]) * self.config.T
        
        inputs = tf.concat([t, S], axis=1)
        u_pred = model(inputs)
        
        # 计算篮子期权的到期价值：max(avg(S_i) - K, 0)
        S_mean = tf.reduce_mean(S, axis=1, keepdims=True)
        u_true = tf.maximum(S_mean - self.config.K, 0.0)
        
        # 初始条件损失
        ic_loss = tf.reduce_mean(tf.square(u_pred - u_true))
        
        return ic_loss


# ============================================================================
# 神经网络模型
# ============================================================================

class PINNModel:
    """
    物理信息神经网络（PINN）模型
    用于求解100维Black-Scholes方程
    """
    
    def __init__(self, config):
        self.config = config
        self.model = self._build_model()
        self.optimizer = optimizers.Adam(learning_rate=config.learning_rate)
        
    def _build_model(self):
        """
        构建神经网络模型
        
        网络结构：
        - 输入层：101个神经元（1维时间 + 100维空间）
        - 隐藏层：4层，每层128个神经元，激活函数为Tanh
        - 输出层：1个神经元（期权价格）
        
        返回：
            model: Keras模型
        """
        model = keras.Sequential()
        
        # 输入层
        model.add(layers.Input(shape=(self.config.dimension + 1,)))
        
        # 隐藏层
        for units in self.config.nn_layers[1:-1]:
            model.add(layers.Dense(units, activation='tanh'))
        
        # 输出层
        model.add(layers.Dense(self.config.nn_layers[-1], activation='linear'))
        
        return model
    
    def train(self, bs_eq, num_epochs):
        """
        训练PINN模型
        
        参数：
            bs_eq: Black-Scholes方程实例
            num_epochs: 训练轮数
        """
        # 生成训练数据
        t_train = np.random.uniform(
            self.config.t_min, 
            self.config.t_max, 
            [self.config.num_samples, 1]
        )
        S_train = np.random.uniform(
            self.config.S_min, 
            self.config.S_max, 
            [self.config.num_samples, self.config.dimension]
        )
        
        # 生成边界条件数据
        t_bc = np.random.uniform(
            self.config.t_min, 
            self.config.t_max, 
            [self.config.num_boundary_samples, 1]
        )
        S_bc = np.random.uniform(
            self.config.S_min, 
            self.config.S_max, 
            [self.config.num_boundary_samples, self.config.dimension]
        )
        
        # 生成初始条件数据
        t_ic = np.ones([self.config.num_initial_samples, 1]) * self.config.T
        S_ic = np.random.uniform(
            self.config.S_min, 
            self.config.S_max, 
            [self.config.num_initial_samples, self.config.dimension]
        )
        
        # 训练循环
        history = {'loss': [], 'pde': [], 'bc': [], 'ic': []}
        
        for epoch in range(num_epochs):
            with tf.GradientTape() as tape:
                # PDE损失
                t_train_tf = tf.convert_to_tensor(t_train, dtype=tf.float32)
                S_train_tf = tf.convert_to_tensor(S_train, dtype=tf.float32)
                pde_loss = bs_eq.pde_residual(self.model, t_train_tf, S_train_tf)
                
                # 边界条件损失
                S_bc_tf = tf.convert_to_tensor(S_bc, dtype=tf.float32)
                bc_loss = bs_eq.boundary_condition(self.model, S_bc_tf)
                
                # 初始条件损失
                S_ic_tf = tf.convert_to_tensor(S_ic, dtype=tf.float32)
                ic_loss = bs_eq.initial_condition(self.model, S_ic_tf)
                
                # 总损失（加权组合）
                total_loss = pde_loss + 0.5 * bc_loss + 0.5 * ic_loss
            
            # 反向传播
            gradients = tape.gradient(total_loss, self.model.trainable_variables)
            self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
            
            # 记录历史
            history['loss'].append(float(total_loss))
            history['pde'].append(float(pde_loss))
            history['bc'].append(float(bc_loss))
            history['ic'].append(float(ic_loss))
            
            # 打印进度
            if (epoch + 1) % 100 == 0:
                print(f"Epoch {epoch + 1}/{num_epochs}, "
                      f"Loss: {total_loss:.6f}, "
                      f"PDE: {pde_loss:.6f}, "
                      f"BC: {bc_loss:.6f}, "
                      f"IC: {ic_loss:.6f}")
        
        return history
    
    def predict(self, t, S):
        """
        使用训练好的模型进行预测
        
        参数：
            t: 时间，形状为[N, 1]
            S: 空间坐标，形状为[N, dimension]
        
        返回：
            u: 预测的期权价格，形状为[N, 1]
        """
        inputs = np.concatenate([t, S], axis=1)
        inputs = tf.convert_to_tensor(inputs, dtype=tf.float32)
        return self.model(inputs).numpy()


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序：训练并验证PINN模型"""
    
    print("=" * 80)
    print("100维Black-Scholes期权定价PINN模型")
    print(f"执行时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)
    
    # 初始化配置
    config = BlackScholesConfig()
    print("\n模型配置：")
    print(f"  维度：{config.dimension}")
    print(f"  无风险利率：{config.r}")
    print(f"  波动率：{config.sigma}")
    print(f"  到期时间：{config.T}年")
    print(f"  行权价格：{config.K}")
    print(f"  训练样本数：{config.num_samples + config.num_boundary_samples + config.num_initial_samples}")
    
    # 初始化Black-Scholes方程
    bs_eq = BlackScholesEquation(config)
    
    # 初始化PINN模型
    print("\n构建神经网络模型...")
    pinn = PINNModel(config)
    print(f"  网络结构：{config.nn_layers}")
    
    # 训练模型
    print("\n开始训练模型...")
    history = pinn.train(bs_eq, config.epochs)
    
    # 显示最终结果
    print("\n训练完成！")
    print(f"  最终总损失：{history['loss'][-1]:.6f}")
    print(f"  最终PDE损失：{history['pde'][-1]:.6f}")
    print(f"  最终BC损失：{history['bc'][-1]:.6f}")
    print(f"  最终IC损失：{history['ic'][-1]:.6f}")
    
    # 测试预测
    print("\n进行测试预测...")
    test_t = np.array([[0.5]])  # 到期前0.5年
    test_S = np.random.uniform(config.S_min, config.S_max, [1, config.dimension])
    test_price = pinn.predict(test_t, test_S)
    print(f"  测试时间：{test_t[0, 0]:.4f}年")
    print(f"  资产平均对数价格：{np.mean(test_S):.4f}")
    print(f"  预测期权价格：{test_price[0, 0]:.6f}")
    
    print("\n模型训练和验证完成！")
    
    return pinn, history, config


if __name__ == "__main__":
    pinn_model, training_history, config = main()
