import sys
import os
import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import warnings
warnings.filterwarnings('ignore')

# 添加路径
 
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '.')))

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from FBSNNs.Utils import set_seed
from FBSNNs import *
from FBSNNs.CallOption import *
class CallOptionOptunaOptimizer:
    """CallOption模型的Optuna超参数优化器"""
    
    def __init__(self, Xi, T, D, device=None):
        """
        初始化优化器
        
        参数:
        Xi: 初始条件
        T: 时间区间
        D: 问题维度
        device: 计算设备
        """
        self.Xi = Xi
        self.T = T
        self.D = D
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 存储最佳试验结果
        self.best_trial = None
        self.best_model = None
        self.study = None
        
    def objective(self, trial):
        """Optuna目标函数"""
        try:
            # 1. 定义超参数搜索空间
            # 网络架构参数
            n_layers = trial.suggest_int("n_layers", 2, 6)
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
            activation = trial.suggest_categorical("activation", ["Sine", "ReLU", "Tanh"])
            mode = trial.suggest_categorical("mode", ["FC", "Naisnet"])
            
            # 训练参数
            M = trial.suggest_int("M", 50, 200, step=50)  # 批次大小
            N = trial.suggest_int("N", 20, 100, step=10)  # 时间步数
            Mm = N ** (1/5)
            
            # 学习率和迭代次数
            learning_rate1 = trial.suggest_float("learning_rate1", 1e-5, 1e-2, log=True)
            learning_rate2 = trial.suggest_float("learning_rate2", 1e-6, 1e-4, log=True)
            n_iter1 = trial.suggest_int("n_iter1", 1000, 5000, step=1000)
            n_iter2 = trial.suggest_int("n_iter2", 2000, 10000, step=2000)
            
            # 构建网络层
            layers = [self.D + 1] + [hidden_size] * n_layers + [1]
            
            # 2. 创建模型
            model = CallOption(self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation)
            
            # 3. 两阶段训练
            start_time = time.time()
            
            # 第一阶段训练
            graph1 = model.train(n_iter1, learning_rate1)
            
            # 第二阶段训练（精细调优）
            graph2 = model.train(n_iter2, learning_rate2)
            
            training_time = time.time() - start_time
            
            # 4. 评估模型
            set_seed(42)  # 固定随机种子用于公平比较
            t_test, W_test = model.fetch_minibatch()
            X_pred, Y_pred = model.predict(self.Xi, t_test, W_test)
            
            # 转换为numpy数组
            if hasattr(t_test, 'cpu'):
                t_test = t_test.cpu().numpy()
            if hasattr(X_pred, 'cpu'):
                X_pred = X_pred.cpu().detach().numpy()
            if hasattr(Y_pred, 'cpu'):
                Y_pred = Y_pred.cpu().detach().numpy()
            
            # 计算基准解（Black-Scholes解析解）
            def black_scholes_call(S, K=1.0, T=1.0, r=0.01, sigma=0.25):
                from scipy.stats import norm
                d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
                d2 = d1 - sigma*np.sqrt(T)
                return S * norm.cdf(d1) - K * np.exp(-r*T) * norm.cdf(d2)
            
            # 计算相对误差
            S_test = np.sum(X_pred[:, -1, :], axis=1, keepdims=True)
            Y_analytical = black_scholes_call(S_test)
            relative_error = np.mean(np.abs(Y_pred[:, -1, 0] - Y_analytical[:, 0]) / (Y_analytical[:, 0] + 1e-8))
            
            # 最终损失（组合误差和训练时间）
            final_loss = relative_error + 0.01 * training_time/60  # 平衡准确性和效率
            
            # 记录试验属性
            trial.set_user_attr("training_time", training_time)
            trial.set_user_attr("relative_error", relative_error)
            trial.set_user_attr("final_loss", final_loss)
            trial.set_user_attr("layers", layers)
            
            print(f"试验 {trial.number}: 误差={relative_error:.6f}, 时间={training_time:.2f}s, 最终损失={final_loss:.6f}")
            
            return final_loss
            
        except Exception as e:
            print(f"试验 {trial.number} 失败: {e}")
            return float('inf')
    
    def optimize(self, n_trials=50, timeout=3600, study_name="call_option_optuna"):
        """执行超参数优化"""
        print("=" * 80)
        print("           CallOption模型超参数优化")
        print("=" * 80)
        
        # 创建研究
        self.study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner()
        )
        
        # 执行优化
        self.study.optimize(self.objective, n_trials=n_trials, timeout=timeout, show_progress_bar=True)
        
        # 保存最佳结果
        self.best_trial = self.study.best_trial
        print(f"\n优化完成! 最佳试验: #{self.best_trial.number}")
        print(f"最佳最终损失: {self.best_trial.value:.6f}")
        print(f"相对误差: {self.best_trial.user_attrs['relative_error']:.6f}")
        print(f"训练时间: {self.best_trial.user_attrs['training_time']:.2f}s")
        
        return self.best_trial
    
    def train_best_model(self):
        """使用最佳超参数训练最终模型"""
        if self.best_trial is None:
            raise ValueError("请先运行优化!")
        
        print("\n" + "=" * 80)
        print("           使用最佳超参数训练最终模型")
        print("=" * 80)
        
        # 提取最佳参数
        params = self.best_trial.params
        print("最佳超参数配置:")
        for key, value in params.items():
            print(f"  {key}: {value}")
        
        # 构建网络层
        n_layers = params["n_layers"]
        hidden_size = params["hidden_size"]
        layers = [self.D + 1] + [hidden_size] * n_layers + [1]
        
        # 创建最终模型
        M = params["M"]
        N = params["N"]
        Mm = N ** (1/5)
        activation = params["activation"]
        mode = params["mode"]
        
        model = CallOption(self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation)
        
        # 两阶段训练
        learning_rate1 = params["learning_rate1"]
        learning_rate2 = params["learning_rate2"]
        n_iter1 = params["n_iter1"]
        n_iter2 = params["n_iter2"]
        
        print("第一阶段训练...")
        model.train(n_iter1, learning_rate1)
        
        print("第二阶段训练...")
        model.train(n_iter2, learning_rate2)
        
        self.best_model = model
        return model
    
    def visualize_results(self):
        """可视化优化结果"""
        if self.study is None:
            print("请先运行优化!")
            return
        
        try:
            # 优化历史
            fig1 = optuna.visualization.plot_optimization_history(self.study)
            fig1.show()
            
            # 超参数重要性
            fig2 = optuna.visualization.plot_param_importances(self.study)
            fig2.show()
            
            # 切片图
            fig3 = optuna.visualization.plot_slice(self.study)
            fig3.show()
            
        except Exception as e:
            print(f"可视化错误: {e}")
    
    def save_study(self, filename="optuna_study.pkl"):
        """保存研究结果"""
        if self.study is None:
            print("没有可保存的研究结果!")
            return
        
        import joblib
        joblib.dump(self.study, filename)
        print(f"研究已保存到: {filename}")
    
    def load_study(self, filename="optuna_study.pkl"):
        """加载研究结果"""
        import joblib
        self.study = joblib.load(filename)
        self.best_trial = self.study.best_trial
        print(f"研究已从 {filename} 加载")

def main_optuna():
    """主函数：运行Optuna优化"""
    # 设置固定参数
    D = 1  # 维度
    Xi = np.array([1.0] * D)[None, :]  # 初始条件
    T = 1.0  # 时间
    
    # 创建优化器
    optimizer = CallOptionOptunaOptimizer(Xi, T, D)
    
    # 运行优化（减少试验次数以快速演示）
    best_trial = optimizer.optimize(
    n_trials=100,           # 试验次数
    timeout=3600,          # 超时时间（秒）
    study_name="FBSNN" )  # 30分钟超时
    
    
    # 训练最佳模型
    best_model = optimizer.train_best_model()
    
    # 可视化结果
    optimizer.visualize_results()
    
    # 保存研究
    optimizer.save_study()
    
    return optimizer, best_model

def main_vanilla_with_optuna():
    """原有的vanilla_call功能，集成了Optuna优化"""
    print("=" * 80)
    print("           CallOption定价与Optuna超参数优化")
    print("=" * 80)
    
    # 设置基本参数
    M = 100
    N = 50
    D = 1
    Mm = N ** (1/5)
    Xi = np.array([1.0] * D)[None, :]
    T = 1.0
    
    # 检查是否有现有的优化结果
    study_file = "optuna_study.pkl"
    if os.path.exists(study_file):
        choice = input("发现现有的优化结果，是否加载? (y/n): ")
        if choice.lower() == 'y':
            optimizer = CallOptionOptunaOptimizer(Xi, T, D)
            optimizer.load_study(study_file)
            best_model = optimizer.train_best_model()
        else:
            optimizer, best_model = main_optuna()
    else:
        # 询问是否进行优化
        choice = input("是否进行超参数优化? (y/n): ")
        if choice.lower() == 'y':
            optimizer, best_model = main_optuna()
        else:
            # 使用默认参数运行原始版本
            print("使用默认参数运行...")
            layers = [D + 1] + 4 * [256] + [1]
            mode = "Naisnet"
            activation = "Sine"
            model = CallOption(Xi, T, M, N, D, Mm, layers, mode, activation)
            
            # 训练模型
            n_iter1 = 2000
            lr1 = 1e-3
            n_iter2 = 5000
            lr2 = 1e-5
            
            print("第一阶段训练...")
            model.train(n_iter1, lr1)
            
            print("第二阶段训练...")
            model.train(n_iter2, lr2)
            
            best_model = model
    
    # 模型评估和可视化（原有功能）
    set_seed(37)
    t_test, W_test = best_model.fetch_minibatch()
    X_pred, Y_pred = best_model.predict(Xi, t_test, W_test)
    
    # 转换为numpy
    if hasattr(t_test, 'cpu'):
        t_test = t_test.cpu().numpy()
    if hasattr(X_pred, 'cpu'):
        X_pred = X_pred.cpu().detach().numpy()
    if hasattr(Y_pred, 'cpu'):
        Y_pred = Y_pred.cpu().detach().numpy()
    
    # 重复采样以获得更多测试数据
    for i in range(15):
        t_test_i, W_test_i = best_model.fetch_minibatch()
        X_pred_i, Y_pred_i = best_model.predict(Xi, t_test_i, W_test_i)
        if hasattr(X_pred_i, 'cpu'):
            X_pred_i = X_pred_i.cpu().detach().numpy()
        if hasattr(Y_pred_i, 'cpu'):
            Y_pred_i = Y_pred_i.cpu().detach().numpy()
        if hasattr(t_test_i, 'cpu'):
            t_test_i = t_test_i.cpu().numpy()
        t_test = np.concatenate((t_test, t_test_i), axis=0)
        X_pred = np.concatenate((X_pred, X_pred_i), axis=0)
        Y_pred = np.concatenate((Y_pred, Y_pred_i), axis=0)
    
    # 截断数据以确保批大小一致
    X_pred = X_pred[:500, :]
    X_preds = X_pred[:, :, 0]
    t_test = t_test[:500, :, :]  # 确保t_test的批大小与X_pred一致
    
    # 计算Black-Scholes解析解（支持整个时间序列）
    def black_scholes_call(S, K=1.0, T_total=1.0, r=0.01, sigma=0.25, q=0, t=None):
        from scipy.stats import norm
        if t is None:
            t = np.zeros_like(S)
        T_remaining = T_total - t
        # 避免除以零或负数
        T_remaining = np.maximum(T_remaining, 1e-10)
        d1 = (np.log(S/K) + (r - q + 0.5*sigma**2)*T_remaining) / (sigma*np.sqrt(T_remaining))
        d2 = d1 - sigma*np.sqrt(T_remaining)
        call_price = S * np.exp(-q*T_remaining) * norm.cdf(d1) - K * np.exp(-r*T_remaining) * norm.cdf(d2)
        return call_price
    
    # 计算整个时间序列的期权价格
    S_test = X_preds  # 形状是 (batch_size, time_steps)
    Y_analytical = black_scholes_call(S_test, T_total=T, t=t_test[:, :, 0])  # 形状是 (batch_size, time_steps)
    
    # 计算最终时刻的误差
    errors = (Y_analytical[:, -1] - Y_pred[:500, -1, 0])**2
    final_error = np.sqrt(errors.mean())
    
    print(f"\n模型评估结果:")
    print(f"RMSE: {final_error:.6f}")
    print(f"误差均值: {errors.mean():.6f}")
    print(f"误差标准差: {errors.std():.6f}")
    
    # 绘图
    plt.figure(figsize=(12, 8))
    
    # 损失曲线
    if hasattr(best_model, 'training_loss') and best_model.training_loss:
        plt.subplot(2, 2, 1)
        plt.plot(best_model.iteration, best_model.training_loss)
        plt.xlabel('迭代次数')
        plt.ylabel('损失值')
        plt.yscale("log")
        plt.title('训练损失曲线')
        plt.grid(True, alpha=0.3)
    
    # 价格对比
    plt.subplot(2, 2, 2)
    samples = min(5, len(t_test))
    for i in range(samples):
        plt.plot(t_test[i, :, 0], Y_pred[i, :, 0], 'b-', alpha=0.7, label='预测解' if i == 0 else "")
        plt.plot(t_test[i, :, 0], Y_analytical[i, :], 'r--', alpha=0.7, label='解析解' if i == 0 else "")
    plt.xlabel('时间 t')
    plt.ylabel('期权价格')
    plt.title('价格对比')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 误差分布
    plt.subplot(2, 2, 3)
    plt.hist(errors.flatten(), bins=50, alpha=0.7)
    plt.xlabel('平方误差')
    plt.ylabel('频数')
    plt.title('误差分布')
    plt.grid(True, alpha=0.3)
    
    # 最终价格散点图
    plt.subplot(2, 2, 4)
    plt.scatter(Y_analytical[:, -1], Y_pred[:500, -1, 0], alpha=0.6)
    plt.plot([Y_analytical[:, -1].min(), Y_analytical[:, -1].max()], [Y_analytical[:, -1].min(), Y_analytical[:, -1].max()], 'r--')
    plt.xlabel('解析解')
    plt.ylabel('预测解')
    plt.title('最终价格对比')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('call_option_results.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return best_model

if __name__ == "__main__":
    # 安装所需库
    try:
        import optuna
    except ImportError:
        print("安装Optuna...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "optuna"])
        import optuna
    
    try:
        import joblib
    except ImportError:
        print("安装joblib...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "joblib"])
        import joblib
    
    # 运行主程序
    model = main_vanilla_with_optuna()