import sys
import os
import optuna
from optuna.trial import TrialState
import numpy as np
import torch
import matplotlib.pyplot as plt
import time
import datetime
import warnings
import optuna
import joblib


sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from cqf.fbsnn.BlackScholesBarenblatt import BlackScholesBarenblatt, u_exact
from cqf.fbsnn.Utils import set_seed

# 解决中文乱码问题
# 在macOS上使用系统自带的中文字体
plt.rcParams["font.sans-serif"] = [
    "Arial Unicode MS",
    "Heiti TC",
    "Microsoft YaHei",
]  # 使用系统支持的中文字体
plt.rcParams["axes.unicode_minus"] = False  # 解决负号显示问题


class BlackScholesBarenblattOptunaOptimizer:
    """BlackScholesBarenblatt模型的Optuna超参数优化器"""

    def __init__(self, Xi, T, D):
        """
        初始化优化器

        参数:
        Xi: 初始条件
        T: 时间区间
        D: 问题维度
        """
        self.Xi = Xi
        self.T = T
        self.D = D

        # 存储最佳试验结果
        self.best_trial = None
        self.best_model = None
        self.study = None
        self.optimization_history = []

    def objective(self, trial):
        """Optuna目标函数 - 优化BlackScholesBarenblatt模型"""
        try:
            # 设置随机种子以确保可重复性

            # 1. 定义超参数搜索空间
            # 网络架构参数
            n_layers = trial.suggest_int("n_layers", 2, 6)
            hidden_size = trial.suggest_categorical("hidden_size", [64, 128, 256, 512])
            activation = trial.suggest_categorical(
                "activation", ["Sine", "ReLU", "Tanh"]
            )
            mode = trial.suggest_categorical("mode", ["FC", "Naisnet"])

            # 训练参数
            M = trial.suggest_int("M", 50, 200, step=50)  # 批次大小
            N = 50  # 固定时间步数为50，不再使用Optuna优化
            Mm = N ** (1 / 5)

            # 学习率和迭代次数（单阶段训练）
            learning_rate1 = trial.suggest_float("learning_rate1", 1e-5, 1e-2, log=True)
            n_iter1 = trial.suggest_int("n_iter1", 500, 25000, step=500)

            # 构建网络层
            layers = [self.D + 1] + [hidden_size] * n_layers + [1]

            # 2. 创建模型
            model = BlackScholesBarenblatt(
                self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation
            )

            # 单阶段训练
            start_time = time.time()

            # 训练模型
            print(f"试验 {trial.number}: 训练中...")
            graph = model.train(n_iter1, learning_rate1)

            training_time = time.time() - start_time

            # 4. 评估模型性能
            t_test, W_test = model.fetch_minibatch()
            X_pred, Y_pred = model.predict(self.Xi, t_test, W_test)

            # 转换为numpy数组
            if hasattr(t_test, "cpu"):
                t_test = t_test.cpu().numpy()
            if hasattr(X_pred, "cpu"):
                X_pred = X_pred.cpu().detach().numpy()
            if hasattr(Y_pred, "cpu"):
                Y_pred = Y_pred.cpu().detach().numpy()

            # 计算解析解作为基准
            Y_analytical = u_exact(t_test, X_pred, self.T)

            # 计算相对误差
            relative_error = np.mean(
                np.abs(Y_pred[:, -1, 0] - Y_analytical[:, -1, 0])
                / (np.abs(Y_analytical[:, -1, 0]) + 1e-8)
            )

            # 最终损失（组合误差和训练时间）
            final_loss = relative_error + 0.001 * training_time  # 平衡准确性和效率

            # 记录试验属性
            trial.set_user_attr("training_time", training_time)
            trial.set_user_attr("relative_error", relative_error)
            trial.set_user_attr("final_loss", final_loss)
            trial.set_user_attr("layers", layers)

            print(
                f"试验 {trial.number}: 误差={relative_error:.6f}, 时间={training_time:.2f}s, 最终损失={final_loss:.6f}"
            )

            # 清理GPU内存
            # 清理所有设备的缓存（无论CUDA还是MPS）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()

            return final_loss

        except Exception as e:
            print(f"试验 {trial.number} 失败: {e}")
            # 清理GPU内存
            # 清理所有设备的缓存（无论CUDA还是MPS）
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if hasattr(torch, "mps") and torch.backends.mps.is_available():
                torch.mps.empty_cache()
            return float("inf")

    def optimize(
        self, n_trials=50, timeout=7200, study_name="black_scholes_barenblatt_optuna"
    ):
        """执行超参数优化"""
        print("=" * 80)
        print("           Black-Scholes-Barenblatt模型超参数优化")
        print("=" * 80)
        print(f"问题维度: {self.D}D")
        print(f"时间区间: [0, {self.T}]")
        print(f"试验次数: {n_trials}")
        print(f"超时时间: {timeout}秒")
        print()

        # 创建研究
        self.study = optuna.create_study(
            direction="minimize",
            study_name=study_name,
            sampler=optuna.samplers.TPESampler(seed=42),
            pruner=optuna.pruners.HyperbandPruner(),
        )

        # 执行优化
        self.study.optimize(
            self.objective,
            n_trials=n_trials,
            timeout=timeout,
            show_progress_bar=True,
            catch=(Exception,),
        )

        # 保存最佳结果
        self.best_trial = self.study.best_trial
        print(f"\n优化完成! 最佳试验: #{self.best_trial.number}")
        print(f"最佳最终损失: {self.best_trial.value:.6f}")
        print(f"相对误差: {self.best_trial.user_attrs['relative_error']:.6f}")
        print(f"训练时间: {self.best_trial.user_attrs['training_time']:.2f}s")

        return self.best_trial

    def train_best_model(self, save_model=True):
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
        Mm = N ** (1 / 5)
        activation = params["activation"]
        mode = params["mode"]

        model = BlackScholesBarenblatt(
            self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation
        )

        # 单阶段训练
        learning_rate1 = params["learning_rate1"]
        n_iter1 = params["n_iter1"]

        print("模型训练中...")
        model.train(n_iter1, learning_rate1)

        self.best_model = model

        # 保存模型
        if save_model:
            self.save_best_model()

        return model

    def evaluate_model(self, model=None, n_samples=1000):
        """评估模型性能"""
        if model is None:
            if self.best_model is None:
                raise ValueError("没有可评估的模型!")
            model = self.best_model

        print("\n" + "=" * 80)
        print("           模型评估结果")
        print("=" * 80)

        # 生成测试数据
        t_test, W_test = model.fetch_minibatch()
        X_pred, Y_pred = model.predict(self.Xi, t_test, W_test)

        # 重复采样以获得更多测试数据
        for i in range(15):
            t_test_i, W_test_i = model.fetch_minibatch()
            X_pred_i, Y_pred_i = model.predict(self.Xi, t_test_i, W_test_i)

            if hasattr(t_test, "cpu"):
                t_test = torch.cat([t_test, t_test_i], dim=0)
                X_pred = torch.cat([X_pred, X_pred_i], dim=0)
                Y_pred = torch.cat([Y_pred, Y_pred_i], dim=0)
            else:
                t_test = np.concatenate([t_test, t_test_i], axis=0)
                X_pred = np.concatenate([X_pred, X_pred_i], axis=0)
                Y_pred = np.concatenate([Y_pred, Y_pred_i], axis=0)

        # 转换为numpy
        if hasattr(t_test, "cpu"):
            t_test = t_test.cpu().numpy()[:n_samples]
            X_pred = X_pred.cpu().detach().numpy()[:n_samples]
            Y_pred = Y_pred.cpu().detach().numpy()[:n_samples]

        # 计算解析解
        Y_analytical = u_exact(t_test, X_pred, self.T)

        # 计算误差统计
        errors = np.abs(Y_pred[:, -1, 0] - Y_analytical[:, -1, 0])
        relative_errors = errors / (np.abs(Y_analytical[:, -1, 0]) + 1e-8)

        rmse = np.sqrt(np.mean(errors**2))
        mean_relative_error = np.mean(relative_errors)
        std_relative_error = np.std(relative_errors)

        print(f"RMSE: {rmse:.6f}")
        print(f"平均相对误差: {mean_relative_error:.6f}")
        print(f"相对误差标准差: {std_relative_error:.6f}")
        print(f"最大相对误差: {np.max(relative_errors):.6f}")
        print(f"最小相对误差: {np.min(relative_errors):.6f}")

        return {
            "rmse": rmse,
            "mean_relative_error": mean_relative_error,
            "std_relative_error": std_relative_error,
            "t_test": t_test,
            "X_pred": X_pred,
            "Y_pred": Y_pred,
            "Y_analytical": Y_analytical,
        }

    def visualize_results(self, save_plots=True):
        """可视化优化结果和模型性能"""
        if self.study is None:
            print("请先运行优化!")
            return

        # 确保总目录存在
        os.makedirs("optuna_outcomes", exist_ok=True)

        # 创建结果目录（重命名为imgs）
        results_dir = "optuna_outcomes/imgs"
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # 设置Plotly中文支持
            import plotly.io as pio
            import plotly.graph_objects as go

            # 使用Arial Unicode MS作为中文字体
            pio.templates.default = "plotly_white"
            go.Layout(font=dict(family="Arial Unicode MS, Heiti TC, Microsoft YaHei"))

            # 获取已完成的试验数量
            completed_trials = [
                t for t in self.study.trials if t.state == TrialState.COMPLETE
            ]

            # 1. 优化历史
            fig1 = optuna.visualization.plot_optimization_history(self.study)

            # 如果只有一个试验，调整x轴范围以确保数据点可见
            if len(completed_trials) == 1:
                fig1.update_layout(xaxis=dict(range=[-0.5, 0.5]))  # 设置合适的x轴范围

            if save_plots:
                fig1.write_image(f"{results_dir}/optimization_history_{timestamp}.png")
            fig1.show()

            # 2. 超参数重要性
            fig2 = optuna.visualization.plot_param_importances(self.study)
            if save_plots:
                fig2.write_image(f"{results_dir}/param_importance_{timestamp}.png")
            fig2.show()

            # 3. 切片图
            fig3 = optuna.visualization.plot_slice(self.study)
            if save_plots:
                fig3.write_image(f"{results_dir}/slice_plot_{timestamp}.png")
            fig3.show()

            # 4. 并行坐标图
            fig4 = optuna.visualization.plot_parallel_coordinate(self.study)
            if save_plots:
                fig4.write_image(f"{results_dir}/parallel_coord_{timestamp}.png")
            fig4.show()

        except Exception as e:
            print(f"可视化错误: {e}")
            # 使用matplotlib创建基本可视化
            self._create_basic_visualizations(results_dir, timestamp)

    def _create_basic_visualizations(self, results_dir, timestamp):
        """使用matplotlib创建基本可视化（备用）"""
        # 优化历史
        plt.figure(figsize=(10, 6))
        trials = [t for t in self.study.trials if t.state == TrialState.COMPLETE]
        values = [t.value for t in trials]
        numbers = [t.number for t in trials]

        plt.plot(numbers, values, "b-", alpha=0.7)
        plt.xlabel("试验次数")
        plt.ylabel("损失值")
        plt.title("优化历史")
        plt.grid(True, alpha=0.3)
        plt.savefig(
            f"{results_dir}/optimization_history_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

        # 模型性能可视化（如果已有最佳模型）
        if self.best_model is not None:
            self._plot_model_performance(results_dir, timestamp)

    def _plot_model_performance(self, results_dir, timestamp):
        """绘制模型性能图"""
        # 评估模型
        results = self.evaluate_model(n_samples=500)

        # 创建综合性能图
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # 1. 预测vs真实值散点图
        ax1 = axes[0, 0]
        y_true = results["Y_analytical"][:, -1, 0]
        y_pred = results["Y_pred"][:, -1, 0]
        ax1.scatter(y_true, y_pred, alpha=0.6)
        ax1.plot(
            [y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2
        )
        ax1.set_xlabel("解析解")
        ax1.set_ylabel("预测解")
        ax1.set_title("预测vs真实值")
        ax1.grid(True, alpha=0.3)

        # 2. 误差分布
        ax2 = axes[0, 1]
        errors = np.abs(y_true - y_pred)
        ax2.hist(errors, bins=50, alpha=0.7, color="red")
        ax2.set_xlabel("绝对误差")
        ax2.set_ylabel("频数")
        ax2.set_title("误差分布")
        ax2.grid(True, alpha=0.3)

        # 3. 时间序列对比（前5个样本）
        ax3 = axes[1, 0]
        n_samples_show = min(5, len(results["t_test"]))
        for i in range(n_samples_show):
            t = results["t_test"][i, :, 0]
            y_pred_sample = results["Y_pred"][i, :, 0]
            y_true_sample = results["Y_analytical"][i, :, 0]
            ax3.plot(t, y_pred_sample, "b-", alpha=0.7, label="预测" if i == 0 else "")
            ax3.plot(
                t, y_true_sample, "r--", alpha=0.7, label="解析解" if i == 0 else ""
            )
        ax3.set_xlabel("时间 t")
        ax3.set_ylabel("解值 Y(t)")
        ax3.set_title("时间序列对比")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 相对误差统计
        ax4 = axes[1, 1]
        relative_errors = np.abs(y_true - y_pred) / (np.abs(y_true) + 1e-8)
        ax4.boxplot(relative_errors)
        ax4.set_ylabel("相对误差")
        ax4.set_title("相对误差统计")
        ax4.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            f"{results_dir}/model_performance_{timestamp}.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.show()

    def save_study(self, filename="bsb_optuna_study.pkl"):
        """保存研究结果"""
        if self.study is None:
            print("没有可保存的研究结果!")
            return

        import joblib

        # 创建目录
        # 确保总目录存在
        os.makedirs("optuna_outcomes", exist_ok=True)
        os.makedirs("optuna_outcomes/studies", exist_ok=True)
        filepath = f"optuna_outcomes/studies/{filename}"
        joblib.dump(self.study, filepath)
        print(f"研究已保存到: {filepath}")

    def load_study(self, filename="bsb_optuna_study.pkl"):
        """加载研究结果"""
        import joblib

        filepath = f"optuna_outcomes/studies/{filename}"
        self.study = joblib.load(filepath)
        self.best_trial = self.study.best_trial
        print(f"研究已从 {filepath} 加载")

    def save_best_model(self, filename=None):
        """保存最佳模型"""
        if self.best_model is None:
            print("没有可保存的最佳模型!")
            return

        if filename is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"bsb_best_model_{timestamp}.pth"

        # 确保总目录存在
        os.makedirs("optuna_outcomes", exist_ok=True)

        # 创建模型目录（重命名为models）
        os.makedirs("optuna_outcomes/models", exist_ok=True)
        filepath = f"optuna_outcomes/models/{filename}"

        # 保存模型状态和超参数
        save_dict = {
            "model_state_dict": self.best_model.model.state_dict(),
            "best_params": self.best_trial.params,
            "training_loss": getattr(self.best_model, "training_loss", []),
            "iteration": getattr(self.best_model, "iteration", []),
            "Xi": self.Xi,
            "T": self.T,
            "D": self.D,
        }

        torch.save(save_dict, filepath)
        print(f"最佳模型已保存到: {filepath}")

    def generate_report(self):
        """生成优化报告"""
        if self.study is None:
            print("请先运行优化!")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        # 确保总目录存在
        os.makedirs("optuna_outcomes", exist_ok=True)
        report_dir = "optuna_outcomes/reports"
        os.makedirs(report_dir, exist_ok=True)

        report_file = f"{report_dir}/bsb_optuna_report_{timestamp}.txt"

        with open(report_file, "w", encoding="utf-8") as f:
            f.write("=" * 80 + "\n")
            f.write("           Black-Scholes-Barenblatt Optuna优化报告\n")
            f.write("=" * 80 + "\n\n")

            f.write(
                f"报告生成时间: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            )
            f.write(f"问题维度: {self.D}D\n")
            f.write(f"时间区间: [0, {self.T}]\n\n")

            f.write("最佳试验结果:\n")
            f.write(f"  试验编号: #{self.best_trial.number}\n")
            f.write(f"  最终损失: {self.best_trial.value:.6f}\n")
            f.write(f"  相对误差: {self.best_trial.user_attrs['relative_error']:.6f}\n")
            f.write(
                f"  训练时间: {self.best_trial.user_attrs['training_time']:.2f}秒\n\n"
            )

            f.write("最佳超参数配置:\n")
            for key, value in self.best_trial.params.items():
                f.write(f"  {key}: {value}\n")
            f.write(f"  网络结构: {self.best_trial.user_attrs['layers']}\n\n")

            f.write("试验统计:\n")
            completed_trials = [
                t for t in self.study.trials if t.state == TrialState.COMPLETE
            ]
            failed_trials = [t for t in self.study.trials if t.state == TrialState.FAIL]
            f.write(f"  完成试验: {len(completed_trials)}\n")
            f.write(f"  失败试验: {len(failed_trials)}\n")
            f.write(f"  总试验数: {len(self.study.trials)}\n")

            if completed_trials:
                errors = [
                    t.user_attrs.get("relative_error", float("inf"))
                    for t in completed_trials
                ]
                times = [t.user_attrs.get("training_time", 0) for t in completed_trials]
                f.write(f"  平均误差: {np.mean(errors):.6f}\n")
                f.write(f"  最佳误差: {np.min(errors):.6f}\n")
                f.write(f"  最差误差: {np.max(errors):.6f}\n")
                f.write(f"  平均时间: {np.mean(times):.2f}秒\n")

        print(f"报告已生成: {report_file}")


if __name__ == "__main__":

    set_seed(42)
    """原有的vanilla功能，集成了Optuna优化"""
    print("=" * 80)
    print("           Black-Scholes-Barenblatt方程求解与Optuna优化")
    print("=" * 80)

    # 直接运行Optuna优化（100次试验）
    print("直接运行Optuna优化（100次试验）...")
    """主函数：运行Black-Scholes-Barenblatt的Optuna优化"""
    # 设置固定参数
    D = 100  # 维度
    Xi = np.array([1.0, 0.5] * (D // 2))[None, :]  # 初始条件
    T = 1.0  # 时间区间

    print("Black-Scholes-Barenblatt方程Optuna超参数优化")
    print("=" * 60)

    # 创建优化器
    optimizer = BlackScholesBarenblattOptunaOptimizer(Xi, T, D)

    # 直接运行优化（不加载现有研究，运行100次新试验）
    print("直接运行100次新试验...")
    best_trial = optimizer.optimize(
        n_trials=100,  # 试验次数（设置为100）
        timeout=3600,  # 1小时超时
        study_name=f"BSB_{D}D_Optuna",
    )

    # 训练最佳模型
    best_model = optimizer.train_best_model(save_model=True)

    # 评估模型
    evaluation_results = optimizer.evaluate_model()

    # 可视化结果
    optimizer.visualize_results(save_plots=True)

    # 生成报告
    optimizer.generate_report()

    # 保存研究
    optimizer.save_study()

    print("\n" + "=" * 60)
    print("优化完成!")
    print("=" * 60)

    # 可视化最终结果
    optimizer.visualize_results()
