#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
整合版Black-Scholes-Barenblatt模型训练器
结合FormalTrainer和run_model的功能
"""

import sys
import os
import numpy as np
import torch
import time
import datetime
import warnings

warnings.filterwarnings("ignore")

# 添加路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from fbsnn.BlackScholesBarenblatt import BlackScholesBarenblatt, u_exact
from fbsnn.Utils import figsize, set_seed, setup_device

# 确保导入matplotlib
import matplotlib.pyplot as plt

# 设置中文支持
plt.rcParams["font.sans-serif"] = ["Arial Unicode MS", "Heiti TC", "Microsoft YaHei"]
plt.rcParams["axes.unicode_minus"] = False


class FormalTrainer:
    """正式训练器：使用Optuna优化得到的最佳模型和参数训练BlackScholesBarenblatt模型"""

    def __init__(self, model_path=None, study_path=None):
        """
        初始化正式训练器

        参数:
        model_path: 已保存的最佳模型路径
        study_path: 已保存的Optuna研究路径
        """
        self.model_path = model_path
        self.study_path = study_path
        self.loaded_model = None
        self.best_params = None
        self.Xi = None
        self.T = None
        self.D = None

    def load_model_and_params(self):
        """
        加载已保存的模型和参数
        """
        import joblib

        if self.model_path:
            # 从模型文件加载
            print(f"从模型文件加载: {self.model_path}")
            
            # 获取最佳设备
            device, _ = setup_device()
            
            # 加载模型文件并指定设备
            save_dict = torch.load(
                self.model_path, 
                weights_only=False,  # 必须为False，需要加载超参数
                map_location=device  # 使用自动检测的设备
            )

            # 提取参数
            self.best_params = save_dict["best_params"]
            self.Xi = save_dict["Xi"]
            self.T = save_dict["T"]
            self.D = save_dict["D"]

            print(f"问题维度: {self.D}D")
            print(f"时间区间: [0, {self.T}]")

            # 构建网络层
            n_layers = self.best_params["n_layers"]
            hidden_size = self.best_params["hidden_size"]
            layers = [self.D + 1] + [hidden_size] * n_layers + [1]

            # 创建模型
            M = self.best_params["M"]
            N = 50
            Mm = N ** (1 / 5)
            activation = self.best_params["activation"]
            mode = self.best_params["mode"]

            self.loaded_model = BlackScholesBarenblatt(
                self.Xi, self.T, M, N, self.D, Mm, layers, mode, activation
            )

            # 加载模型权重
            self.loaded_model.model.load_state_dict(save_dict["model_state_dict"])
            print("✓ 模型权重加载完成")

            # 打印最佳参数
            print("\n最佳超参数配置:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")

        elif self.study_path:
            # 从研究文件加载
            print(f"从研究文件加载: {self.study_path}")
            study = joblib.load(self.study_path)
            self.best_params = study.best_trial.params

            # 打印最佳参数
            print("\n最佳超参数配置:")
            for key, value in self.best_params.items():
                print(f"  {key}: {value}")

        else:
            raise ValueError("必须提供模型路径或研究路径")

    def get_model(self):
        """
        获取加载的模型
        """
        if not self.loaded_model:
            raise ValueError("请先加载模型和参数")
        return self.loaded_model

    def get_params(self):
        """
        获取最佳参数
        """
        if not self.best_params:
            raise ValueError("请先加载模型和参数")
        return self.best_params

    def get_problem_params(self):
        """
        获取问题参数
        """
        if self.Xi is None or self.T is None or self.D is None:
            raise ValueError("请先加载模型和参数")
        return self.Xi, self.T, self.D


def run_model(model, N_Iter1, learning_rate1, Xi, T, D, M):
    # 快速复现：跳过重新训练，直接使用预训练模型
    tot = time.time()
    samples = 5
    print(model.device)
    # 生成时间戳
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
 

    t_test, W_test = model.fetch_minibatch()
    X_pred, Y_pred = model.predict(Xi, t_test, W_test)

    if type(t_test).__module__ != "numpy":
        t_test = t_test.cpu().numpy()
    if type(X_pred).__module__ != "numpy":
        X_pred = X_pred.cpu().detach().numpy()
    if type(Y_pred).__module__ != "numpy":
        Y_pred = Y_pred.cpu().detach().numpy()

    Y_test = np.reshape(
        u_exact(
            np.reshape(t_test[0:M, :, :], [-1, 1]),
            np.reshape(X_pred[0:M, :, :], [-1, D]),
            T,
        ),
        [M, -1, 1],
    )

    # 创建输出目录
    save_dir = "Figures"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    # 由于跳过重新训练，不生成训练损失图
  

    plt.figure(figsize=figsize(1))
    plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, "b", label="Learned $u(t,X_t)$")
    plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, "r--", label="Exact $u(t,X_t)$")
    plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], "ko", label="$Y_T = u(T,X_T)$")

    plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, "b")
    plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, "r--")
    plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], "ko")

    plt.plot([0], Y_test[0, 0, 0], "ks", label="$Y_0 = u(0,X_0)$")

    plt.xlabel("$t$")
    plt.ylabel("$Y_t = u(t,X_t)$")
    plt.title(
        "D="
        + str(D)
        + " Black-Scholes-Barenblatt, "
        + model.mode
        + "-"
        + model.activation
    )
    plt.legend()
    plt.savefig(
        f"{save_dir}/BSB_{model.D}D_{model.mode}_{model.activation}_{timestamp}.png",
        dpi=300,
        bbox_inches="tight",
    )

    errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test**2)
    mean_errors = np.mean(errors, 0)
    std_errors = np.std(errors, 0)

    plt.figure(figsize=figsize(1))
    plt.plot(t_test[0, :, 0], mean_errors, "b", label="mean")
    plt.plot(
        t_test[0, :, 0],
        mean_errors + 2 * std_errors,
        "r--",
        label="mean + two standard deviations",
    )
    plt.xlabel("$t$")
    plt.ylabel("relative error")
    plt.title(
        "D="
        + str(D)
        + " Black-Scholes-Barenblatt, "
        + model.mode
        + "-"
        + model.activation
    )
    plt.legend()
    plt.savefig(
        f"{save_dir}/BSB_{model.D}D_{model.mode}_{model.activation}_{timestamp}_Errors.png",
        dpi=300,
        bbox_inches="tight",
    )

  

    plt.show()


if __name__ == "__main__":
    # 配置参数
    MODEL_PATH = "optuna_outcomes/models/bsb_best_model_20260118_150740.pth"  # 已保存的最佳模型路径
    STUDY_PATH = "optuna_outcomes/studies/bsb_optuna_study.pkl"  # 已保存的研究路径
    REPORT_PATH = "optuna_outcomes/reports/bsb_optuna_report_20260118_150807.txt"  # Optuna优化报告路径

    # 设置随机种子
    set_seed(42)
    
    """主函数"""
    print("=" * 80)
    print("           Black-Scholes-Barenblatt模型整合训练器")
    print("           基于Optuna优化结果")
    print("=" * 80)

    # 检查文件是否存在
    if os.path.exists(MODEL_PATH):
        use_model_path = True
        print(f"✓ 找到模型文件: {MODEL_PATH}")
    elif os.path.exists(STUDY_PATH):
        use_model_path = True
        print(f"✓ 找到研究文件: {STUDY_PATH}")
    else:
        print("❌ 未找到模型文件或研究文件")
        sys.exit(1)

    try:
        # 创建正式训练器
        if use_model_path:
            trainer = FormalTrainer(model_path=MODEL_PATH)
        else:
            trainer = FormalTrainer(study_path=STUDY_PATH)
            # 如果使用研究文件，需要提供问题的基本参数
            trainer.Xi = np.array([1.0, 0.5] * (50 // 2))[None, :]  # 50维示例
            trainer.T = 1.0
            trainer.D = 50

        # 加载模型和参数
        trainer.load_model_and_params()

        # 获取模型和参数
        model = trainer.get_model()
        best_params = trainer.get_params()
        Xi, T, D = trainer.get_problem_params()

       

        # 自动从Optuna报告文件读取最优参数
        def load_optuna_report_params(report_path):
            """
            从Optuna报告文件中读取最优参数
            """
            import re
            
            with open(report_path, 'r') as f:
                content = f.read()
            
            params = {}
            
            # 提取最佳超参数配置部分
            match = re.search(r'最佳超参数配置:(.*?)\n\n试验统计:', content, re.DOTALL)
            if not match:
                # 如果没有找到试验统计部分，尝试匹配文件末尾
                match = re.search(r'最佳超参数配置:(.*?)$', content, re.DOTALL)
            
            if match:
                params_section = match.group(1)
                
                # 提取各个参数
                param_patterns = [
                    (r'n_layers: (\d+)', 'n_layers', int),
                    (r'hidden_size: (\d+)', 'hidden_size', int),
                    (r'activation: ([\w]+)', 'activation', str),
                    (r'mode: ([\w]+)', 'mode', str),
                    (r'M: (\d+)', 'M', int),
                    (r'learning_rate1: ([\d.e+-]+)', 'learning_rate1', float),
                    (r'n_iter1: (\d+)', 'n_iter1', int),
                ]
                
                for pattern, key, dtype in param_patterns:
                    match = re.search(pattern, params_section)
                    if match:
                        params[key] = dtype(match.group(1))
            
            # 添加Optuna.py中硬编码的时间步数N
            params['N'] = 50  # 从Optuna.py中获取的固定值
            
            return params
        
        # 使用Optuna报告中的最优参数（覆盖模型文件中的参数）
        print("\n🔧 从Optuna报告文件加载最优参数...")
        report_best_params = load_optuna_report_params(REPORT_PATH)
        print(f"✓ 从报告文件读取到的参数: {report_best_params.keys()}")

        # 更新最佳参数
        best_params.update(report_best_params)
        print("✓ Optuna报告参数已应用")

        # 设置训练参数
        N_Iter1 = best_params.get("n_iter1", 20000)  # 使用报告中的20000次
        learning_rate1 = best_params.get("learning_rate1", 0.00023345864076016249)  # 使用报告中的学习率
        
        # 同时更新模型的N值（时间步数）
        model.N = best_params.get("N", model.N)
        print(f"\n📊 更新后的关键参数:")
        print(f"   N (时间步数): {model.N}")
        print(f"   n_iter1 (训练步数): {N_Iter1}")
        print(f"   learning_rate1: {learning_rate1}")

        print("\n" + "=" * 80)
        print("           开始训练模型")
        print("=" * 80)
        print(f"训练阶段: {N_Iter1}次迭代, 学习率={learning_rate1}")

        # 获取问题参数
        Xi, T, D = trainer.get_problem_params()
        M = model.M  # 获取批次大小

        # 打印run_model调用前的入参
        print("\n" + "=" * 60)
        print("run_model 入参:")
        print("=" * 60)
        print(f"N_Iter1: {N_Iter1}")
        print(f"learning_rate1: {learning_rate1}")
        print(
            f"Xi shape: {Xi.shape}, Xi first few values: {Xi[0, :3] if Xi.size > 0 else 'empty'}"
        )
        print(f"T: {T}")
        print(f"D: {D}")
        print(f"M: {M}")
        print(f"model.mode: {model.mode}")
        print(f"model.activation: {model.activation}")
        print(f"model.D: {model.D}")
        print(f"model.M: {model.M}")
        print(f"model.N: {model.N}")
        print("=" * 60)

        # 将参数传递给run_model函数
        # 运行模型训练和评估
        final_model = run_model(
            model, N_Iter1, learning_rate1, Xi, T, D, M
        )

        print("\n" + "=" * 80)
        print("           训练完成!")
        print("=" * 80)

    except Exception as e:
        print(f"\n❌ 训练过程中出现错误: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
