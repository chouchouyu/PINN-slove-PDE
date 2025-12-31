"""
梯度裁剪（Gradient Clipping）原理与示例
演示梯度裁剪的作用、实现方式以及在训练中的应用
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

def demonstrate_gradient_clipping():
    """演示梯度裁剪的基本原理和作用"""
    
    print("梯度裁剪（Gradient Clipping）原理与作用")
    print("=" * 60)
    
    # 1. 什么是梯度爆炸问题
    print("\n1. 梯度爆炸（Gradient Explosion）问题")
    print("-" * 40)
    
    print("""梯度爆炸是指在深度神经网络训练过程中，梯度值变得异常巨大的现象。

产生原因：
1. 网络层数过深，梯度在反向传播中连续相乘
2. 权重初始化不当
3. 学习率设置过高
4. 损失函数曲面在某些区域非常陡峭

后果：
1. 参数更新过大，模型"跳跃"到不良区域
2. 损失函数出现NaN（非数值）
3. 训练过程不稳定，无法收敛
4. 在混合精度训练中更容易发生溢出""")
    
    # 2. 梯度裁剪的基本思想
    print("\n\n2. 梯度裁剪的基本思想")
    print("-" * 40)
    
    print("""梯度裁剪的核心思想：限制梯度的大小，防止梯度爆炸。

具体操作：
1. 计算所有参数的梯度范数（通常使用L2范数）
2. 如果梯度范数超过预设的阈值（max_norm）
3. 将所有梯度按比例缩小，使得总范数等于阈值
4. 公式：gradient = gradient * (max_norm / max(norm, max_norm))

数学表达：
if total_norm > max_norm:
    clip_coef = max_norm / (total_norm + 1e-6)
    for param in model.parameters():
        param.grad *= clip_coef""")
    
    # 3. 在PyTorch中的使用
    print("\n\n3. 在PyTorch中的使用")
    print("-" * 40)
    
    print("""PyTorch提供torch.nn.utils.clip_grad_norm_函数：

基本用法：
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

常用参数：
- parameters: 模型参数
- max_norm: 最大梯度范数阈值
- norm_type: 范数类型（默认2，L2范数）
- error_if_nonfinite: 如果梯度为NaN/Inf是否报错""")
    
    # 4. 实际演示梯度裁剪效果
    print("\n\n4. 梯度裁剪效果演示")
    print("-" * 40)
    
    # 创建一个简单的模型
    torch.manual_seed(42)
    
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(10, 50)
            self.fc2 = nn.Linear(50, 20)
            self.fc3 = nn.Linear(20, 1)
            self.relu = nn.ReLU()
        
        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu(self.fc2(x))
            return self.fc3(x)
    
    # 创建模型、优化器和数据
    model = SimpleModel()
    optimizer = optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()
    
    # 创建一些测试数据
    batch_size = 8
    x = torch.randn(batch_size, 10)
    y = torch.randn(batch_size, 1)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    print(f"批量大小: {batch_size}")
    
    # 模拟几次训练迭代，观察梯度变化
    gradient_norms_no_clip = []
    gradient_norms_with_clip = []
    
    for iteration in range(5):
        print(f"\n--- 迭代 {iteration+1} ---")
        
        # 复制模型用于对比
        model_no_clip = SimpleModel()
        model_no_clip.load_state_dict(model.state_dict())
        optimizer_no_clip = optim.SGD(model_no_clip.parameters(), lr=0.1)
        
        model_with_clip = SimpleModel()
        model_with_clip.load_state_dict(model.state_dict())
        optimizer_with_clip = optim.SGD(model_with_clip.parameters(), lr=0.1)
        
        # 前向传播
        output_no_clip = model_no_clip(x)
        loss_no_clip = criterion(output_no_clip, y)
        
        output_with_clip = model_with_clip(x)
        loss_with_clip = criterion(output_with_clip, y)
        
        # 反向传播
        optimizer_no_clip.zero_grad()
        loss_no_clip.backward()
        
        optimizer_with_clip.zero_grad()
        loss_with_clip.backward()
        
        # 计算梯度范数（裁剪前）
        def calculate_total_grad_norm(model):
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            return total_norm ** 0.5
        
        grad_norm_before = calculate_total_grad_norm(model_no_clip)
        gradient_norms_no_clip.append(grad_norm_before)
        print(f"裁剪前梯度范数: {grad_norm_before:.6f}")
        
        # 应用梯度裁剪
        torch.nn.utils.clip_grad_norm_(model_with_clip.parameters(), max_norm=1.0)
        
        # 计算梯度范数（裁剪后）
        grad_norm_after = calculate_total_grad_norm(model_with_clip)
        gradient_norms_with_clip.append(grad_norm_after)
        print(f"裁剪后梯度范数: {grad_norm_after:.6f}")
        
        # 执行优化器步骤
        optimizer_no_clip.step()
        optimizer_with_clip.step()
        
        # 更新模型状态
        model.load_state_dict(model_with_clip.state_dict())
    
    # 5. 可视化梯度裁剪效果
    print("\n\n5. 梯度裁剪效果可视化")
    print("-" * 40)
    
    # 创建一个更大的演示
    torch.manual_seed(42)
    
    # 创建一个容易产生大梯度的场景
    class UnstableModel(nn.Module):
        def __init__(self):
            super().__init__()
            # 使用较大的初始权重，模拟梯度爆炸
            self.fc1 = nn.Linear(5, 10)
            nn.init.normal_(self.fc1.weight, mean=0.0, std=10.0)  # 大权重
            self.fc2 = nn.Linear(10, 5)
            nn.init.normal_(self.fc2.weight, mean=0.0, std=10.0)  # 大权重
            self.fc3 = nn.Linear(5, 1)
        
        def forward(self, x):
            x = torch.tanh(self.fc1(x))
            x = torch.tanh(self.fc2(x))
            return self.fc3(x)
    
    # 训练并记录梯度
    def train_and_record(model, use_clipping=False, max_norm=1.0):
        """训练模型并记录梯度信息"""
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        criterion = nn.MSELoss()
        
        x = torch.randn(4, 5)
        y = torch.randn(4, 1)
        
        gradient_norms = []
        loss_values = []
        
        for epoch in range(20):
            # 前向传播
            output = model(x)
            loss = criterion(output, y)
            loss_values.append(loss.item())
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            
            # 计算并记录梯度范数
            total_norm = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            gradient_norms.append(total_norm ** 0.5)
            
            # 应用梯度裁剪
            if use_clipping:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_norm)
            
            # 更新参数
            optimizer.step()
        
        return gradient_norms, loss_values
    
    # 运行有/无梯度裁剪的训练
    print("运行有/无梯度裁剪的训练对比...")
    
    model_no_clip = UnstableModel()
    model_with_clip = UnstableModel()
    
    # 确保两个模型初始状态相同
    model_with_clip.load_state_dict(model_no_clip.state_dict())
    
    grad_norms_no_clip, losses_no_clip = train_and_record(model_no_clip, use_clipping=False)
    grad_norms_with_clip, losses_with_clip = train_and_record(model_with_clip, use_clipping=True, max_norm=5.0)
    
    # 绘制结果
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 梯度范数对比
    axes[0, 0].plot(grad_norms_no_clip, 'r-', linewidth=2, label='无梯度裁剪')
    axes[0, 0].plot(grad_norms_with_clip, 'b-', linewidth=2, label='有梯度裁剪')
    axes[0, 0].axhline(y=5.0, color='g', linestyle='--', label='裁剪阈值=5.0')
    axes[0, 0].set_xlabel('训练迭代')
    axes[0, 0].set_ylabel('梯度范数')
    axes[0, 0].set_title('梯度范数变化对比')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # 损失函数对比
    axes[0, 1].plot(losses_no_clip, 'r-', linewidth=2, label='无梯度裁剪')
    axes[0, 1].plot(losses_with_clip, 'b-', linewidth=2, label='有梯度裁剪')
    axes[0, 1].set_xlabel('训练迭代')
    axes[0, 1].set_ylabel('损失值')
    axes[0, 1].set_title('损失函数变化对比')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_yscale('log')
    
    # 梯度分布直方图（最后一次迭代）
    axes[1, 0].hist(grad_norms_no_clip, bins=15, alpha=0.7, color='red', edgecolor='black', label='无梯度裁剪')
    axes[1, 0].hist(grad_norms_with_clip, bins=15, alpha=0.7, color='blue', edgecolor='black', label='有梯度裁剪')
    axes[1, 0].set_xlabel('梯度范数')
    axes[1, 0].set_ylabel('频数')
    axes[1, 0].set_title('梯度范数分布直方图')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # 训练稳定性分析
    stability_no_clip = np.std(grad_norms_no_clip) / np.mean(grad_norms_no_clip)
    stability_with_clip = np.std(grad_norms_with_clip) / np.mean(grad_norms_with_clip)
    
    axes[1, 1].axis('off')
    axes[1, 1].text(0.5, 0.8, '训练稳定性分析', ha='center', va='center', fontsize=16, fontweight='bold')
    axes[1, 1].text(0.5, 0.7, f'无裁剪梯度变异系数: {stability_no_clip:.4f}', ha='center', va='center', fontsize=12)
    axes[1, 1].text(0.5, 0.6, f'有裁剪梯度变异系数: {stability_with_clip:.4f}', ha='center', va='center', fontsize=12)
    axes[1, 1].text(0.5, 0.5, f'稳定性提升: {(stability_no_clip/stability_with_clip):.2f}倍', 
                   ha='center', va='center', fontsize=12, color='green')
    axes[1, 1].text(0.5, 0.4, '结论:', ha='center', va='center', fontsize=14, fontweight='bold')
    axes[1, 1].text(0.5, 0.3, '梯度裁剪显著提高训练稳定性', ha='center', va='center', fontsize=12, color='green')
    
    plt.tight_layout()
    plt.savefig('gradient_clipping_effect.png', dpi=150, bbox_inches='tight')
    print("梯度裁剪效果图已保存为 'gradient_clipping_effect.png'")
    plt.show()
    
    # 6. 梯度裁剪在期权定价模型中的应用
    print("\n\n6. 在期权定价模型中的应用")
    print("-" * 40)
    
    print("""在您的GPUCallOption期权定价模型中，梯度裁剪特别重要：

为什么需要梯度裁剪：
1. 数值敏感性：期权定价涉及金融衍生品计算，对数值稳定性要求高
2. 蒙特卡洛模拟：随机路径可能产生极端值，导致梯度爆炸
3. 深度网络：复杂的神经网络结构容易积累梯度
4. 混合精度训练：float16数值范围小，更容易溢出

在训练循环中的位置（以您的代码为例）：
def train(self, n_iter, learning_rate):
    for it in range(n_iter):
        optimizer.zero_grad()
        
        # AMP训练
        if self.use_amp and self.device_type == 'cuda':
            with torch.cuda.amp.autocast():
                loss = self.loss_function(...)
            
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)  # 必须先反缩放
            
            # 梯度裁剪（关键步骤！）
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            # 非AMP训练
            loss = self.loss_function(...)
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            optimizer.step()

参数选择建议：
1. max_norm: 通常设为0.5-5.0之间，需根据具体任务调整
2. 在期权定价中，建议从1.0开始尝试
3. 监控梯度范数，确保裁剪阈值合理
4. 与AMP配合使用时，注意顺序：先unscale，再clip""")
    
    return gradient_norms_no_clip, gradient_norms_with_clip, losses_no_clip, losses_with_clip

def common_problems_and_solutions():
    """常见问题与解决方案"""
    print("\n\n7. 常见问题与解决方案")
    print("-" * 40)
    
    problems = [
        {
            "问题": "梯度裁剪后训练变慢",
            "原因": "阈值设置过小，限制了梯度更新",
            "解决方案": "适当增大max_norm，如从0.5调到1.0或2.0"
        },
        {
            "问题": "梯度仍然出现NaN",
            "原因": "裁剪阈值仍然太大，或模型其他问题",
            "解决方案": "减小max_norm，检查数据预处理，添加梯度裁剪前的检查"
        },
        {
            "问题": "不知道如何设置max_norm",
            "原因": "缺乏梯度范数监控",
            "解决方案": """在裁剪前记录梯度范数：
total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
print(f"梯度范数: {total_norm}")"""
        },
        {
            "问题": "与AMP一起使用时出错",
            "原因": "顺序错误",
            "解决方案": "正确顺序：scaler.scale(loss).backward() → scaler.unscale_(optimizer) → clip_grad_norm_ → scaler.step(optimizer)"
        },
        {
            "问题": "在期权定价中效果不明显",
            "原因": "阈值不适合您的具体模型",
            "解决方案": "尝试不同的max_norm值，监控训练损失和梯度分布"
        }
    ]
    
    for i, problem in enumerate(problems, 1):
        print(f"\n{i}. {problem['问题']}")
        print(f"   原因: {problem['原因']}")
        print(f"   解决方案: {problem['解决方案']}")
    
    return problems

if __name__ == "__main__":
    print("梯度裁剪原理与作用详解")
    print("=" * 60)
    
    # 运行主演示
    grad_norms_no_clip, grad_norms_with_clip, losses_no_clip, losses_with_clip = demonstrate_gradient_clipping()
    
    # 常见问题
    problems = common_problems_and_solutions()
    
    print("\n" + "=" * 60)
    print("总结")
    print("=" * 60)
    
    print("""
梯度裁剪（Gradient Clipping）核心总结：

1. 主要作用：
   - 防止梯度爆炸，稳定训练过程
   - 避免参数更新过大，提高收敛性
   - 在混合精度训练中防止数值溢出

2. 工作原理：
   - 计算所有梯度的总范数
   - 如果范数超过阈值，按比例缩小所有梯度
   - 保持梯度方向不变，只改变幅度

3. 在期权定价模型中的重要性：
   - 金融计算对数值稳定性要求高
   - 蒙特卡洛模拟可能产生极端梯度
   - 深度网络结构容易积累梯度
   - 与AMP配合使用可显著加速训练

4. 使用建议：
   - 通常在反向传播后、优化器更新前使用
   - 与AMP一起使用时注意操作顺序
   - 根据任务调整max_norm（通常0.5-5.0）
   - 监控梯度范数以选择合适的阈值

梯度裁剪是一个简单但强大的工具，能显著提高深度学习训练的稳定性和可靠性，
特别是在像期权定价这样对数值精度敏感的应用中。
    """)
