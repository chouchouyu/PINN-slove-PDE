"""
梯度缩放（Gradient Scaling）原理与流程演示
"""

import torch
import torch.nn as nn

def demonstrate_gradient_scaling():
    """演示梯度缩放的基本原理和流程"""
    
    print("梯度缩放（Gradient Scaling）工作流程演示")
    print("=" * 60)
    
    # 1. 模拟问题：小梯度在float16中下溢
    print("\n1. 问题：小梯度在float16中可能下溢为0")
    print("-" * 40)
    
    # 模拟一个非常小的梯度（在float16的表示范围内接近0）
    true_gradient_fp32 = torch.tensor([1e-7, 2e-7, 5e-8], dtype=torch.float32)
    print(f"真实的梯度（float32）: {true_gradient_fp32}")
    
    # 转换为float16（模拟AMP中的计算）
    gradient_fp16 = true_gradient_fp32.to(torch.float16)
    print(f"转换为float16后    : {gradient_fp16}")
    print(f"注意：部分小梯度值在float16中可能被表示为0，导致信息丢失")
    
    # 2. 解决方案：梯度缩放
    print("\n\n2. 解决方案：在float16计算前放大损失")
    print("-" * 40)
    
    # 创建一个简单的模型和虚拟数据
    model = nn.Linear(3, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    
    # 模拟使用AMP训练（关键步骤分解）
    print("\nAMP训练中的关键步骤分解：")
    
    # 步骤1: 创建梯度缩放器（GradScaler）
    scaler = torch.cuda.amp.GradScaler(enabled=True)
    print(f"a) 创建梯度缩放器，初始缩放因子: {scaler.get_scale()}")
    
    # 虚拟输入和目标
    input_data = torch.randn(4, 3)
    target = torch.randn(4, 1)
    
    # 步骤2: 前向传播（在autocast上下文中，使用float16计算）
    print("b) 前向传播：在autocast上下文中，模型计算使用float16")
    
    # 步骤3: 计算损失（假设损失很小，模拟小梯度场景）
    with torch.cuda.amp.autocast():
        output = model(input_data)
        loss = nn.MSELoss()(output, target)
    
    print(f"   计算得到的损失值: {loss.item():.6e}")
    
    # 步骤4: 缩放损失并反向传播
    print("c) 缩放损失并反向传播")
    scaled_loss = scaler.scale(loss)  # 损失被放大（例如乘以2^16=65536倍）
    print(f"   缩放后的损失值: {scaled_loss.item():.6e}")
    print(f"   缩放因子约为: {scaled_loss.item() / loss.item():.0f}")
    
    # 反向传播（计算梯度）
    optimizer.zero_grad()
    scaled_loss.backward()  # 梯度也会被等比例放大
    
    # 检查缩放后的梯度
    scaled_grad = model.weight.grad.clone()
    print(f"   缩放后的梯度范数: {scaled_grad.norm().item():.6e}")
    
    # 步骤5: 将缩放后的梯度转换回优化器所需的精度
    print("d) 将缩放后的梯度‘反缩放’回原始范围")
    scaler.unscale_(optimizer)  # 将梯度除以缩放因子
    
    # 步骤6: 梯度裁剪（在反缩放后执行）
    print("e) 梯度裁剪（防止梯度爆炸）")
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
    
    # 步骤7: 优化器更新权重
    print("f) 优化器使用‘正确缩放’后的梯度更新权重")
    scaler.step(optimizer)
    
    # 步骤8: 更新缩放因子
    print("g) 更新缩放因子（基于本迭代中梯度是否出现inf/NaN）")
    scaler.update()
    print(f"   更新后的缩放因子: {scaler.get_scale()}")
    
    # 3. 动态缩放因子调整
    print("\n\n3. 动态缩放因子调整策略")
    print("-" * 40)
    
    print("""
梯度缩放器的智能调整策略：
1. 初始缩放因子较大（如2^16），确保小梯度不被下溢
2. 每次迭代后检查梯度：
   - 如果梯度中出现Inf/NaN：减少缩放因子（如乘以0.5）
   - 如果连续N次迭代没有Inf/NaN：增加缩放因子（如乘以2.0）
3. 目标：使用尽可能大的缩放因子，同时避免溢出
""")
    
    # 4. 在您期权定价模型中的具体应用
    print("\n\n4. 在期权定价GPU模型中的应用")
    print("-" * 40)
    
    print("""在您的GPUCallOption.train()方法中：

def train(self, n_iter, learning_rate):
    # ... 初始化 ...
    
    # 关键AMP代码段：
    if self.use_amp and self.device_type == 'cuda':
        with torch.cuda.amp.autocast():                    # 1) 自动精度转换
            loss, Y0_pred, Y1_pred = self.loss_function(...)
        
        self.scaler.scale(loss).backward()                # 2) 缩放损失并反向传播
        self.scaler.unscale_(self.optimizer)              # 3) 反缩放梯度
        torch.nn.utils.clip_grad_norm_(...)               # 4) 梯度裁剪
        self.scaler.step(self.optimizer)                  # 5) 优化器更新
        self.scaler.update()                              # 6) 更新缩放因子
    else:
        # 非AMP训练路径
        loss.backward()
        optimizer.step()

为什么这对期权定价很重要：
1. 数值稳定性：蒙特卡洛模拟可能产生非常小或非常大的中间值
2. 加速训练：float16计算比float32快得多
3. 内存效率：可以处理更大的批量或更复杂的模型
""")
    
    return scaler

def show_fp16_limitations():
    """展示float16的数值范围限制"""
    
    print("\n" + "=" * 60)
    print("float16数值范围限制演示")
    print("=" * 60)
    
    # float16的数值范围
    print("\nfloat16的表示范围：")
    print(f"  最小正规格化数: {torch.finfo(torch.float16).tiny:.6e}")
    print(f"  最大正数: {torch.finfo(torch.float16).max:.6e}")
    print(f"  机器精度: {torch.finfo(torch.float16).eps:.6e}")
    
    # 创建一些可能下溢的值
    test_values = torch.tensor([1e-8, 1e-7, 1e-6, 1e-5, 1e-4], dtype=torch.float32)
    print(f"\n测试值（float32）: {test_values}")
    
    # 转换为float16
    test_values_fp16 = test_values.to(torch.float16)
    print(f"转换为float16后 : {test_values_fp16}")
    
    # 识别下溢的值
    for i, (fp32, fp16) in enumerate(zip(test_values, test_values_fp16)):
        if fp16 == 0 and fp32 != 0:
            print(f"  ✓ 值 {fp32:.2e} 在float16中下溢为0")
        else:
            print(f"  ✓ 值 {fp32:.2e} 在float16中保持为 {fp16:.2e}")
    
    print("\n结论：")
    print("1. 当梯度值 < 约6e-8 时，在float16中可能变为0")
    print("2. 梯度缩放通过放大损失值，等比例放大所有梯度")
    print("3. 放大后的梯度 > float16的最小表示范围，避免下溢")
    print("4. 权重更新前，梯度被缩回原始比例")

if __name__ == "__main__":
    # 设置随机种子以便重现
    torch.manual_seed(42)
    
    # 演示梯度缩放原理
    scaler = demonstrate_gradient_scaling()
    
    # 展示float16限制
    show_fp16_limitations()
    
    print("\n" + "=" * 60)
    print("关键要点总结")
    print("=" * 60)
    
    summary = """
梯度缩放（Gradient Scaling）的本质：

1. **解决什么问题？**
   - float16数值范围小（约6e-5 ~ 65504），小梯度容易下溢为0
   - 梯度下溢会导致训练停滞（权重无法更新）

2. **如何解决？**
   - 前向/反向计算时：将损失放大S倍 → 梯度也放大S倍
   - 优化器更新前：将梯度缩小S倍 → 恢复原始比例
   - 动态调整S：基于梯度是否溢出（出现Inf/NaN）

3. **在PyTorch中如何实现？**
   - 使用 torch.cuda.amp.GradScaler()
   - scaler.scale(loss).backward()    # 缩放并反向传播
   - scaler.step(optimizer)           # 更新权重
   - scaler.update()                  # 调整缩放因子

4. **为什么对期权定价模型重要？**
   - 期权定价涉及大量数值计算，部分中间值可能很小
   - AMP+梯度缩放能显著加速训练（2-8倍）而不损失精度
   - 允许使用更大批量或更复杂模型进行蒙特卡洛模拟

简而言之，梯度缩放是AMP训练的"安全阀"，它让float16训练既快又稳。
"""
    print(summary)
