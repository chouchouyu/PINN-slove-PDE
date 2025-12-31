"""
跨设备混合精度训练实现完整版
支持CUDA、MPS、CPU等多种设备的自动混合精度训练
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
import warnings
warnings.filterwarnings('ignore')

def setup_device():
    """检测并设置最佳计算设备"""
    device = None
    device_type = None
    
    # 1. 检测CUDA
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        device_type = "cuda"
        print(f"检测到CUDA设备: {device}, {torch.cuda.get_device_name(0)}")
        return device, device_type
    
    # 2. 检测MPS
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
        device_type = "mps"
        print(f"检测到MPS设备: {device}")
        return device, device_type
    
    # 3. 回退到CPU
    device = torch.device("cpu")
    device_type = "cpu"
    print(f"使用CPU设备: {device}")
    return device, device_type

def create_test_model(input_size=10, hidden_size=50, output_size=1):
    """创建一个简单的测试模型"""
    return nn.Sequential(
        nn.Linear(input_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, output_size)
    )

def test_mps_amp_support():
    """测试MPS设备的AMP支持情况"""
    print("\n" + "="*60)
    print("测试MPS设备的AMP支持情况")
    print("="*60)
    
    if not (hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()):
        print("MPS设备不可用，跳过测试")
        return False
    
    print("PyTorch版本:", torch.__version__)
    print("设备类型: MPS")
    
    # 测试MPS是否支持autocast
    mps_amp_available = False
    try:
        # 尝试导入MPS相关的AMP
        if hasattr(torch, 'autocast'):
            # 测试MPS autocast
            with torch.autocast(device_type='mps', dtype=torch.float16):
                x = torch.randn(4, 10, device='mps')
                model = create_test_model().to('mps')
                _ = model(x)
            mps_amp_available = True
            print("✓ MPS支持torch.autocast")
    except Exception as e:
        print(f"✗ MPS autocast不支持: {e}")
    
    # 测试MPS是否支持GradScaler
    mps_scaler_available = False
    try:
        # PyTorch 1.12+ 有通用的GradScaler
        if hasattr(torch.amp, 'GradScaler'):
            scaler = torch.amp.GradScaler('mps')
            mps_scaler_available = True
            print("✓ MPS支持torch.amp.GradScaler")
        else:
            print("✗ 当前PyTorch版本不支持torch.amp.GradScaler")
    except Exception as e:
        print(f"✗ MPS GradScaler不支持: {e}")
    
    return mps_amp_available and mps_scaler_available

def train_with_cuda_amp(model, data_loader, criterion, optimizer, num_epochs=5):
    """使用CUDA AMP进行训练"""
    print("\n" + "="*60)
    print("CUDA AMP训练")
    print("="*60)
    
    from torch.cuda.amp import autocast, GradScaler
    
    model.train()
    scaler = GradScaler()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            
            optimizer.zero_grad()
            
            # 使用autocast进行前向传播
            with autocast():
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 缩放损失并反向传播
            scaler.scale(loss).backward()
            
            # 取消缩放，梯度裁剪，优化器步骤
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
    
    return model

def train_with_mps_amp(model, data_loader, criterion, optimizer, num_epochs=5):
    """使用MPS AMP进行训练"""
    print("\n" + "="*60)
    print("MPS AMP训练")
    print("="*60)
    
    model.train()
    
    # PyTorch 2.0+ 支持通用autocast
    autocast_available = hasattr(torch, 'autocast')
    
    # 使用通用GradScaler（如果可用）
    if hasattr(torch.amp, 'GradScaler'):
        scaler = torch.amp.GradScaler('mps')
        use_scaler = True
    else:
        use_scaler = False
        print("警告: torch.amp.GradScaler不可用，将不使用梯度缩放")
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to('mps'), targets.to('mps')
            
            optimizer.zero_grad()
            
            if autocast_available:
                # 使用MPS autocast
                with torch.autocast(device_type='mps', dtype=torch.float16):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                # 手动混合精度
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            if use_scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
    
    return model

def train_with_cpu_fallback(model, data_loader, criterion, optimizer, num_epochs=5):
    """CPU回退训练（无AMP）"""
    print("\n" + "="*60)
    print("CPU训练（无AMP）")
    print("="*60)
    
    model.train()
    
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        start_time = time.time()
        
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            optimizer.zero_grad()
            
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
        
        epoch_time = time.time() - start_time
        avg_loss = epoch_loss / len(data_loader)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.6f}, Time: {epoch_time:.2f}s")
    
    return model

def create_dummy_data(batch_size=32, input_size=10, num_batches=10):
    """创建虚拟训练数据"""
    data_loader = []
    for _ in range(num_batches):
        inputs = torch.randn(batch_size, input_size)
        targets = torch.randn(batch_size, 1)
        data_loader.append((inputs, targets))
    return data_loader

def benchmark_amp_performance(device_type):
    """基准测试不同设备上AMP的性能"""
    print("\n" + "="*60)
    print(f"{device_type.upper()}设备AMP性能基准测试")
    print("="*60)
    
    # 设置设备
    if device_type == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda:0")
    elif device_type == "mps" and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
        device_type = "cpu"
    
    # 创建模型和数据
    model = create_test_model().to(device)
    data_loader = create_dummy_data(batch_size=32, input_size=10, num_batches=5)
    
    # 将数据移动到设备
    for i in range(len(data_loader)):
        inputs, targets = data_loader[i]
        data_loader[i] = (inputs.to(device), targets.to(device))
    
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # 测试有AMP和无AMP的性能
    results = {}
    
    if device_type == "cuda":
        # CUDA AMP测试
        print("测试CUDA AMP性能...")
        
        # 有AMP
        start_time = time.time()
        model_amp = create_test_model().to(device)
        optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
        model_amp = train_with_cuda_amp(model_amp, data_loader, criterion, optimizer_amp, num_epochs=3)
        amp_time = time.time() - start_time
        
        # 无AMP
        start_time = time.time()
        model_no_amp = create_test_model().to(device)
        optimizer_no_amp = optim.Adam(model_no_amp.parameters(), lr=0.001)
        model_no_amp = train_with_cpu_fallback(model_no_amp, data_loader, criterion, optimizer_no_amp, num_epochs=3)
        no_amp_time = time.time() - start_time
        
        results = {
            'device': device_type,
            'amp_time': amp_time,
            'no_amp_time': no_amp_time,
            'speedup': no_amp_time / amp_time if amp_time > 0 else 0
        }
        
    elif device_type == "mps":
        # MPS AMP测试
        print("测试MPS AMP性能...")
        
        # 检查AMP支持
        mps_amp_supported = test_mps_amp_support()
        
        if mps_amp_supported:
            # 有AMP
            start_time = time.time()
            model_amp = create_test_model().to(device)
            optimizer_amp = optim.Adam(model_amp.parameters(), lr=0.001)
            model_amp = train_with_mps_amp(model_amp, data_loader, criterion, optimizer_amp, num_epochs=3)
            amp_time = time.time() - start_time
            
            # 无AMP
            start_time = time.time()
            model_no_amp = create_test_model().to(device)
            optimizer_no_amp = optim.Adam(model_no_amp.parameters(), lr=0.001)
            model_no_amp = train_with_cpu_fallback(model_no_amp, data_loader, criterion, optimizer_no_amp, num_epochs=3)
            no_amp_time = time.time() - start_time
            
            results = {
                'device': device_type,
                'amp_supported': True,
                'amp_time': amp_time,
                'no_amp_time': no_amp_time,
                'speedup': no_amp_time / amp_time if amp_time > 0 else 0
            }
        else:
            results = {
                'device': device_type,
                'amp_supported': False,
                'amp_time': 0,
                'no_amp_time': 0,
                'speedup': 0
            }
            print("MPS AMP不支持，跳过性能测试")
    
    else:
        # CPU测试（无AMP）
        print("CPU设备不支持AMP，跳过性能测试")
        results = {
            'device': device_type,
            'amp_supported': False,
            'amp_time': 0,
            'no_amp_time': 0,
            'speedup': 0
        }
    
    return results

def unified_amp_training_logic():
    """统一的AMP训练逻辑，自动检测设备并选择最佳AMP策略"""
    print("\n" + "="*60)
    print("统一AMP训练逻辑实现")
    print("="*60)
    
    # 检测设备
    device, device_type = setup_device()
    print(f"使用设备: {device_type} ({device})")
    
    class UnifiedAMPTrainer:
        """统一的AMP训练器，自动适配不同设备"""
        def __init__(self, device, device_type):
            self.device = device
            self.device_type = device_type
            self.scaler = None
            self.autocast_context = None
            
            # 初始化AMP相关组件
            self._init_amp_components()
        
        def _init_amp_components(self):
            """根据设备类型初始化AMP组件"""
            if self.device_type == "cuda":
                # CUDA设备：使用torch.cuda.amp
                from torch.cuda.amp import autocast, GradScaler
                self.autocast_context = lambda: autocast()
                self.scaler = GradScaler()
                print("✓ 初始化CUDA AMP组件")
                
            elif self.device_type == "mps":
                # MPS设备：使用通用AMP（如果可用）
                if hasattr(torch, 'autocast'):
                    self.autocast_context = lambda: torch.autocast(
                        device_type='mps', 
                        dtype=torch.float16
                    )
                    print("✓ 初始化MPS autocast")
                else:
                    self.autocast_context = None
                    print("✗ MPS autocast不可用")
                
                if hasattr(torch.amp, 'GradScaler'):
                    self.scaler = torch.amp.GradScaler('mps')
                    print("✓ 初始化MPS GradScaler")
                else:
                    self.scaler = None
                    print("✗ MPS GradScaler不可用")
                
            else:
                # CPU设备：无AMP支持
                self.autocast_context = None
                self.scaler = None
                print("ℹ CPU设备不支持AMP")
        
        def train_step(self, model, optimizer, criterion, inputs, targets):
            """执行一个训练步骤"""
            # 将数据移动到设备
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad()
            
            # 前向传播（使用AMP）
            if self.autocast_context is not None:
                with self.autocast_context():
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = model(inputs)
                loss = criterion(outputs, targets)
            
            # 反向传播和优化
            if self.scaler is not None:
                # 使用梯度缩放
                self.scaler.scale(loss).backward()
                
                # 取消缩放并应用梯度裁剪
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 优化器步骤
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                # 无梯度缩放
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            return loss.item()
        
        def inference_step(self, model, inputs):
            """推理步骤"""
            inputs = inputs.to(self.device)
            
            with torch.no_grad():
                if self.autocast_context is not None:
                    with self.autocast_context():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)
            
            return outputs
    
    # 使用示例
    trainer = UnifiedAMPTrainer(device, device_type)
    
    # 创建测试模型和数据
    model = create_test_model().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 创建测试数据
    test_inputs = torch.randn(4, 10).to(device)
    test_targets = torch.randn(4, 1).to(device)
    
    # 执行训练步骤测试
    print("\n执行训练步骤测试...")
    loss = trainer.train_step(model, optimizer, criterion, test_inputs, test_targets)
    print(f"训练步骤完成，损失: {loss:.6f}")
    
    # 执行推理步骤测试
    print("\n执行推理步骤测试...")
    outputs = trainer.inference_step(model, test_inputs)
    print(f"推理完成，输出形状: {outputs.shape}")
    
    return trainer

def main():
    """主函数"""
    print("跨设备混合精度训练实现")
    print("="*60)
    
    # 1. 测试MPS AMP支持
    mps_amp_supported = test_mps_amp_support()
    
    # 2. 运行性能基准测试
    device_results = []
    
    # 测试CUDA（如果可用）
    if torch.cuda.is_available():
        cuda_results = benchmark_amp_performance("cuda")
        device_results.append(cuda_results)
    
    # 测试MPS（如果可用）
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        mps_results = benchmark_amp_performance("mps")
        device_results.append(mps_results)
    
    # 测试CPU
    cpu_results = benchmark_amp_performance("cpu")
    device_results.append(cpu_results)
    
    # 3. 显示结果摘要
    print("\n" + "="*60)
    print("性能测试结果摘要")
    print("="*60)
    
    for result in device_results:
        print(f"\n{result['device'].upper()}设备:")
        if result.get('amp_supported', False) or result['device'] == 'cuda':
            print(f"  AMP支持: 是")
            print(f"  AMP训练时间: {result['amp_time']:.2f}s")
            print(f"  无AMP训练时间: {result['no_amp_time']:.2f}s")
            if result['speedup'] > 0:
                print(f"  加速比: {result['speedup']:.2f}x")
        else:
            print(f"  AMP支持: 否")
    
    # 4. 统一AMP训练逻辑演示
    print("\n" + "="*60)
    print("统一AMP训练逻辑演示")
    print("="*60)
    
    unified_trainer = unified_amp_training_logic()
    
    # 5. 在您的期权定价模型中的建议
    print("\n" + "="*60)
    print("在期权定价模型中的应用建议")
    print("="*60)
    
    print("""
对于苹果M4芯片（MPS设备）的AMP使用建议：

1. AMP支持情况:
   - PyTorch 2.0+ 支持MPS的autocast
   - 可以使用通用torch.amp.GradScaler
   - 但MPS的AMP加速效果可能不如CUDA明显

2. 代码适配建议:
   - 使用统一的AMP训练逻辑，自动检测设备
   - 在CUDA设备上使用torch.cuda.amp
   - 在MPS设备上使用torch.autocast(device_type='mps')
   - 在CPU设备上不使用AMP

3. 在GPUCallOption模型中的实现:
   可以参考上面的统一AMP训练逻辑，根据设备类型自动选择AMP策略。
    """)
    
    return device_results, unified_trainer

def test_amp_in_option_pricing():
    """测试在期权定价模型中应用AMP"""
    print("\n" + "="*60)
    print("在期权定价模型中应用AMP测试")
    print("="*60)
    
    # 检测设备
    device, device_type = setup_device()
    
    # 创建模拟的期权定价网络
    class OptionPricingNet(nn.Module):
        def __init__(self, input_size=2, hidden_size=64, output_size=1):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, output_size)
            )
        
        def forward(self, x):
            return self.net(x)
    
    # 创建模型
    model = OptionPricingNet().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # 创建模拟的期权数据 (t, S)
    batch_size = 32
    t = torch.rand(batch_size, 1)  # 时间维度
    S = torch.randn(batch_size, 1)  # 资产价格
    inputs = torch.cat([t, S], dim=1).to(device)
    targets = torch.randn(batch_size, 1).to(device)  # 期权价格
    
    # 创建统一AMP训练器
    trainer = UnifiedAMPTrainer(device, device_type)
    
    # 执行一个训练步骤
    print("执行期权定价模型训练步骤...")
    loss = trainer.train_step(model, optimizer, criterion, inputs, targets)
    print(f"训练步骤完成，损失: {loss:.6f}")
    
    # 执行推理
    print("\n执行期权定价模型推理...")
    test_inputs = torch.cat([torch.rand(4, 1), torch.randn(4, 1)], dim=1).to(device)
    outputs = trainer.inference_step(model, test_inputs)
    print(f"推理完成，输出形状: {outputs.shape}")
    
    return model, trainer

if __name__ == "__main__":
    # 运行主函数
    device_results, trainer = main()
    
    # 运行期权定价模型测试
    option_model, option_trainer = test_amp_in_option_pricing()
    
    print("\n" + "="*60)
    print("程序执行完成")
    print("="*60)
    
    # 显示设备信息总结
    print("\n设备信息总结:")
    device, device_type = setup_device()
    print(f"最终使用的设备: {device_type} ({device})")
    print(f"PyTorch版本: {torch.__version__}")
    
    if device_type == "cuda":
        print(f"GPU名称: {torch.cuda.get_device_name(0)}")
        print(f"GPU内存: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
    elif device_type == "mps":
        print("MPS设备: Apple Silicon GPU")
