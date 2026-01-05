import numpy as np
import inspect
from typing import Callable, Any, Union, Optional

class PathsGenerator:
    """
    路径生成器基类
    这是一个示例类，您可以根据实际需求定义真正的PathsGenerator类
    """
    
    def __init__(self, S0: float, r: float, sigma: float, T: float, dt: float, asset_num: int = 1):
        """
        初始化路径生成器
        
        参数:
        - S0: 初始价格
        - r: 无风险利率
        - sigma: 波动率
        - T: 总时间
        - dt: 时间步长
        - asset_num: 资产数量
        """
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.dt = dt
        self.asset_num = asset_num
    
    def simulate(self, n_paths: int) -> np.ndarray:
        """
        模拟价格路径
        
        参数:
        - n_paths: 模拟路径数量
        
        返回: 三维数组，形状 (n_paths, asset_num, n_time_steps+1)
        """
        n_steps = int(self.T / self.dt) + 1
        
        # 创建初始路径
        paths = np.zeros((n_paths, self.asset_num, n_steps))
        paths[:, :, 0] = self.S0
        
        # 模拟路径
        for t in range(1, n_steps):
            Z = np.random.randn(n_paths, self.asset_num)
            paths[:, :, t] = paths[:, :, t-1] * np.exp(
                (self.r - 0.5 * self.sigma**2) * self.dt + 
                self.sigma * np.sqrt(self.dt) * Z
            )
        
        return paths

def standard_call_payoff(K: float = 100.0) -> Callable:
    """
    创建标准看涨期权收益函数
    
    参数:
    - K: 行权价
    
    返回: 收益函数
    """
    def payoff(*prices: float) -> float:
        """
        看涨期权收益函数
        
        参数:
        - *prices: 可变数量的价格参数
        
        返回: 收益值
        """
        # 将所有价格求和（对于多资产情况）
        total_price = np.sum(prices)
        return max(total_price - K, 0)
    
    return payoff

def standard_put_payoff(K: float = 100.0) -> Callable:
    """
    创建标准看跌期权收益函数
    
    参数:
    - K: 行权价
    
    返回: 收益函数
    """
    def payoff(*prices: float) -> float:
        """
        看跌期权收益函数
        
        参数:
        - *prices: 可变数量的价格参数
        
        返回: 收益值
        """
        total_price = np.sum(prices)
        return max(K - total_price, 0)
    
    return payoff

def validate_payoff_func(payoff_func: Callable) -> bool:
    """
    验证收益函数是否符合要求
    
    参数:
    - payoff_func: 要验证的收益函数
    
    返回: 布尔值，表示是否通过验证
    
    验证标准:
    1. 必须是可调用对象
    2. 必须接受可变位置参数
    3. 返回一个数值
    """
    if not callable(payoff_func):
        return False
    
    try:
        # 获取函数签名
        sig = inspect.signature(payoff_func)
        
        # 检查参数
        params = sig.parameters
        
        # 应该至少有一个参数，并且是可变位置参数（*args）
        has_varargs = False
        for param in params.values():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                has_varargs = True
                break
        
        if not has_varargs:
            return False
        
        # 测试函数调用
        # 用不同数量的参数测试
        test_cases = [
            (100.0,),  # 单个价格
            (50.0, 50.0),  # 两个价格
            (30.0, 30.0, 30.0)  # 三个价格
        ]
        
        for args in test_cases:
            result = payoff_func(*args)
            # 检查返回值是否为数值
            if not isinstance(result, (int, float, np.number)):
                return False
        
        return True
        
    except Exception:
        return False

def validate_paths_generator(paths_generator: Any, expected_class: type = None) -> bool:
    """
    验证路径生成器是否符合要求
    
    参数:
    - paths_generator: 要验证的路径生成器
    - expected_class: 期望的类类型（可选）
    
    返回: 布尔值，表示是否通过验证
    
    验证标准:
    1. 如果是expected_class提供，检查类型
    2. 必须具有simulate方法
    3. simulate方法必须返回numpy数组
    """
    if expected_class is not None and not isinstance(paths_generator, expected_class):
        return False
    
    # 检查是否有simulate方法
    if not hasattr(paths_generator, 'simulate'):
        return False
    
    simulate_method = getattr(paths_generator, 'simulate')
    if not callable(simulate_method):
        return False
    
    # 测试simulate方法
    try:
        result = simulate_method(10)  # 模拟10条路径
        if not isinstance(result, np.ndarray):
            return False
        return True
    except Exception:
        return False

class MC_American_Option:
    """
    美式期权蒙特卡洛定价类
    使用最小二乘蒙特卡洛方法
    """
    
    def __init__(self, paths_generator, payoff_func: Callable, K: Optional[float] = None):
        """
        初始化美式期权定价器
        
        参数:
        - paths_generator: 路径生成器实例
        - payoff_func: 收益函数，必须接受可变位置参数
        - K: 行权价（可选，如果payoff_func中已包含）
        
        抛出:
        - TypeError: 如果参数类型不正确
        - ValueError: 如果参数值无效
        """
        print("正在初始化MC_American_Option...")
        
        # 验证paths_generator
        print("验证paths_generator...")
        if not validate_paths_generator(paths_generator, PathsGenerator):
            raise TypeError(
                "paths_generator必须是PathsGenerator类的实例，并且具有simulate方法。"
            )
        print("✓ paths_generator验证通过")
        
        # 验证payoff_func
        print("验证payoff_func...")
        if not validate_payoff_func(payoff_func):
            raise TypeError(
                "payoff_func必须是可调用函数，接受可变位置参数，并返回数值。"
            )
        print("✓ payoff_func验证通过")
        
        # 存储参数
        self.paths_generator = paths_generator
        self.payoff_func = payoff_func
        self.K = K
        
        # 打印参数信息
        self._print_initialization_info()
    
    def _print_initialization_info(self):
        """打印初始化信息"""
        print("\n" + "=" * 60)
        print("MC_American_Option初始化完成")
        print("=" * 60)
        
        # 获取payoff_func信息
        payoff_name = self.payoff_func.__name__ if hasattr(self.payoff_func, '__name__') else "匿名函数"
        payoff_module = self.payoff_func.__module__ if hasattr(self.payoff_func, '__module__') else "__main__"
        
        # 获取paths_generator信息
        paths_gen_name = self.paths_generator.__class__.__name__
        
        print(f"参数配置:")
        print(f"  1. 路径生成器: {paths_gen_name}")
        print(f"  2. 收益函数: {payoff_name} (来自模块: {payoff_module})")
        if self.K is not None:
            print(f"  3. 行权价K: {self.K}")
        
        # 测试收益函数
        print(f"\n收益函数测试:")
        test_prices = [100.0, 105.0, 95.0]
        for i, price in enumerate(test_prices, 1):
            result = self.payoff_func(price)
            print(f"  测试{i}: payoff({price}) = {result}")
        
        # 测试路径生成器
        print(f"\n路径生成器测试:")
        test_paths = self.paths_generator.simulate(5)
        print(f"  模拟5条路径，形状: {test_paths.shape}")
        print(f"  资产数量: {test_paths.shape[1]}")
        print(f"  时间步数: {test_paths.shape[2]}")
    
    def price(self, n_paths: int = 1000, n_time_steps: int = None) -> float:
        """
        计算美式期权价格
        
        参数:
        - n_paths: 模拟路径数量
        - n_time_steps: 时间步数（如果为None，则使用paths_generator的默认值）
        
        返回: 期权价格
        """
        # 这里应该实现美式期权的定价逻辑
        # 为了示例，我们返回一个模拟值
        print(f"\n开始计算美式期权价格...")
        print(f"模拟参数: 路径数={n_paths}")
        
        # 这里应该实现完整的LSM算法
        # 目前返回一个占位值
        return 10.0  # 示例值

def test_mc_american_option():
    """测试MC_American_Option类的初始化"""
    print("MC_American_Option类测试")
    print("=" * 60)
    
    # 创建路径生成器
    paths_gen = PathsGenerator(
        S0=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        dt=0.1,
        asset_num=1
    )
    
    # 测试1: 使用正确的参数
    print("\n测试1: 正确参数")
    print("-" * 40)
    
    try:
        # 创建标准看涨期权收益函数
        K = 100.0
        payoff = standard_call_payoff(K)
        
        # 创建美式期权定价器
        option = MC_American_Option(paths_gen, payoff, K)
        
        print(f"初始化成功!")
        
        # 测试定价
        price = option.price(n_paths=100)
        print(f"计算出的期权价格: {price:.4f}")
        
    except Exception as e:
        print(f"初始化失败: {type(e).__name__}: {e}")
    
    # 测试2: 使用错误的paths_generator
    print("\n测试2: 错误的paths_generator")
    print("-" * 40)
    
    try:
        class WrongGenerator:
            def __init__(self):
                self.S0 = 100
        
        wrong_gen = WrongGenerator()
        payoff = standard_call_payoff(100)
        option = MC_American_Option(wrong_gen, payoff, 100)
    except TypeError as e:
        print(f"预期错误: {type(e).__name__}: {e}")
    
    # 测试3: 使用错误的payoff_func
    print("\n测试3: 错误的payoff_func")
    print("-" * 40)
    
    try:
        # 错误的收益函数：不接受可变参数
        def wrong_payoff(price):
            return max(price - 100, 0)
        
        option = MC_American_Option(paths_gen, wrong_payoff, 100)
    except TypeError as e:
        print(f"预期错误: {type(e).__name__}: {e}")
    
    # 测试4: 使用lambda函数
    print("\n测试4: 使用lambda函数")
    print("-" * 40)
    
    try:
        # 使用lambda函数（应该失败，因为lambda没有可变参数）
        lambda_payoff = lambda *prices: max(np.sum(prices) - 100, 0)
        
        # 实际上这个lambda函数是正确的，因为它有*args
        option = MC_American_Option(paths_gen, lambda_payoff, 100)
        print(f"lambda函数验证通过!")
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}")
    
    # 测试5: 多资产收益函数
    print("\n测试5: 多资产收益函数")
    print("-" * 40)
    
    try:
        # 创建多资产路径生成器
        multi_asset_gen = PathsGenerator(
            S0=100.0,
            r=0.05,
            sigma=0.2,
            T=1.0,
            dt=0.1,
            asset_num=3
        )
        
        # 多资产收益函数：三个资产价格之和减去行权价
        def multi_asset_payoff(*prices):
            return max(np.sum(prices) - 300, 0)
        
        option = MC_American_Option(multi_asset_gen, multi_asset_payoff, 300)
        print(f"多资产收益函数验证通过!")
    except Exception as e:
        print(f"错误: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)

def test_payoff_func_validation():
    """测试收益函数验证"""
    print("\n收益函数验证测试")
    print("=" * 60)
    
    # 正确的收益函数
    def correct_payoff(*prices):
        K = 100
        return max(np.sum(prices) - K, 0)
    
    # 错误的收益函数1：没有可变参数
    def wrong_payoff1(price):
        return max(price - 100, 0)
    
    # 错误的收益函数2：返回非数值
    def wrong_payoff2(*prices):
        return "not a number"
    
    # 错误的收益函数3：不可调用
    not_callable = 123
    
    test_cases = [
        ("正确的收益函数", correct_payoff, True),
        ("没有可变参数的函数", wrong_payoff1, False),
        ("返回非数值的函数", wrong_payoff2, False),
        ("非可调用对象", not_callable, False),
    ]
    
    for name, func, expected in test_cases:
        result = validate_payoff_func(func)
        status = "通过" if result == expected else "失败"
        print(f"{name}: 验证{status} (结果: {result}, 期望: {expected})")

if __name__ == "__main__":
    # 运行测试
    test_payoff_func_validation()
    test_mc_american_option()
    
    # 使用示例
    print("\n" + "=" * 60)
    print("使用示例")
    print("=" * 60)
    
    # 创建路径生成器
    pg = PathsGenerator(
        S0=100.0,
        r=0.05,
        sigma=0.2,
        T=1.0,
        dt=0.1,
        asset_num=2
    )
    
    # 定义收益函数
    K = 200.0
    def my_payoff(*prices):
        """自定义收益函数，两个资产价格之和减去行权价"""
        return max(np.sum(prices) - K, 0)
    
    # 创建美式期权定价器
    try:
        american_option = MC_American_Option(pg, my_payoff, K)
        print(f"美式期权定价器创建成功!")
    except Exception as e:
        print(f"创建失败: {e}")
