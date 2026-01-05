import numpy as np
from typing import List, Tuple, Union

class RegressionModel:
    """
    回归模型类，用于最小二乘蒙特卡洛（LSM）期权定价
    通过基函数和回归系数来评估继续持有期权的价值
    """
    
    def __init__(self, basis_functions: List[str] = None, degree: int = 2):
        """
        初始化回归模型
        
        输入参数：
        basis_functions: 基函数类型列表，如['poly', 'exp', 'log']
        degree: 多项式阶数（如果使用多项式基函数）
        
        内部状态：
        self.basis_funcs: 存储基函数类型
        self.degree: 存储多项式阶数
        self.coefficients: 存储回归系数（训练后设置）
        self.feature_names: 存储特征名称（训练后设置）
        """
        self.basis_funcs = basis_functions or ['poly']
        self.degree = degree
        self.coefficients = None
        self.feature_names = None
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> 'RegressionModel':
        """
        训练回归模型
        
        输入参数：
        X: 特征矩阵，形状 (n_samples, n_features)
        y: 目标值，形状 (n_samples,)
        
        处理流程：
        1. 创建设计矩阵（应用基函数变换）
        2. 使用最小二乘法计算回归系数
        3. 存储系数和特征名称
        
        输出：训练好的模型自身（支持链式调用）
        """
        # 1. 创建设计矩阵
        design_matrix = self._create_design_matrix(X)
        
        # 2. 计算回归系数 (使用伪逆避免奇异矩阵问题)
        # 输入：design_matrix (n_samples, n_basis)，y (n_samples,)
        # 处理：(X^T X)^(-1) X^T y
        # 输出：coefficients (n_basis,)
        self.coefficients = np.linalg.pinv(design_matrix.T @ design_matrix) @ design_matrix.T @ y
        
        # 3. 存储特征名称（用于调试和可视化）
        self.feature_names = self._generate_feature_names(X.shape[1])
        
        return self
    
    def evaluate(self, X: np.ndarray) -> np.ndarray:
        """
        评估方法：计算给定输入的特征值
        
        输入参数：
        X: 特征值数组，形状可以是:
           - 1D: (n_features,) 单个样本
           - 2D: (n_samples, n_features) 多个样本
        
        输出：预测值，形状与输入样本数对应
        """
        # 步骤1: 输入验证和维度处理
        # 输入：X (可以是1D或2D数组)
        # 处理：确保X是2D数组 (n_samples, n_features)
        # 输出：X_2d (2D数组)
        if X.ndim == 1:
            X_2d = X.reshape(1, -1)
        else:
            X_2d = X
        
        # 步骤2: 创建设计矩阵
        # 输入：X_2d (n_samples, n_features)
        # 处理：对每个样本应用基函数变换
        # 输出：design_matrix (n_samples, n_basis)
        design_matrix = self._create_design_matrix(X_2d)
        
        # 步骤3: 检查模型是否已训练
        # 输入：self.coefficients
        # 处理：验证系数是否存在
        # 输出：无（如果未训练则抛出异常）
        if self.coefficients is None:
            raise ValueError("模型未训练，请先调用fit方法")
        
        # 步骤4: 计算预测值
        # 输入：design_matrix (n_samples, n_basis), self.coefficients (n_basis,)
        # 处理：矩阵乘法 y_pred = X * beta
        # 输出：predictions (n_samples,)
        predictions = design_matrix @ self.coefficients
        
        # 步骤5: 返回与输入形状匹配的结果
        # 输入：predictions (n_samples,)
        # 处理：如果原始输入是1D，则返回标量
        # 输出：标量或1D数组
        if X.ndim == 1:
            return predictions[0]
        return predictions
    
    def _create_design_matrix(self, X: np.ndarray) -> np.ndarray:
        """
        创建设计矩阵：应用基函数变换
        
        输入参数：
        X: 原始特征矩阵，形状 (n_samples, n_features)
        
        输出：设计矩阵，形状 (n_samples, n_basis)
        """
        n_samples, n_features = X.shape
        basis_columns = []
        
        for basis_func in self.basis_funcs:
            if basis_func == 'poly':
                # 多项式基函数：1, x, x^2, ..., x^d
                for d in range(self.degree + 1):
                    if d == 0:
                        # 常数项
                        basis_columns.append(np.ones(n_samples))
                    else:
                        # 各特征的多项式项
                        for feat_idx in range(n_features):
                            basis_columns.append(X[:, feat_idx] ** d)
            
            elif basis_func == 'interaction':
                # 交互项：x_i * x_j
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        basis_columns.append(X[:, i] * X[:, j])
            
            elif basis_func == 'exp':
                # 指数基函数：exp(-x)
                for feat_idx in range(n_features):
                    basis_columns.append(np.exp(-X[:, feat_idx]))
            
            elif basis_func == 'log':
                # 对数基函数：log(1+x)
                for feat_idx in range(n_features):
                    basis_columns.append(np.log1p(X[:, feat_idx]))
        
        # 将所有基函数列堆叠成设计矩阵
        if basis_columns:
            return np.column_stack(basis_columns)
        else:
            return np.ones((n_samples, 1))  # 只有常数项
    
    def _generate_feature_names(self, n_features: int) -> List[str]:
        """
        生成特征名称（用于调试）
        
        输入参数：
        n_features: 原始特征数量
        
        输出：特征名称列表
        """
        feature_names = []
        
        for basis_func in self.basis_funcs:
            if basis_func == 'poly':
                for d in range(self.degree + 1):
                    if d == 0:
                        feature_names.append("const")
                    else:
                        for feat_idx in range(n_features):
                            feature_names.append(f"x{feat_idx}^{d}")
            
            elif basis_func == 'interaction':
                for i in range(n_features):
                    for j in range(i + 1, n_features):
                        feature_names.append(f"x{i}*x{j}")
            
            elif basis_func == 'exp':
                for feat_idx in range(n_features):
                    feature_names.append(f"exp(-x{feat_idx})")
            
            elif basis_func == 'log':
                for feat_idx in range(n_features):
                    feature_names.append(f"log(1+x{feat_idx})")
        
        return feature_names
    
    def get_coefficients(self) -> Tuple[np.ndarray, List[str]]:
        """
        获取回归系数和对应的特征名称
        
        输出：(系数数组, 特征名称列表)
        """
        if self.coefficients is None:
            raise ValueError("模型未训练")
        return self.coefficients, self.feature_names


# 测试代码
if __name__ == "__main__":
    print("=== 测试1: 单样本评估 ===")
    
    # 创建回归模型
    model = RegressionModel(basis_functions=['poly', 'interaction'], degree=2)
    
    # 生成模拟数据
    np.random.seed(42)
    n_samples = 100
    n_features = 2
    X_train = np.random.randn(n_samples, n_features)
    
    # 创建目标变量（模拟继续持有价值）
    y_train = 2 * X_train[:, 0] + 3 * X_train[:, 1] + 0.5 * X_train[:, 0] * X_train[:, 1] + np.random.randn(n_samples) * 0.1
    
    # 训练模型
    print("训练模型中...")
    model.fit(X_train, y_train)
    
    # 获取系数
    coefficients, feature_names = model.get_coefficients()
    print(f"回归系数数量: {len(coefficients)}")
    print(f"特征名称: {feature_names}")
    print("前5个系数:", coefficients[:5])
    
    # 测试单个样本评估
    test_sample = np.array([1.5, 2.0])
    print(f"\n测试样本: {test_sample}")
    
    # 调用evaluate方法
    prediction = model.evaluate(test_sample)
    print(f"预测值: {prediction:.4f}")
    
    # 验证计算过程
    print("\n验证计算过程:")
    design_matrix = model._create_design_matrix(test_sample.reshape(1, -1))
    print(f"设计矩阵形状: {design_matrix.shape}")
    print(f"手动计算: {np.sum(design_matrix[0] * coefficients):.4f}")
    
    print("\n=== 测试2: 多样本评估 ===")
    
    # 测试多个样本
    test_samples = np.array([[1.0, 1.0], [2.0, 1.5], [0.5, 2.5]])
    print(f"测试样本形状: {test_samples.shape}")
    
    predictions = model.evaluate(test_samples)
    print(f"预测值: {predictions}")
    
    print("\n=== 测试3: 不同基函数测试 ===")
    
    # 测试指数和对数基函数
    model2 = RegressionModel(basis_functions=['exp', 'log'], degree=1)
    
    # 使用正数数据（对数和指数需要正数输入）
    X_train2 = np.abs(np.random.randn(50, 2)) + 0.1
    y_train2 = np.exp(-X_train2[:, 0]) + np.log1p(X_train2[:, 1]) + np.random.randn(50) * 0.05
    
    model2.fit(X_train2, y_train2)
    
    test_sample2 = np.array([0.5, 1.5])
    prediction2 = model2.evaluate(test_sample2)
    print(f"测试样本: {test_sample2}")
    print(f"使用exp/log基函数的预测值: {prediction2:.4f}")
    
    print("\n=== 测试4: 错误处理 ===")
    
    # 测试未训练模型
    untrained_model = RegressionModel()
    try:
        untrained_model.evaluate(test_sample)
    except ValueError as e:
        print(f"预期错误: {e}")
    
    print("\n所有测试完成!")
