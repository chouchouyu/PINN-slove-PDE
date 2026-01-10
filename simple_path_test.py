import sys, os
import numpy as np

# 添加原始实现的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/hdp-master 2/src")

# 导入原始实现的GBM
from blackscholes.utils.GBM import GBM

# 设置相同的参数
np.random.seed(555)
T = 1
days = 5
asset_num = 2
init_price_vec = 100*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.05
dividend_vec = 0.1*np.ones(asset_num)
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3
n_simulations = 1

print("=== 简单路径测试 ===")

# 原始实现的路径生成
print("\n1. 原始实现 (GBM.simulateV2):")
original_gbm = GBM(T, days, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)

# 手动执行原始实现的模拟逻辑
original_paths = []
original_drift_vec = original_gbm.ir - original_gbm.dividend_vec
original_L = np.linalg.cholesky(original_gbm.corr_mat)
for _ in range(n_simulations):
    sim = np.zeros([original_gbm.asset_num, original_gbm.N+1])
    sim[:, 0] = original_gbm.init_price_vec
    print(f"  路径 {_+1}:")
    for i in range(1, original_gbm.N+1):
        original_dW = original_L.dot(np.random.normal(size=original_gbm.asset_num))*np.sqrt(original_gbm.dt)
        original_rand_term = np.multiply(original_gbm.vol_vec, original_dW)
        sim[:, i] = np.multiply(sim[:, i-1], np.exp((original_drift_vec-original_gbm.vol_vec**2/2)*original_gbm.dt + original_rand_term))
        print(f"    t={i}: dW={original_dW}, 价格={sim[:, i]}")
    original_paths.append(sim)
original_paths = np.array(original_paths)

# 用户实现的路径生成
print("\n2. 用户实现 (手动执行):")

# 手动执行用户实现的模拟逻辑
user_paths = []
user_dt = T / days
user_drift_vec = ir - dividend_vec
user_L = np.linalg.cholesky(corr_mat)

# 重置随机种子
np.random.seed(555)

for _ in range(n_simulations):
    sim = np.zeros([asset_num, days + 1])
    sim[:, 0] = init_price_vec
    print(f"  路径 {_+1}:")
    for t in range(1, days + 1):
        user_dW = user_L.dot(np.random.normal(size=asset_num))*np.sqrt(user_dt)
        user_rand_term = np.multiply(vol_vec, user_dW)
        sim[:, t] = np.multiply(sim[:, t-1], np.exp((user_drift_vec-vol_vec**2/2)*user_dt + user_rand_term))
        print(f"    t={t}: dW={user_dW}, 价格={sim[:, t]}")
    user_paths.append(sim)
user_paths = np.array(user_paths)