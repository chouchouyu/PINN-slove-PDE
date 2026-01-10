import sys, os
import numpy as np

# 添加原始实现的路径
sys.path.append(os.path.dirname(os.path.abspath(__file__))+"/hdp-master 2/src")

# 导入原始实现的GBM
from blackscholes.utils.GBM import GBM

# 导入用户实现的Paths_generater
import cqf_mc_American

# 设置相同的参数
np.random.seed(555)
T = 1
days = 300
asset_num = 2
init_price_vec = 100*np.ones(asset_num)
vol_vec = 0.2*np.ones(asset_num)
ir = 0.05
dividend_vec = 0.1*np.ones(asset_num)
corr_mat = np.eye(asset_num)
corr_mat[0, 1] = 0.3
corr_mat[1, 0] = 0.3
n_simulations = 2  # 少量模拟路径便于比较

print("=== 比较路径生成结果 ===")

# 原始实现的路径生成
print("\n1. 原始实现 (GBM.simulateV2):")
original_gbm = GBM(T, days, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
original_paths = original_gbm.simulateV2(n_simulations)
print(f"原始实现路径形状: {original_paths.shape}")
print(f"第1个路径的初始价格: {original_paths[0, :, 0]}")
print(f"第1个路径的中间价格 (t=100): {original_paths[0, :, 100]}")
print(f"第1个路径的到期价格 (t=300): {original_paths[0, :, -1]}")

# 用户实现的路径生成
print("\n2. 用户实现 (Paths_generater.gbm):")
user_gbm = cqf_mc_American.Paths_generater(T, days, init_price_vec, ir, vol_vec, dividend_vec, corr_mat)
user_paths = user_gbm.gbm(n_simulations)
print(f"用户实现路径形状: {user_paths.shape}")
print(f"第1个路径的初始价格: {user_paths[0, :, 0]}")
print(f"第1个路径的中间价格 (t=100): {user_paths[0, :, 100]}")
print(f"第1个路径的到期价格 (t=300): {user_paths[0, :, -1]}")

# 详细比较
print("\n3. 详细比较:")
print(f"\n路径形状差异: {original_paths.shape} vs {user_paths.shape}")

# 比较路径的每个元素
print(f"\n路径元素差异 (第1个路径，t=0到t=5):")
for t in range(0, 6):
    original_price = original_paths[0, :, t]
    user_price = user_paths[0, :, t]
    diff = original_price - user_price
    print(f"t={t}: 原始={original_price}, 用户={user_price}, 差异={diff}")

# 检查参数是否一致
print(f"\n4. 参数检查:")
print(f"原始实现 dt: {original_gbm.dt}, 用户实现 dt: {user_gbm.dt}")
print(f"原始实现 drift: {original_gbm.ir - original_gbm.dividend_vec}")
print(f"用户实现 drift: {user_gbm.r - user_gbm.dividend_vec}")
print(f"原始实现 vol: {original_gbm.vol_vec}")
print(f"用户实现 vol: {user_gbm.vol_vec}")
print(f"原始实现 corr_mat:")
print(original_gbm.corr_mat)
print(f"用户实现 corr_mat:")
print(user_gbm.corr_mat)

# 检查随机数生成
print(f"\n5. 随机数生成检查:")
# 重置随机种子
np.random.seed(555)
# 原始实现的随机数生成
L_original = np.linalg.cholesky(corr_mat)
dW_original = L_original.dot(np.random.normal(size=asset_num))*np.sqrt(original_gbm.dt)
print(f"原始实现的dW: {dW_original}")

# 重置随机种子
np.random.seed(555)
# 用户实现的随机数生成
L_user = np.linalg.cholesky(corr_mat)
dW_user = L_user.dot(np.random.normal(size=asset_num))*np.sqrt(user_gbm.dt)
print(f"用户实现的dW: {dW_user}")

# 检查价格更新公式
print(f"\n6. 价格更新公式检查:")
t = 1
# 原始实现的价格更新
original_drift = original_gbm.ir - original_gbm.dividend_vec
original_vol = original_gbm.vol_vec
original_prev_price = original_paths[0, :, t-1]
original_dW = L_original.dot(np.random.normal(size=asset_num))*np.sqrt(original_gbm.dt)
original_rand_term = np.multiply(original_vol, original_dW)
original_new_price = np.multiply(original_prev_price, np.exp((original_drift-original_vol**2/2)*original_gbm.dt + original_rand_term))
print(f"原始实现价格更新: t={t}, 前价格={original_prev_price}, 新价格={original_new_price}, 实际路径价格={original_paths[0, :, t]}")

# 用户实现的价格更新
user_drift = user_gbm.r - user_gbm.dividend_vec
user_vol = user_gbm.vol_vec
user_prev_price = user_paths[0, :, t-1]
user_dW = L_user.dot(np.random.normal(size=asset_num))*np.sqrt(user_gbm.dt)
user_rand_term = np.multiply(user_vol, user_dW)
user_new_price = np.multiply(user_prev_price, np.exp((user_drift-user_vol**2/2)*user_gbm.dt + user_rand_term))
print(f"用户实现价格更新: t={t}, 前价格={user_prev_price}, 新价格={user_new_price}, 实际路径价格={user_paths[0, :, t]}")