import numpy as np

# 创建一个形状为[5, 8]的零数组，以便支持索引4
cash_flow = np.zeros([5, 8])

# 生成一个形状为1行8列的随机数数组
random_array = np.random.rand(1, 8)
print(f"生成的随机数数组：\n{random_array}")

# 直接使用索引列表
selected_values1 = random_array[0, [1, 4]]
print(f"直接使用索引列表取出的数据：\n{selected_values1}")

# 将索引存储在变量A中
A = [1, 4]
selected_values2 = random_array[0, A]
print(f"使用变量A作为索引取出的数据：\n{selected_values2}")

# 验证两种方法结果相同
print(f"两种方法结果是否相同：{np.array_equal(selected_values1, selected_values2)}")

# 将这些数据赋值给cash_flow[:, -2]的索引1和4的位置
cash_flow[A, -2] = selected_values2

print(f"最终的cash_flow数组：\n{cash_flow}")
