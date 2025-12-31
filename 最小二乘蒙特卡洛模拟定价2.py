# https://zhuanlan.zhihu.com/p/665538917

import numpy as np
import matplotlib.pyplot as plt
import numba
import pandas as pd
import scipy.stats as stats
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #正常显示负号

# S_path = pd.DataFrame(data={"Path":np.arange(1,9),"t=0":np.ones(8),\
#                      "t=1":np.array([1.09,1.16,1.22,0.93,1.11,0.76,0.92,0.88]),\
#                      "t=2":np.array([1.08,1.26,1.07,0.97,1.56,0.77,0.84,1.22]),\
#                      "t=3":np.array([1.34,1.54,1.03,0.92,1.52,0.90,1.01,1.34])})
# S_path.set_index("Path",inplace =True)
# S_path


# # 可以计算出相应的内在价值。
# exercise_value = np.maximum(1.1-S_path,0)
# exercise_value

# itm_index = (exercise_value["t=2"]  > 0)
# X = S_path.loc[itm_index,"t=2"]#回归的X为t=2的股票价格，需要注意的是只选取实值路径
# Y = exercise_value.loc[itm_index,"t=3"]*np.exp(-0.06)#回归的Y为t=3的现金流量折现，需要注意的是只选取实值路径
# reg_data = pd.DataFrame({"X=股票价格":X,"Y=未来现金流折现":Y})  #这里仅仅是方便列示
# print(reg_data)

# reg = np.polyfit(x = X,y = Y,deg = 2)  #XY做2次最小二乘法
# print(f"回归方程\n{reg[0]:.3f}X^2+{reg[1]:.3f}X {reg[2]:.3f}")# 回归的系数
# reg_data["holding_value"] = np.polyval(reg,S_path.loc[itm_index,"t=2"])  #求出期望的继续持有价值
# reg_data


# reg_data["exercise_value"] = exercise_value.loc[itm_index,"t=2"]
# reg_data["early_exercise"]= np.where(reg_data["holding_value"]<reg_data["exercise_value"],1,0)
# reg_data


# 先定义一下上一期可以通用的模拟路径
def geo_brownian(steps,paths,T,S0,b,sigma):  
    #用来生成模拟的路径
    dt = T / steps # 求出时间间隔dt
    S_path = np.zeros((steps+1,paths))   #创建一个矩阵，用来准备储存模拟情况
    S_path[0] = S0  #起点设置
    #rn = np.random.standard_normal(S_path.shape) # 也可以一次性创建出需要的正态分布随机数，当然也可以写在循环里每次创建一个时刻的随机数
    for step in range(1,steps+1):
        rn = np.random.standard_normal(paths) #创造随机数
        S_path[step] = S_path[step - 1] * np.exp((b-0.5*sigma**2)*dt +sigma*np.sqrt(dt)*rn) #几何布朗运动的解
    return S_path
def LSM(steps,paths,CP,S0,X,sigma,T,r,b):
    #代码也可以多写几行计算出所有的提前行权节点，这里为了逻辑清晰就没有列出
    S_path = geo_brownian(steps,paths,T,S0,b,sigma) # 价格生成路径
    dt = T / steps
    cash_flow = np.zeros_like(S_path)  #实现创建好现金流量的矩阵，后续使用
    df = np.exp(-r * dt) #每一期的折现因子
    if CP == "C":
        cash_flow[-1] = np.maximum(S_path[-1] - X,0)   #先确定最后一期的价值，就是实值额
        exercise_value = np.maximum(S_path - X,0)
    else:
        cash_flow[-1] = np.maximum(X - S_path[-1] ,0)   #先确定最后一期的价值，就是实值额
        exercise_value = np.maximum(X - S_path,0)

    for t in range(steps-1,0,-1):  #M-1为倒数第二个时点，从该时点循环至1时点
        df_cash_flow = cash_flow[t + 1] * df
        S_price = S_path[t]  #标的股价，回归用的X
        itm_index = (exercise_value[t] > 0)  #确定实值的index，后面回归要用，通过index的方式可以不破坏价格和现金流矩阵的大小
        reg = np.polyfit(S_price[itm_index], df_cash_flow[itm_index],2) # 实值路径下的标的股价X和下一期的折现现金流Y回归
        holding_value = exercise_value[t].copy()  # 创建一个同长度的向量，为了保持index一致，当然也可以用np.zeros_like等方式，本质一样
        holding_value[itm_index] = np.polyval(reg, S_price[itm_index])  # 回归出 holding_value，其他的值虽然等于exercise_value，但是后续判断会去除
        ex_index = itm_index & (exercise_value[t] > holding_value)  #在实值路径上，进一步寻找出提前行权的index

        df_cash_flow[ex_index] = exercise_value[t][ex_index]  # 将cash_flow中提前行权的替换为行权价值，其他保持下一期折现不变
        cash_flow[t] = df_cash_flow       
    value = cash_flow[1].mean() * df
    return value
# P=LSM(steps = 1000,paths = 50000,CP = "P",S0 = 40,X = 40,sigma = 0.2,T = 1,r = 0.06,b=0.06)
P=LSM(steps = 1000,paths = 50000,CP = "P",S0 = 1,X = 1.1,sigma = 0.2,T = 0.3333,r = 0.03,b=0.06) #0.10443593279494927
    # p=LSM(S0=1,K=1.1,vol=0.2,r=0.03,q=0,t=0.3333,n=100000,call_or_put=0)

print(P) # 2.2881226971718096
# 2.3063932226893673