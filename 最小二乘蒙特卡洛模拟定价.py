# https://mp.weixin.qq.com/s/bKc3bdP1BR6yrIwIZeDkyQ

import numpy as np
import pandas as pd
from scipy.stats import norm


# 定义价格路径函数：

#         此处生成的价格路径矩阵和前2篇文章相同，是一个n行t列的矩阵，在后面的计算中为了方便，转置为了t行n列

def price_path(S0, vol, r, q, t,n): 
    dt = 1 / 252
    rt = r - q - 0.5 * vol ** 2
    days=round(t*252)
    S = np.zeros((n, days))
    S[:, 0] = S0
    for i in range(1, days):
        rand = np.random.normal(size=n)
        S[:, i] = S[:, i-1] * np.exp(rt * dt + vol * np.sqrt(dt) * rand)
    return S


# 定义LSM美式期权定价函数：
 

def LSM(S0,K,vol,r,q,t,n,call_or_put):
    
    """
    S0：标的初始价格
    K：行权价格
    vol：波动率
    r：无风险利率
    q：股息率
    t：期限，年化表示，一年默认为252个交易日，按照t*252四舍五入换算为天数
    n：蒙特卡洛模拟次数
    call_or_put：call=1，put=-1
    """
    
    price = price_path(S0, vol, r, q, t,n).T #此处转置，得到的是一个t行n列的价格路径矩阵
    dt = 1 / 252 #定义时间间隔，此处设置为以1个交易日为时间单位，一年252个交易日，可以修改
    df = np.exp(-r * dt) #每一期的折现因子
    days=round(t*252) #输入的t是年化期限，转化为交易日天数
    cash_flow = np.zeros_like(price)  #初始化现金流矩阵
    
    if call_or_put == 1:
        cash_flow[-1] = np.maximum(price[-1] - K,0)   #最后一天的现金流就是当天的内在价值
        exercise_value = np.maximum(price - K,0) #创建内在价值矩阵
    else:
        cash_flow[-1] = np.maximum(K - price[-1] ,0)   #最后一天的现金流就是当天的内在价值
        exercise_value = np.maximum(K - price,0) #创建内在价值矩阵

    for i in range(days-2,0,-1):  #从倒数第2天循环至第1天
        
        discounted_cashflow = cash_flow[i + 1] * df #将下一期的现金流贴现至当前，作为因变量
        S_price = price[i]  #当前时点的标的资产价格，作为自变量
        ITM_index = (exercise_value[i] > 0)  #筛选出实值的点的index
        
        reg = np.polyfit(S_price[ITM_index], discounted_cashflow[ITM_index],2) # 对实值的点，进行回归，reg储存的是二项式回归的三个系数
        continuation_value = exercise_value[i].copy()  # 创建一个相同长度的向量，储存存续价值
        continuation_value[ITM_index] = np.polyval(reg, S_price[ITM_index])  # 对实值的点，回归出存续价值。其他的值虽然等于exercise_value，但实际没有用到
        exercise_index = ITM_index & (exercise_value[i] > continuation_value)  #判断是否行权，并在实值的点的基础上，进一步筛选出行权的index
        discounted_cashflow[exercise_index] = exercise_value[i][exercise_index]  # 将现金流向量中行权的点替换为行权价值，其他保持下一期折现不变
        cash_flow[i] = discounted_cashflow       
   
    value = cash_flow[1].mean() * df  
    
    return value

if __name__ == "__main__":

    p=LSM(S0=1,K=1.1,vol=0.2,r=0.03,q=0,t=0.3333,n=100000,call_or_put=0) #0.10824100246493358
    print(p)