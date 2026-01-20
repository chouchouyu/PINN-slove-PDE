正确答案
https://github.com/batuhanguler/Deep-BSDE-Solver/blob/master/BlackScholesBarenblatt100D.py

https://github.com/Alysium/mlmc-undergrad-thesis/blob/main/mmc_runtime_code.py
正确答案

https://github.com/alexander-dybdahl/deep-fbsde/blob/main/README.md
https://github.com/TheOlyle/Deep-NN-FBSDE-solver/blob/main/src/BlackScholesBarenblatt100D.py  可以采用

https://github.com/zhoumeng-creater/FBSNN/blob/master/TensorFlow%202%20%E5%AE%9E%E7%8E%B0%20FBSNN%EF%BC%9A%E9%AB%98%E7%BB%B4%20FBSDE%20%E6%95%B0%E5%80%BC%E5%AE%9E%E9%AA%8C%E5%A4%8D%E7%8E%B0%E6%8A%A5%E5%91%8A.md


https://github.com/xiankangW-gi/FBSNN_torch/blob/main/main.py
最简单的demo
使用Euler-Maruyama方法求解Black-Scholes-Barenblatt(BSB)方程需要结合随机控制的思想，因为BSB方程本质上是Hamilton-Jacobi-Bellman(HJB)方程。以下是详细的实现方法：



https://github.com/rukikotoo/FBSNNs-in-TF2 最好

https://github.com/AJ-04/cqf_raj/tree/main/Raj  最好


https://ar5iv.labs.arxiv.org/html/1804.07010?_immersive_translate_auto_translate=1
基础论文


https://www.bilibili.com/video/BV1MW4y167S2/?spm_id_from=333.337.search-card.all.click&vd_source=93593b3b4cb42a6c21d9929535b5a591
Physics-informed neural networks（PINNs）代码部分讲解，嵌入物理知识神经网络
要看完

https://www.bilibili.com/video/BV1kWJ1zAELv?spm_id_from=333.788.videopod.sections&vd_source=93593b3b4cb42a6c21d9929535b5a591
0-引言-PINN入门详细学习规划【PINN入门30讲系列课程】
要看完

https://www.bilibili.com/video/BV1gJHYeREEj?spm_id_from=333.788.videopod.sections&vd_source=93593b3b4cb42a6c21d9929535b5a591&p=18
【中文配音】科学与工程领域的 AI （2024）- 苏黎世联邦理工学院



https://github.com/rukikotoo/DSGE/blob/main/BGG.pdf


https://github.com/EmoryMLIP/FBSNNs
进一步优化


class SimpleNetwork(nn.Module):
    def __init__(self, input_dim=10, hidden_dim=20, output_dim=1):
        super().__init__()
        # 三个线性层
        self.layer1 = nn.Linear(input_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.layer3 = nn.Linear(hidden_dim, output_dim)
        
        # 激活函数
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # 第一层: 10维 -> 20维
        x = self.layer1(x)  # 执行: x = xW1^T + b1
        x = self.relu(x)    # 非线性激活
        
        # 第二层: 20维 -> 20维
        x = self.layer2(x)  # 执行: x = xW2^T + b2
        x = self.relu(x)
        
        # 第三层: 20维 -> 1维
        x = self.layer3(x)  # 执行: x = xW3^T + b3
        
        return x

# 使用示例
model = SimpleNetwork(input_dim=10, hidden_dim=20, output_dim=1)
print("网络结构:")
print(model)

# 查看参数数量
total_params = sum(p.numel() for p in model.parameters())
print(f"\n总参数数量: {total_params}")

# 详细查看各层参数
for name, param in model.named_parameters():
    print(f"{name}: 形状={param.shape}, 参数量={param.numel()}")

----
layer1.weight: 形状=(20, 10), 参数量=200
layer1.bias: 形状=(20,), 参数量=20
layer2.weight: 形状=(20, 20), 参数量=400
layer2.bias: 形状=(20,), 参数量=20
layer3.weight: 形状=(1, 20), 参数量=20
layer3.bias: 形状=(1,), 参数量=1
总参数量: 661





https://github.com/Alysium/mlmc-undergrad-thesis/blob/446a0cd7856681287d4384d5647c4cb617ac0f1a/mmc_raissi_nn_model.py#L57-L64

https://github.com/Aadhithya-06/Final-Year-Project/blob/745261643bbc89c30d832068042e2753c7efd637/Pytorch/Models.py#L24-L31


https://www.research-collection.ethz.ch/entities/publication/8f47707f-a58a-4ff4-86e7-cbc3560b9779


PINN 直接求解 PDE 的強形式（Strong Form），依賴於空間採樣。
FBSNN 則是利用機率論，通過模擬隨機路徑（SDE）來求解，在處理超高維度且具有特定擴散結構的問題時，FBSNN 有時比純 PINN 更穩定。


----https://zhuanlan.zhihu.com/p/35938271

多层次蒙特卡洛(multilevel mc) · 自适应多层次蒙特卡洛(adaptive MLMC)
file:///Users/susan/Downloads/OPRE_2008.pdf


https://www.zhihu.com/question/266175980/answer/389439392
在使用蒙特卡洛模拟进行回望期权定价时，如何消除bias?


https://ins.sjtu.edu.cn/people/shijin/PS/HuJinLiZhang.pdf


https://www.bilibili.com/video/BV1n54y157rS/?spm_id_from=333.337.search-card.all.click&vd_source=93593b3b4cb42a6c21d9929535b5a591
ROS 2D导航原理系列（二）|自适应蒙特卡罗定位AMCL


https://www.bilibili.com/video/BV1DS4y177G8/?spm_id_from=333.337.search-card.all.click&vd_source=93593b3b4cb42a6c21d9929535b5a591
【搬运】美式期权定价-最小二乘蒙特卡洛法
https://www.bilibili.com/video/BV1nCWczjEMt/?spm_id_from=333.337.search-card.all.click&vd_source=93593b3b4cb42a6c21d9929535b5a591
【学术】多标的资产期权蒙特卡洛定价分析
https://www.youtube.com/watch?v=E9VfVd6-OjA&t=702s
Plamen Turkedjiev：用于近似 BSDES 和半线性偏微分方程的最小二乘回归蒙特卡洛

https://www.bilibili.com/video/BV1PVnfz3EPb?spm_id_from=333.788.player.switch&vd_source=93593b3b4cb42a6c21d9929535b5a591
20-【补充代码】-PINN构造--完整总损失函数【PINN入门30讲系列课程】


研究了广义自回归条件异方差(GARCH)模型下方差衍生产品的加速模拟定价理论.基于Black-Scholes模型下的产品价格解析解以及对两类标的过程的矩分析,提出了一种GARCH模型下高效控制变量加速技术,并给出最优控制变量的选取方法.数值计算结果表明,提出的控制变量加速模拟方法可以有效地减小Monte Carlo模拟误差,提高计算效率.该算法可以方便地解决GARCH随机波动率模型下其他复杂产品的计算问题,如亚式期权、篮子期权、上封顶方差互换、Corridor方差互换以及Gamma方差互换等计算问题.

https://www.zhihu.com/question/357517051
最小二乘蒙特卡洛模拟对期权定价原理是什么？


。在对奇异期权定价时，

通常是多个随机变量的的联合概率密度函数。例如美式期权（American option）和障碍期权（barrier option）定价要考虑停时（stopping time）问题；亚式期权（Asian option）和回望期权（lookback option）则要考虑其路径依赖（path-dependence）《金融随机分析2》第七、第八章）。
from https://zhuanlan.zhihu.com/p/22754445



 


"这两个代码 有什么区别？为什么要有这些区..."点击查看元宝的回答
https://yb.tencent.com/s/CkKYzx1vrClk



https://docs.sciml.ai/HighDimPDE/dev/examples/blackscholes/
求解100维布莱克-斯科尔斯-巴伦布拉特方程

s13662-019-2135-z.pdf

"BlackScholesBarenbla..."点击查看元宝的回答
https://yb.tencent.com/s/GpGNuzkbYcM3

"把这段 d = 100 # number..."点击查看元宝的回答
https://yb.tencent.com/s/7UVeeePI9ew8


"写出具体的数学公式，我会使用标准的随机微..."点击查看元宝的回答
https://yb.tencent.com/s/Nh3vakdSl5i3

"BlackScholesBarenbla..."点击查看元宝的回答
https://yb.tencent.com/s/FOfiFJDUkllN

"@testset "DeepBSDE_H..."点击查看元宝的回答
https://yb.tencent.com/s/3gjUHdpLONWQ
必看

"前面自适应性蒙特卡洛怎么用在美式期权上？"点击查看元宝的回答
https://yb.tencent.com/s/BeDZB6MqIWNe

https://mp.weixin.qq.com/s/bKc3bdP1BR6yrIwIZeDkyQ
美式期权(一)：最小二乘蒙特卡洛模拟定价(LSM)


https://github.com/ananyo49/Least-Square-Monte-Carlo-American-Options-?tab=readme-ov-file
Least-Square-Monte-Carlo-America-Options-
