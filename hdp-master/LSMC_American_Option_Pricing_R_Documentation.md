# LSMC美式期权定价R代码文档

## 1. 概述

本文档详细解释了 `/Users/susan/Downloads/hdp-master/LSMC American Option Pricing.R` 文件中的代码实现，该文件使用最小二乘蒙特卡洛(Least Squares Monte Carlo, LSMC)方法对美式期权进行定价。

### 1.1 核心功能
- 股票路径生成（含对偶变量技术）
- 三种基函数实现（拉盖尔多项式、埃尔米特多项式、单项式）
- 美式看跌期权定价
- 远期开始美式期权定价
- 测试矩阵生成
- 快慢两种实现方式比较

### 1.2 技术要点
- **LSMC方法**：结合蒙特卡洛模拟和最小二乘回归估计延续价值
- **对偶变量技术**：减少蒙特卡洛模拟方差
- **反向递归**：从到期日向前计算最优执行策略
- **基函数选择**：不同多项式函数对定价精度的影响

## 2. 股票路径生成函数

### 2.1 stock_path_generator函数

```R
#stock path generator function

stock_path_generator=function(s0,r,sig,paths,timesteps,T){
  
  stock_paths=matrix(nrow=paths,ncol=timesteps)
  delta=T/timesteps
  for(i1 in 1:(nrow(stock_paths)/2)){
    i=2*i1-1
    z_pos=rnorm(timesteps)
    z_neg=-z_pos #antithetic 
    stock_paths[i,] = s0*exp((r-0.5*sig^2)*T + sig*sqrt(delta)*cumsum(z_pos)) #simulating ith path
    stock_paths[(i+1),] = s0*exp((r-0.5*sig^2)*T + sig*sqrt(delta)*cumsum(z_neg)) #simulating (i+1)th path using antithetic values
  }
  
  stock_paths=cbind(s0,stock_paths) #making the first row as the stock price
}
```

#### 逐行解释

1.  **第3行**：定义函数`stock_path_generator`，参数包括：
    - `s0`：初始股票价格
    - `r`：无风险利率
    - `sig`：波动率
    - `paths`：模拟路径数量
    - `timesteps`：每个路径的时间步数
    - `T`：到期时间（年）

2.  **第5行**：创建`stock_paths`矩阵，行数为路径数，列数为时间步数，用于存储模拟的股票价格路径。

3.  **第6行**：计算时间步长`delta`，即每两个时间点之间的时间间隔。

4.  **第7行**：开始循环，由于使用对偶变量技术，每次循环生成两条路径，因此循环次数为路径数的一半。

5.  **第8行**：计算当前路径索引`i`，确保生成的是奇数索引路径。

6.  **第9行**：生成`timesteps`个标准正态分布随机数`z_pos`，用于模拟股票价格的随机波动。

7.  **第10行**：生成`z_pos`的相反数`z_neg`，这是**对偶变量技术**的核心，用于减少模拟方差。

8.  **第11-12行**：使用几何布朗运动(GBM)模型模拟股票价格路径：
    - 公式：`S_t = S_0 * exp( (r - 0.5*σ²)*T + σ*√(Δt)*cumsum(z) )`
    - 其中`cumsum(z_pos)`和`cumsum(z_neg)`分别是正随机数和负随机数的累积和
    - 这两行分别生成第`i`条和第`i+1`条路径

9.  **第15行**：使用`cbind`函数在矩阵左侧添加初始股票价格`s0`，这样每条路径的第一个元素就是初始价格。

10. **第16行**：函数结束，返回生成的股票价格路径矩阵。

### 2.2 对偶变量技术说明

对偶变量技术是一种方差减少技术，通过生成正负对称的随机数对来降低蒙特卡洛模拟的方差。这种方法的原理是：如果一条路径的股票价格高于预期，另一条路径的股票价格会低于预期，两者的平均值会更接近真实值，从而减少整体方差。

## 3. 基函数实现

基函数用于LSMC方法中的回归分析，用于估计期权的延续价值。代码实现了三种基函数：拉盖尔多项式(Laguerre)、埃尔米特多项式(Hermite)和单项式(Monomials)。

### 3.1 拉盖尔多项式：laguerre_upto_k函数

```R
#laguerre definition
laguerre_upto_k=function(x,k){
  f1=exp(-x/2)
  f2=f1*(1-x)
  f3=f1*(1 - 2*x + (0.5*x^2))
  f4=f1*(1 - 3*x + (1.5*x^2) - (x^3)/6)
  
  if(k==2){
    cbind(f1,f2)
  } else if(k==3){
    cbind(f1,f2,f3)
  } else if(k==4){
    cbind(f1,f2,f3,f4)
  }
}
```

#### 逐行解释

1.  **第19行**：定义函数`laguerre_upto_k`，参数：
    - `x`：输入变量（通常是股票价格）
    - `k`：多项式阶数（2-4）

2.  **第20-23行**：计算拉盖尔多项式的前4阶：
    - `f1 = exp(-x/2)`：0阶拉盖尔多项式（权重函数）
    - `f2 = exp(-x/2)*(1-x)`：1阶拉盖尔多项式
    - `f3 = exp(-x/2)*(1 - 2x + 0.5x²)`：2阶拉盖尔多项式
    - `f4 = exp(-x/2)*(1 - 3x + 1.5x² - x³/6)`：3阶拉盖尔多项式

3.  **第25-31行**：根据参数`k`返回相应阶数的多项式矩阵：
    - `k=2`：返回0-1阶多项式
    - `k=3`：返回0-2阶多项式
    - `k=4`：返回0-3阶多项式

### 3.2 埃尔米特多项式：hermite_upto_k函数

```R
hermite_upto_k=function(x,k){
  f1=1
  f2=2*x
  f3=4*(x^2) - 2
  f4=8*(x^3) - 12*x
  
  if(k==2){
    cbind(f1,f2)
  } else if(k==3){
    cbind(f1,f2,f3)
  } else if(k==4){
    cbind(f1,f2,f3,f4)
  }
}
```

#### 逐行解释

1.  **第34行**：定义函数`hermite_upto_k`，参数与拉盖尔多项式函数相同。

2.  **第35-38行**：计算埃尔米特多项式的前4阶（物理学家埃尔米特多项式）：
    - `f1 = 1`：0阶埃尔米特多项式
    - `f2 = 2x`：1阶埃尔米特多项式
    - `f3 = 4x² - 2`：2阶埃尔米特多项式
    - `f4 = 8x³ - 12x`：3阶埃尔米特多项式

3.  **第40-46行**：根据参数`k`返回相应阶数的多项式矩阵。

### 3.3 单项式：monomials_upto_k函数

```R
monomials_upto_k=function(x,k){
  f1=1
  f2=x
  f3=x^2
  f4=x^3
  
  if(k==2){
    cbind(f1,f2)
  } else if(k==3){
    cbind(f1,f2,f3)
  } else if(k==4){
    cbind(f1,f2,f3,f4)
  }
}
```

#### 逐行解释

1.  **第49行**：定义函数`monomials_upto_k`，参数与前两个函数相同。

2.  **第50-53行**：计算单项式的前4阶：
    - `f1 = 1`：常数项（0次幂）
    - `f2 = x`：一次项（1次幂）
    - `f3 = x²`：二次项（2次幂）
    - `f4 = x³`：三次项（3次幂）

3.  **第55-61行**：根据参数`k`返回相应阶数的多项式矩阵。

### 3.4 基函数比较

| 基函数类型 | 特点 | 适用场景 |
|-----------|------|----------|
| 拉盖尔多项式 | 指数衰减权重，适合非负变量 | 股票价格等非负金融变量 |
| 埃尔米特多项式 | 对称性质，适合正态分布变量 | 对数收益率等对称分布变量 |
| 单项式 | 简单直观，计算效率高 | 快速计算或初步分析 |

## 4. 美式看跌期权定价核心函数

### 4.1 american_put_price_lsmc函数

```R
#1
#using lecture notes method
american_put_price_lsmc=function(s0,r,sig,paths,timesteps,T,func,strike,k){
  stock=stock_path_generator(s0,r,sig,paths,timesteps,T)
  
  #defining index matrix
  index=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  
  #defining Y matrix (payoff)
  Y=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  exercisevalues=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  exercisevalues=pmax((strike-stock),0)
  Y[,ncol(Y)]=exercisevalues[,ncol(stock)]
  
  index[which(Y[,ncol(Y)]>0),ncol(index)]=1
  #View(cbind(Y[,ncol(Y)],index[,ncol(index)]))
  
  delta=T/timesteps
  pb=txtProgressBar(min=0,max=(ncol(Y)-2),style=3)
  for(j in 1:(ncol(Y)-2)){

    #start from the end
    j_reverse=ncol(Y)-j
    
    #Y = ind1*exp(-rd)*ev1 + ind2*exp(-2rd)*ev2 + . . .
    Y[,j_reverse]=rowSums(as.matrix(index[,((j_reverse+1):ncol(index))]*exp(-r*(((j_reverse+1):ncol(index))-j_reverse)
                                                          *delta)*exercisevalues[,((j_reverse+1):ncol(index))]))
    
    #find all positions in the current column for which Y=0 (we might try exercising here, hoping to get something>0)
    exercise_positions=which(Y[,j_reverse]==0)
    
    #convert all index values to zero (for now)
    index[exercise_positions,j_reverse]=1
    
    #make all Ys equal to corresponding exercise values
    Y[exercise_positions,j_reverse]=exercisevalues[exercise_positions,j_reverse]
    
    #even after exercising, some of the values in the current column will remain zero if option is out of the money
    exercise_positions=which(Y[,j_reverse]==0)
    
    #we won't exercise here and change those index values back to 0 for which even after exercising we get 0
    index[exercise_positions,j_reverse]=0
    
    #intuitive but slower for loop
    
    
    #now take all the values for which option is in the money
    nodes=which(Y[,j_reverse]>0)
    ys=Y[nodes,j_reverse]
    
    #and corresponding stock price
    xs=stock[nodes,j_reverse]
    
    #choose the appropriate L function
    if(func=="L"){
      L_xs=laguerre_upto_k(xs,k)
    }else if(func=="H"){
      L_xs=hermite_upto_k(xs,k)
    }else if(func=="M"){
      L_xs=monomials_upto_k(xs,k)
    }
    
    A=matrix(nrow=k,ncol=k)
    A=t(L_xs)%*%(L_xs) #sum of all combinations of L function (summation Li(x)Lj(x))
     
    b=matrix(nrow=k,ncol=1)
    b=t(t(as.matrix(ys))%*%L_xs)
    
    a=solve(A,b,tol = 1e-300)
    ecv=rep(0,length=nrow(Y))
    ecv[nodes]=L_xs%*%a #get the continuation values
     
    ev=rep(0,length=nrow(Y))
    ev[nodes]=exercisevalues[nodes,j_reverse]
    
    index[which(ev>ecv),j_reverse]=1 #wherever exercise value > continuation value, make that index 1
    
    index[which(index[,j_reverse]==1),(j_reverse+1):ncol(index)]=0 #making all index values zero after 1 is seen
    
    #for loop to do the same operation as line above (slower)
    #for(i in 1:nrow(index)){
    # index[i,(which(index[i,]==1)[-1])]=0
    #}
    
    #put the maximum of exercise value and the value that was there before (discounted payoff from future)
    Y[,j_reverse]=pmax(ev,Y[,j_reverse])
    #View(cbind(ev,ecv,index[,j_reverse]))
    
    setTxtProgressBar(pb,j)
  }
  close(pb)
  
  
  #multiply index by exercise matrix to get only the exercise values where it is optimal
  #and discount by appropriate multiple of delta (col 1 will have 0 as multiple and so on)
  payoff_each_path=rowSums(exp(-r*seq(0,(ncol(Y)-1))*delta)*(index*exercisevalues))
  
  #take payoff by monte carlo

  p=mean(payoff_each_path)
  p
}
```

#### 函数参数
- `s0`：初始股票价格
- `r`：无风险利率
- `sig`：波动率
- `paths`：模拟路径数量
- `timesteps`：时间步数
- `T`：到期时间（年）
- `func`：基函数类型（"L"=拉盖尔，"H"=埃尔米特，"M"=单项式）
- `strike`：执行价格
- `k`：基函数阶数

#### 逐行解释

1.  **第69行**：定义函数`american_put_price_lsmc`，使用LSMC方法为美式看跌期权定价。

2.  **第70行**：调用`stock_path_generator`函数生成股票价格路径。

3.  **第73行**：创建`index`矩阵，用于标记每条路径的最优执行时间（1表示执行，0表示不执行）。

4.  **第76-79行**：
    - 创建`Y`矩阵存储期权价值
    - 创建`exercisevalues`矩阵存储各时间点的执行价值（`pmax((strike-stock),0)`确保价值非负）
    - 在到期日（最后一列），期权价值等于执行价值

5.  **第81行**：在到期日，将所有执行价值大于0的路径标记为执行（`index`矩阵最后一列设为1）。

6.  **第84-85行**：
    - 计算时间步长`delta`
    - 创建进度条`pb`，用于显示计算进度

7.  **第86-156行**：反向循环（从倒数第二列开始向前计算）
    - **第89行**：计算当前处理的时间点`j_reverse`
    - **第92-93行**：计算路径在当前时间点的延续价值，即未来最优执行收益的现值
    - **第96-108行**：处理当前时间点可能执行的路径：
      - 找到延续价值为0的路径（`Y[,j_reverse]==0`）
      - 暂时标记这些路径为执行（`index[exercise_positions,j_reverse]=1`）
      - 更新这些路径的期权价值为当前执行价值
      - 对于执行后价值仍为0的路径，重新标记为不执行
    - **第114-127行**：
      - 找到当前时间点有正价值的路径（`nodes`）
      - 根据`func`参数选择相应的基函数
    - **第129-135行**：
      - 构建回归矩阵`A`和向量`b`
      - 求解回归系数`a`（使用`solve`函数，设置`tol=1e-300`避免数值问题）
    - **第136-140行**：
      - 计算延续价值`ecv`（estimated continuation value）
      - 提取当前时间点的执行价值`ev`
    - **第142行**：比较执行价值和延续价值，将执行价值大于延续价值的路径标记为执行
    - **第144行**：对于标记为执行的路径，将其后续时间点的标记设为0（因为期权一旦执行就结束）
    - **第152行**：更新当前时间点的期权价值为执行价值和延续价值的最大值
    - **第155行**：更新进度条

8.  **第157行**：关闭进度条

9.  **第162行**：计算每条路径的贴现收益：
    - `exp(-r*seq(0,(ncol(Y)-1))*delta)`生成各时间点的贴现因子
    - `index*exercisevalues`得到每条路径在最优执行时间的收益
    - `rowSums`计算每条路径的总收益

10. **第166行**：计算所有路径收益的平均值，即期权的蒙特卡洛估计值

11. **第167行**：返回期权价格

## 5. 测试矩阵

代码中生成了多个测试矩阵，用于比较不同基函数和不同阶数对定价结果的影响。

### 5.1 拉盖尔多项式测试矩阵

```R
#a
laguerrek2=matrix(c(
  american_put_price_lsmc(s0=36,r=0.06,sig=0.2,T=0.5,func="L",strike=40,k=2,paths=100000,timesteps = 126)
  ,american_put_price_lsmc(s0=36,r=0.06,sig=0.2,T=1,func="L",strike=40,k=2,paths=100000,timesteps = 200)
  ,american_put_price_lsmc(s0=36,r=0.06,sig=0.2,T=2,func="L",strike=40,k=2,paths=100000,timesteps = 200)  
  ,american_put_price_lsmc(s0=40,r=0.06,sig=0.2,T=0.5,func="L",strike=40,k=2,paths=100000,timesteps = 126)
  ,american_put_price_lsmc(s0=40,r=0.06,sig=0.2,T=1,func="L",strike=40,k=2,paths=100000,timesteps = 200)
  ,american_put_price_lsmc(s0=40,r=0.06,sig=0.2,T=2,func="L",strike=40,k=2,paths=100000,timesteps = 200)
  ,american_put_price_lsmc(s0=44,r=0.06,sig=0.2,T=0.5,func="L",strike=40,k=2,paths=100000,timesteps = 126)
  ,american_put_price_lsmc(s0=44,r=0.06,sig=0.2,T=1,func="L",strike=40,k=2,paths=100000,timesteps = 200)
  ,american_put_price_lsmc(s0=44,r=0.06,sig=0.2,T=2,func="L",strike=40,k=2,paths=100000,timesteps = 200))
  ,nrow=3,ncol=3)

colnames(laguerrek2)=c("S0=36","S0=40","S0=44")
rownames(laguerrek2)=c("T=0.5","T=1","T=2")
```

这段代码生成了使用2阶拉盖尔多项式的测试矩阵，测试了三种初始价格（36、40、44）和三种到期时间（0.5年、1年、2年）的组合。

类似地，代码中还生成了3阶和4阶拉盖尔多项式的测试矩阵（`laguerrek3`和`laguerrek4`）。

### 5.2 埃尔米特多项式测试矩阵

代码中生成了埃尔米特多项式的测试矩阵（`hermitek2`、`hermitek3`、`hermitek4`），结构与拉盖尔多项式测试矩阵相同，只是将`func`参数改为"H"。

### 5.3 单项式测试矩阵

代码中生成了单项式的测试矩阵（`monomialk2`、`monomialk3`、`monomialk4`），结构与前两种测试矩阵相同，只是将`func`参数改为"M"。

### 5.4 测试矩阵结构

所有测试矩阵都是3x3矩阵，行表示不同的到期时间，列表示不同的初始股票价格：

| 到期时间 | S0=36 | S0=40 | S0=44 |
|---------|-------|-------|-------|
| T=0.5   | 价格1 | 价格2 | 价格3 |
| T=1     | 价格4 | 价格5 | 价格6 |
| T=2     | 价格7 | 价格8 | 价格9 |

## 6. 远期开始美式期权定价

### 6.1 欧洲远期开始期权计算

```R
#2
#a

r=0.06
T_put=1
timesteps_put=100
t_put=0.2
stock_2a=stock_path_generator(s0=65,r=0.06,sig=0.2,paths=100000,timesteps=timesteps_put,T=T_put)
europut=exp(-r*T_put)*pmax((stock_2a[,(t_put*timesteps_put)]-stock_2a[,ncol(stock_2a)]),0)
europut_value=mean(europut)
```

这段代码计算了一个远期开始的欧式看跌期权的价值：
- 初始价格`s0=65`
- 波动率`sig=0.2`
- 无风险利率`r=0.06`
- 到期时间`T_put=1`年
- 远期开始时间`t_put=0.2`年（即期权在0.2年后开始，此时执行价格确定）
- 执行价格等于`t_put`时刻的股票价格
- 期权价值为到期日收益的现值平均值

### 6.2 fwd_start_american_put_price_lsmc函数

```R
#b
fwd_start_american_put_price_lsmc=function(s0,r,sig,paths,timesteps,T,func,k,t){

  stock=stock_path_generator(s0,r,sig,paths,timesteps,T)
  
  #defining index matrix
  index=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  
  strike=stock[,(t*timesteps)]
  #defining Y matrix (payoff)
  Y=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  exercisevalues=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  exercisevalues[,(t*timesteps):ncol(exercisevalues)]=pmax((strike-stock[,(t*timesteps):ncol(exercisevalues)]),0)
  Y[,ncol(Y)]=exercisevalues[,ncol(stock)]
  
  index[which(Y[,ncol(Y)]>0),ncol(index)]=1
  #View(cbind(Y[,ncol(Y)],index[,ncol(index)]))
  
  delta=T/timesteps
  for(j_reverse in (ncol(Y)-1):(t*timesteps + 1)){ #exercising starts right after t=0.2
    #starting from the end
    
    #Y = ind1*exp(-rd)*ev1 + ind2*exp(-2rd)*ev2 + . . .
    Y[,j_reverse]=rowSums(as.matrix(index[,((j_reverse+1):ncol(index))]*exp(-r*(((j_reverse+1):ncol(index))-j_reverse)
                                                                            *delta)*exercisevalues[,((j_reverse+1):ncol(index))]))
    
    #find all positions in the current column for which Y=0 (we might try exercising here, hoping to get something>0)
    exercise_positions=which(Y[,j_reverse]==0)
    
    #convert all index values to zero (for now)
    index[exercise_positions,j_reverse]=1
    
    #make all Ys equal to corresponding exercise values
    Y[exercise_positions,j_reverse]=exercisevalues[exercise_positions,j_reverse]
    
    #even after exercising, some of the values in the current column will remain zero if option is out of the money
    exercise_positions=which(Y[,j_reverse]==0)
    
    #we won't exercise here and change those index values back to 0 for which even after exercising we get 0
    index[exercise_positions,j_reverse]=0
    
    
    #now take all the values for which option is in the money
    nodes=which(Y[,j_reverse]>0)
    ys=Y[nodes,j_reverse]
    
    #and corresponding stock price
    xs=stock[nodes,j_reverse]
    
    #choose the appropriate L function
    if(func=="L"){
      L_xs=laguerre_upto_k(xs,k)
    }else if(func=="H"){
      L_xs=hermite_upto_k(xs,k)
    }else if(func=="M"){
      L_xs=monomials_upto_k(xs,k)
    }
    
    A=matrix(nrow=k,ncol=k)
    A=t(L_xs)%*%(L_xs) #sum of all combinations of L function (summation Li(x)Lj(x))
    
    b=matrix(nrow=k,ncol=1)
    b=t(t(as.matrix(ys))%*%L_xs)
    
    a=solve(A,b,tol = 1e-30)
    ecv=rep(0,length=nrow(Y))
    ecv[nodes]=L_xs%*%a #get the continuation values
    
    ev=rep(0,length=nrow(Y))
    ev[nodes]=exercisevalues[nodes,j_reverse]
    
    index[which(ev>ecv),j_reverse]=1 #wherever exercise value > continuation value, make that index 1
    
    index[which(index[,j_reverse]==1),(j_reverse+1):ncol(index)]=0 #making all index values zero after 1 is seen
    
    #for loop to do the same operation as line above (slower)
    #for(i in 1:nrow(index)){
    # index[i,(which(index[i,]==1)[-1])]=0
    #}
    
    #put the maximum of exercise value and the value that was there before (discounted payoff from future)
    Y[,j_reverse]=pmax(ev,Y[,j_reverse])
    #View(cbind(ev,ecv,index[,j_reverse]))
    
  }

  #multiply index by exercise matrix to get only the exercise values where it is optimal
  #and discount by appropriate multiple of delta (col 1 will have 0 as multiple and so on)
  payoff_each_path=rowSums(exp(-r*seq(0,(ncol(Y)-1))*delta)*(index*exercisevalues))
  
  #take payoff by monte carlo

  p=mean(payoff_each_path)
  p
}
americanput=fwd_start_american_put_price_lsmc(s0=65,sig=0.2,r=0.06,k=3,func="M",t=0.2,timesteps=100000,paths=100,T=1)
```

#### 函数参数
- `s0`：初始股票价格
- `r`：无风险利率
- `sig`：波动率
- `paths`：模拟路径数量
- `timesteps`：时间步数
- `T`：到期时间（年）
- `func`：基函数类型
- `k`：基函数阶数
- `t`：远期开始时间（年）

#### 关键差异

与普通美式期权定价函数相比，远期开始期权函数的主要差异在于：

1.  **执行价格确定**：执行价格等于远期开始时间`t`的股票价格（`strike=stock[,(t*timesteps)]`）

2.  **执行时间限制**：期权只能在远期开始时间`t`之后执行（`exercisevalues[,(t*timesteps):ncol(exercisevalues)]`）

3.  **循环范围**：反向循环从到期日开始，只计算到远期开始时间`t`（`for(j_reverse in (ncol(Y)-1):(t*timesteps + 1))`）

#### 函数调用示例

```R
americanput=fwd_start_american_put_price_lsmc(s0=65,sig=0.2,r=0.06,k=3,func="M",t=0.2,timesteps=100000,paths=100,T=1)
```

这个示例计算了一个远期开始的美式看跌期权的价值：
- 初始价格`s0=65`
- 波动率`sig=0.2`
- 无风险利率`r=0.06`
- 基函数阶数`k=3`
- 基函数类型`func="M"`（单项式）
- 远期开始时间`t=0.2`年
- 时间步数`timesteps=100000`
- 路径数量`paths=100`
- 到期时间`T=1`年

## 7. 慢速实现：slow_american_put_price_lsmc函数

```R
#################################################################################################################
#################################################################################################################
#################################################################################################################

#here is a slower function that does the same thing for Q1. However, I observed that the 
# results are more accurate using this function (but only that it is slower)

slow_american_put_price_lsmc=function(s0,r,sig,paths,timesteps,T,func,strike,k){
  stock=stock_path_generator(s0,r,sig,paths,timesteps,T)
  
  #defining index matrix
  index=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  
  #defining Y matrix (payoff)
  Y=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  exercisevalues=matrix(0,nrow=nrow(stock),ncol=ncol(stock))
  exercisevalues=pmax((strike-stock),0)
  Y[,ncol(Y)]=exercisevalues[,ncol(stock)]
  
  index[which(Y[,ncol(Y)]>0),ncol(index)]=1
  #View(cbind(Y[,ncol(Y)],index[,ncol(index)]))
  
  delta=T/timesteps
  pb=txtProgressBar(min=0,max=(ncol(Y)-2),style=3)
  for(j in 1:(ncol(Y)-2)){
    #j=1
    j_reverse=ncol(Y)-j
    for(i in 1:nrow(Y)){
      if(length(which(index[i,]==1))>0){ #not considering out of the money paths
        exerc=which(index[i,]==1)[1] #taking the first value for which index=1
        Y[i,j_reverse]=index[i,exerc]*exp(-r*(exerc-j_reverse)*delta)*exercisevalues[i,exerc]
      }else {
        Y[i,j_reverse]=exercisevalues[i,j_reverse]
        if(Y[i,j_reverse]>0){
          index[i,j_reverse]=1
        }
      }
    }
    nodes=which(Y[,j_reverse]>0)
    ys=Y[nodes,j_reverse]
    xs=stock[nodes,j_reverse]
    if(func=="L"){
      L_xs=laguerre_upto_k(xs,k)
    }else if(func=="H"){
      L_xs=hermite_upto_k(xs,k)
    }else if(func=="M"){
      L_xs=monomials_upto_k(xs,k)
    }
    
    A=matrix(nrow=k,ncol=k)
    A=t(L_xs)%*%(L_xs) #sum of all combinations of L function
    
    b=matrix(nrow=k,ncol=1)
    b=t(t(as.matrix(ys))%*%L_xs)
    
    a=solve(A,b,tol = 1e-30)
    ecv=rep(0,length=nrow(Y))
    ecv[nodes]=L_xs%*%a
    
    ev=rep(0,length=nrow(Y))
    ev[nodes]=exercisevalues[nodes,j_reverse]
    
    index[which(ev>ecv),j_reverse]=1
    for(i in 1:nrow(index)){
      index[i,(which(index[i,]==1)[-1])]=0
    }
    Y[,j_reverse]=pmax(ev,ecv)
    #View(cbind(ev,ecv,index[,j_reverse]))
    
    setTxtProgressBar(pb,j)
  }
  close(pb)
  
  payoff_each_path=vector(length = nrow(Y))
  
  for(i in 1:length(payoff_each_path)){
    ind=which(index[i,]==1)
    if(length(ind)>0){
      payoff_each_path[i]=exp(-r*ind*delta)*exercisevalues[i,ind]
    }
  }
  p=mean(payoff_each_path)
  p
}
```

### 7.1 与快速实现的主要差异

| 特性 | 快速实现 (american_put_price_lsmc) | 慢速实现 (slow_american_put_price_lsmc) |
|------|-----------------------------------|----------------------------------------|
| 延续价值计算 | 使用向量操作 `rowSums` | 使用嵌套循环逐路径计算 |
| 索引更新 | 向量操作 `index[which(index[,j_reverse]==1),(j_reverse+1):ncol(index)]=0` | 循环 `for(i in 1:nrow(index)){...}` |
| 收益计算 | 向量操作 `rowSums` | 循环逐路径计算 |
| 计算速度 | 更快（向量操作效率高） | 更慢（嵌套循环效率低） |
| 数值稳定性 | 可能受向量操作精度影响 | 更高（逐路径计算更精确） |

### 7.2 性能与精度权衡

作者在代码注释中提到："我观察到使用这个函数结果更准确，但只是速度较慢"。这反映了数值计算中常见的性能与精度权衡：
- 向量操作利用了R的向量化优势，计算速度快，但可能在某些情况下损失精度
- 嵌套循环虽然速度慢，但逐路径计算提供了更高的数值稳定性和精度

## 8. 代码分析与改进建议

### 8.1 优点

1.  **结构清晰**：代码模块化程度高，不同功能分离到不同函数中
2.  **注释详细**：关键步骤都有注释说明，便于理解
3.  **灵活性强**：支持多种基函数和参数组合
4.  **方差减少**：使用对偶变量技术提高模拟效率
5.  **结果验证**：提供快慢两种实现方式进行结果验证

### 8.2 改进建议

1.  **参数检查**：添加参数有效性检查，避免无效输入导致错误
2.  **并行计算**：对于路径数较多的情况，可以使用R的并行计算功能提高速度
3.  **内存优化**：对于大规模模拟，可以考虑使用更高效的内存管理方式
4.  **基函数扩展**：可以添加更多类型的基函数，如切比雪夫多项式、样条函数等
5.  **结果输出**：增加结果的统计信息（如标准差、置信区间等）
6.  **代码复用**：提取公共代码部分（如回归计算、索引更新等）为独立函数

### 8.3 示例改进：参数检查

```R
# 添加参数检查的示例
american_put_price_lsmc=function(s0,r,sig,paths,timesteps,T,func,strike,k){
  # 参数检查
  if(s0 <= 0) stop("初始股票价格s0必须大于0")
  if(r < 0) stop("无风险利率r不能为负")
  if(sig <= 0) stop("波动率sig必须大于0")
  if(paths <= 0) stop("路径数paths必须大于0")
  if(timesteps <= 0) stop("时间步数timesteps必须大于0")
  if(T <= 0) stop("到期时间T必须大于0")
  if(!func %in% c("L","H","M")) stop("基函数类型func必须是'L'、'H'或'M'")
  if(strike <= 0) stop("执行价格strike必须大于0")
  if(k < 2 || k > 4) stop("基函数阶数k必须在2-4之间")
  
  # 原有代码...
}
```

## 9. 总结

### 9.1 核心实现

本文件实现了基于LSMC方法的美式期权定价，主要包括：
- 股票路径生成（含对偶变量技术）
- 三种基函数（拉盖尔多项式、埃尔米特多项式、单项式）
- 美式看跌期权定价
- 远期开始美式期权定价
- 测试矩阵生成
- 快慢两种实现方式

### 9.2 技术贡献

1.  **完整实现**：提供了LSMC方法的完整R语言实现，包括路径生成、基函数、定价逻辑等
2.  **多种选择**：支持三种不同的基函数，便于比较不同基函数对定价结果的影响
3.  **扩展应用**：实现了远期开始期权这一复杂衍生品的定价
4.  **结果验证**：通过快慢两种实现方式验证结果的准确性

### 9.3 应用价值

该代码可用于：
- 美式期权定价研究
- 基函数选择对定价精度的影响分析
- 远期开始期权等复杂衍生品的定价
- 蒙特卡洛模拟方差减少技术的应用研究
- 金融工程教学与学习

## 10. 数学附录

### 10.1 几何布朗运动

股票价格遵循几何布朗运动：

$$dS_t = rS_t dt + \sigma S_t dW_t$$

其解析解为：

$$S_t = S_0 \exp\left( (r - \frac{1}{2}\sigma^2)t + \sigma W_t \right)$$

其中：
- $S_t$：t时刻的股票价格
- $S_0$：初始股票价格
- $r$：无风险利率
- $\sigma$：波动率
- $W_t$：标准布朗运动

### 10.2 LSMC方法原理

LSMC方法的核心思想是使用最小二乘回归估计期权的延续价值：

1.  模拟大量股票价格路径
2.  从到期日开始反向计算
3.  在每个时间点，对有正收益的路径进行回归，估计延续价值
4.  比较执行价值和延续价值，确定最优执行策略
5.  计算所有路径的贴现收益，取平均值作为期权价格

### 10.3 拉盖尔多项式

拉盖尔多项式的定义：

$$L_0(x) = 1$$
$$L_1(x) = 1 - x$$
$$L_2(x) = 1 - 2x + \frac{1}{2}x^2$$
$$L_3(x) = 1 - 3x + \frac{3}{2}x^2 - \frac{1}{6}x^3$$

带权重的拉盖尔多项式：

$$\tilde{L}_n(x) = e^{-x/2}L_n(x)$$

### 10.4 埃尔米特多项式（物理学家）

埃尔米特多项式的定义：

$$H_0(x) = 1$$
$$H_1(x) = 2x$$
$$H_2(x) = 4x^2 - 2$$
$$H_3(x) = 8x^3 - 12x$$

## 参考文献

1. Longstaff, F. A., & Schwartz, E. S. (2001). Valuing American options by simulation: A simple least-squares approach. The Review of Financial Studies, 14(1), 113-147.
2. Glasserman, P. (2004). Monte Carlo methods in financial engineering. Springer Science & Business Media.
3. Hull, J. C. (2018). Options, futures, and other derivatives (10th ed.). Pearson.