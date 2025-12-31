function [price, computational_cost] = adaptive_mlmc_european_call(S0, K, r, sigma, T, M, Lmin, Lmax, eps_target)
% 参数说明
% S0: 初始股价
% K: 行权价
% r: 无风险利率
% sigma: 波动率
% T: 到期时间
% M: 层级精度倍增因子（例如，M=2表示每一级时间步翻倍）
% Lmin: 最粗糙层级
% Lmax: 最精细层级限制
% eps_target: 目标均方根误差 (RMSE)

% 初始化
Vl = zeros(Lmax, 1); % 存储各层级方差估计
Cl = zeros(Lmax, 1); % 存储各层级单次模拟成本
Nl = zeros(Lmax, 1); % 存储各层级模拟次数
sums = zeros(Lmax, 3); % 用于统计: [P0, Pl, Pl-1]

computational_cost = 0;

% 初始运行少量模拟，以估计各级别方差和成本
fprintf('初步估计各级别方差和成本...\n');
for l = Lmin:Lmax
    Np = 1000; % 初始用于估计的模拟次数
    [cost_temp, variance_temp, Pfine, Pcoarse] = mlmc_level_l(l, M, Np, S0, K, r, sigma, T);
    computational_cost = computational_cost + cost_temp;
    Vl(l) = variance_temp;
    Cl(l) = cost_temp / Np;
    if l == Lmin
        sums(l, 1) = sums(l, 1) + sum(Pfine);
        sums(l, 2) = sums(l, 2) + sum(Pfine);
    else
        sums(l, 1) = sums(l, 1) + sum(Pfine - Pcoarse); % 估计差值 E[Pl - Pl-1]
        sums(l, 2) = sums(l, 2) + sum(Pfine);
        sums(l, 3) = sums(l, 3) + sum(Pcoarse);
    end
    fprintf('层级 l=%d, 估计方差 Vl=%.4e, 单次成本 Cl=%.4e\n', l, Vl(l), Cl(l));
end

% 自适应确定最优模拟次数 Nl
fprintf('\n根据目标误差自适应分配模拟次数...\n');
theta = 0.95; % 分配给粗糙层级的误差权重（例如95%）
for iter = 1:10 % 迭代次数限制，防止无限循环
    total_variance = 0;
    for l = Lmin:Lmax
        total_variance = total_variance + sqrt(Vl(l) * Cl(l));
    end
    
    for l = Lmin:Lmax
        % 根据方差和成本，使用MLMC最优分配公式计算Nl
        Nl(l) = ceil((1/eps_target^2) * sqrt(Vl(l)/Cl(l)) * total_variance / theta);
    end

    % 运行新增的模拟次数
    for l = Lmin:Lmax
        if Nl(l) > 0
            [cost_temp, variance_temp, Pfine, Pcoarse] = mlmc_level_l(l, M, Nl(l), S0, K, r, sigma, T);
            computational_cost = computational_cost + cost_temp;
            % 更新方差估计 (简化处理，实际中可能需要更复杂的合并)
            Vl(l) = (Vl(l) + variance_temp) / 2;
            if l == Lmin
                sums(l, 1) = sums(l, 1) + sum(Pfine);
                sums(l, 2) = sums(l, 2) + sum(Pfine);
            else
                sums(l, 1) = sums(l, 1) + sum(Pfine - Pcoarse);
                sums(l, 2) = sums(l, 2) + sum(Pfine);
                sums(l, 3) = sums(l, 3) + sum(Pcoarse);
            end
        end
    end

    % 检查是否达到目标精度 (简化版，实际应计算置信区间)
    current_eps = sqrt(sum(Vl(l) / Nl(l)));
    fprintf('迭代 %d: 当前估计误差 = %.4e\n', iter, current_eps);
    if current_eps < eps_target
        fprintf('已达到目标精度。\n');
        break;
    end
end

% 计算最终的期权价格估计
price = 0;
for l = Lmin:Lmax
    if l == Lmin
        price = price + sums(l, 1) / (Nl(l) + 1000); % 加上初始的1000次
    else
        price = price + sums(l, 1) / (Nl(l) + 1000);
    end
end
price = exp(-r * T) * price; % 折现

fprintf('\n最终期权价格估计: %.4f\n', price);
fprintf('总计算成本 (任意单位): %.2f\n', computational_cost);
end

function [cost, variance, Pfine, Pcoarse] = mlmc_level_l(l, M, Nl, S0, K, r, sigma, T)
% 模拟特定层级 l 的路径和收益
% l: 当前层级
% M: 精度倍增因子
% Nl: 模拟次数
% 返回:
% cost: 总计算成本
% variance: 本层级差值的方差
% Pfine: 精细层级收益
% Pcoarse: 粗糙层级收益 (对于l=Lmin，Pcoarse=0)

Nfine = M^l; % 精细层级的时间步数
dt_fine = T / Nfine;

Pfine = zeros(Nl, 1);
Pcoarse = zeros(Nl, 1);

for i = 1:Nl
    % 生成布朗运动路径（使用相同的随机数源进行耦合）
    rng('shuffle'); % 为演示起见，实际耦合需要更精细的控制
    W_fine = cumsum([0, sqrt(dt_fine) * randn(1, Nfine)]);
    S_fine = S0 * exp((r - 0.5 * sigma^2) * (0:dt_fine:T) + sigma * W_fine);
    Pfine(i) = max(S_fine(end) - K, 0);

    if l > 1 % 如果存在更粗糙的层级
        Ncoarse = M^(l-1);
        dt_coarse = T / Ncoarse;
        % 从精细路径中抽取对应粗糙时间点的路径（实现耦合）
        W_coarse = W_fine(1:M:end); % 抽取
        S_coarse = S0 * exp((r - 0.5 * sigma^2) * (0:dt_coarse:T) + sigma * W_coarse);
        Pcoarse(i) = max(S_coarse(end) - K, 0);
    else
        Pcoarse(i) = 0; % 最粗糙层级没有更粗的一级
    end
end

% 计算本层级差值的方差
if l == 1
    Y = Pfine; % 最粗糙层级的"差值"就是其本身
else
    Y = Pfine - Pcoarse;
end
variance = var(Y);

% 计算成本：假设成本与时间步数（即模拟路径的精细程度）成正比
cost = Nl * Nfine;
end

% 示例调用
S0 = 100; K = 110; r = 0.05; sigma = 0.2; T = 1;
M = 2; Lmin = 1; Lmax = 5; eps_target = 0.01;
[price, cost] = adaptive_mlmc_european_call(S0, K, r, sigma, T, M, Lmin, Lmax, eps_target);