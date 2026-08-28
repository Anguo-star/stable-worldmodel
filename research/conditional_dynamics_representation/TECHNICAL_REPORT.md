# 从边缘非坍缩到条件联合动力学

## ContextWorld 的诊断结论、条件重叠联合对齐与自回归滚动一致性

### 摘要

世界模型即使具有方差充足、无明显坍缩的 latent，也未必会从历史中识别当前 episode 的隐藏
动力学。我们用 ContextWorld 检验一个更严格的问题：固定当前观测和查询动作，只改变能够揭示
隐藏动力学的历史，模型是否会预测不同且方向正确的未来。原生 LeWM、stop-gradient、SIGReg、
VISReg 以及多种全局协方差目标表明，改善 latent 的边缘分布并不能自动建立这种条件关系。

本文将失败拆为六个可独立测量的环节：历史信息是否可读、目标未来是否可分、Predictor 是否使用
历史、连续响应是否校准、单步响应能否在自回归 rollout 中保持，以及新能力是否保留原任务规划。
分析进一步发现，连续 Motion Damping 中存在两个先后发生的断点：有限的 query 模板会被共享
Predictor 记忆；即使单步条件响应已经学会，它也可能在模型把自己的预测重新作为输入后迅速失真。

基于这一诊断，我们得到一个不改变 LeWM 参数量和推理结构的方法：**条件重叠联合对齐**
（Conditional-Overlap Joint Alignment，COJA）。训练数据中选取具有相同可见当前条件
`(Q,A)`、不同历史 `H` 和各自真实未来的样本组；原生 MSE 继续拟合绝对未来，新增的训练期辅助项
只对齐组内条件响应，并用一个在真实 target 处停止的 assignment barrier 防止条件分支停在共同
中点。该方法不把隐藏动力学标签输入模型，不增加 encoder、adapter、head 或推理计算。

COJA 在 ActionDelay、Contact Friction、Portal Exit 和 Motion Damping 上给出正信号；其中
ActionDelay 已有三个训练 seed 的独立复现。Motion 的全 query 训练把 NRE 从 matched no-aux
的 `1.130` 降到 `0.767`，证明连续条件响应可由原 LeWM 学到。可是，一步预测已经明显依赖历史
的 checkpoint 在两步隐藏动力学规划中仍近乎失败：正确历史物理误差约为 `103.17 px`，交换历史
反而略好。这说明**单步条件可辨识性不等于 rollout 条件可辨识性**。

基于这一新断点，我们把 COJA 扩展为 **rollout-consistent COJA（RC-COJA）**：保持模型、参数、
推理路径和单步 COJA 不变，只把一部分已有的 hidden-row 原生 MSE 分配给真实的第二步自回归预测。
无需第二步 COJA，仅使用第二步 native MSE，就把两步隐藏规划误差从匹配的一步训练对照的
`102.37 px` 降到 `58.54 px`，并在从未训练的三步评测中从 `111.45 px` 降到 `80.53 px`；
两项 paired-bootstrap 95% interval 均远离零。一步误差增加 `3.04 px`，显示仍有短期—长程
权衡。相同 checkpoint 的标准 PushT 保留性结果为 `60/100`，matched placebo 为 `57/100`，
配对区间 `[-6,+12]` 个百分点，未显示可辨认的退化。本文因此分别报告直接条件响应、隐藏动力学
规划和原任务保持，不再用一个标准 CEM 点数替代前两者。

固定 `ρ=0.25`、不在第二个任务上调参的 Contact Friction 迁移进一步表明，这一机制并非 Motion
特例。最初重复 query action 的实现虽改善两步隐藏规划，却使标准 PushT 降低
`5.0 [0.67,9.33]` 个百分点。单因素替换为普通 PushT 训练总体中的连续 action block 后，两步和
未训练五步物理误差相对 matched control 分别改善 `4.39 [2.93,5.90]` 和
`4.37 [1.87,6.84] px`，标准 PushT 则为 `216/300` 对 `212/300`，差值
`+1.33 [-3.33,+6.00]` 个百分点。该结果把当前失败定位到 continuation action support，而非
RC-COJA、`ρ` 或模型复杂度，并得到首个 Contact 单 seed ICL—hidden planning—retention Pareto
正例。

随后从公开 PushT 初始化直接进行一次 4,096-step 单阶段训练，排除了“两阶段续训才有效”的解释。
相对同数据、同预算的一步 COJA，完整 RC-COJA 在两步和未训练五步隐藏规划上分别改善
`3.16 [1.58,4.79]` 与 `3.09 [0.65,5.46] px`，正确历史收益的 difference-in-differences 也分别
增加 `1.89 [1.28,2.52]` 与 `3.61 [1.87,5.40] px`。一步 ICL 基本不变，标准 PushT 为
`207/300` 对 `206/300`，差 `+0.33 [-4.33,+5.00]pp`。公开原始数据参考为 `237/300`；这约
`10pp` 的差距已完整出现在一步 COJA 对照中，因此是共享混合数据/适配路径的独立代价，不是
RC 的方法效应。

Motion 的最终单阶段复核给出更强的同类证据。从同一个公开初始化直接训练 4,096 steps 后，
RC-COJA 相对同数据、同预算的一步 COJA，在两步和未训练三步隐藏规划上分别改善
`57.39 [52.19,62.42]` 与 `39.08 [33.90,44.21] px`；正确历史收益的
difference-in-differences 分别为 `4.12 [2.90,5.34]` 与 `2.28 [0.28,4.01] px`。较弱的一步
子集退化 `3.01 [1.86,4.13] px`，因此仍需如实报告短期—长程权衡。标准 PushT 的同 300 queries
为 no-aux/一步 COJA/RC=`203/188/194`；RC−一步 COJA=`+2.00 [-2.67,+6.67]pp`，没有识别出
RC 特异的 retention 损伤，也不能据此宣称规划提升。至此，RC 的活性机制、单阶段可训练性及
Motion/Contact 跨任务性均已有完整证据，发现期不再修改模型、loss 或 action sampler。

冻结配方的独立 seed `14322` 进一步给出近乎逐值复现：h1/h2/h3 的 RC−COJA 物理改善分别为
`-3.30 [-4.53,-2.14]`、`+57.60 [52.55,62.55]` 和
`+41.56 [36.02,47.02] px`；seed `14321` 对应为 `-3.01/+57.39/+39.08 px`。两 training seed
的标准 CEM 方法效应分别为 `+2.00/-1.33pp`，层级 bootstrap 均值为 `+0.33pp`、95% 区间
`[-4.0,+4.5]pp`。因此大幅 h2/h3 收益与小幅 h1 代价均可复现，而标准环境 retention 没有识别出
稳定的 RC 方向。

Contact 的独立 seed `13314` 同样复现了完整单阶段结果。RC−COJA 在 h2 和未训练 h5 上分别改善
`3.67 [2.06,5.34]` 与 `4.37 [1.76,6.92] px`，正确历史收益的 difference-in-differences 为
`2.14 [1.53,2.78]` 与 `3.73 [1.87,5.66] px`；seed `13313` 的对应主效应为
`3.16/3.09 px`。h1 为 `+0.08 [-0.27,0.42] px`，没有检测到短期损伤。两个 seed 的 direct NRE
平均变化仅 `+0.00010`，gain 平均增加 `0.00230`；标准 PushT 的方法效应为 `+0.33/+2.00pp`，
training-seed 分层 bootstrap 均值为 `+1.17 [-2.17,+4.50]pp`。因此 Contact 的长程收益也不是
单 seed 偶然性，且没有识别出 RC 特异的一步校准或原任务保持代价。

## 1. 问题定义

设

- `H`：动作—观测历史；
- `Q`：当前观测；
- `A`：查询动作序列；
- `M`：episode 内固定、但不直接提供给模型的动力学因素；
- `O+`：执行查询动作后的未来观测。

研究对象是

\[
p(O^+\mid H,Q,A),
\]

而不是 latent 的边缘分布 `p(Z)`。在 ContextWorld 的 matched-history 评测中，`Q` 和 `A`
保持相同，只有历史所揭示的 `M` 不同。正确模型应满足

\[
p(O^+\mid H_0,Q,A)\neq p(O^+\mid H_1,Q,A).
\]

如果模型对两段历史输出同一个条件均值，它可能仍具有健康的 latent 方差和较低的平均训练损失，
却没有学会 episode-level system identification。

### 1.1 直接测量条件响应

令两个 matched 条件的 target latent 和 prediction response 为

\[
r=z_1^+-z_0^+,\qquad
\hat r=\hat z_1^+-\hat z_0^+.
\]

本文同时报告三项连续量：

\[
\text{gain}=\frac{\langle\hat r,r\rangle}{\lVert r\rVert^2},
\]

\[
\text{alignment}=\frac{\langle\hat r,r\rangle}
{\lVert\hat r\rVert\lVert r\rVert},
\]

\[
\text{NRE}=\frac{\lVert\hat r-r\rVert^2}{\lVert r\rVert^2}.
\]

`NRE=1` 对应完全不随历史变化的 prediction response。低于 `1` 表示模型相对“忽略历史”
取得连续改善；它是有物理含义的参考点，不是用来一票否决模型的硬阈值。离散的 future、history、
switch 与 worst-condition accuracy 用来测 assignment，不能替代 gain、alignment 和 NRE。

### 1.2 规划能力包含两个不同问题

直接 ICL 指标回答“模型是否根据历史改变未来预测”。规划还必须拆成两个估计量：

1. **隐藏动力学规划**：在 ContextWorld 环境中，正确历史是否使 CEM 选择更合适的动作；
2. **原任务保持**：在没有隐藏动力学的标准环境中，训练后的 checkpoint 是否保留原有规划能力。

标准 PushT CEM 只回答第二个问题。若候选与 published source 使用不同的 `50/50` 混合数据，
二者的差值还同时混入数据 recipe 和训练曝光效应。为分离这些因素，本文使用三个模型臂：

- `source`：published 原始数据 checkpoint；
- `native`：相同混合数据、预算和训练流程，但不使用新方法；
- `candidate`：在 matched native 基础上加入新方法。

由此分别报告 `native-source` 的数据效应、`candidate-native` 的方法效应和
`candidate-source` 的部署总效应。在隐藏动力学规划中，还对同一 checkpoint 比较正确历史与
交换历史；这是判断规划收益是否真正来自 history use 的直接干预。Motion 的标准 CEM 下降仍是
真实 retention 问题，但不能据此推断候选在隐藏动力学环境中没有价值。

## 2. 为什么边缘分布约束不够

### 2.1 置换不变性

考虑只依赖 target latent 经验分布的正则

\[
R(\{z_i^+\}_{i=1}^n).
\]

对任意样本置换 `π`，都有

\[
R(\{z_i^+\})=R(\{z_{\pi(i)}^+\}).
\]

也就是说，只要 latent 的集合不变，该正则无法判断“哪段历史应该对应哪个未来”。它可以防止
全局坍缩、调整尺度或改善协方差，却不能单独保证

\[
I(\hat Z^+;M\mid Q,A)>0.
\]

这不是说 conditional MSE 在理论上永远学不会条件期望。实际问题是 Encoder、Projector 和
Predictor 联合优化时，预测条件均值、移动 target geometry 或利用与任务无关的 latent 方差，
通常比学习微弱的 history-conditioned response 更容易。

### 2.2 ContextWorld 的受控反例

以下现象共同排除了“只需更强 anti-collapse regularizer”的解释：

- ActionDelay 中，原生 LeWM 在完整 1,024 步后仍接近三分类随机；同规模 PLDM 能明显学习，
  说明任务信号和模型容量并非根本不足。
- `target stop-gradient + SIGReg` 保持全部 target pair 非重合，但 `959/960` 个 query 仍对不同
  历史作出相同选择。
- VISReg 的 1,024-step ActionDelay 结果为 macro `0.3316`、worst group `0`、仅 `8/960`
  个 history-responsive query，同时 `0/2,880` 个 target pair collapse。
- 多个 checkpoint 的 history 信息可以被 probe 读取，却仍没有方向正确的 prediction response。

结论是：**历史可读、target 可分和边缘非坍缩都是必要信息，但都不等于 Predictor 建立了正确的
history–action–future 对应关系。**

## 3. 根因分析框架

为避免把一种症状误写成唯一原因，我们从六个层面诊断模型。

| 层面 | 问题 | 主要测量 | 当前结论 |
|---|---|---|---|
| 数据可辨识性 | 是否存在固定 `(Q,A)` 后改变历史的观测支持 | exact/near conditional overlap | 普通 marginal 数据不提供这种反事实对应；当前正例依赖 overlap |
| 表征可用性 | 历史是否保留隐藏动力学信息 | query/scene leave-out probe | 部分 checkpoint 可读，但可读性不足以保证预测使用 |
| 条件耦合 | Predictor 是否真正使用历史 | switch、Jacobian、模块互换 | 主要断点位于 Predictor trunk，而非输出 projection |
| 响应校准 | 变化方向与幅值是否正确 | gain、alignment、NRE | assignment accuracy 可在 NRE 很差时虚高，必须直接测响应 |
| Rollout 一致性 | 单步历史响应能否在 self-rollout 后保持 | 1/2/3-step hidden planning、correct/swapped history | 单步 COJA 不充分；真实自回归 MSE 可修复两步并迁移到三步 |
| 总体泛化与保持 | 响应是否跨 query 泛化，原任务规划是否保留 | train–Development gap、matched standard CEM | query 覆盖可修复 NRE；保留性必须与同数据 placebo 比较 |

### 3.1 Motion 的 Predictor 记忆现象

旧 Cartesian 训练在 8,192 步中反复使用 2,048 个 query 模板。其校准轨迹为：

| checkpoint | training NRE | Development NRE | train–Development gap |
|---:|---:|---:|---:|
| 2,048 steps | `0.756` | `0.901` | `+0.144` |
| 8,192 steps | `0.207` | `1.120` | `+0.913` |

训练继续进行时，in-sample response 几乎被精确拟合，held-out response 却变差。模块互换显示，
joint checkpoint 的效果全部随 Predictor trunk 转移，`pred_proj` 的影响很小；逐层替换又表明该
变化分布在六个 Transformer block 中，而不是一个可简单冻结的末层。这些结果支持
“query-specific response memorization”解释，而不是“训练不足”解释。

### 3.2 校准与规划并非同一断点

用全部 8,192 个 matched query 训练后，Motion 的 NRE 和 assignment 均显著改善，证明 query
覆盖确实是旧校准失败的主因。然而标准 CEM 仍低于 matched no-aux 模型。因此不能继续把所有失败
归为一个笼统的“Predictor 不读历史”：当前 Predictor 已读历史并能泛化到 Development，但条件
梯度仍会改变 planner 所依赖的原任务预测函数。

### 3.3 单步条件可辨识性不保证自回归保持

规划器消费的不是孤立的一步预测，而是把模型自己的输出重新放回历史。令

\[
\hat z_{t+1}=F_\theta(H_t,A_t),\qquad
\hat z_{t+k}=F_\theta(\hat H_{t+k-1},A_{t+k-1}).
\]

即使第一步的历史响应方向正确，多步响应仍包含一串对模型生成状态的 Jacobian：

\[
\frac{\partial \hat z_{t+k}}{\partial H_t}
=
\left(\prod_{j=1}^{k-1}
\frac{\partial F_\theta}{\partial \hat z_{t+j}}\right)
\frac{\partial F_\theta}{\partial H_t}.
\]

第一步 conditional objective 只约束最右侧项；后续 Jacobian 可以缩小、放大或旋转历史效应。
Motion 的四动作 COJA checkpoint 正好给出这个反例：单步 switch=`0.938`，但两步正确历史规划
误差约 `103 px`，交换历史并不更差。故根因不是“模型仍完全不读历史”，而是**训练分布只校准
真实历史上的一步映射，部署却要求模型在自己生成的历史上保持同一个条件动力学解释**。

## 4. 条件重叠联合对齐

### 4.1 数据条件

COJA 使用包含如下二元组的数据：

\[
(H_0,Q,A,O_0^+),\qquad(H_1,Q,A,O_1^+).
\]

两个样本具有相同的可见 `Q,A`，历史和真实未来不同。训练实现可以仅以 query RGB 与 raw action
字节构造 key；在当前资产中，这一可见 key 恰好恢复原 matched groups。因此：

- hidden dynamics label 不进入模型或 loss；
- pair id 和人工 low/high 名称不是必需输入；
- 模型推理时不需要 pair；
- 仍需要训练数据具有 conditional overlap。

最后一点是当前方法的数据假设，而不是应被隐藏的实现细节。对普通 unmatched replay，如果同一
`(Q,A)` 几乎不重复，则需要主动收集 overlap、可靠的近邻/条件核，或额外结构假设；仅靠边缘正则
无法凭空恢复未被观测的反事实关系。

### 4.2 Center-free response objective

对组内 predictions `p_0,p_1` 和 stop-gradient targets `t_0,t_1`，先去除各自公共中心：

\[
\tilde p_i=p_i-\tfrac12(p_0+p_1),\qquad
\tilde t_i=t_i-\tfrac12(t_0+t_1).
\]

响应项为

\[
L_{\mathrm{resp}}=
\frac{\operatorname{MSE}(\tilde p,\operatorname{sg}(\tilde t))}
{\operatorname{MSE}(\operatorname{sg}(\tilde t),0)+\epsilon}.
\]

它直接训练同一 `(Q,A)` 下由历史引起的相对 future response。原生逐样本 MSE 继续负责绝对未来
和公共中心，因此辅助项不再重复放大 center error。

### 4.3 在真实 target 处停止的 assignment barrier

令 `d=t_1-t_0`，定义沿真实 response 轴的 prediction gain 与中心偏移：

\[
\alpha=\frac{\langle p_1-p_0,d\rangle}{\lVert d\rVert^2},\qquad
\beta=\frac{\langle \tfrac12(p_0+p_1)-\tfrac12(t_0+t_1),d\rangle}
{\lVert d\rVert^2}.
\]

两个正确 assignment margin 为

\[
m_0=\alpha/2-\beta,\qquad m_1=\alpha/2+\beta.
\]

真实 targets 对应 `α=1, β=0`，两侧 margin 都是 `1/2`。因此采用

\[
L_{\mathrm{assign}}=
\tfrac12\sum_{i=0}^1[\max(0,\tfrac12-m_i)]^2.
\]

与普通 ranking/contrastive loss 不同，该 barrier 在真实 target 处为零，不会持续把两个
predictions 推得比真实 future 更远。

### 4.4 总目标与部署边界

当前训练目标为

\[
L=L_{\mathrm{native\ MSE}}+0.09L_{\mathrm{SIGReg}}
+0.09(L_{\mathrm{resp}}+L_{\mathrm{assign}}).
\]

辅助分支只更新 LeWM 已有的 Predictor 和 prediction projection；target、history embedding 和
action embedding 在该分支中 stop-gradient。Encoder、Projector 和 action encoder 不增加任何
新模块。训练完成后仍保存标准 LeWM state dict，规划器和推理调用保持不变。

当前实现每个 optimizer step 使用 64 条原始环境数据和 64 条 ContextWorld 数据；后 64 条组成
32 个完整 matched pairs。这个 `50/50` 指样本曝光，不表示各 loss 的数值贡献相等。

### 4.5 Rollout-consistent 扩展

COJA 解决“同一个 query 在不同历史下应对应哪个一步未来”，但不约束模型在自生成历史上的后续
映射。RC-COJA 不增加新的网络或推理分支，而是在训练时展开现有 Predictor。对 hidden rows，令

\[
\hat z_{t+1}=F_\theta(H_t,A_t),\qquad
\hat z_{t+2}=F_\theta(\operatorname{shift}(H_t,\hat z_{t+1}),A_{t+1}),
\]

并使用

\[
L_{\mathrm{hidden}}^{\mathrm{RC}}
=(1-\rho)L_{\mathrm{MSE}}^{(1)}
+\rho L_{\mathrm{MSE}}^{(2,\mathrm{AR})}
+\lambda L_{\mathrm{COJA}}^{(1)}.
\]

梯度不在 `\hat z_{t+1}` 处停止，因此第二步误差同时校准第一步输出和 Predictor 在自生成状态上的
响应。总 hidden native-MSE 权重保持不变；`ρ` 只重新分配已有监督，不叠加一个越来越强的 loss。
当前折中候选使用 `ρ=0.25`：在总体 `0.5` 的 hidden native 权重中，一步/两步分别为
`0.375/0.125`，一步 COJA 仍为 `0.09`。第二步只使用 native MSE，不使用额外 relation，因而
新增可学习参数、模块和部署计算均为零。

这里仍需真实的轨迹 continuation target，也仍保留 COJA 的 conditional-overlap 数据假设。
简洁性来自“不改变 LeWM”，不是声称无需组织训练数据。训练数据提供的对象从单步联合分布
`(H,Q,A,O^+)` 延伸到短自回归轨迹；模型推理接口没有变化。

## 5. 实验结果

### 5.1 边缘正则负对照

| 方法 | 任务 | 训练预算 | 关键结果 |
|---|---|---:|---|
| native LeWM | ActionDelay | 1,024 | macro 约 `0.333`，几乎无 history response |
| VISReg | ActionDelay | 1,024 | macro `0.3316`，`8/960` responsive queries，target 无坍缩 |
| PLDM | ActionDelay | 1,024 | macro `0.8257`，说明任务信号可学习 |

该对照不否定 VISReg 或 PLDM 在其原始目标上的价值；它只说明更健康的 latent marginal 不是
ContextWorld conditional identifiability 的充分条件。

### 5.2 跨任务条件响应

| 任务与训练 | 直接条件能力 | 连续响应 | 标准环境原任务保持 |
|---|---|---|---|
| ActionDelay，3 training seeds | macro `0.940/0.936/0.942`；worst `0.906/0.901/0.911` | `960` 类 matched query 对历史稳定响应 | 已有同 checkpoint paired CEM 保持证据 |
| Contact Friction，4,096 steps | joint/native future `0.771/0.521`，switch `1.000/0.668` | NRE `0.579/0.984`，gain `0.447/0.015` | CEM100 `72/73`，差值 `-1pp [-9,+7]pp` |
| Portal Exit，4,096 steps | joint/native future `0.752/0.584`，worst `0.746/0.512` | NRE `0.284/0.471`，gain `0.604/0.366` | CEM50 `47/46`；样本量不足以主张提升 |
| Motion Damping，8,192 full-release | joint/native future `0.660/0.494`，switch `0.969/0.441` | NRE `0.767/1.130`，gain `0.370/0.0065` | CEM300 `213/232`，差值 `-6.33pp [-11.00,-1.67]pp` |

ActionDelay 证明离散条件 assignment 可以稳定学习；Contact 与 Portal 说明同一类联合条件信号并不
必然损害规划；Motion 则证明连续校准与 planner-function retention 仍需分别解决。

### 5.3 Motion full-release 单因素结果

最新 Motion 比较固定标准 PushT 初始化、模型、optimizer、训练 seed、8,192 steps、`64+64`
batch、原生 MSE+SIGReg 和 auxiliary 公式，只改变训练 query 覆盖：删除旧 Cartesian action
overlay，直接使用冻结 release 的全部 8,192 个 matched query。结果为：

| variant | future | history | switch | worst | gain | alignment | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched no-aux | `0.494` | `0.500` | `0.441` | `0.230` | `0.0065` | `0.017` | `1.130` |
| COJA | **`0.660`** | **`0.668`** | **`0.969`** | **`0.570`** | **`0.370`** | **`0.520`** | **`0.767`** |

COJA 的 NRE paired-bootstrap 95% interval 为 `[0.724,0.814]`，完整位于零响应参考 `1.0`
以下。该结果与旧训练集的 train–Development gap 一起，建立了“覆盖不足导致 response
memorization”的因果解释。

规划结果使用两个互补 catalog：

- evaluator seeds `42/43/44` 各 100 条：COJA/no-aux 为 `70/79`、`66/75`、`76/82`；
  等 seed 平均差值 `-8.0pp`，95% paired interval `[-12.67,-3.67]pp`；
- 与既有完整结果相同的 seed42×300 catalog：`213/232`，差值 `-6.33pp`，95% paired
  interval `[-11.00,-1.67]pp`，exact McNemar `p=0.0145`。

第二组更适合与旧 CEM300 横向比较；第一组表明不利方向并非单一 evaluator seed 的偶然点数。
这些统计用于描述效应稳定性，不作为单阈值式的模型生死判定。

### 5.4 Contact 隐藏动力学规划：正确历史开始产生实际收益

标准 PushT retention 不能检验 Contact Friction ICL 是否对规划有用。新的 Development-only
评测固定同一 query action family，并让 CEM 只选择连续 action scale；动作随后在真实 low/high
friction 模拟器中执行。目标是 low-friction、scale `1` 的五个 action block 后未来。物理 oracle
表明 `244/256` 个 query 的 low/high acceptable action region 不重叠，平均最优 scale gap 为
`0.358`，因此该评测确实需要识别隐藏条件。

| arm | 正确历史物理误差↓ | 正确历史 scale regret↓ | 错误历史带来的物理损失↑ | 错误历史带来的 regret↑ |
|---|---:|---:|---:|---:|
| published source | `87.08` | `0.3386` | `-0.37` | `-0.0023` |
| matched native | `94.72` | `0.4102` | `+0.28` | `+0.0001` |
| **COJA** | **`89.79`** | **`0.3783`** | **`+1.70 [0.80,2.68]`** | **`+0.0121 [0.0059,0.0192]`** |

相对 matched native，COJA 的方法效应为物理误差改善 `4.94 px [2.31,7.58]`、scale regret
改善 `0.0319 [0.0121,0.0519]`。其正确历史收益相对 native 又分别增加
`1.41 px [0.33,2.54]` 和 `0.0120 [0.0053,0.0198]`。因此 COJA 不只是改变直接 ICL
分数；它使历史在实际规划中具有可测的正价值。

source 的绝对误差仍低于 COJA。三臂分解显示，`50/50` 数据与 native continuation 相对 source
带来 `7.64 px` 和 `0.0716` regret 的不利数据效应，COJA 追回了其中一部分但尚未全部追回。
此外，本评测使用 `64` samples × `6` iterations 的一维 CEM screen，而不是标准 PushT 的完整
`300×30` 多维 planner；它支持估计量方向和 checkpoint 排序，不应被写成最终 deployment 成功率。

### 5.5 历史候选重评：哪些应恢复，哪些仍然失败

修正规划估计量后，旧候选不能按一次 source-only 标准 CEM 点数机械淘汰，但也不能全部复活。
重评首先复用冻结 checkpoint，只对“直接 ICL 已有明显正效应，且旧否决依赖错误 comparator 或
波动硬门”的方案补 hidden-planning 对照。

| 候选 | 重评状态 | 理由 |
|---|---|---|
| COJA / 早期 PCJA exact-overlap family | **恢复为主候选** | ActionDelay 三 seed、Contact/Motion 直接响应为正；Contact hidden planning 已显示 matched-native 方法收益 |
| DynamicsResponseSIGReg | **恢复为强任务特异基线** | Action Strength 三 seed future `0.966`、switch `0.996`；同场 hidden planning 相对 matched native 将 mode classification 提高 `15.23pp [12.30,18.16]`，regret 降低 `0.0706 [0.0544,0.0872]`。但 Motion 4,096-step future/history 仅 `0.412/0.418`，不是通用解 |
| target-JTCov | **不恢复为主候选** | 同场 Action Strength hidden planning 相对 matched native 的 classification 低 `8.79pp [5.86,11.91]`，regret 高 `0.0725 [0.0530,0.0921]`，执行距离高 `2.66px [1.91,3.43]`；结合 ActionDelay/Speed 与多任务 retention 失败，错误 comparator 没有救回它 |
| terminal ConditionalSIGReg | **保留近门诊断，不晋级** | 旧 1,024-step checkpoint 的 classification=`0.801`，但相对 4,096-step matched native 的差为 `-2.15pp [-5.47,1.17]`，regret 和执行距离更差。因训练 release/预算不匹配，这不是严格方法归因，但已不足以支持优先重训 |
| function/action-function anchor | **性能上界** | 直接 ICL 与 retention 较强，但 frozen teacher/额外约束不符合最终简洁目标 |
| VISReg、stop-gradient+SIGReg、Motion PCJR/CCRM、continuous transition basis | **维持否决** | 直接 history-conditioned response 本身失败，改变 CEM comparator 无法救回 |

这一重评还澄清了一个理论层级。DynamicsResponseSIGReg 虽先用 matched pairs 构造 response，
随后仍把不同 query 的 target/prediction contrasts 放入总体分布统计；它没有逐 query 强制
`\hat r_i` 对齐 `r_i`。这解释了它能解决方向较一致的 Action Strength，却在 query-dependent
Motion response 上失败。COJA 的增量不是“再加一个更强 SIGReg”，而是从 response marginal
matching 转为 matched `(Q,A)` 下的逐实例 conditional correspondence。

Action Strength 的补充复评使用完全相同的 `256` 对条件、CEM seed、物理 oracle 与真实执行。
source、matched native、target-JTCov、terminal ConditionalSIGReg 和 DynamicsResponseSIGReg 的
mode classification 分别为 `0.488/0.822/0.734/0.801/0.975`。其中 native、JTCov 与
DynamicsResponse 使用相同 4,096-step 训练 recipe，可作方法归因；terminal checkpoint 来自旧
1,024-step release，只作 checkpoint screen。这个结果真正救回的是 DynamicsResponseSIGReg，
而不是所有曾在旧硬门附近的候选。

### 5.6 Motion rollout 一致性：从一步正例到多步规划

我们从同一个已完成的一步四动作 COJA checkpoint 出发，给所有 continuation arm 使用相同的
`50/50` 数据、batch stream、optimizer 和 1,024 个额外训练步。匹配对照仍只优化一步；
其余 arm 只重新分配既有 hidden native MSE 和 relation 权重。所有 checkpoint 的参数量与部署
调用完全相同。

| continuation objective | 1-step 物理误差↓ | 2-step 物理误差↓ | 3-step 物理误差↓ |
|---|---:|---:|---:|
| 匹配的一步训练对照 | `22.86` | `102.37` | `111.45` |
| rollout2 relation only (`0.5`) | `23.18` | `86.17` | — |
| rollout2 native MSE (`ρ=0.25`) | `25.90` | `58.54` | `80.53` |
| rollout2 native MSE (`ρ=0.50`) | `29.56` | `44.40` | `71.21` |
| rollout2 native MSE + relation (`0.5/0.5`) | `32.67` | `43.18` | `69.69` |

三个 horizon 的物理 oracle 分别留下 `34/156/134` 个可辨识 pair，因此只在同一 horizon 内作
paired 比较，不把原始距离跨列平均。折中候选 `ρ=0.25` 相对匹配对照的 1/2/3-step 物理误差
改善分别为 `-3.04 [-4.21,-1.81]`、`+43.83 [39.57,48.01]` 和
`+30.92 [25.32,36.55]` px。负号表示一步略差；两步和三步区间为 paired-bootstrap 95%
interval。该候选只训练到第二步，却在第三步仍显著改善，排除了“只记住第二个 target”的简单
解释。

正确历史的价值也在 rollout 后出现。对 `ρ=0.25`，交换历史相对正确历史使两步/三步物理误差
增加 `1.88/0.70 px`；匹配对照对应为 `+0.63/-0.89 px`。更大的 `ρ=0.50` 把两步/三步收益进一步
提高，但一步损失也更大。这说明当前主要机制是可解释的短期—长程权衡，而不是某个随机点数：
更多 self-rollout 监督使 Predictor 在自己生成的状态上更稳定，同时减少了真实一步状态上的监督。

因果拆分还表明，第二步 native MSE 是主要活性成分。relation-only 虽把两步绝对误差改善约
`16.20 px`，但交换历史效应为负；native-MSE-only 改善约 `57.97 px`，并给出正的历史收益。
在其上再加第二步 relation 只带来较小的绝对增益。故当前最小方法不是“每个 horizon 都再加一套
COJA”，而是**一步 COJA 建立条件对应，短自回归 native MSE 负责把该对应传递到规划 horizon**。

标准 PushT 保留性只作独立副作用检查。同一 100 个 episode 上，一步训练对照、`ρ=0.50` 和
`ρ=0.25` 分别成功 `57/100`、`53/100` 和 `60/100`；折中候选相对 placebo 的配对差为
`+3pp [-6,+12]pp`。点估计有利但区间很宽，合理结论是没有检测到 retention 损伤，而不是方法已
提高标准 PushT。与旧的 source-only 比较不同，这里模型共享相同数据、起点、额外步数和评测
episode，可分离 rollout objective 的方法效应。

### 5.7 Contact rollout 一致性迁移：action support 是 Pareto 关键

跨任务实验不在 Contact 上重新选权重。所有 continuation arm 都从同一个 4,096-step 一步 COJA
checkpoint 出发，继续训练 1,024 steps；共享 seed、`50/50` 数据、optimizer、batch streams 和
一步 COJA。对照把 hidden native MSE 权重 `0.5` 全用于一步；两个 RC arm 都固定采用 Motion 的
`ρ=0.25`，即一步/两步权重 `0.375/0.125`。它们唯一的区别是第二个 action block：

- `repeated-action RC` 再执行一次已发布 query action；
- `empirical-action RC` 从原始 PushT 训练总体的每个 episode 确定性抽取一个连续五步 block。

每个 block 在 low/high friction 下完全相同；8,192 个模板全部保留，不按 future、contact、模型
输出或 hidden label 筛选。两者都不增加参数、模块、teacher 或推理计算。direct one-step 也
近乎重合：control/repeated/empirical 的 future=`0.781/0.779/0.783`、
history=`0.854/0.846/0.850`、switch 均为 `1.000`、NRE=`0.617/0.619/0.618`。

| horizon | oracle 可辨识 pair | control 误差↓ | repeated RC 误差↓ | empirical RC 误差↓ | empirical 相对 control 改善 | empirical 历史收益相对 control 的增加 |
|---|---:|---:|---:|---:|---:|---:|
| 2（训练） | `256/256` | `29.31` | `24.84` | **`24.92`** | **`4.39 [2.93,5.90]`** | **`1.26 [0.67,1.87]`** |
| 5（未训练） | `244/256` | `93.41` | `91.87` | **`89.04`** | **`4.37 [1.87,6.84]`** | **`2.58 [0.86,4.34]`** |

empirical 与 repeated 在 h2 绝对误差上不可区分（差 `-0.08 [-1.26,1.04] px`）；在 h5 上
empirical 点估计再改善 `2.82 px`，区间 `[-0.79,6.49]`。repeated arm 的 correct-vs-swapped
历史效应更大，但 empirical arm 相对 control 在 h2/h5 仍分别增加
`1.26 [0.67,1.87]` 和 `2.58 [0.86,4.34] px`。因此经验动作没有靠忽略历史换取较低绝对误差；
它同时保留了正的 history-use 因果效应。

标准无隐藏摩擦 PushT 给出决定性的单因素结果：

| arm | 300 matched queries | 相对 control 的配对效应 |
|---|---:|---:|
| one-step continuation control | `212/300` | — |
| repeated-action RC | `197/300` | `-5.00 [-9.33,-0.67]pp` |
| **empirical-action RC** | **`216/300`** | **`+1.33 [-3.33,+6.00]pp`** |

empirical 相对 repeated 为 `+6.33 [1.67,11.00]pp`。因为三臂只改变第二步 action support，
这完成了因果定位：重复 action 让 hidden rollout 监督覆盖过窄，进而损害普通多动作规划；RC
原则和 `ρ=0.25` 本身不是该损伤的充分原因。经验动作版本在单 training seed 上同时保持 direct
ICL、改善 h2 与未训练 h5 hidden planning，并未检测到 standard retention 损伤，是当前第一个
Contact Pareto 正例。其局限仍是 training-only conditional overlap 和短 trajectory continuation，
而不是模型参数或部署复杂度。

为检验上述结果是否依赖“先训练一步 COJA、再续训 RC”的阶段安排，我们又从公开初始化直接训练
4,096 steps。三臂分别是公开原始数据参考、同 `50/50` mixture 的一步 COJA，以及同 mixture 的
empirical-action RC-COJA；后两者共享初始化、预算、一步辅助项和评测 queries。

| horizon | 公开原始数据参考误差↓ | 一步 COJA↓ | 单阶段 RC-COJA↓ | RC 相对一步 COJA改善 | RC 历史收益相对一步 COJA的增加 |
|---|---:|---:|---:|---:|---:|
| 2（训练） | `25.99` | `28.45` | **`25.29`** | **`3.16 [1.58,4.79]`** | **`1.89 [1.28,2.52]`** |
| 5（未训练） | `94.73` | `89.82` | **`86.72`** | **`3.09 [0.65,5.46]`** | **`3.61 [1.87,5.40]`** |

单阶段 RC 的 direct future/history/switch/worst=`0.775/0.844/1.000/0.738`，一步 COJA 为
`0.771/0.850/1.000/0.734`；NRE=`0.619/0.617`，没有可辨认的一步校准变化。标准 PushT 的
公开参考/一步 COJA/RC=`237/206/207`（300 个完全相同的 queries）。真正的方法对比
RC−一步 COJA 为 `+0.33 [-4.33,+5.00]pp`，且 discordant pairs 为 `25/24`；因此没有检测到
RC retention 代价。RC−公开参考为 `-10.00 [-15.00,-5.00]pp`，但一步 COJA−公开参考已经是
`-10.33 [-15.33,-5.67]pp`。这严格区分了两个 estimand：RC 改善 rollout 的增量效应，与混合
数据/一步适配相对原始训练的共同代价。完整单阶段结果说明两阶段 schedule 不是必要组成。

冻结配方的独立 training seed `13314` 只改变 seed 与输出目录，其余训练和评测协议保持不变：

| training seed | direct future RC−COJA | direct NRE RC−COJA | h2 物理改善 | h5 未训练时域物理改善 | standard CEM300 RC−COJA |
|---:|---:|---:|---:|---:|---:|
| `13313` | `+0.20pp` | `+0.00046` | `+3.16 [1.58,4.79] px` | `+3.09 [0.65,5.46] px` | `+0.33 [-4.33,+5.00]pp` |
| `13314` | `-0.78pp` | `-0.00027` | `+3.67 [2.06,5.34] px` | `+4.37 [1.76,6.92] px` | `+2.00 [-2.33,+6.33]pp` |

seed `13314` 的 h2/h5 正确历史收益 DID 分别为
`+2.14 [1.53,2.78]` 与 `+3.73 [1.87,5.66] px`，h1 方法效应为
`+0.08 [-0.27,+0.42] px`。两个训练 seed 的 direct future/history 平均变化为
`-0.29/-0.59pp`，NRE/gain 平均变化为 `+0.00010/+0.00230`，均远小于长程规划效应。标准 CEM
使用同一 `da974c821e3f…` query catalog；按 training seed 分层、再在 seed 内重采样 paired
queries，方法效应均值为 `+1.17pp`，95% interval `[-2.17,+4.50]pp`。因此当前可重复的机制是：
RC 保留一步条件映射，却显著提高正确历史在多步规划中的价值；它不是用一步 ICL 或原任务规划
退化换来的假改善。

### 5.8 Motion 单阶段闭环：部署相关 action support，而非无条件多样性

Contact 表明经验动作能修复重复动作造成的 retention 损伤，但这不意味着“动作越多样越好”。
为作单因素检验，我们在 Motion 中保持起点、`ρ=0.25`、loss、seed、batch stream 和 1,024-step
预算不变，只把与评测动作族一致的 zero hold 换成从普通 PushT replay 无条件抽取的五步 action
block。该 empirical arm 相对一步 placebo 在 h2/h3 仍改善 `21.17 [17.80,24.64]` 与
`13.28 [9.08,17.99] px`，说明自回归 native MSE 仍是活性成分；但它相对 zero hold 分别差
`23.36 [20.33,26.40]` 与 `16.70 [12.28,21.11] px`，而且 correct-vs-swapped history benefit
从 `+1.93/+0.63 px` 翻成 `-0.64/-1.95 px`。

因此，普适原则不是最大化无条件 action diversity，而是让 rollout 监督覆盖**与 query 和部署
相关的 action support**。Motion 的 zero hold 正好属于其隐藏规划评测动作族；Contact 的
多动作 planner 则需要普通轨迹中的连续 action block。该负对照阻止我们把某个任务上的数据实现
误写成统一采样公式。

随后删除两阶段 continuation，从公开 PushT 初始化直接执行完整 4,096-step RC-COJA。三臂共享
相同初始化、`50/50` 数据、optimizer、预算和评测 query：matched no-aux、一步 COJA，以及
RC-COJA。保存模型仍是原 LeWM state dict；第二步只重用现有 Predictor 和 native MSE。

| horizon | matched no-aux↓ | 一步 COJA↓ | 单阶段 RC-COJA↓ | RC 相对一步 COJA改善 | RC 历史收益 DID |
|---|---:|---:|---:|---:|---:|
| h1 | `23.28` | **`23.23`** | `26.24` | `-3.01 [-4.13,-1.86]` | `-0.24 [-0.46,-0.03]` |
| h2（训练） | `100.20` | `103.17` | **`45.78`** | **`57.39 [52.19,62.42]`** | **`4.12 [2.90,5.34]`** |
| h3（未训练） | `106.23` | `108.34` | **`69.26`** | **`39.08 [33.90,44.21]`** | **`2.28 [0.28,4.01]`** |

这里 h1/h2/h3 的物理 oracle 分别保留 `34/156/134` 个可辨识 pair，故只作列内配对比较。
RC 的 direct future/history/switch/worst=`0.553/0.617/0.938/0.258`。h2/h3 的绝对误差和
history intervention 同时改善，说明收益不是靠忽略历史取得；h1 则保留一个明确的短期代价。

独立 training seed `14322` 在完全相同的 31 点 action grid 上复现如下：

| horizon | seed14321 RC−COJA 改善 | seed14322 RC−COJA 改善 | seed14322 历史收益 DID |
|---|---:|---:|---:|
| h1 | `-3.01 [-4.13,-1.86]` | `-3.30 [-4.53,-2.14]` | `-0.06 [-0.28,+0.21]` |
| h2（训练） | `+57.39 [52.19,62.42]` | `+57.60 [52.55,62.55]` | `+4.24 [3.16,5.37]` |
| h3（未训练） | `+39.08 [33.90,44.21]` | `+41.56 [36.02,47.02]` | `+2.91 [0.90,4.95]` |

两个 seed 的 h2/h3 点估计差异远小于各自主效应，h1 代价也同向且同量级。这把结论从发现期
单 seed 趋势升级为冻结配方复现；它仍不把两个 seed 的 query 简单 pooling 成一个伪大样本。

标准无隐藏 damping 的 300-query retention 结果为：

| arm | successes | paired effect |
|---|---:|---:|
| matched no-aux | `203/300` | — |
| one-step COJA | `188/300` | `-5.00 [-11.00,+0.67]pp` vs no-aux |
| RC-COJA | `194/300` | `+2.00 [-2.67,+6.67]pp` vs COJA；`-3.00 [-8.67,+2.67]pp` vs no-aux |

所有臂使用完全相同的 query catalog。区间不支持用 2–3pp 点差宣称严格优劣；它支持的结论是：
RC 相对一步 COJA 没有检测到独立的大幅 retention 代价，而 no-aux 到 COJA 的共享变化必须与
rollout 方法效应分开。结合 Contact 完整单阶段结果，两阶段 schedule 和 continuation-only
optimizer trajectory 均已被排除为必要机制。

seed `14322` 的 matched RC/COJA 为 `192/196`，即 `-1.33 [-6.67,+4.00]pp`；与 seed `14321`
的 `+2.00pp` 方向相反但幅度都小。按 training seed 分层、再在 seed 内重采样 300 个 paired
queries，两个 seed 的层级均值为 `+0.33pp`，50%/80%/95% 区间分别为
`[-1.17,+1.83]`、`[-2.50,+3.17]`、`[-4.00,+4.50]pp`。这正是不能用单个 CEM 点数硬门裁决
候选的实证例子。

## 6. 关键消融与被排除的解释

| 替代解释 | 关键实验 | 结论 |
|---|---|---|
| target 完全坍缩导致失败 | stop-gradient、VISReg、target pair separation | target 可分仍可能完全忽略历史 |
| Encoder 丢失全部历史信息 | cross-fitted history probes | 部分信息可读，但 Predictor 不一定使用 |
| 任意 pair/contrast 都有效 | Contact shifted-pair control | 错误 `(Q,A)` 配对退回近 native；需要条件对应 |
| 直接拟合 pair center 可补全 absolute future | exact-center control | center 项数值占优并压低 gain；不能与 response 同权直接加入 |
| Motion 只需训练更久 | 2,048→8,192 train–Development 轨迹 | in-sample 大幅改善而 held-out 恶化，训练更久会加重记忆 |
| 输出 head 是主要故障 | Predictor/pred_proj module swap | 效应与误差主要位于 Predictor trunk |
| NRE 改善会自动恢复 CEM | full-release Motion | NRE 降到 `0.767`，CEM 仍下降约 `6–8pp` |
| 标准 PushT CEM 下降说明 ICL 对规划无用 | Contact 三臂 hidden CEM | COJA 相对 matched native 同时改善物理误差、regret 和正确历史收益 |
| 旧硬门失败都等于方法失败 | 历史 checkpoint 重评 | 近门比例与错误 source comparator 会误伤；直接 response 失败则不会被重评救回 |
| 单步 COJA 学会 history response 后可直接用于多步规划 | Motion 1/2/3-step hidden planning | 一步 switch 高，但两步 history benefit 可为负；单步可辨识性不自动通过自回归复合 |
| 每个 rollout horizon 都必须增加配对 relation | rollout2 MSE/relation 因子拆分 | 第二步 native MSE 是主要活性成分；第二步 relation 不是当前最小解所必需 |
| rollout2 只记住第二步终点 | 未训练的 3-step hidden planning | 两步训练在三步仍显著改善，支持可迁移的 rollout 校准 |
| 标准 CEM 点数下降可直接否决 rollout 方法 | 同数据、同预算、同 episode 的一步训练对照 | `ρ=0.25` 为 `60/100` 对 `57/100`，区间跨零；此前需区分数据效应、方法效应与 hidden planning |
| RC 只是 Motion 特例 | 固定 `ρ=0.25` 的 Contact h2/h5 迁移 | h2 绝对误差与 h2/h5 正确历史收益均改善，支持跨任务 rollout 机制 |
| Contact retention 差只是 50/50 数据或 CEM 波动 | 同数据 continuation control、300 matched standard queries | RC 比 control 低 `5.0 [0.67,9.33]pp`；这是当前重复动作 RC 的真实方法代价 |
| 重复动作与普通 action support 无关 | 只替换 h2 action 的 empirical/repeated 因果对照 | empirical 保留 h2/h5 收益，并将 CEM300 从 `197` 恢复到 `216`；action support 是 Pareto 关键 |
| RC 只在先练好 COJA 后续训才有效 | 从公开初始化单阶段训练 4,096 steps | RC 相对一步 COJA 在 h2/h5 改善 `3.16/3.09 px`，标准 CEM 为 `207/206`；两阶段 schedule 非必要 |
| RC 导致相对公开参考的约 `10pp` 标准 CEM 差距 | 完整三臂的同数据方法对照 | 一步 COJA 已为 `206/300`，RC 为 `207/300`；该差距来自共享 mixture/适配路径，不是 RC 增量 |
| continuation action 越多样越好 | Motion zero-hold 与无条件 replay block 单因素对照 | 无条件 replay 仍优于 placebo，但显著弱于部署相关 zero hold，并把正确历史收益翻负；关键是 support match，不是最大多样性 |
| Motion 结果依赖 continuation warm start | 从公开初始化单阶段 4,096-step 三臂 | RC 相对一步 COJA的 h2/h3 改善为 `57.39/39.08 px`，两阶段 schedule 非必要 |
| Motion RC 必然损害标准规划 | 同 query 的 no-aux/COJA/RC CEM300 | RC−COJA=`+2.00 [-2.67,+6.67]pp`；未识别出 RC 特异损伤，也未证明提升 |

## 7. 统计呈现原则

本研究不再用一个统一 suite 硬指标决定方法是否有效。

1. **任务内效应**：对 matched queries 报告候选减对照的点估计、50%/80%/95% paired
   intervals 和逐条件结果。
2. **训练 seed**：发现阶段先跑一个完整 seed；方法固定后，再把 training seed 作为独立层级，
   不把多个 seed 的样本简单拼接成一个更大的 query 集。
3. **suite 汇总**：每个任务权重相等。层级 bootstrap 先在任务内重采样 query 和 evaluator
   seed，再对任务级 effect 取均值与中位数；报告分布和最差任务尾部，不要求每项跨过同一个阈值。
4. **能力分轴**：assignment、NRE 和 CEM 分别呈现。suite 平均提升不能掩盖某个任务
   `NRE>1`，CEM 提升也不能代替历史条件响应。
5. **实际差异参考**：同时报告 `P(delta>=0)`、`P(delta>=-2pp)` 和
   `P(delta>=-5pp)` 的 bootstrap 重采样频率，并明确它们不是 Bayesian posterior probability。
6. **规划三效应**：分别报告 `native-source`、`candidate-native` 和 `candidate-source`；原任务
   retention 与 ContextWorld hidden planning 不互相替代。hidden planning 还必须报告同 checkpoint
   的 correct-vs-swapped history effect。

## 8. 当前边界与下一问题

### 8.1 已建立的结论

- latent 边缘非坍缩不保证 history-conditioned future identifiability；
- 相同可见 `(Q,A)` 下的联合条件关系是有效训练信号；
- 不增加模型参数或推理结构，原 LeWM 可以学习离散和连续隐藏动力学响应；
- Motion 的旧 NRE 失败主要来自有限 query 模板的总体泛化问题；
- 单步条件可辨识性不保证在 self-conditioned rollout 中保持；
- 一步 COJA 加真实自回归 native MSE 能大幅修复两步规划，并迁移到未训练的第三步；
- 固定 `ρ=0.25` 的同一原则在 Contact 两步和未训练五步上增强正确历史的规划价值，排除
  Motion-specific 解释；
- Contact 单因素对照确认 continuation action support 决定 retention：empirical-action RC 保留
  h2/h5 收益，并把 standard CEM300 从 repeated 的 `197` 恢复到 `216`，matched control 为 `212`；
- Contact 从公开初始化的单阶段 4,096-step 训练复现了 RC 的 h2/h5 增益，排除了必须先训练 COJA
  checkpoint 再续训的解释；
- 在完整训练中，RC 相对同 mixture 一步 COJA 的标准 CEM 为 `207/300` 对 `206/300`，说明 RC
  增量没有检测到 retention 代价；公开原始数据参考 `237/300` 的差距属于共同训练路径；
- Motion 从公开初始化的单阶段 4,096-step 训练把 h2/h3 相对一步 COJA 改善
  `57.39/39.08 px`，同时正确历史收益的 DID 为正，排除了 continuation warm start；
- 冻结配方的独立 seed 将 h2/h3 改善复现为 `57.60/41.56 px`，h1 代价也复现为约 `3.3 px`；
- Motion 的无条件 replay-action 负对照说明 action diversity 不是单调有益，rollout target 必须
  落在 query/deployment-relevant support；
- Motion 同 query CEM300 中 RC/COJA/no-aux=`194/188/203`，RC−COJA 区间跨零，未检测到
  RC 特异 retention 损伤；
- 第二个 training seed 的 RC−COJA CEM300 为 `-1.33 [-6.67,+4.00]pp`；两 seed 层级均值
  `+0.33 [-4.00,+4.50]pp`，标准 retention 没有稳定方向；
- 第二步 COJA/relation 不是当前主要活性成分，最小桥梁是现有 MSE 的 horizon 重分配；
- 条件响应、隐藏动力学规划和标准环境原任务保持是三个独立 estimand；
- Contact 上，COJA 的直接 ICL 已转化为相对 matched native 的隐藏动力学规划收益；
- 旧 DynamicsResponseSIGReg 是有效的 Action Strength 正例，但其 Motion 失败说明 response
  marginal matching 不等于逐 query conditional alignment。
- Action Strength 同场复评确认 DynamicsResponseSIGReg 的 hidden-planning 方法效应；同时否定
  了“target-JTCov 只是被 source comparator 误伤”的解释。

### 8.2 尚未建立的结论

- COJA 尚未在普通 unmatched offline replay 上删除 conditional-overlap 数据假设；
- RC-COJA 仍需要真实 trajectory continuation；它不增加参数，但不是“无需组织数据”；
- Motion 单阶段候选在较弱的一步规划子集上仍比匹配的一步 COJA 差
  `3.01 [1.86,4.13] px`；
- Motion 标准 PushT 的 RC−COJA=`+2.00 [-2.67,+6.67]pp`，区间仍不足以证明提升或严格等价；
- Contact h2 上 RC 相对公开原始数据参考的绝对规划改善区间仍跨零，尽管 h5 已明确更好；
- 同 mixture 的一步 COJA 与 RC 相对公开原始数据参考仍有约 `10pp` 标准 PushT 差距；这不是 RC
  方法效应，但仍是训练数据/适配 recipe 需要单独解决的 suite-level 代价；
- rollout-consistency 已在 Motion 和 Contact 各两个 training seed 上得到机制正例，但尚未扩展
  到 ActionDelay、Portal 或非 PushT 动力学域；两个 seed 仍不足以精确估计训练方差；
- Contact h1 的完整单阶段 horizon 对照目前只有 seed `13314`；
- continuation 仍需真实短轨迹，而且不同任务需要与 query/deployment 相关的 action support；
  Motion 已否定从无条件 replay marginal 随意抽动作可作为普适规则；
- Portal 的最终方法仍缺多 training-seed 结果；
- 现有结果不支持“所有 marginal regularizer 都无用”或“PLDM 整体弱于 LeWM”的宽泛主张。

### 8.3 下一项最小实验

Motion 的 action-support 资格检验和从公开初始化的单阶段 4,096-step 训练均已完成。无条件 replay
action 弱于部署相关 zero hold，但单阶段 zero-hold RC 在 h2/h3 给出强正效应，且 matched CEM300
没有检测到 RC 特异损伤。因此发现阶段现在停止修改 `ρ`、schedule、模型、loss 和 sampler。

Motion 与 Contact 的首个独立 seed 均已完成：两个任务的长程方法效应同向复现，direct response
与 matched standard retention 没有出现 RC 特异的大幅代价。发现期因此结束，下一步转入冻结的
suite-level 完整验证：保持任务各自已限定的 deployment-support action target，统一报告 direct
response、correct-vs-swapped hidden planning、同训练数据的 COJA 对照和标准原任务 retention；
任务与 training seed 作为独立层级，不用单一比例硬门裁决。Motion 的 h1 代价作为真实 horizon
tradeoff 保留，不重新开启候选搜索。

不再扩展 margin、SIGReg、VISReg、额外 horizon relation、encoder/adapter 或梯度投影家族。
当前方法仍需解决的是怎样从普通 offline trajectory 自动形成 conditional overlap，而不是模型或
loss 复杂度。

## 9. 结论

ContextWorld 揭示的不是普通 representation collapse，而是**条件联合关系缺失**：模型可以拥有
健康的 latent marginal，却没有学会“这段历史、这个动作，应对应哪个未来”。COJA 用相同
`(Q,A)` 下的历史干预直接打破这一置换对称性，并在不改变 LeWM 部署结构的前提下显著改善多个
隐藏动力学任务。

最新结果把方法又推进了一步。COJA 解决第一步的 conditional correspondence；RC-COJA 用同一个
Predictor 的短自回归 native MSE 缩小部署时的 rollout mismatch。固定配方已在 Motion 与 Contact
两个连续隐藏动力学任务上增强多步正确历史的规划价值，排除了单任务偶然性。Contact 的单因素
对照进一步证明，最初 `-5pp` retention 代价来自重复 action 的窄 support；经验动作版本在 h2/h5
保持正效应，并以 `216/300` 对 matched control `212/300` 消除了可辨认损伤。Motion 的单阶段
复核又在 h2/h3 给出 `57.39/39.08 px` 的大幅改善，独立 seed 复现为
`57.60/41.56 px`；h1 代价也稳定在 `3.01/3.30 px`。Contact 的 h2/h5 则从
`3.16/3.09 px` 复现为 `3.67/4.37 px`，未检测到 h1 损伤。Motion 两 seed 标准 CEM 方法效应的
层级均值为 `+0.33 [-4.00,+4.50]pp`，Contact 为 `+1.17 [-2.17,+4.50]pp`。当前最合理的结论是：
**条件重叠联合对齐是有效的单步条件可辨识方法，deployment-support-matched 的短自回归一致性
是把它转化为规划的可迁移机制；它不增加模型或部署复杂度，已在 Motion 与 Contact 的公开初始化
单阶段完整训练中得到双 seed 复现，但显式 conditional overlap、短轨迹数据、horizon 权衡、
有限 seed 与共享混合训练的原任务代价仍是边界。**

## 主要机器可读证据

- [Motion 旧 8,192-step 校准分解](artifacts/pusht_motion_damping_full8192_response_calibration_v1/analysis.json)
- [Motion 旧 2,048-step 校准分解](artifacts/pusht_motion_damping_full8192_response_calibration_v1/step2048_analysis.json)
- [Motion Predictor module swap](artifacts/pusht_motion_damping_full8192_module_swap_v1/module_swap_v2.json)
- [Motion full-release COJA Development](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_step8192_v1/s14321_step8192_v1/development_response_analysis_v1.json)
- [Motion full-release matched no-aux Development](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_native_control_step8192_v1/s14321_step8192_v1/development_response_analysis_v1.json)
- [Motion multi-catalog CEM continuous effect](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_cem_seeds42_43_44_n100_runtimefix_v2/continuous_paired_effect_v1.json)
- [Motion seed42×300 CEM continuous effect](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_cem300_seed42_current_runtime_v1/continuous_paired_effect_v1.json)
- [Contact Friction transfer summary](artifacts/pusht_contact_friction_visible_joint_transfer_v1/summary.json)
- [Contact source/native/COJA hidden-dynamics CEM](artifacts/pusht_contact_friction_hidden_cem_h5_three_arm_development256_cpu_v1/summary.json)
- [历史候选 comparator 重评](artifacts/conditional_joint_comparator_validity_v2/summary.json)
- [Action Strength DynamicsResponseSIGReg 三 seed 结果](../../../ContextWorld/configs/benchmark/pusht_hidden_actuation_replay_matched_results_v2.yaml)
- [Action Strength 历史候选同场 hidden-planning 复评](artifacts/historical_candidate_reevaluation_v1/summary.json)
- [target-JTCov 任务广度与隐藏规划结果](results/joint_temporal_covariance_sigreg_task_breadth_summary_v1.json)
- [VISReg ActionDelay summary](artifacts/visreg_action_delay_discovery_v1/summary.json)
- [Motion rollout-consistency 因果拆分与折中候选汇总](artifacts/pusht_motion_damping_rollout_consistency_mve_v1/summary.json)
- [Motion 单阶段 RC-COJA h1 hidden planning](artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected34_blocks1_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json)
- [Motion 单阶段 RC-COJA h2 hidden planning](artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected156_blocks2_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json)
- [Motion 单阶段 RC-COJA h3 hidden planning](artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected134_blocks3_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json)
- [Motion 单阶段 standard CEM300 配对统计](artifacts/pusht_motion_damping_full4096_standard_cem300_paired_v1/paired_analysis_v1.json)
- [Motion RC-COJA 两 training-seed 复现汇总](artifacts/pusht_motion_damping_rc_coja_full4096_replication_v1/replication_summary_v1.json)
- [Contact rollout-consistency 跨任务汇总](artifacts/pusht_contact_friction_rollout_consistency_transfer_v1/summary.json)
- [Contact h2 hidden-dynamics CEM](artifacts/pusht_contact_friction_rollout_consistent_hidden_cem_h2_dev256_v1/summary.json)
- [Contact h5 hidden-dynamics CEM](artifacts/pusht_contact_friction_rollout_consistent_hidden_cem_h5_dev256_v1/summary.json)
- [Contact standard PushT CEM100](artifacts/pusht_contact_friction_rollout_consistent_standard_cem100_v1/aggregate.json)
- [Contact empirical-action h2 hidden-dynamics CEM](artifacts/pusht_contact_friction_empirical_action_hidden_cem_h2_dev256_v1/summary.json)
- [Contact empirical-action h5 hidden-dynamics CEM](artifacts/pusht_contact_friction_empirical_action_hidden_cem_h5_dev256_v1/summary.json)
- [Contact empirical-action standard CEM300](artifacts/pusht_contact_friction_empirical_action_standard_cem300_v1/aggregate.json)
- [Contact 完整单阶段 h2 hidden-dynamics CEM](artifacts/pusht_contact_friction_empirical_action_rc_full4096_hidden_cem_h2_dev256_v1/summary.json)
- [Contact 完整单阶段 h5 hidden-dynamics CEM](artifacts/pusht_contact_friction_empirical_action_rc_full4096_hidden_cem_h5_dev256_v1/summary.json)
- [Contact 完整单阶段 RC standard CEM300](artifacts/pusht_contact_friction_empirical_action_rc_full4096_standard_cem300_v1/aggregate.json)
- [Contact 一步 COJA standard CEM300](artifacts/pusht_contact_friction_coja4096_standard_cem300_v1/aggregate.json)
- [Contact 公开原始数据参考 standard CEM300](artifacts/pusht_contact_friction_native4096_standard_cem300_v1/aggregate.json)
- [Contact 完整单阶段 response decomposition](artifacts/pusht_contact_friction_empirical_action_rc_full4096_center_response_vs_coja_v1.json)
- [Contact RC-COJA 两 training-seed 复现汇总](artifacts/pusht_contact_friction_rc_coja_full4096_replication_v1/replication_summary_v1.json)
- [rollout2 真实 target builder](scripts/build_pusht_motion_damping_planner_curve_rollout2_targets_v1.py)
- [RC-COJA continuation 实现](scripts/run_pusht_motion_damping_planner_curve_rollout_consistent_continuation_v1.py)
- [Motion 无条件 empirical-action target builder](scripts/build_pusht_motion_damping_planner_curve_rollout2_empirical_action_targets_v1.py)
- [Motion empirical-action RC continuation](scripts/run_pusht_motion_damping_planner_curve_rollout_consistent_empirical_action_continuation_v1.py)
- [Motion 单阶段 RC-COJA 实现](scripts/run_pusht_motion_damping_rollout_consistent_zero_hold_full4096_v1.py)
- [Motion 冻结配方独立 seed runner](scripts/run_pusht_motion_damping_rc_coja_full4096_replication_v1.py)
- [Contact rollout2 target builder](scripts/build_pusht_contact_friction_rollout2_targets_v1.py)
- [Contact RC-COJA continuation](scripts/run_pusht_contact_friction_rollout_consistent_continuation_v1.py)
- [Contact empirical-action target builder](scripts/build_pusht_contact_friction_rollout2_empirical_action_targets_v1.py)
- [Contact empirical-action RC continuation](scripts/run_pusht_contact_friction_rollout_consistent_empirical_action_continuation_v1.py)
- [Contact 冻结配方独立 seed runner](scripts/run_pusht_contact_friction_rc_coja_full4096_replication_v1.py)
