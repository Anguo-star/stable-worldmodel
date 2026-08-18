# 条件动力学 ICL：从边缘非坍缩到条件联合响应

> 阶段状态（2026-08-17）：ActionDelay 上的 **predictor-only PCJA** 已由同一个
> 1,024-step checkpoint 通过冻结 Private Development、正式 Public Test 和原任务 CEM
> 非劣门；它仍是当前第一个兼顾该任务 ICL 与规划保持的简洁条件关系正例。Motion Damping
> 上，`K=4` exact-native bridge 已通过并消费了唯一授权的 seed `14321`、256-step replay。
> 相对严格 deterministic worker-0 parent，K4 的 future/history 各增加 `6/512`，其中 future
> 的 paired-bootstrap 95% CI 为 `[+1,+12]/512`，说明聚合不是完全无效；但 switch 减少
> `10/256`，alignment 从 `0.480` 降至 `0.399`，normalized response error 从 `0.787`
> 升至 `0.843`。因此它没有通过 Development，也没有授权续训。
> 随后的零训练 `2×2×2` 模块互换把小 assignment 收益和主要方向损伤都定位到
> **Predictor trunk**，而不是 Encoder/target path、`pred_proj` 或模块拼接失配。当前最强
> 机制判断是：persistent PCJR 确实增强了历史响应，但主要沿错误的 held-out target-response
> 方向放大；配对/PCJR 方法族没有被否决。下一步只执行 Predictor 梯度与冻结 Development
> estimand 的零训练对齐审计，区分训练 panel 覆盖/相消、native MSE 竞争和 objective
> 方向错位；不增加 loss、不切换新候选。ContextWorld suite v2 additive integrity reseal 已
> 正式通过；Public、CEM、Contact 与额外 seed 仍关闭。

## 摘要

本研究关注世界模型能否从动作—观测历史中识别 episode 内隐藏动力学，并据此预测同一
当前状态、同一 query action 下的不同未来：

\[
p(O^+\mid H,Q,A).
\]

其中 `H` 是历史，`Q` 是当前观测，`A` 是未来查询动作。目标是学习
**历史、动作与未来的条件联合关系**，不是只让 latent 的边缘分布 `p(Z_t)` 具有方差或
接近某个先验。

迄今证据形成了一条较清楚的因果链：

1. 原生 SIGReg、stop-gradient、JTCov、Encoder-only 以及多种全局/paired 统计能够改变
   latent geometry，却不能稳定保证 Predictor 对历史作出方向正确的响应。
2. ActionDelay 中，native LeWM 在 1,024 步后仍近似三分类随机，而同预算 PLDM 可以学会，
   排除了任务信号或模型规模根本不可学的解释。
3. full-gradient PCJA 能学会 ActionDelay，但相对 native 的 CEM 下降 `13.33` 个百分点，
   说明“创造条件耦合”与“保持原规划能力”必须同时约束。
4. 将同一个 PCJA 辅助项的梯度限制到 `predictor + pred_proj` 后，ActionDelay 的冻结
   Private macro 达到 `0.9309`、正式 Public macro 达到 `0.9452`，累计 900 次配对 CEM
   的差值为 `-2.67` 个百分点，单侧 95% lower 为 `-3.67` 个百分点，通过 `-5` 个百分点
   的非劣界。
5. 预注册的跨任务证伪已经给出反例：Motion Damping 的 symmetric PCJA 在 step `8192`
   达到 `context switch=1.000`，但 `future=0.7793`、`history=0.7129`、`worst=0.6836`，
   三项主门失败。相对旧 endpoint，
   history preference 明显增加，future 与 worst-group accuracy 却明显下降。
6. 四个既有 checkpoint 用同一 256-pair Development scorer 重评分后，旧 ranking 虽有
   `future=0.9746`，其 normalized response error 却为 `378.15`、response gain 为 `9.23`、
   calibrated pair 为 `0%`；离散 target selection 会掩盖极严重的响应失配。
7. PCJA 把 gain 拉回 `1.167`、alignment 提高到 `0.680`、error 降到 `1.614`，但仍没有
   过门。其 error 可分解为 `0.028` 的沿 target 轴幅值误差与 `1.587` 的正交误差，后者占
   `98.3%`。因而当前 PCJA 不是通用修复；最强机制判断是：**条件分配与条件响应校准是
   两个不同问题**，且该 Motion-PCJA candidate 的主要剩余 residual 是方向错误，不是简单
   margin 过大。
8. CCRM 在同一 seed/预算下把 normalized response error 降到 `0.468`、alignment 提高到
   `0.751`，但 response gain 只有 `0.430`；其 common-center coefficient `beta` 的加权均值
   为 `1.690`，导致 `245/256` pairs 单侧越过 faster boundary，只有 `7/256` 同时满足两侧。
9. TA-CCRM 精确加入 `4 beta^2` 后，signed `beta` 加权均值降至 `0.137`、中位数降至
   `0.041`，两侧同时正确增至 `64/256`；但最终仅有 `future=0.6211`、`history=0.6855`、
   `switch=0.9219`、`worst=0.5859`，response alignment/gain 同时退化。
10. 零训练步终点梯度分解排除了公式符号错误：CCRM 与 anchor 的 prediction-space gradient
    cosine 精确为 `0`，但其 Predictor 参数梯度范数为 `3.75` 对 `34.90`；加入 anchor 后
    全局梯度范数从 `1.02` 升至 `3.30`。相反，PCJA 与 CCRM 的参数梯度范数为 `3.08` 对
    `3.75`、cosine 为 `+0.166`，组合后的全局范数仅 `1.08`。
11. 冻结的 `PCJA+CCRM` 组合确实把 CCRM 的中心偏置从 `1.690` 降到 `0.627`，并让
    `94.53%` pairs 的 response 优于完全不响应历史的基线；但 `208/256` pairs 仍满足
    `|beta| >= gain/2`，只有 `48/256` 同时跨过两个 assignment boundary。三项主门分别还差
    `183/512` 个 future decisions、`135/512` 个 history decisions 和 `147/256` 个
    worst-mode pairs，不属于阈值抖动。
12. 组合 checkpoint 的零训练终点审计显示，PCJA 与 CCRM 在 output space 仍为正 cosine
    `+0.315`，映射到共享 Predictor 参数后却变为 `-0.113`；PCJA 梯度范数 `10.06`，CCRM
    为 `3.07`，完整更新方向与 `native+PCJA` 的 cosine 为 `0.959`。因此 `1:1` loss sum
    并没有形成平衡的优化折中；这解释当前 recipe 的失败，但不足以声称所有权重、seed 或
    条件关系目标均不可能。
13. 为检验 target geometry 漂移是否就是 CCRM 的剩余根因，只冻结 Encoder/Projector、
    保留相同 CCRM 的单因素候选完成了正式 8,192 步。中心偏置从 `1.690` 降至 `0.690`，
    但 response gain/alignment 从 `0.430/0.751` 降到 `0.309/0.523`，normalized error 从
    `0.468` 恶化至 `0.731`；Development 明确失败。这否定“冻结表示足够”，不否定条件
    关系目标这一方法族。
14. 同一冻结 checkpoint 的 exact training batch 上，native hidden-terminal MSE 对中心
    偏置给出正确且较强的一阶方向（每单位学习率 `beta^2` 变化 `-0.573`），但对 response
    error 和 assignment margin 的作用分别只有 `-0.0039` 与 `+4.7e-7`；CCRM 强力改善
    response（`-1.813`）却单独恶化中心（`+0.650`）。按固定 `0.09` 加权后的完整方向三项
    都改善且不触发 clipping，说明剩余问题更像**条件信号尺度与跨 batch 泛化/优化效率**，
    不是终点符号错误或简单分量冲突。
15. 最新九任务矩阵显示根因不能再写成单一的 “LeWM failure”：ActionDelay 偏向 PLDM，
    Action Strength、Reacher 与 Cube 偏向当前 LeWM reference，Contact/Motion/Portal 又呈现
    不同的低响应、中心偏置或绝对未来失配。跨任务稳定的陈述只有：全局 non-collapse、
    target separation 和 history readability 均不足以保证正确的 `H,Q,A -> future` 映射。
16. pair-normalized exact-future 虽同时约束 centered response 与公共 center，正式终点却退化到
    `future/history/switch/worst=0.475/0.412/0.320/0.063`，gain=`-0.041`、error=`1.291`；
    “把弱 native future MSE 放大成单一归一化 residual 就足够”已被真实反例否定。
17. squared-hinge BC-CCRM 不再拉动已可行 pair，并在 8,192 步把同训练 batch 的 boundary
    loss 降至初始的 `4.90%`；Development 也单调改善到 `future=0.635`、`history=0.723`、
    `switch=0.973`、error=`0.508`。但只有 `69/256` pairs 同时可行，故该固定 recipe 失败，
    而“配对/CCRM/可行性约束整个家族无效”并未被证伪。
18. 固定 prediction center、只全局缩放 response 的结果级反事实不能使任何已测 endpoint 同时
    通过任务门和 response 门；PLDM Contact/Motion 还分别有 `53/256`、`91/256` 个负投影
    response。因而 PLDM 的失败不只是 under-gain，BC-CCRM 的剩余失败也不只是全局尺度。
19. linear exact penalty 确实改变了优化，但没有修复 Motion：相对 squared 版，switch/gain/
    alignment 略升，future/worst 反而下降，两侧 assignment 同时可行的 Development pair 从
    `69/256` 降到 `57/256`；同一首批训练样本在两个终点都仍有 `22/32` active pair。因此
    “squared hinge 在边界附近梯度衰减是充分根因”已被否定，penalty power 不再搜索。
20. 三个冻结快照的逐 pair 参数梯度审计显示，linear auxiliary 的合成率
    `||sum_i g_i||/sum_i||g_i||` 从初始化 `0.207` 降到终点 `0.173`，终点 `7/32` pair 在全
    参数空间反对 aggregate、Predictor 子空间为 `9/32`，负 signed-share mass 为
    `0.098/0.157`。同一 batch 的 target axis 从初始化到 linear 终点的中位 cosine 只有
    `0.547`、normalized delta 为 `0.862`；但 squared 与 linear 终点 target axis 的中位
    cosine 为 `0.966`。这说明两种 penalty 都进入了相近的移动 target geometry，并在共享
    Predictor 上产生低相干更新；它是新的机制证据，不足以单独宣称因果根因。
21. 成功的 ActionDelay predictor-only PCJA 正对照在终点具有正的 weighted pair-gradient
    cosine=`0.02875` 且负 signed-share mass=`0`，但 target axis 从初始化到终点的中位
    cosine/normalized delta 仍为 `0.599/0.817`，几乎复现 Motion 的 `0.600/0.851`。因此
    endpoint target 漂移不是失败特异现象；hard-freeze 与 EMA 也不能再因该现象直接晋级。
22. 失败 Motion-PCJA 终点的 weighted pair-gradient cosine 仅 `0.00046`，但负 signed-share
    mass 只有 `0.00366`，既不像成功 ActionDelay 的正相干，也不像失败 BC-CCRM 的强负质量。
    更关键的是，binary PCJA 在 `prediction=target` 时 loss 仍为 `0.31326`，沿真实 response
    gain 的导数为 `-0.26894`；相同解析控制下 CCRM 为 loss=`0`、导数=`0`。这证明 PCJA 是
    条件分配目标而非 proper response-calibration loss，能解释“switch 学会但连续 future 未
    校准”，却不能单独解释 Motion 中占 `98.3%` 的正交误差。
23. proper `PCJR-CV` 通过了零训练步驻点、梯度路由与非重复性审计，随后只执行一条
    seed `14321`、1,024-step 确定性机制轨迹；所有带仪器重放在 step `256/512/1024` 的
    model-state SHA 与父轨迹精确一致，排除了审计改变训练的解释。
24. 该轨迹的 response gain/alignment/error 从 `0.166/0.480/0.787` 改善到
    `0.295/0.625/0.633`，但 future/history 仅从 `0.502/0.518` 到 `0.514/0.535`。
    普通 prediction-center MSE 下降 `18.8%`，沿真实 target axis 的 center MSE 却基本不变，
    且 target response energy 下降 `20.7%`；改善主要发生在 assignment 无关的正交子空间。
25. 13 个时点的真实 AdamW 归因显示，Predictor-only 当前 batch 的 12 个非零更新全部降低
    `beta^2`，clip 与 AdamW 没有把局部方向翻转。单锚点 first-order 与 finite-difference 的
    12 个非零符号完全一致，因而“有限步长曲率导致反转”也不是主要解释。
26. 六个从真实 training-loss boundary 绑定、记录 input SHA 的多时点锚点给出跨 batch 反例：
    Predictor-only 当前 diagonal 为 `12/12` 改善，历史格却为 `17/42` 改善、`25/42` 恶化，
    累计 `beta^2` 变化 `+0.281`；live re-encode 的历史累计变化扩大到 `+1.616`。step 1024
    时 Predictor-only 有 `5/6`、live 有 `6/6` 历史锚点被同一更新伤害。
27. MSE–PCJR 分量归因锁定了冲突来源。三个采样点上，两项对当前 diagonal 都是 `3/3`
    改善；对 13 个历史格，native MSE 的 `beta^2` 为 `10` 改善、`3` 恶化、净方向
    `-0.361`，加权 PCJR 为 `2` 改善、`11` 恶化、净方向 `+0.837`。PCJR 对历史 response
    仍为 `9/13` 改善，证明它不是“无效”，而是在共享 Predictor 中呈现 response-corrective、
    target-axis-assignment-antagonistic 的跨 query 分裂。
28. 由此当前只否决固定 Motion `PCJR-CV@0.09` 的“单 batch properness 足以保证连续动力学
    ICL”解释，不否决 pairing、PCJR、SIGReg、LeWM 或 PLDM。最小下一步不是再加 `beta` loss：
    既有 TA-CCRM 已证明强 target-axis 项会产生新的参数空间支配；先对完全相同 PCJR 公式做
    current-only、current+previous 与固定 uniform panel 的零更新梯度矩阵证伪。

因此当前主问题不再是“再设计一个更强的边缘正则”，而是：

> 对相同 query，如何用一个仍然简洁的条件关系目标，使当前 batch 学到的 history-conditioned
> response 能跨 query/batch 保持 target-axis assignment，而不是只在局部 pair 上正确、更新后
> 又伤害此前条件关系？

## 1. 研究对象与证据合同

### 1.1 条件可辨识性不是边缘非坍缩

隐藏动力学任务满足：

\[
p(O^+\mid H_1,Q,A)\ne p(O^+\mid H_2,Q,A),
\]

其中 `Q,A` 固定，只有能够揭示 episode dynamics `M` 的历史不同。模型即使具有健康的
全局 latent 方差，也可能对两段历史输出同一个条件均值。任何只依赖 target latent 经验
边缘分布、且对样本重排不变的正则，都不能单独保证

\[
I(\hat Z^+;M\mid Q,A)>0.
\]

这不是说 MSE 理论上不能学习条件期望，而是说在 Encoder、target geometry 与 Predictor
联合可训练时，存在更容易的、忽略历史的优化路径。

### 1.2 三个可独立检验的环节

为避免把最终症状误写成唯一根因，本文把能力拆为：

1. **历史可读**：`H` 中关于隐藏动力学的信息被 Encoder/context 保留；
2. **条件未来可用**：不同动力学的 target future 具有可读的尺度、方向与投影几何；
3. **预测耦合**：Predictor 使用 `H,A`，产生与真实 target response 对齐的变化。

历史 probing、target pair distance 或全局方差只能检验必要条件，不能替代第三项。最终
estimand 必须在 held-out matched query 上直接测量。

### 1.3 冻结晋级顺序

每个候选必须依次通过：

1. release、data、runtime、sampler、参数集合与 step-0 梯度路由审计；
2. 固定训练预算与 seed 的冻结 Development ICL 门；
3. Development 通过后，在 **同一 checkpoint SHA** 上运行原任务 CEM；
4. 单 checkpoint 的 ICL 与 CEM 都通过后，才运行额外训练 seeds；
5. 三 seed Development 全通过后，才打开相应 Public Test，并仍绑定各自固定 checkpoint。

prefix、训练 loss、梯度方向、target separation 和 probing 均为诊断量，不得替代冻结终点。
实现或审计失败只允许修复实现；完整冻结终点真实失败时，首先停止的是该固定
recipe/预算/seed 的晋级。只有跨预注册变体、seed 与任务均出现同一失败，才允许把结论扩大
到机制或算法族；单个远离阈值的终点可以停止继续消耗下游 CEM/Public 预算，却不能写成
“理论无效”。

## 2. 已排除的解释

### 2.1 benchmark 与数据不是根本不可学

ActionDelay H7 的同预算参考结果为：

| 方法 | optimizer steps | macro | 最差组 | bootstrap lower | history-responsive queries |
|---|---:|---:|---:|---:|---:|
| native LeWM | 1,024 | 0.3333 | 0.0000 | 0.3333 | 2/960 |
| PLDM | 1,024 | 0.8257 | 0.4875 | 0.8149 | 950/960 |

PLDM 使用同一 Predictor 主体且能学会，说明数据包含可学习的历史条件信号。它并不证明
PLDM 的全部 std/cov/temporal 目标都是必要的，也不排除 LeWM-specific 的优化盆地。

### 2.2 stop-gradient 与 target separation 都不充分

`target stop-gradient + native SIGReg` 的 256-step falsification 中，2,880 个 target pair
按冻结距离判据均未重合，但 959/960 个 query 对三种历史选择相同输出，macro 仍为
`0.3337`。stop-gradient 关闭了 MSE 直接拉动在线 target 的捷径，却没有自动创造
history-to-future coupling。

### 2.3 全局联合统计仍不等于条件联合关系

Temporal-centered SIGReg、target/predictive JTCov、Encoder-only 与多种 paired/global
统计在局部任务上有正例，但没有同时满足跨任务 ICL 与原能力保持。尤其 target-JTCov
只通过六项 ICL 中的三项，九项原任务 CEM 中仅两项非劣；predictive JTCov 虽把梯度送入
Predictor，却压缩了真正的条件 response。

关键区别不是正则是否被称为“联合”，而是它是否按 `H,Q,A` 识别正确的 future 对应关系。

## 3. Predictor-only PCJA

### 3.1 条件分配目标

对共享 query、具有 `N` 个不同历史/真实未来的配对组，定义 auxiliary prediction 与
stop-gradient target 的代价矩阵：

\[
D_{ij}=\frac{d\!\left(\hat z_i^+,\operatorname{sg}(z_j^+)\right)}
{\max\!\left(\frac{1}{N(N-1)}\sum_{a\ne b}d\!\left(z_a^+,z_b^+\right),\epsilon\right)}.
\]

`D` 的行回答“给定历史 `i`，哪个 future 正确”；列回答“给定 future `j`，哪个历史产生
它”。Paired Conditional Joint Assignment（PCJA）使用对称交叉熵：

\[
L_{\mathrm{PCJA}}=\tfrac12\left[
\operatorname{CE}(-D,I_N)+
\operatorname{CE}(-D^\top,I_N)
\right].
\]

第一项直接对应 benchmark 的 `correct_future`，第二项对应 `correct_history`。同时约束两者
能排除“所有 predictions 靠近同一个容易 target”或“一个 prediction 吸收多个 histories”
的多对一捷径。

ActionDelay 使用 `N=3` 的 delay `0/4/8` triplets。跨任务实现只把同一公式泛化到任意
`N>=2`；Motion Damping 使用已有的二元 pair，即 `N=2`，没有改变 assignment 公式。

### 3.2 总目标与梯度边界

主训练路径保持原生 LeWM：

\[
L=L_{\mathrm{native\ live\ MSE}}
+0.09L_{\mathrm{SIGReg}}
+0.09L_{\mathrm{PCJA}}.
\]

PCJA 分支具有以下边界：

- target terminal latent 在 assignment 内 detach；
- history embeddings 与 action embeddings 在进入 auxiliary Predictor 前 detach；
- auxiliary 梯度只允许更新 `predictor` 与 `pred_proj`；
- 原生 MSE 与 SIGReg 的全 live 梯度路径保持不变；
- 每个 batch 只增加一次 deterministic Predictor 调用；不增加 encode 或 SIGReg 调用；
- 不新增参数、head 或 inference path；pair metadata 与隐藏动力学标签不进入模型。

这个路由不是工程细节。full-gradient PCJA 在 ActionDelay Private 达到 macro `0.9451`，但
相对 native 的 300 次配对 CEM 确认下降 `13.33` 个百分点。predictor-only 路由把辅助项
限制为“教 Predictor 使用已经可用的条件信息”，避免借辅助损失重塑通用视觉与动作表示。

### 3.3 简洁性与监督边界

当前 PCJA 需要训练数据中能够组成 matched counterfactual group 的结构。隐藏 mode/delay
只用于采样与确定矩阵对角线，不作为网络输入，也不需要新增推理时 context module。因此它
比 PLDM 的多项 std/cov/temporal objective 更直接对齐条件 ICL estimand，且方法与推理均
更小。

但“显式 privileged 配对”不是理论上唯一的实现：同一归纳偏置未来可由 episode 内
support/query 切分、可验证的 context swap 或数据增强产生。当前证据只支持：某种直接的
条件关系/干预信号很可能是简单 LeWM 稳定获得条件可辨识性的关键；它还不支持“任何配对
都有效”或“隐藏标签是理论必需”。Motion Damping 已有严格 pair，而旧 ranking loss 仍
失败，正好可以区分普通配对与 PCJA 的对称 assignment/梯度隔离机制。

### 3.4 Motion Damping 暴露的 assignment–calibration 缺口

PCJA 的交叉熵只要求正确 assignment 的 logit 优于错误 assignment；它不是 response
regression。二元情形最清楚：若两个 predictions 已经精确等于各自 targets，归一化距离矩阵
为

\[
D=\begin{bmatrix}0&1\\1&0\end{bmatrix}.
\]

此时每个方向的交叉熵仍为

\[
\log(1+e^{-1})\approx0.313,
\]

而不是零；辅助梯度仍有动力继续扩大 matched/counterfactual margin。native MSE 会把输出
拉回真实 target，最终平衡点未必等于真实 response。这个性质在离散 ActionDelay 上没有
阻止成功，但在 target difference 较小、要求连续响应校准的 Motion Damping 上可能造成
明显冲突。

目标函数的非驻点性质已经由解析推导与实现数值梯度共同确认；但冻结终点并不支持“PCJA
独自造成了校准失败”这个更强说法。对同一 256 个 Development pairs 的新重评分为：

| endpoint | future | history | response gain | alignment | normalized error | calibrated pairs |
|---|---:|---:|---:|---:|---:|---:|
| legacy ranking | 0.9746 | 0.5293 | 9.2349 | 0.4643 | 378.1542 | 0.0% |
| legacy multi-term fit | 0.6973 | 0.7012 | 0.4941 | 0.5343 | **0.8669** | 63.3% |
| legacy projected geometry | 0.7422 | 0.6602 | 0.6934 | 0.4809 | 1.6926 | 13.3% |
| predictor-only PCJA | **0.7793** | **0.7129** | **1.1671** | **0.6797** | 1.6145 | 27.7% |

旧 ranking 是最清楚的 anti-spoofing 反例：prediction response 的 RMS 是 target response 的
`19.89` 倍，即使远离两个真实 futures，只要稍微更接近正确一侧，future selection 仍可
接近满分。legacy fit 则给出相反失败：aggregate error 已低于“不响应历史”的 `1.0`
基线，但 gain 仅 `0.4941`，刚低于冻结的 `0.5` 下限，且主能力门仍失败。

对拼接后的 response 做 target 轴投影，有

\[
e_{\mathrm{norm}}=(g-1)^2+e_{\perp}.
\]

PCJA 的两项分别为 `0.0279` 与 `1.5865`；正交项占总 error 的 `98.27%`。因此“把 `0.09`
再调小”或“提前 cutoff”只针对很小的轴向误差，不是当前证据支持的修复。PCJA 相对 ranking
在 256/256 pairs 上都有更低 normalized error，但这个比较仍是 endpoint comparison：三个
legacy 方法彼此固定了 frozen encoder/projector、权重 `1.0` 与梯度路由；PCJA 则使用 live
representation、权重 `0.09` 和 predictor-only auxiliary route，不能冒充单因素因果消融。

## 4. ActionDelay 的冻结正证据

### 4.1 同一 checkpoint 的 ICL 与 CEM

候选 `action_delay_h7_a0_aux_pcja_predictor_only_v1` 使用 seed `3072`，固定 1,024 optimizer
steps；PCJA 从 step 1 到 1,024 持续启用。checkpoint 为：

```text
SHA256 9d2dc13a70c1eda350daf8e68306401b2c7fa2817b3e278ddb6f3f299d0d2e65
```

| 冻结评测 | macro / point diff | 最差组 / lower | 结论 |
|---|---:|---:|---|
| Private Development | 0.9309 | min 0.8933；bootstrap lower 0.9133 | 四项 gate 全过 |
| Public Test | 0.9452 | min 0.9167；bootstrap lower 0.9282 | 正式独立重评分一致 |
| CEM，累计 900 paired trials | -2.67 pp | one-sided 95% lower -3.67 pp | 通过 -5 pp 非劣界 |

Private 的 4,500 个 physical-group target comparisons 中 collapsed pair 为 `0`。Public、
Private 与 CEM 均绑定上述同一 checkpoint SHA；Public 结果不是从多个候选或 checkpoint
中按分数选择。

### 4.2 当前结论边界

可以确认：

- 原生 LeWM 架构能够学会 ActionDelay ICL；
- 直接、对称的 condition assignment 是目前第一个同时通过该任务 ICL 与 CEM 的简洁
  LeWM 辅助目标；
- predictor-only 梯度隔离对 Pareto 结果很重要。

尚不能确认：

- 多训练 seed 稳定性；
- 跨连续动力学、接触动力学或环境域的通用性；
- 对称列项与梯度隔离各自是否必要；
- 显式 pair 能否被无 privileged metadata 的训练构造替代；
- PCJA 是否整体优于 PLDM，而不是互补或诊断性上界。

因此本阶段不把单 seed Public 正例写成方法级 SOTA，也不跳过跨任务 falsification 直接扫
更多 seed。

## 5. 跨任务证伪阶梯

### 5.1 为什么先做 Motion Damping，而不是 Speed

最新 ContextWorld 文档把能力扩展为九项。它们不是一个可平均成单分数的同质任务，也没有
出现“一个模型族全面修复另一个模型族”的结果。suite v2 additive integrity reseal 已由
`contextworld_icl_suite_v2_integrity_reseal_v2` 正式通过；当前 13-row scoreboard 是加法扩展，
没有重跑历史 Public 或改选 checkpoint。下表只保留两个证据层级：`[P]` 为冻结正式 Public，
`[D]` 为按停止规则停在 Development 的结果，不能把 `[D]` 写成 Public。

| 能力与任务 | LeWM reference | PLDM reference | 当前失败表型 |
|---|---|---|---|
| 即时连续响应：Speed | `[P]` 95.26%，3/3 通过并保持 CEM | `[P]` 96.70%，3/3 且配对 CEM 保持 | 当前不是 coupling 失败；只作 retention control |
| 即时连续响应：Action Strength | `[P]` 96.61%，3/3 通过并保持 CEM | `[P]` 94.27%，0/3；CEM 未授权 | H/switch 并非主要失败，absolute future/最弱 margin 不足 |
| 即时连续响应：Reacher Mass | `[P]` 76.11%，3/3 通过并保持 CEM | `[P]` 63.15%，0/3 | history/switch 可高而 future response 不准或偏弱 |
| 时间延迟：ActionDelay | `[P]` 32.43%，0/3；CEM 保持 | `[P]` 93.36%，3/3 且 CEM 保持 | LeWM 近乎 coupling silence；PCJA 正面对应该表型 |
| 接触动力学：Contact Friction | `[D]` future 96.09%、history 90.23%，失败 | `[D]` future 52.73%、gain 0.033，失败 | LeWM 接近门但 history 使用不足；PLDM response 极弱 |
| 接触动力学：Motion Damping | `[D]` future 97.46%、history 52.93%，失败 | `[D]` future 51.95%、gain 0.160，失败 | reference 显示 assignment/response 失败；跨 batch transfer 是 PCJR-CV prefix 的机制证据 |
| 接触/附着：Cube Carry | `[P]` 78.45%，3/3 且 CEM 保持 | `[D]` 50.13%，0/3 | 当前 paired exact-future LeWM 配方有效；PLDM 近 chance |
| 结构规则：Door | `[P]` 100%，3/3 | `[P]` 99.33%，3/3 | 两者 ICL 都学会，却都严重损害原 TwoRoom CEM |
| 结构规则：Portal Exit | `[P]` 83.92%，0/3；CEM 保持 | `[P]` 59.31%，0/3 | LeWM 可能有差分响应但 absolute future 未过；PLDM 欠增益 |

suite reseal 只确认当前 release、scoreboard 与历史 predecessor 的加法绑定完整，不把
Development failure 升格成 Public，也不提供方法训练归因。九任务仍应按失败表型分层，而不是
平均成一个榜分。只有零更新 kill test 支持、重新预注册的 Motion candidate 通过 Development
与同 SHA CEM 后，首个跨任务鉴别器才是 **Contact Friction**：它与 Motion 同为 PushT、H3、
matched paired query、相同 common init 与 batch 结构，最能区分通用 cross-batch coupling
缺口和 Motion 特有的 post-contact damping calibration。Reacher/Portal
留作跨域验证；Speed、Action Strength 与 Cube-LeWM 只在候选跨失败任务后作为 retention 或
negative control，不用于调参。

### 5.2 Phase A：二元 predictor-only PCJA 已冻结

实现只做了一个方法变化：将 ActionDelay 的硬编码三元 PCJA 泛化为 `groups: (P,N)`，并在
Motion Damping 上取 `N=2`。没有移植 ActionDelay sampler、H7 temporal weighting 或额外
side view。通用实现通过了旧 `N=3` 前向/梯度 parity、二元手算、置换不变与 fail-closed
测试；真实 Motion batch 的 CUDA BF16 step-0 审计确认 auxiliary target/input detach、一次
额外 Predictor、零额外 encode/SIGReg、无新增参数，并只向 `predictor/pred_proj` 提供梯度。

必须保持：

- 原 H3 identifiable-future 主损失：original rows 的三步 MSE、hidden rows 的 terminal
  MSE，以及两部分固定组合；
- native SIGReg 恰好调用一次且权重 `0.09`；PCJA 权重固定 `0.09`，不扫参；
- AdamW `lr=5e-5`、`wd=1e-3`、clip `1.0`、BF16、batch `128`、step `8192`；
- 初始化 checkpoint SHA `9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59`；
- 每 batch `64 original + 64 hidden`，hidden 中恰好 32 个完整 binary pairs；
- Motion Damping 的 `16 twin groups × 4 rows` 与 forward/reverse x0 label exchange；
- 无新参数、optimizer parameter set 不变、推理输出在 overlay 未启用时逐位一致。

除主候选外只保留一个非候选机制对照：相同 predictor-only 路由下的 **row-only
assignment**。它不参与候选选择，只用于区分：

| 终点模式 | 机制解释 |
|---|---|
| symmetric 通过，row-only 失败 | 缺失的 future→history 列约束是关键 |
| 两者都通过 | predictor-only 梯度隔离比列项更关键 |
| 两者都失败 | ActionDelay 正例未跨任务泛化；先分析，不发明新 loss |

### 5.3 Phase B 结果：Motion Damping 终点失败

第一训练 seed 固定为 `14321`，终点固定 step `8192`；没有依据 prefix 选 checkpoint。
checkpoint SHA256 为：

```text
50eba192e60eae77070f9ee74e44375c1e2122ee5951ecd37b3da828de102bcc
```

冻结 monitor 呈现如下轨迹（终端打印保留三位小数）：

| optimizer step | correct future | correct history | context switch | worst group |
|---:|---:|---:|---:|---:|
| 1,024 | 0.529 | 0.545 | 0.668 | 0.246 |
| 2,048 | 0.564 | 0.627 | 0.895 | 0.324 |
| 4,096 | 0.719 | 0.705 | 0.977 | 0.559 |
| 8,192 monitor | **0.777** | **0.713** | **1.000** | **0.680** |
| 同 checkpoint Development-only 重评分 | **0.7793** | **0.7129** | **1.0000** | **0.6836** |

冻结门要求 future/history/switch/worst 分别至少为 `0.95/0.95/0.95/0.90`。因此只有
switch 通过，Development 判定已经终止；即使后续精确 latent-response 重评分通过，也
不能反转三项主门失败。

与旧 LeWM endpoint 比较，PCJA 的影响不是简单“无效”：

| endpoint | correct future | correct history | context switch | worst group |
|---|---:|---:|---:|---:|
| 旧 paired-ranking LeWM | 0.9746 | 0.5293 | 0.9766 | 0.9727 |
| predictor-only PCJA | 0.7793 | 0.7129 | 1.0000 | 0.6836 |

它把 history preference 提高约 `18.4` 个百分点并使 switch 完全正确，却以 future 下降约
`19.5` 个百分点、worst group 下降约 `28.9` 个百分点为代价。这个模式说明 PCJA 确实
打开了 history-conditioned response，但没有得到 Pareto-safe 的真实 future fit。它与
此前 paired-fit/matching 类目标的“history 增强、target calibration 退化”谱系一致，只是
在当前 predictor-only 路由下更强。

同 checkpoint 的 256-pair Development-only 记录进一步给出：target separation 256/256
通过，response gain 为 `1.1671`、alignment 为 `0.6797`、normalized response error 为
`1.6145`；只有 `27.73%` pair 优于“完全不响应历史”的 error=`1` 基线，joint ICL pair
success 仅 `7.03%`。问题不是 target collapse，也不主要是 aggregate gain 的轻微过冲；
error decomposition 显示 `98.27%` 是正交于真实 response 的剩余能量。`switch=1` 只要求
点积符号为正，远弱于方向和幅值都复制真实 response。

训练后 wrapper 因把原生结果键 `loss_trace` 写成 `trace`，在 checkpoint 和终点 monitor
落盘后发生 postfit `KeyError`；完整 research overlay receipt 因而缺失。随后只加载不可变
checkpoint 的 Development-only recovery 与 record rescore 均完成，模型评分前后 state SHA
一致，Public 未打开、没有新增 optimizer step。monitor 与正式 adapter 在 future/worst 上
相差 1–2 个边界 decision，但两条路径均明确失败，不影响结论。

### 5.4 Phase C：Contact Friction 迁移已按门控停止

原计划只有 Motion Damping 的 ICL 与 CEM 均通过，才把完全相同的 `N=2` 公式、`0.09`
权重和 gradient route 迁移到 Contact Friction seed `13313`、step `8192`。Motion 已失败，
所以这项无调参迁移没有启动；对应 CEM 与 Public 也保持关闭。Contact 不再被当作继续消耗
预算的第二次同类尝试，而是在形成并冻结有明确 calibration 依据的新候选后，作为更困难的
接触动力学检验。

### 5.5 Phase D：跨域与 PLDM 失败形态

若两个 PushT 任务均通过，下一步按数据可用性进入 Portal Exit，检验是否只是 PushT 域内
修复。随后在 Reacher/Cube 的 PLDM 失败项上使用同预算、同初始化和同数据的对照，区分
PCJA 是独立简洁修复、PLDM 的互补项，还是仅有诊断价值。

结果解释预先固定为：

| 结果 | 允许的结论 |
|---|---|
| Damping 失败 | PCJA 目前是 ActionDelay task-local 正例 |
| Damping ICL 过、CEM 失败 | 能创建 coupling，但不是 Pareto 修复 |
| Damping 过、Friction 失败 | 对平滑动力学有效，尚不能处理接触机制 |
| 两个 PushT 过、Portal 失败 | PushT 域内方法 |
| 跨域通过但系统性弱于 PLDM | PCJA 更适合作为辅助/诊断目标 |
| 跨域通过且 CEM 保留、可匹配或补足 PLDM | 才支持通用主方法主张 |

现在已经出现五个真实的 Motion Damping 终点，不再训练 row-only control，也不扫 cutoff、
权重或 seed。五者使用同一 seed `14321`、8,192 steps、native MSE/SIGReg 和完整 twin batch；
前四者只改变 predictor-only 配对辅助目标，第五者在原始 CCRM 上单独冻结
Encoder/Projector：

| 辅助目标 | future | history | switch | worst | gain | alignment | norm. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| PCJA | 0.7793 | 0.7129 | 1.0000 | 0.6836 | 1.167 | 0.680 | 1.614 |
| CCRM | 0.5137 | 0.5293 | 0.9961 | 0.0430 | 0.430 | 0.751 | 0.468 |
| TA-CCRM | 0.6211 | 0.6855 | 0.9219 | 0.5859 | 0.296 | 0.499 | 0.759 |
| PCJA+CCRM | 0.5938 | 0.6875 | 0.9883 | 0.3281 | 0.520 | 0.717 | 0.486 |
| frozen Encoder/Projector + CCRM | 0.5332 | 0.5781 | 0.9805 | 0.1211 | 0.309 | 0.523 | 0.731 |

CCRM 对同一 `Q,A` 的 `N` 个历史—未来组定义为

\[
\tilde p_{gi}=\hat z^+_{gi}-\frac1N\sum_j\hat z^+_{gj},\qquad
\tilde t_{gi}=z^+_{gi}-\frac1N\sum_j z^+_{gj},
\]

并只使用一个 target-normalized 项：

\[
L_{\mathrm{CCRM}}=
\frac1P\sum_g
\frac{\frac1N\sum_i\left\|\tilde p_{gi}-\operatorname{sg}(\tilde t_{gi})\right\|^2}
{\frac1N\sum_i\left\|\operatorname{sg}(\tilde t_{gi})\right\|^2+\epsilon}.
\]

二元组中它精确等于 scorer 的 normalized response error；真实 response 的 loss 与梯度为
零，zero response 的 loss 为 `1`。实际终点证明它确实能学到 response geometry，但由于
中心化，它不约束公共 prediction center。CCRM checkpoint 的 target-axis center coefficient
`beta` 加权均值为 `1.690`，这精确重构了 faster `98.4%`、no-extra `4.3%` 的单侧选择。

TA-CCRM 试图用一个残差加入 `4 beta^2`。它把 signed `beta` 加权均值降到 `0.137`，说明
诊断与符号均正确；失败原因是优化尺度而不是公式错误。终点真实 batch 上：

- CCRM 与 anchor 的 prediction-space gradient 正交；
- 映射到共享 `predictor + pred_proj` 后 cosine 为 `-0.021`，不是强方向冲突；
- 但 anchor 参数梯度范数 `34.90`，CCRM 仅 `3.75`；
- 固定 clip=`1` 下，总梯度由 `1.02` 增至 `3.30`，响应校准更新被中心项淹没。

这否定了继续调二次 anchor 的权重。随后预注册的最小组合把两项已被独立验证的必要信号
放进**同一个辅助前向**：

\[
L_{\mathrm{pair}}=L_{\mathrm{CCRM}}+L_{\mathrm{PCJA}},\qquad
L=L_{\mathrm{native\ pred}}+0.09L_{\mathrm{SIGReg}}+0.09L_{\mathrm{pair}}.
\]

PCJA 提供有界的正确 assignment 梯度，CCRM 提供全向 response calibration；二者共享同一
matched group、prediction 和 detached target，不新增 head、参数、数据源或推理路径，也只
保留一个外部权重 `0.09`。这不是重新堆叠 PLDM 式统计正则，而是对应两个已被反例证明不可
互相替代的条件 estimand。训练前在 TA-CCRM endpoint 上看到两者参数梯度范数为 `3.08` 与
`3.75`、cosine 为 `+0.166`，因此没有事前证据显示 TA anchor 式数量级支配。

PCJA 在 exact response 处仍有有限 margin 梯度，因此组合的理想二元轴向稳态不是 gain=`1`
而是唯一解 `g*=1.12275`，满足 `2(g-1)=sigmoid(-g)`；对应 CCRM error 仅 `0.0151`。这项
可解析偏移已在 step-0 冻结，没有把组合伪称为 exact-response 零梯度目标。

完整训练后的 checkpoint SHA 为
`fad2016b9e250eb2bf4922c35fc4449424aa2b5a1b319237bfb4ae566267aa1d`。
checkpoint-only 精确重评分保持 Public 关闭、增加 optimizer step 为 `0`，并确认当前
Development manifest、Loader Validation table 与训练时完全一致。组合通过 switch、target
separation、gain 和 normalized-error 门，却明显未过三个直接门：

| 主门 | 观察值 | 阈值 | 固定 catalog 仍需增加的正确数 |
|---|---:|---:|---:|
| correct future | 0.5938（304/512） | 0.95 | 183 |
| correct history | 0.6875（352/512） | 0.95 | 135 |
| worst damping | 0.3281（84/256） | 0.90 | 147 |

这不是轻微阈值误差。它也不是“组合完全没学到”：gain=`0.520`、alignment=`0.717`、
normalized error=`0.486`，`242/256` pairs 优于 zero-response baseline；target-axis
weighted `beta` 从 CCRM 的 `1.690` 降至 `0.627`。但 `208/256` pairs 的中心偏置幅度仍不小于
半个 response gain，只有 `48/256` 同时越过两个 target boundary，所以绝对 future
assignment 仍失败。

终点真实训练 batch 的零训练梯度分解解释了为何简单相加没有形成预期折中：

- output-space PCJA/CCRM gradient cosine 为 `+0.315`；
- 通过共享 Predictor Jacobian 后，参数梯度 cosine 变为 `-0.113`；
- PCJA/CCRM 参数梯度范数为 `10.06/3.07`，比值 `3.28×`；
- 完整更新方向与 `native+PCJA` 的 cosine 为 `0.959`，与 `native+CCRM` 仅 `0.343`；
- 完整全局梯度范数为 `1.349`，超过固定 clip=`1.0`。

因此停止的是预注册的 `1:1 PCJA+CCRM` recipe 及其下游迁移，不是三项机制判断：PCJA
确实能创造 history-conditioned coupling，CCRM 确实能校准 centered response，公共中心
仍确实是一个可精确重构的失败方向。单 seed、单权重也不能证明所有比例或训练时长都不可能；
但当前 recipe 离门很远、终点又出现参数空间冲突和 clipping，没有足够的事前因果依据授权
事后权重 sweep。继续扫参会偏离“一个简洁、无需调组件平衡的 LeWM 修复”这一研究目标。

### 5.6 冻结表示单因素实验：漂移是放大器，不是完整根因

为避免把 `PCJA+CCRM` 的参数空间冲突误当成 CCRM 家族本身的失败，下一项只改变一件事：
在相同 seed、数据、batch、8,192-step 预算、native MSE/SIGReg 与 `0.09 CCRM` 下冻结
Encoder/Projector；Predictor、pred_proj 与 action encoder 仍由原生路径训练，CCRM 路径仍只
更新 Predictor/pred_proj。当前 Motion release 相对训练预注册时只发生 public API 自哈希
更新；训练数据、Development table/order、threshold 与 scorer 均未改变。追加 identity
reseal 后重新执行完整 CUDA/BF16 exact-batch step-0，所有 batch bytes、loss、response、
route 和方向 gate 与旧 receipt 完全一致，随后才获得单 seed 训练授权。

正式终点 checkpoint `735781a7566b…` 的独立 Development 重评分与原评分逐字节一致。
虽然 target 256/256 非重合、switch=`0.9805`、normalized error=`0.7309 < 1`，但
future/history/worst=`0.5332/0.5781/0.1211`，gain=`0.3089`，明显失败；Public、CEM 与
额外 seed 均保持关闭。相对 live CCRM，冻结将 target-energy-weighted center coefficient
从 `1.690` 降到 `0.690`，绝对偏置减少 `59.18%`，却同时降低 response gain 与 alignment。
`239/256` pairs 的 `|beta|` 仍至少为 gain 的一半，只有 `17/256` 同时跨过两侧 target
boundary。因此冻结表示只减弱了一个放大因素，没有恢复绝对 future assignment。

终点 exact training batch 的零优化步梯度审计进一步排除了几种容易误判的解释：

- 完整目标梯度范数为 `0.1745`，小于 clip=`1.0`，不是裁剪屏障；
- 完整下降方向预测 response error `-0.1651`、`beta^2 -0.2341`、assignment margin
  `+0.000618`，三项在该 batch 上没有直接冲突；
- native hidden-terminal MSE 对 `beta^2` 的方向很强且正确（`-0.5729`），但对 response
  error 与 margin 的作用只有 `-0.0039` 与 `+4.7e-7`；
- CCRM 单独强力降低 response error（`-1.8132`），却因中心化 null direction 使
  `beta^2` 增加 `+0.6501`。

### 5.7 Exact-future 与 squared-hinge BC-CCRM 的真实反例

frozen-CCRM 的终点审计曾指出：native exact-future 梯度主要修正 center，CCRM 主要修正
response。由此得到的 pair-normalized exact-future 用同一个 target-energy 分母拟合每个
absolute future；它在解析上等于 CCRM 加 normalized common-center error，不含可调分量权重。
但正式 8,192-step 终点得到
`future/history/switch/worst=0.475/0.412/0.320/0.063`、gain=`-0.041`、
alignment=`-0.091`、error=`1.291`。因此停止的是“放大 absolute future MSE 足够”这一
固定主张；它没有否定 matched pair 中确实存在可学习的 response 与 assignment 信号。

下一项没有再拟合任意 center，而是保留 CCRM，并只惩罚二元 pair 落在错误 target Voronoi
cell 的部分。令 prediction response 在 target axis 上的投影为 `alpha`，prediction/target
公共中心差的投影为 `beta`，两个正确 assignment 的必要充分边界为

\[
m_0=\alpha/2-\beta>0,\qquad m_1=\alpha/2+\beta>0.
\]

squared-hinge BC-CCRM 使用
`L = L_CCRM + mean(relu(-m)^2)`；pair 一旦进入两个正确 cell，边界项精确归零。该方法在
Development 上相对所有 CCRM 类前项取得最强的共同折中：future=`0.635`、history=`0.723`、
switch=`0.973`、gain=`0.4999`、alignment=`0.702`、error=`0.508`。训练 monitor 从 step 0
到 `1024/2048/4096/8192` 的 future 为 `0.500/0.510/0.561/0.602/0.637`，不是“完全没学”。
但冻结主门仍明显失败，只有 `69/256` Development pairs 同时越过两侧边界。

同一个训练 batch 的终点审计显示：boundary loss 从 `1.1167` 降到 `0.0547`，CCRM 从
`1.0412` 降到 `0.4726`，可行 pair 从 `0/32` 增至 `10/32`；余下 `22/32` 仍 active。
终点完整方向继续降低 boundary、response error 和 `beta^2`，clip scale=`0.913`，所以失败
不能归为符号错误、严重裁剪或只在 Development 泛化失败。严格否决范围是：**equal-weight、
zero-margin、predictor-only、squared-hinge、8,192-step 的固定 recipe**；不扩大到 pairing、
CCRM、conditional intervention 或边界条件本身。

### 5.8 全局 response-scale 反事实

为避免把“继续增大 response”误当新方法，对现有 endpoint 做了不运行模型的结果级反事实：
固定每对 prediction center，只将 response 乘全局标量 `s`。没有一个 endpoint 能在保持
gain/error 门的同时通过任务门。BC-CCRM 的 response-error 最优尺度为 `0.985`，几乎就是
当前幅值；即使在 response 门允许的最佳尺度 `1.969`，future/worst 也只有 `0.742/0.582`。
PLDM Contact/Motion 分别有 `53/256` 与 `91/256` 个 projected response 非正，且不存在同时
满足 gain `>=0.5`、error `<1` 的全局尺度。因此两类失败都包含方向/正交或 center 误差，
不是统一的 under-gain。

### 5.9 Linear exact penalty：真实终点与严格否决范围

squared hinge 的梯度随违规量线性衰减：初始化时少数大违规可主导更新，接近边界后每个
active pair 的压力又趋近零。当前候选只把平方罚改成线性 exact penalty：

\[
L_{\mathrm{linear\text{-}BC}}=L_{\mathrm{CCRM}}+
\operatorname{mean}\left[\operatorname{ReLU}(-m)\right].
\]

它保持相同数据、margin、target/input detach、`predictor+pred_proj` 路由、外部权重 `0.09`
和 inference；新增参数、head 与 loss 成分均为零。冻结的同批次双快照 comparator 8/8 通过：

- 初始化 boundary/CCRM 梯度比从 squared 的 `9.806` 降至 linear 的 `3.127`，完整 clip scale
  从 `0.536` 改善到 `0.624`；
- squared-BC 失败终点的比值从 `2.390` 提高到 `4.006`，完整 clip scale 仍为 `0.754`；
- 两个快照的 linear 完整方向都同时降低 linear violation 与 normalized response error；
- predecessor scalar、batch bytes、route、buffer/mode/RNG restoration 与零 optimizer/
  Development/Public/CEM guard 全部通过。

这些量只授权了一次从原始初始化开始的 seed `14321`、8,192-step 冻结 Development endpoint，
并不构成成功证据。正式终点 checkpoint `e47976be4641…` 的独立评分与重评分逐字节一致：
future/history/switch/worst=`0.6113/0.7266/0.9922/0.3906`，gain/alignment/error=
`0.5239/0.7213/0.4797`。它只通过 switch、target separation、gain 与 error 门，future、
history 与 worst 三个直接门失败；Public、CEM 与额外 seed 均未打开。

linear 不是“完全没学”：monitor 的 future 从 step 0 的 `0.500` 单调增至
`0.504/0.539/0.590/0.611`，同一首批训练样本的 boundary loss 降至初始的 `16.7%`，可行
pair 从 `0/32` 增至 `10/32`。但相对 squared 终点，它仅把 switch/gain/alignment 分别提高
`0.0195/0.0240/0.0197`，future/worst 下降 `0.0234/0.0352`，Development 两侧同时可行
pair 从 `69` 降到 `57`，target-axis center bias 反而增加 `0.0454`。终点的完整方向仍同时
降低 boundary violation、response error 与 `beta^2`，所以正确否决范围是：**固定 seed、
预算、权重、zero-margin、predictor-only 的 linear exact-penalty recipe，以及“只换 penalty
shape 即可修复”**。不得扩大到 pairing、CCRM 或 conditional intervention 整个家族。

### 5.10 逐 pair 梯度与 target endpoint 漂移

为了区分“局部方向正确但跨 pair 抵消”和“Predictor 一直追逐变化 target”，在完全相同的
首批训练数据上复放初始化、squared-BC 终点和 linear-BC 终点。对每个 pair 单独计算
Predictor/pred_proj 参数梯度，并定义

\[
r_{\mathrm{cancel}}=\frac{\left\|\sum_i g_i\right\|}
{\sum_i\left\|g_i\right\|}.
\]

`r=1` 表示完全同向，`r=0` 表示完全抵消。初始化的 linear auxiliary 为 `0.207`；linear
终点降为 `0.173`，weighted pairwise cosine 为 `-0.0112`。终点全参数空间有 `7/32` pair
反对 aggregate，负 signed-share mass=`0.0977`；只看 Predictor 则为 `9/32` 与 `0.1571`。
squared 终点也有 `7/32` pair 反对其 active objective。因而终点更新并非“没有梯度”，而是
由大量近正交、部分反向的 pair 梯度低效合成。该比例符合预注册的强抵消判据，但负质量仍属
中等，不能单独解释全部 Development 缺口。

同一批 target response 轴从初始化到 squared/linear 终点的中位 cosine 分别为
`0.538/0.547`，中位 normalized delta 为 `0.855/0.862`，轴长中位缩至初始的
`0.702/0.761`；说明 stop-gradient 只阻断单步反传，并没有让跨训练步 target 静止。与此同时，
squared 与 linear 两终点 target axis 高度相似（中位 cosine=`0.966`、norm ratio=`1.007`），
表明 penalty shape 的差别没有把 Encoder/Projector 带到不同的条件几何盆地。

成功 ActionDelay 的同定义正对照已经完成。其 target axis 漂移与 Motion 几乎同量级，却仍
通过 Development、Public 与 CEM，因此 endpoint target 漂移的失败特异性被否定。成功终点的
weighted pair-gradient cosine=`0.02875`、负 signed-share mass=`0`；失败 Motion-PCJA 分别为
`0.00046/0.00366`，而失败 linear BC-CCRM 为 `-0.01124/0.09773`。这形成三种可区分表型：
正相干成功、近正交的 assignment-only 失败，以及带实质反向质量的 boundary-objective 失败。

因此，BC-CCRM 的共享参数冲突是真实局部机制，却不是 Motion-PCJA 的通用解释；也不能因
target 漂移引入 hard-freeze、EMA 或 lagged target。当前更窄的联合机制是：**PCJA 能提供
history-conditioned assignment 信号，但它不把真实连续 response 设为驻点；Motion 的主要
正交误差还暴露出 native absolute-future 路径、Predictor Jacobian 或跨 query 泛化中的另一
断点。** 这一步授权了 proper conditional-joint 的零训练步 kill test，但不直接授权第九个
8,192-step recipe。

### 5.11 Proper conditional-joint 的轨迹归因

`PCJR-CV` 将每个 matched group 的 centered response residual 按真实 response energy
标准化，并用 batch-global target-center variation 标准化公共 center residual。它在
`prediction=target` 时 loss 与 output gradient 均为零，不再有 PCJA 的非驻点 margin；零训练步
审计也确认 target/input detach、只更新 `predictor+pred_proj`、不等价于 native MSE，因而只
放行一条 seed `14321`、1,024-step 确定性机制轨迹。

该轨迹没有形成 Development 正例。step `256/512/1024` 的 future/history 仅为
`0.502/0.518`、`0.508/0.527`、`0.514/0.535`；response gain 从 `0.166` 增到 `0.295`，
normalized error 从 `0.787` 降到 `0.633`，但 target-energy weighted `beta^2` 从 `1.323`
增到 `1.479`。普通 prediction-center MSE 下降 `18.8%`，target-axis center projection MSE
却没有改善，说明训练主要清理了 assignment 无关的正交误差。

为避免再次用 endpoint 局部梯度误判，随后完成三层同轨归因：

1. 13 个采样点中，实际 AdamW 对 train-mode 当前 batch 的 Predictor-only `beta^2` 有
   12 个非零改善、0 个恶化；clip/AdamW 不是普遍翻转源。
2. 六个从真实 loss boundary 绑定并哈希的训练输入锚点显示，eval-mode Predictor-only 当前
   diagonal 同样为 `12/12` 改善；42 个历史 transfer cells 却只有 `17` 改善、`25` 恶化，
   净变化 `+0.281`。live re-encode 的历史净变化扩大到 `+1.616`。
3. 在 step `256/512/1024` 分开 MSE 与 PCJR 后，两项对当前 diagonal 都为 `3/3` 改善；
   历史 `beta^2` 的分量方向如下：

| Predictor-side 分量 | 历史改善/恶化 | 累计每单位学习率方向 | 历史 response 改善/恶化 |
|---|---:|---:|---:|
| native MSE | 10 / 3 | `-0.361` | 5 / 8 |
| `0.09 * PCJR-CV` | 2 / 11 | `+0.837` | 9 / 4 |
| prediction total | 7 / 6 | `+0.476` | 11 / 2 |

负值表示改善。这个分解给出了该 frozen prefix 内此前缺失的机制归因：native MSE 主要保住
center/target-axis，
但不稳定地学 response；PCJR 明显修复 response，却把当前 batch 的条件几何局部化，经过共享
Predictor 后对多数历史 query 的 target-axis assignment 产生反向作用。两项之和可以同时降低
普通 response/center loss，却仍恶化 `beta^2`，所以 broad fit 与正确条件选择不能互换。

这不是 PCJR 或 pairing 的家族否决。它保留了 ActionDelay 的成功贡献，也解释了为什么每次
“初步局部有效”到完整轨迹会失效：被优化的是当前 batch 的正确关系，而不是跨 query 一致的
关系。下一步只允许一个零更新矩阵：用完全相同 PCJR 公式比较 current-only、
current+previous 与冻结 uniform multi-query aggregate；若聚合仍净伤害历史 `beta^2` 或丢失
response 修正，则在训练前停止。既有 TA-CCRM 已显示直接新增强 `beta` 项会产生参数空间支配，
因此不把“再加一个 loss”作为默认修复。

完整 full-path 审计另有一个必须保留的数值边界：step1024 的
`joint = response + center + cross` 梯度重建相对误差为 `4.017%`，没有通过注册的 BF16
`2%` 门；旧顶层汇总字段没有覆盖这一 nested failure。该细分只能作描述性证据，不参与上表
MSE–PCJR 分量结论；后者有独立的 prediction-gradient 与真实 pre-clip Predictor gradient
重建闸门。

### 5.12 K=4 multi-query estimator：小幅分配收益与错误方向放大

为直接检验“单批正确、跨批冲突”是否只是随机梯度估计问题，冻结 `K=4`：当前 batch 加三个
最近的 persistent historical batches。每个 slot 固定 raw `pixels/action`，在当前 checkpoint
的同一 eval mode 下独立重编码；PCJR 逐 batch 计算后均匀平均。`D` 仅把当前 PCJR 除以四，
`A` 则平均当前与三批历史 PCJR，所以 `A-D` 精确隔离历史信息的增量，没有改 loss 公式、权重、
参数或 inference。

三个冻结快照的一阶方向如下；负值表示对应 metric 改善：

| step | `A` 历史 response | `A` 历史 center | `A` 历史 `beta^2` | `A-D` 历史 `beta^2` |
|---:|---:|---:|---:|---:|
| 256 | `-0.02241` | `-0.01309` | `-0.14152` | `-0.08854` |
| 512 | `-0.02223` | `-0.00832` | `-0.08136` | `-0.05232` |
| 1024 | `-0.01867` | `-0.00755` | `-0.13878` | `-0.05991` |

九个 historical cells 合并后，`A` 的 response/center 为 `9/9` 改善，`beta^2` 为 `8/9`
改善、mean=`-0.12055`；`A-D` 的 `beta^2` 为 `9/9` 改善、mean=`-0.06692`。当前 diagonal
的三项 metric 在三个快照也全部不受损，单 slot 最大梯度范数占比只有 `0.272–0.278`，四种
treatment 均处于相同 clipping regime。因而在这个共同 eval-reencoded 世界中，聚合提供了
真实、非单纯降权的正向机制线索，历史 antagonism 不是 PCJR 代数上的必然。

原始零步面板的 validity gate 没有通过：原训练路径 current PCJR 与 eval-reencoded current
PCJR 的梯度 cosine 在 step `256/512/1024` 只有 `0.9224/0.9112/0.8969`，相对 L2 差约
`39–45%`。因此没有用这个结果直接授权训练。append-only bridge 随后把 current slot 绑定到
原训练 graph，并把历史 raw batch 按 K4 实际 train-mode 语义重编码；它通过了 state、batch、
route、buffer、RNG、gradient reconstruction 和 native replay 的全部硬门，才授权唯一一条
256-step replay。

训练 recipe 保持 native identifiable-future MSE 与 `0.09` native SIGReg。step `1–128`
仍为 current-batch predictor-only PCJR `0.09`；step `129–256` 改为
`B_current/B1/B32/B128` 四个 complete-pair batch 的 PCJR 均值，总权重仍为 `0.09`。历史
raw input 在每一步以当前参数重编码，history/action/target detach，辅助梯度只进入
`predictor + pred_proj`；没有新增参数、head、inference path 或额外 optimizer step。

严格比较对象是相同 worker-0 deterministic trajectory 的 step-256 state，不是早期使用不同
worker/runtime 的 exploratory screen：

| checkpoint | future | history | worst | switch | gain | alignment | normalized error | weighted `beta` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| deterministic parent | `0.50195` | `0.51758` | `0.04297` | `0.97656` | `0.1664` | `0.4804` | `0.7872` | `1.3230` |
| persistent K4 | `0.51367` | `0.52930` | `0.06250` | `0.93750` | `0.1745` | `0.3988` | `0.8425` | `1.3064` |

paired delta 中，future 与 balanced macro 均为 `+6/512`，future 的 95% CI 为
`[+1,+12]/512`；history 的区间跨零。与此同时，switch 为 `-10/256`，alignment 明确下降，
normalized error 明确上升，校准 pair 从 `238` 降为 `202`。prediction response energy ratio
从 `0.1198` 增至 `0.1915`，其中正交于真实 target response 的 normalized energy 从
`0.0922` 增至 `0.1610`。所以 K4 产生了可检测的 assignment 信号，却主要把 response 沿
错误方向放大；这比“有效/无效”的二分判断更精确。

为判断损伤来自 target geometry、Predictor 还是跨模块 basis 不兼容，固定 parent(`P`) 与
K4(`K`) 两个 checkpoint，把 `C=encoder+projector+action_encoder`、Predictor trunk 和
`R=pred_proj` 做了八格零训练互换。所有 cell 均严格加载、target separation 全过；这里只
展示决定性格：

| `C/P/R` | future | history | worst | switch | gain | alignment | normalized error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `PPP` | `0.5020` | `0.5176` | `0.0430` | `0.9766` | `0.1664` | `0.4804` | `0.7872` |
| `PKP` | `0.5156` | `0.5391` | `0.0977` | `0.9375` | `0.2168` | `0.3992` | `0.8613` |
| `KPK` | `0.5059` | `0.5273` | `0.0547` | `0.9727` | `0.1368` | `0.4789` | `0.8080` |
| `KKK` | `0.5137` | `0.5293` | `0.0625` | `0.9375` | `0.1745` | `0.3988` | `0.8425` |

对另外两个因子取平均后，K4 Predictor trunk 的主效应为：future `+0.0088`、history
`+0.0132`、worst `+0.0332`，同时 switch `-0.0361`、gain `+0.0440`、alignment
`-0.0806`、normalized error `+0.0537`。conditioning/target path 与 `pred_proj` 的效应小得多，
且部分抵消 Predictor 的损伤。`PKP` 已在纯 parent basis 中复现该 phenotype，`KPK` 则仍
接近 parent，因而主要解释不是模块拼接失配。该矩阵是机制定位，不是可部署 hybrid，也不
授权从中挑 checkpoint。

旧 parent score 与 K4 score 曾分别在 default 和 deterministic CUDA runtime 下产生。最终矩阵
在一个统一 deterministic runtime 中重评分；两个 source diagonal 的 256 个 pair 离散决策
全部精确复现，连续差异被预先冻结的 `1e-5–2e-4` runtime envelope 覆盖。两次未通过对角
重现的实现尝试没有产出科学结果，只保留失败回执。这一数值恢复不改变上述效应方向。

因此当前不增加 K、权重或训练预算。唯一下一 kill test 在 parent 与 K4 各自 native latent
坐标中，将 `B256` 或 `[B1,B32,B128,B256]` 的 predictor-only PCJR 梯度投影到冻结 Motion
Development 的 `beta`、response gain、normalized response error 与 matched-vs-cross
assignment surrogate；同时单列 native MSE 与 `pred_proj`。如果 PCJR trunk 梯度本身在
held-out pairs 上方向错误，就停止延长该 estimator；如果 PCJR 正确而 MSE/聚合/clip 后方向
错误，则保留条件关系目标，把根因转向目标竞争或采样覆盖。两种结果都不能否决 pairing
这一更宽方法族。

## 6. Step-0 与冻结身份

下面九项是 PCJA+CCRM 在训练前已经通过的冻结审计；它们本身不替代终点评测：

1. release/code/data/init SHA 全匹配，完整 release audit PASS；
2. primary batch bytes/order、sampler、32 pairs 及 Motion twin invariants 不变；
3. 每 pair 的 query RGB 与完整 action 相同，history/future 不同，privileged key 不进入模型；
4. auxiliary 恰好一次 Predictor、零次 encode、零次 SIGReg；调用前后 RNG、module mode 与
   buffers 恢复；
5. PCJA 与 CCRM 均 finite、所有 target scales `>1e-8`、target gradient 为 `None`；
6. auxiliary 全参数梯度仅 `predictor/pred_proj` finite nonzero，其余三组为 `None`；
7. primary loss/gradient 与无 overlay control 一致，总损失精确为
   `primary + 0.09 * (PCJA + CCRM)`；
8. `N=3` 前向与梯度对旧 ActionDelay 实现 parity，`N=2` 通过手算与 chance `log(2)`，
   同时置换不变；非法 shape、重复/越界 group 与非有限值 fail closed；
9. optimizer step 仍为 0，且 Development/Public 在 preflight 中从未读取。

组合的 step-0 receipt 还证明：只执行一次辅助 Predictor；组合值与梯度精确等于既有
PCJA+CCRM；binary gain 稳态与解析根 `1.12275` 一致；zero response loss 为
`1+log(2)`；real BF16 batch 上 PCJA/CCRM 均只进入 `predictor/pred_proj`，且不存在类似
TA-CCRM 的分量数量级支配；optimizer、Development/Public/CEM 调用均为零。全部检查通过后
才执行固定 8,192-step 训练；终点失败没有被 step-0 诊断量覆盖。

本次组合实验的关键冻结身份：

| 对象 | SHA256 |
|---|---|
| 训练时 Motion release | `a04cbb76d28d614c8f95f8c8fc2ed68ab03c05b2f57beb2cf682b0d31fc0031c` |
| recovery 时 release metadata/API snapshot | `0b34365e6dee8d02af321f3af2cb4c4a6b2db1fcf304d01f22dfe462fc3334e0` |
| Development manifest | `48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b` |
| Loader Validation table | `64d43c931f106c2d53e3c3084e62381d2f2640c9d943e269475f3fb76aaa2de4` |
| 初始化 checkpoint | `9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59` |
| 终点 checkpoint | `fad2016b9e250eb2bf4922c35fc4449424aa2b5a1b319237bfb4ae566267aa1d` |
| implementation freeze | `f9e142bd02de76928bc921261a9bf8d4f95ff3b1523fa02b5ddb65ac980dfbb6` |

当前 linear exact-penalty 实验另用 append-only 身份，不改写上述失败记录：

| 对象 | SHA256 |
|---|---|
| 双快照 comparator prereg | `a323f8cff4fdf4f3df7764c81b27eb7ff401398ae3fc46e269ec075c7e4153e3` |
| linear objective | `57aba9e18363bd8e2142105bcd7b3c544b4720caaa7dd674c1e6e8af0a8c0976` |
| CUDA/BF16 comparator receipt | `3212698e4f88898e1aecc0bae1420f2cfb98f1a3a2c489c265b3ed1820756fd8` |
| 独立 conditional-GO review | `f1785641018e3dcbf6e87b47d976e094e57327cb4d63e0428c9b754897d1e99a` |
| 单次训练授权 | `6d27035d531de3ca4baa5d8371c74883bdc1dbab5341f107e935327be76cde0a` |
| 固定 runner | `45d7f8c5863a87afc300751258f772d006de5368b33880f423d80fcb15cca2ee` |
| linear 终点 checkpoint | `e47976be464175551ec58b787857bdced5b4d725f76cf37f3e74c4b399eef54a` |
| linear Development decision | `589b2b7a3b0f0b0cc06e984ce66128e3c6cb7104d5a173c862dfafd158a9ebb6` |
| linear 终点分量审计 | `e151420082e6e5ff17c7c3dd223f52d015aff550603290b8e5d02509f95f2a24` |
| linear 严格失败分解 | `46c7828bca0ad8b06ea88f04f3a13249fd806d615d8aa1cee2a881e3862a1f2e` |
| pair-gradient 审计预注册 | `12d5a2a6cf7ca902057ce1735fb0c0ececc27c9a7c129f766a4202cc664e56ac` |
| pair-gradient 审计结果 | `2b125f9ecb0dbf4bcf04aa8d0f16d3b65372c495734dca3a15217badd002500c` |
| Motion 根因收敛判定 | `c84200df7f608462512a78448241e54ecd127d2a1e64899f7b5ff8354535566b` |

追加正对照与 Motion-PCJA 审计使用 append-only 身份，不改写上述训练记录：

| 对象 | SHA256 |
|---|---|
| ActionDelay 成功正对照结果 | `1d6a8030d7c3b093d97ac6bc4ff38d0b7ce1e6a7f3c55fe636dbf3328bbab14f` |
| Motion-PCJA pair-gradient 预注册 | `4929b74030f9575e9118419471fb23bced7c5179536312b4b5082f90252a61c5` |
| Motion-PCJA 审计实现 | `62af17c71d2137756e01662563b99bec318cb8072439a50b5c51d465fb28c5b3` |
| BF16 stationarity 恢复 addendum | `bb2adb42ee11d68ebf051f0f78a5382d5c9856c60af3eb40d61a2340ebcdc1de` |
| Motion-PCJA pair-gradient 结果 | `066c7e736f6be69e0281d2766aafc9c8e0dd1639193cb674b4f77d726c53a233` |
| 跨成功/失败对照根因判定 v2 | `9af53389fd9e51cc846ee97092a7b3e3ba7dded2a1acc89eae0539bc93f50892` |

独立 review 的实际 reviewer 身份与权限边界记录在原始回执中；它只复核是否值得执行一次
固定训练，不作为方法有效性证据。

训练完成后，ContextWorld 的 release metadata 与通用 adapter API 有更新。checkpoint-only
recovery 将训练 release 与当前评分 release 分开记录，并确认 release id、Development split、
pair count、manifest 与 Loader Validation table 全部不变；Public Test 保持
`closed_not_read_not_scored`。因此终点差异不能归因于 benchmark 数据漂移，也没有把两个
release SHA 混写成同一训练身份。

实验 overlay 只写在本 research 目录，不修改 ContextWorld release 或其失败 reference receipt。
新的 append-only prereg/addendum 必须明确授权 follow-up Development；不得通过改旧 YAML
绕过 `failed_development` 状态。

## 7. 研究主张与创新边界

本项目当前最稳的主线应收窄为：

> 全局健康的 JEPA latent 仍可能对 episode dynamics 条件失明；对 matched query 的
> prediction–future 代价矩阵做对称条件分配，并将辅助梯度隔离到 Predictor 路径，能够在
> 不改推理模型的情况下恢复某些 history-conditioned response；但**条件可分不等于条件
> 校准，单 batch properness 也不等于跨 query 一致性**。连续隐藏动力学还要求 prediction
> response 的方向与幅值匹配 real response，并且一个 query 上的更新不能系统性破坏其他
> query 的 target-axis assignment；这些要求都不能由 target selection accuracy、训练 loss
> 或全局 non-collapse 代替。

这与只改变 action branch 的 action sensitivity、只改善 latent marginal geometry，或仅仅
增加 context token 不同。真正需要验证的贡献包括：

- matched history intervention benchmark、row/column 双向 estimand 与 target-normalized
  response anti-spoofing gate；
- history readable / future usable / predictor coupled 的机制分解；
- PCJA 在 ActionDelay 成功、在 Motion Damping 呈现“switch 正确但 future 不准”的受控
  反例，旧 ranking 的“selection 近满分但 response error=378”反例，以及由此得到的
  assignment–calibration 区分；
- CCRM→TA-CCRM 的单因素反例：修复一个可精确重构的 output-space null direction 仍可能因
  共享 Predictor Jacobian 与 gradient clipping 破坏另一必要响应分量；
- PCJA+CCRM 的组合反例：两个 output-space 信号可以局部兼容，经过共享 Predictor Jacobian
  后却在参数空间冲突并由一项主导；因此“把两个必要 estimand 相加”不等于得到充分目标；
- PCJR-CV 的跨 batch 反例：一个在真实 paired future 上严格驻点的 proper residual，可以在
  当前 batch 同时改善 response/center/assignment，却在共享 Predictor 上改善历史 response、
  反向伤害历史 target-axis assignment；
- current/anchor finite-transfer 与 MSE–PCJR 分量矩阵，使“局部公式正确”“优化器是否翻转”
  和“跨 query 随机梯度估计是否一致”成为三个可独立检验的机制层；
- 极小、reconstruction-free、无新增参数的条件关系辅助项；
- ICL 与原任务 CEM 的同-checkpoint Pareto 证据；
- 后续将显式 pair 替换为无 hidden-label context intervention 的可能路径。

跨任务实验现已给出真实反例，因此论文贡献应暂时收缩为诊断、ActionDelay 机制发现与
assignment–calibration/optimization-geometry 反例，不用更多候选掩盖反证。matched
LeWM/PLDM 分量与梯度路由审计已经完成：Motion 上两类模型的 prediction route 局部有利，
而作用于 shared target/Encoder 的边缘表示 route 都会反向伤害条件 response；同时，最新
九任务矩阵表明 PLDM 只在 ActionDelay 明显占优，在 Action Strength、Reacher、Cube 和
Portal 并不是通用修复。因此不再把“移植完整 PLDM objective”当作默认下一方法。

exact-future 与 squared-BC 的相反反例进一步收窄了方法空间：直接回归 absolute future 会
丢失 response coupling；只校准 centered response 又允许公共中心漂移；用 squared feasibility
补中心虽有真实、单调的作用，却在固定预算内留下大量 active pairs。linear exact penalty
证明即使为每个未过界样本保持常量 output-space 压力，仍不能修复 absolute conditional
future；因此 margin、权重、cutoff 与 penalty power 的后验搜索全部停止。

成功 ActionDelay 正对照、失败 Motion-PCJA、proper PCJR-CV 轨迹与跨锚点分量审计现已完成。
它们否定了三个会导致方案发散的判断：target 漂移不是失败特异信号；Motion-PCJA 不是
BC-CCRM 式强负质量；把 pair loss 改成 proper residual 也不会自动获得 population-consistent
assignment。因此不引入 gradient surgery，不因漂移单独引入 EMA/lagged target，也不继续
搜索 PCJA margin、weight、cutoff、CCRM penalty power 或新的 `beta` 权重。

已经完成一个**零训练步 PCJR estimator kill test**。它保持完全相同的 PCJR 公式，固定
`K=4`（当前 batch 加严格早于它的三个最近 persistent anchors），并选择唯一主语义：固定
raw `pixels/action`，在每个当前 checkpoint 的同一 eval 世界重编码且全部 detach。它不是
stale-latent replay；两种语义不得混写。矩阵比较：

- `C0_actual`：native MSE + 原训练路径当前 batch PCJR，只作语义复现参考；
- `C0_panel`：native MSE + 当前 eval-reencoded batch PCJR；
- `D`：native MSE + `C0_panel` 中 PCJR 的 `1/K`，用于排除“只是降权”；
- `A`：native MSE + 四个 complete-pair batches 分别计算后再 uniform 平均的 PCJR。

因此 `A-D` 恰好隔离三个历史 batch 的增量；禁止把四批 concat 后调用一次 PCJR，因为那会
改变 batch-global scale，成为另一 loss。

四行在冻结 step `256/512/1024` 上报告 current diagonal 与历史 panel 的 response、center、
`beta^2` 方向、full-gradient norm/clip 与 route/identity。所有科学方向门都通过，只有预注册的
actual-current/eval-panel 语义 parity 未过，因此当时的三态判定落在 `inconclusive`。后续
exact-native bridge 没有放宽 `0.95` 门，而是直接复现训练 graph、raw batch 与 train-mode
重编码语义，并通过了唯一 replay 的授权门。实际 K4 训练给出小而可信的 assignment 增益，
但 response alignment/NRE 同时恶化；模块互换又把两者共同定位到 Predictor trunk。

当前唯一允许的后续是零训练的 native-coordinate gradient-to-estimand 审计。它不会把终点
梯度冒充历史上的 step-256 update：只回答在 parent/K4 各自冻结终点，原 PCJR source 若作
一次无穷小下降，会把已打开的 256-pair Development `beta/gain/error/assignment margin` 推向
何处。只有这一审计支持明确、预注册的最小修复，才允许产生一个新训练候选；否则停止 K4
延长并保留 ActionDelay PCJA 作为条件关系正例。1,024、Contact、Public、CEM 与额外 seed
继续关闭。

## 8. 证据入口

- [ContextWorld ICL Suite v2 完整性重封判定](../../../ContextWorld/configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_decision_v2.json)
- [ContextWorld ICL Suite v2 当前 13-row scoreboard](../../../ContextWorld/artifacts/evaluation/contextworld_icl_suite_v2_release_addendum_v1/public_scoreboard.json)
- [predictor-only PCJA 预注册与精确定义](configs/action_delay_h7_a0_aux_pcja_predictor_only_v1.yaml)
- [PCJA 核心实现](scripts/paired_conditional_joint_assignment_v5.py)
- [predictor-only 梯度路由实现](scripts/run_action_delay_h7_a0_aux_pcja_predictor_only_v1.py)
- [ActionDelay Private Development 结果](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1/candidate_results/action_delay_h7_a0_aux_pcja_predictor_only_v1.json)
- [ActionDelay Public 摘要与独立重评分](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_public_v2/public_summary_v2.json)
- [ActionDelay CEM 最终回执](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_cem_resolution_v1/consumption_receipt.json)
- [ActionDelay PCJA 逐 pair 成功正对照](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_v1/analysis/pair_gradient_positive_control_v1.json)
- [full-gradient PCJA Private 结果](artifacts/action_delay_h7_full_f_pcja_private_development_v1/candidate_results/full-f-pcja-s3072-step1024.json)
- [full-gradient PCJA CEM 反证](artifacts/action_delay_full_f_pcja_cem_retention_v1/consumption_receipt.json)
- [Motion Damping binary PCJA 预注册](configs/pusht_motion_damping_binary_pcja_v1.yaml)
- [Motion Damping v5 step-0 PASS receipt](artifacts/pusht_motion_damping_binary_pcja_v1/preflight/cuda_bf16_real_batch_v5.json)
- [Motion Damping exact Development-only terminal result](artifacts/pusht_motion_damping_binary_pcja_v1/recovery/development_only_result_v3.json)
- [Motion Damping terminal decision receipt](artifacts/pusht_motion_damping_binary_pcja_v1/recovery/terminal_decision_v1.json)
- [Motion Damping 四-checkpoint response matrix](artifacts/pusht_motion_damping_binary_pcja_v1/recovery/response_matrix_analysis_v1.json)
- [Motion-PCJA pair-gradient 审计预注册](configs/pusht_motion_damping_binary_pcja_pair_gradient_v1.yaml)
- [Motion-PCJA 逐 pair 梯度与 properness 审计](artifacts/pusht_motion_damping_binary_pcja_v1/analysis/pair_gradient_cancellation_v1.json)
- [BF16 stationarity 实现失败留痕](artifacts/pusht_motion_damping_binary_pcja_v1/preflight/pair_gradient_bf16_stationarity_failure_v1.json)
- [BF16 stationarity 恢复边界](configs/pusht_motion_damping_binary_pcja_pair_gradient_v1_bf16_stationarity_recovery.yaml)
- [response matrix 可复现分析脚本](scripts/analyze_pusht_motion_damping_response_matrix_v1.py)
- [legacy ranking 同 scorer 重评分](artifacts/pusht_motion_damping_binary_pcja_v1/recovery/legacy_ranking_development_response_v1.json)
- [legacy multi-term fit 同 scorer 重评分](artifacts/pusht_motion_damping_binary_pcja_v1/recovery/legacy_fit_development_response_v1.json)
- [legacy projected geometry 同 scorer 重评分](artifacts/pusht_motion_damping_binary_pcja_v1/recovery/legacy_projected_geometry_development_response_v2.json)
- [Motion Damping terminal postfit recovery 边界](configs/pusht_motion_damping_binary_pcja_v1_terminal_postfit_recovery_addendum_v1.yaml)
- [CCRM exact Development 结果](artifacts/pusht_motion_damping_ccrm_v1/recovery/development_only_result_v1.json)
- [CCRM target-axis center-bias 分解](artifacts/pusht_motion_damping_ccrm_v1/recovery/center_bias_analysis_v1.json)
- [TA-CCRM exact Development 结果](artifacts/pusht_motion_damping_ta_ccrm_v1/recovery/development_only_result_v1.json)
- [TA-CCRM/CCRM/PCJA 终点梯度分解](artifacts/pusht_motion_damping_ta_ccrm_v1/analysis/terminal_component_gradient_audit_v2.json)
- [PCJA+CCRM 预注册](configs/pusht_motion_damping_pcja_ccrm_step0_v1.yaml)
- [PCJA+CCRM 真实 CUDA BF16 step-0](artifacts/pusht_motion_damping_pcja_ccrm_v1/preflight/cuda_bf16_real_batch_v1.json)
- [PCJA+CCRM exact Development 结果](artifacts/pusht_motion_damping_pcja_ccrm_v1/recovery/development_only_result_v1.json)
- [PCJA+CCRM 终点分量梯度审计](artifacts/pusht_motion_damping_pcja_ccrm_v1/analysis/terminal_component_gradient_audit_v1.json)
- [四个 paired-objective 端点的冻结停止判定](artifacts/pusht_motion_damping_pcja_ccrm_v1/analysis/paired_objective_decision_v1.json)
- [独立 Luna Max 停止标准复核](artifacts/pusht_motion_damping_pcja_ccrm_v1/analysis/independent_luna_rejection_audit_v1.md)
- [LeWM/PLDM 正则梯度路由分解](artifacts/pusht_regularizer_route_decomposition_v1/result_v1.json)
- [frozen Encoder/Projector + CCRM 预注册](configs/pusht_motion_damping_frozen_ccrm_v1.yaml)
- [current-release exact CUDA/BF16 step-0](artifacts/pusht_motion_damping_frozen_ccrm_v1/preflight/cuda_bf16_step0_release_reseal_v2.json)
- [frozen CCRM 正式 Development 停止判定](artifacts/pusht_motion_damping_frozen_ccrm_v1/development/decision_v1.json)
- [live/frozen CCRM center-response 分解](artifacts/pusht_motion_damping_frozen_ccrm_v1/analysis/endpoint_decomposition_v1.json)
- [frozen CCRM 终点 exact-batch 梯度审计](artifacts/pusht_motion_damping_frozen_ccrm_v1/analysis/terminal_component_gradient_audit_v1.json)
- [frozen CCRM 机制边界与下一 kill test](artifacts/pusht_motion_damping_frozen_ccrm_v1/analysis/mechanism_decision_v1.json)
- [pair-normalized exact-future 正式失败判定](artifacts/pusht_motion_damping_pair_normalized_exact_future_v1/development/decision_v1.json)
- [pair-normalized exact-future 失败机制分解](artifacts/pusht_motion_damping_pair_normalized_exact_future_v1/analysis/failure_mechanism_v1.json)
- [squared-hinge BC-CCRM 正式失败判定](artifacts/pusht_motion_damping_boundary_constrained_ccrm_v1/development/decision_v1.json)
- [BC-CCRM 同训练批次终点梯度审计](artifacts/pusht_motion_damping_boundary_constrained_ccrm_v1/analysis/terminal_component_gradient_audit_v1.json)
- [BC-CCRM 严格否决范围](artifacts/pusht_motion_damping_boundary_constrained_ccrm_v1/analysis/failure_mechanism_v1.json)
- [跨 endpoint 全局 response-scale 反事实](artifacts/binary_response_scale_counterfactual_v1/result_v1.json)
- [linear exact-penalty 双快照零步 comparator](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/preflight/comparator_v1.json)
- [linear exact-penalty 独立 conditional-GO 复核](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/preflight/independent_review_v1.md)
- [linear exact-penalty 正式 Development 失败判定](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/development/decision_v1.json)
- [linear exact-penalty 同训练批次终点梯度审计](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/analysis/terminal_component_gradient_audit_v1.json)
- [linear exact-penalty 严格否决范围](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/analysis/failure_mechanism_v1.json)
- [三快照逐 pair 梯度抵消与 target 漂移审计](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/analysis/pair_gradient_cancellation_v1.json)
- [Motion 当前根因收敛判定与下一正对照](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/analysis/root_cause_synthesis_v1.json)
- [跨成功/失败正对照后的根因判定 v2](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/analysis/root_cause_synthesis_v2.json)
- [pair-gradient 数值边界与失败留痕](artifacts/pusht_motion_damping_exact_penalty_boundary_ccrm_v1/preflight/pair_gradient_numerics_addendum_v1.md)
- [PCJR-CV 1,024-step 确定性轨迹](artifacts/pusht_motion_damping_pcjr_cv_deterministic_trajectory1024_v1/training/s14321_step1024_v1/deterministic_trajectory1024_execution_receipt_v1.json)
- [PCJR-CV 时序 AdamW 归因](artifacts/pusht_motion_damping_pcjr_cv_temporal_optimizer_attribution_v1/training/s14321_step1024_v1/temporal_optimizer_attribution_v1.json)
- [PCJR-CV 单锚点有限更新迁移](artifacts/pusht_motion_damping_pcjr_cv_anchor_transfer_attribution_v1/training/s14321_step1024_v1/anchor_transfer_attribution_v1.json)
- [PCJR-CV 多锚点有限更新迁移](artifacts/pusht_motion_damping_pcjr_cv_multi_anchor_transfer_attribution_v1/training/s14321_step1024_v1/multi_anchor_transfer_attribution_v1.json)
- [native MSE–PCJR 跨锚点分量归因](artifacts/pusht_motion_damping_pcjr_cv_component_transfer_attribution_v1/training/s14321_step1024_v1/component_transfer_attribution_v1.json)
- [Motion 根因收敛判定 v3](artifacts/pusht_motion_damping_pcjr_cv_component_transfer_attribution_v1/analysis/root_cause_synthesis_v3.json)
- [K=4 current-reencoded PCJR estimator 零步预注册](configs/pusht_motion_damping_pcjr_cv_multi_query_estimator_kill_test_v1.yaml)
- [K=4 current-reencoded PCJR estimator 执行器](scripts/run_pusht_motion_damping_pcjr_cv_multi_query_estimator_kill_test_v1.py)
- [K=4 exact-native bridge 结果](artifacts/pusht_motion_damping_pcjr_cv_native_semantics_bridge_recovery_v4/training/s14321_step1024_v1/native_semantics_bridge_recovery_v4.json)
- [K=4 persistent 256-step 预注册](configs/pusht_motion_damping_pcjr_cv_k4_persistent_screen_v1.yaml)
- [K=4 persistent 训练与梯度路由回执](artifacts/pusht_motion_damping_pcjr_cv_k4_persistent_screen_recovery_v2/training/s14321_step256_v1/k4_persistent_execution_receipt_v1.json)
- [K=4 strict Development 结果](artifacts/pusht_motion_damping_pcjr_cv_k4_persistent_screen_recovery_v2/development/s14321_step256_v1.json)
- [K=4 2×2×2 模块互换预注册](configs/pusht_motion_damping_pcjr_cv_k4_module_swap_v1.yaml)
- [K=4 统一 runtime 模块互换归因](artifacts/pusht_motion_damping_pcjr_cv_k4_module_swap_recovery_v3/development/s14321_step256_v1/k4_module_swap_attribution_recovery_v3.json)
- [target stop-gradient V8 反证](artifacts/paired_terminal_target_stopgrad_sigreg_v8_validation/development/action_delay_stage1_s3072_step256_v1_gate_decision.json)
- [Encoder-only / history-value 阶段报告](results/history_value_encoder_only_stage_report_v1.md)
- [target-JTCov 任务广度报告](results/joint_temporal_covariance_sigreg_task_breadth_report_v1.md)
- [conditional identifiability 理论说明](results/identifiability_corrected_mse_theory_v1.md)

机器回执保留完整路径、输入哈希、checkpoint 身份和 gate 字段；本文只呈现支撑当前研究
判断与下一步证伪协议所需的结果，避免把 recovery/version 执行流水写成方法叙事。
