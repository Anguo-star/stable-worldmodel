# 条件动力学 ICL：从边缘非坍缩到条件联合响应

> 面向外部读者的论文式技术报告见
> [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)。它按“问题定义—理论分析—根因拆分—方法—
> 跨任务验证—局限”组织，并纳入 2026-08-28 的统一完整训练对照以及 Motion/Contact hidden planning 与
> rollout-consistency 结果。本文件保留扩展实验、实现边界和历史证据，作为可复核的补充材料。

> **2026-08-28 统一完整训练对照。** 通过同一云端入口完成 seed 3072、10 epoch、8 卡、
> `50/50` joint-scratch 的 matched native/COJA。Robot Arm Mass 的完整 native 已达到
> future/worst/NRE=`0.842/0.793/0.141`，COJA 为 `0.861/0.848/0.233`：早期短预算失败不能
> 再被归因为方法缺失，COJA 在该任务只有较小分配改善并伴随校准权衡。Portal Exit 则给出明确
> 方法效应：native→COJA 的 future=`0.508→0.717`、worst=`0.441→0.660`、joint pair=
> `0.016→0.324`、gain=`0.043→0.605`、NRE=`0.917→0.329`；两臂 switch 均为 `1.0`，说明
> 真正被修复的是响应幅值与真实 future 对齐，而非任意历史敏感性。Portal COJA 的标准 TwoRoom
> CEM 为 `300/300`。对相同 256 个 query pair 做配对 bootstrap 后，future、worst 与 joint-pair
> 的 COJA−native 95% 区间分别为 `[+17.77,+24.02]pp`、`[+13.67,+30.08]pp` 和
> `[+25.00,+36.72]pp`，NRE 改善区间为 `[0.529,0.637]`。这些区间刻画 query 抽样不确定性，
> 不是训练随机种子不确定性；结果仍不替代后续多种子和 Public Test。
> 机器可读身份见
> [完整训练摘要](artifacts/contextworld_joint_scratch_full_single_seed_v1/summary.json)。

> **2026-08-27 Motion 单阶段闭环。** 从公开 PushT 初始化直接训练 4,096 steps，保持原 LeWM
> state dict、`50/50` 数据、一步 COJA 与 `ρ=0.25`，只把既有 hidden native MSE 的 `0.125`
> 分给真实第二步 self-rollout。相对同数据、同预算的一步 COJA，RC-COJA 在 h2 和未训练 h3
> 隐藏规划上分别改善 `57.39 [52.19,62.42]` 与 `39.08 [33.90,44.21] px`，正确历史收益 DID
> 分别为 `4.12 [2.90,5.34]` 与 `2.28 [0.28,4.01] px`；h1 则退化
> `3.01 [1.86,4.13] px`。标准 PushT 的同 300 queries 为 no-aux/一步 COJA/RC=`203/188/194`，
> RC−COJA=`+2.00 [-2.67,+6.67]pp`。因此两阶段 warm start 不是必要机制，RC 也没有被识别出
> 独立的大幅 retention 损伤；但 h1 权衡与区间不确定性必须保留。另一个单因素负对照表明，
> 从普通 replay 无条件抽取多样 action 明显弱于 Motion 的部署相关 zero hold，甚至把正确历史
> 收益翻负。因此通用原则是 **support-matched rollout consistency**，不是最大化 action diversity。
> 冻结配方的独立 seed `14322` 已完成精确 31 点协议：h1/h2/h3 的 RC−COJA 改善为
> `-3.30/+57.60/+41.56 px`，seed `14321` 为 `-3.01/+57.39/+39.08 px`；h2/h3 正确历史收益
> DID 在两个 seed 都明确为正。标准 CEM 的两 seed 方法效应为 `+2.00/-1.33pp`，层级均值
> `+0.33 [-4.00,+4.50]pp`。因此长程收益与短程代价均已复现，retention 则没有稳定 RC 方向。
> 发现期配方至此冻结，下一步进入 publication-level 跨任务统计。

> **2026-08-27 Contact rollout 跨任务更新。** 固定 Motion 选出的 `ρ=0.25`，从同一个
> 4,096-step Contact COJA checkpoint 出发做 1,024-step matched continuation。重复 query
> action 的 RC 虽把 h2 物理误差从 `29.31` 降到 `24.84 px`，却使标准 CEM300 从 `212` 降到
> `197`。只把第二个 action block 换成原始 PushT 训练总体中的连续五步动作后，h2 和未训练 h5
> 相对 control 分别改善 `4.39 [2.93,5.90]` 与 `4.37 [1.87,6.84] px`，标准 CEM300 恢复到
> `216`，相对 control 为 `+1.33 [-3.33,+6.00]pp`。因此短自回归一致性是跨 Motion/Contact 的
> 真实机制，重复动作造成的窄 action support 才是 retention 损伤根因。当前 empirical-action
> RC-COJA 是双 seed Pareto 正例；仍不增加 loss family、参数、encoder、adapter、head 或部署
> 计算，但保留 conditional overlap 与短轨迹数据假设。从公开初始化直接训练 4,096 steps 后，
> RC 相对同 mixture 一步 COJA 在 h2/h5 再改善 `3.16 [1.58,4.79]` 与
> `3.09 [0.65,5.46] px`，标准 CEM300 为 `207/206`，证明两阶段续训不是必要条件，也未检测到
> RC 的增量 retention 代价。公开原始数据参考为 `237/300`；这约 `10pp` 的共同差距在一步
> COJA 中已经存在，必须与 RC 方法效应分开。
> 冻结配方的独立 seed `13314` 进一步复现了 h2/h5 改善
> `3.67 [2.06,5.34]` / `4.37 [1.76,6.92] px`，h1 效应为
> `+0.08 [-0.27,+0.42] px`。两个 training seed 的标准 CEM 方法效应为
> `+0.33/+2.00pp`，分层 bootstrap 均值 `+1.17 [-2.17,+4.50]pp`。因此 Contact
> 长程收益不是单 seed 偶然，也没有检测到 RC 特异的一步校准或原任务保持代价。

> **2026-08-26 rollout 一致性更新。** 最新实验发现，单步 COJA 已经让 Predictor 使用历史，
> 但该条件响应不会自动在 self-conditioned rollout 中保持。只把既有 hidden native MSE 的
> `25%` 分配给真实第二步自回归 target、保留一步 COJA 且不增加参数/模块/推理计算后，Motion
> 两步隐藏规划相对同数据同预算 placebo 改善 `43.83 px [39.57,48.01]`，未训练的三步改善
> `30.92 px [25.32,36.55]`；一步则退化 `3.04 px [1.81,4.21]`。标准 PushT 同 100 episode
> 为 candidate/placebo=`60/57`，配对区间 `[-6,+12]pp`，没有检测到 retention 损伤。因子拆分
> 表明第二步 native MSE 是主要活性成分，第二步 relation 不是必要项。当前最简候选因此更新为
> **一步 COJA + 短自回归原生 MSE（RC-COJA）**；仍需显式 conditional overlap 和真实 trajectory
> continuation，跨任务与最终多 seed 尚未完成。

> **2026-08-26 更新。** 将 Motion 训练从反复使用 2,048 个 Cartesian query 模板改为覆盖
> 全部 8,192 个 matched query 后，center-free joint variant 相对同数据 no-aux 的
> future/switch/gain/alignment/NRE 从 `0.494/0.441/0.0065/0.017/1.130` 改善为
> `0.660/0.969/0.370/0.520/0.767`。这确认旧 `NRE=1.120` 主要包含 Predictor 的
> query memorization，而不是连续 response 不可学习。与既有完整结果相同的 seed42×300
> matched CEM 为 joint/native=`213/232`，paired difference `-6.33pp`，95% interval
> `[-11.00,-1.67]pp`。因此连续校准已明显修复，Motion 的 planner-function retention 仍是
> 独立未解问题；suite 统计改为报告连续效应分布，不再用单一硬阈值裁决方法。

> 最新结论（2026-08-25）：此前用 published source 判断 Contact planning 保持混合了**训练数据
> 与方法效应**。严格 matched-native 对照已经纠正这一点。4,096-step exact visible-overlap joint
> 相对同初始化、同 `64+64` 数据、同 sampler、同优化曝光的 no-aux LeWM，把 Development
> future/history/switch/worst 从 `0.521/0.594/0.668/0.406` 提高到
> `0.771/0.850/1.000/0.734`，gain 从 `0.015` 提高到 `0.447`，NRE 从 `0.984` 降到
> `0.579`。同进程、同 100 个 CEM query 的 source/matched-native/exact-joint 为
> `70/73/72`；joint 相对 matched-native 仅 `-1pp`，paired 95% interval `[-9,+7]pp`。
> 因此旧的 `75/69` source 点差不能再承担方法归因；当前证据是**大幅 ICL 提升且规划近似中性**。
>
> 2,048-step 四臂同进程 CEM 又完成了更强拆分：matched-native/exact-overlap/
> QA-only-approximate/history-diverse-RGB-approximate=`67/67/56/59`。exact-overlap 与 native
> 点数完全相同，而两个 approximate graph 均出现不利规划信号；同时它们在 Development 上又确实
> 比 native 学到条件响应。QA-only 相对 native 的 future/history/switch/gain 分别提高
> `+0.078/+0.227/+0.332/+0.120`，NRE 降低 `0.149`；history-diverse RGB 在其上继续改善
> gain `+0.137`、NRE `-0.107`。因此真问题不是 50/50 合成配比，也不是 joint auxiliary 本身，
> 而是当前 approximate objective 把 **Q/A mismatch nuisance 当作 hidden-dynamics response**。
> 但这只解释 Contact approximate matching，不是全部 planning 冲突。
>
> Motion 2,048-step 的严格 matched native 又补上了另一半。相同 `64+64` 数据、Cartesian
> sampler、初始化和预算下，no-aux 的 future/history/switch/worst 为
> `0.449/0.385/0.098/0.070`，gain=`-0.161`、NRE=`1.701`；exact joint 则为
> `0.533/0.609/0.918/0.184`，gain=`0.170`、NRE=`0.901`，确认 joint relation 是 ICL
> 改善的活性成分。然而同进程 CEM 的 source/native/joint=`70/66/55`，joint−native=`-11pp`
> （paired 95% interval `[-22,0]pp`），而 native−source 仅 `-4pp [-14,+6]pp`。所以 Motion
> 的 planning 损伤不能归给 50/50 数据；**exact conditional overlap 能教会 ICL，却仍不自动
> 保证共享 Predictor 在 planner-consumed queries 上保持原函数。**
>
> 当前最简且未被误杀的活性机制是：原 LeWM、原参数与推理结构、native MSE+SIGReg，加一个训练期
> center-free joint relation；pair 可由可见 `(query RGB, raw action)` 精确恢复，不需要 hidden
> label 或 pair annotation。它在 ActionDelay、Motion 与 Contact 都有 ICL 正例，在 Contact
> 2,048/4,096 又与 planning 共存；但 Motion matched-native 已明确表明它还不是通用 Pareto
> 方法。剩余简洁性/通用性缺口有两个：如何从普通 unmatched replay 获得足够精确的 conditional
> overlap，以及如何避免这种条件梯度在 overlap support 之外改写原规划函数。原 RGB
> qualification 曾因一个 noisy NRE `<1` 硬门停止；
> 后续训练属于明确的探索性 override，不能改写成“原资格门通过”。正式 Public、多 seed 与最终
> non-inferiority 仍保持关闭。
>
> 以下为历史阶段索引：Motion Damping 已获得首个**零新增参数、模块和推理计算**的
> ICL–planning Pareto 正例。function-anchored conditional bootstrap 在同一 checkpoint 上通过
> 六个条件响应门，100-query paired PushT CEM 与 source 同为 `57/100`。这证明现有 LeWM 容量
> 足够，额外 encoder、adapter、head 并非必要；paired joint signal 也不必然损害规划，真正风险是
> 它通过无保护的共享参数路径改写原 planning function。
>
> 固定配方随后以单 seed、无调参方式迁移到 Contact Friction。相对 residual-transition source，
> 8,192-step 正式预算把 future/history/switch/worst 从
> `0.498/0.568/0.590/0.391` 提高到 `0.750/0.867/0.980/0.688`，response gain 从 `0.016`
> 提高到 `0.374`，NRE 从 `1.016` 降到 `0.716`。因此跨任务 conditional signal 是真实的，
> 不是 Motion 特例或局部梯度假象。
>
> 但 Contact 终点仍未达到正式的 future/history/worst=`0.95/0.95/0.90` 与 gain=`0.50` 门，
> 故不运行 CEM、不打开 Public，也不补多 seed。对向条件共批 control 还略差于普通 matched
> sampler，排除了“只需改 batch 配平”的解释。当前配方应准确归类为 **Motion Pareto 正例 +
> Contact 部分迁移**，不能升格为通用方法；显式 matched pair、两阶段 warm start 与训练期
> frozen teacher 也仍未满足最终最简目标。
>
> 最新 canonical-margin 单因素验证又给出了一个清楚边界：在 exact-future loss 上加入到真实
> target 恰好归零的 `0.5` assignment barrier，2,048 step 的七项指标全部改善；8,192 step
> 也把 future/worst/gain 提到 `0.775/0.703/0.427`，同时 NRE 保持 `0.713`。这证明
> assignment bootstrap 可以无过冲地改善，但正式门仍失败四项，且 exact-future residual 几乎
> 未变。因此剩余瓶颈不是 margin 不够，而是跨 query 的 exact conditional future fit；margin
> 权重、形状和 schedule 到此停止。完整结果见 §5.22–§5.24。
>
> §5.25 随后用一次 privileged common-center oracle 和四个严格 2,048-step MVE 闭合该断点。
> 只把 canonical 8,192 prediction center 换成真实 target center，即可把
> future/history/worst 从 `0.775/0.873/0.703` 同时提高到 `0.984/0.984/0.984`，而 response
> 完全不变；但把 center loss 权重提高到 `4` 会使 gain 坍到 `0.022`，pair-midpoint center
> forward 与原 canonical 几乎等价，解冻现有 Encoder/Projector（无论 online target 还是冻结
> target copy）也显著更差。因此 common-center 是**充分的结果级瓶颈定位**，却不是可由标量
> 加权、barycentric mixup 或简单 target-network 修复的训练处方；这些近邻候选均已停止。
>
> §5.26 进一步验证了更接近原始目标的 `response-only` 联合关系。它在不增加部署参数、模块或
> 推理计算的情况下稳定通过连续 Motion 六门；冻结 action encoder 与 pred-proj BN 后，标准
> PushT CEM 从无保护训练的 `37/100` 恢复到 `54/100`，但仍未达到同次 source 的 `58/100`。
> 将它与既有 source-function anchor 合并也只有 `52/100` 对 `57/100`。因此直接条件响应确实
> 能教会 ICL，但只保护一个标量 function-drift 或删除 absolute-center 项都不足以保留规划；
> 当前唯一通过同 checkpoint ICL–CEM Pareto 的仍是保留 exact pair center 的 function-anchor
> 配方。response-only 的权重、步数和路由分支到此停止。
>
> §5.27 进一步直接保护 ordinary action-conditioned future difference。Motion 六门保持全过，
> paired CEM 从 point-anchor 的 `52 vs 57` 收窄到 `56 vs 57`；逐 episode exact McNemar
> `p=1.0`、95% paired bootstrap 为 `[-9,+7]pp`。这支持 action geometry 比平均 point drift
> 更接近 planner 所需量，但固定的“成功数不得低于同次 source”门仍差 1 个 episode，故不晋级、
> 不调 cycle/权重/步数，也不将统计不确定的 `-1pp` 误写成通用失败。
>
> §5.28 去掉 frozen teacher，首次用 simulator-real 的 `2 histories × 2 actions × futures`
> 四元组训练原 LeWM。256-template 版本因覆盖过拟合失败 NRE；扩到旧 legacy-prefix 的
> 2,048-template MVE 规模后，当前完整 recipe 在单 seed Motion 六门全部通过。这给出一个
> teacher-free 连续 ICL 存在性正例，但没有把收益单因素归因于四元组。candidate 的同一 SHA
> 在 CEM 为 `51/100`，source 为 `58/100`；全部 alternate action 分支都处于零接触状态。二者与
> intervention/planner support 失配假设一致，尚未完成 contact-matched 因果对照。
>
> §5.29 已完成这个 contact-matched 证伪。新的 2,048-template overlay 保证每个 alternate
> rollout 都发生 query contact、保持场地边界，且每个 template 都有非零 History×Action
> interaction；同一个零参数、无 teacher recipe 仍通过 Motion 六门。但同 SHA CEM 只有
> `40/100`，低于 source 的 `58/100`（paired 95% interval `[-28,-8]pp`），也低于预算匹配的
> 零接触候选 `51/100`（`[-23,+1]pp`，方向不利但统计未定）。source 与 candidate 累计更新步数
> 不同，故这些结果不能写成“contact 导致退化”；它们足以说明当前 contact 构造没有闭合规划门。
> 因此“是否 contact”不是 planner support 的充分统计量；一个重复的
> toward-block action ray 不能代表 CEM 消费的宽、多方向、多步 action-conditioned function。
> 当前接触实现关闭，不做幅值、权重、步数或模板 sweep；真实 History×Action pairing 家族保留。
>
> §5.30 重新检验了此前最强的 action-function anchor，而不是继续训练新 loss。固定 checkpoint
> 在 seeds `42/43/44` 上完成 300 个同 query paired CEM：candidate `180/300`、source
> `187/300`，点差 `-2.33pp`。事前固定的 `-5pp` 实用非劣界下，一侧 95% paired-bootstrap
> 区间为 `[-6.33,+1.67]pp`，结论是 **inconclusive**：既不能证明非劣，也不能证明实质劣化。
> 更直接地，同一 seed42、同一 100 个 query 的 source 重跑由 `57` 变成 `58`，7 个 episode
> outcome bit 发生翻转，实证说明“candidate 成功数必须逐字节不低于 source”只能作保守流程门，
> 不能作科学否决标准。该候选保留为最接近成功的方法，但不晋级，也不靠事后增加 episode 救援。
>
> §5.31 将 teacher-free `2×2` 四元组中的单一 action ray 换成原始 PushT replay 中无放回抽取的
> 多方向五步真实 action block；模型、目标、seed、预算和部署边界均不变。新候选再次通过 Motion
> 六门（future/history/switch/worst=`0.600/0.660/0.941/0.328`，gain=`0.307`，NRE=`0.908`）。
> 同一 100-query catalog 上 source/零接触 ray/replay-pair 为 `58/51/55`：replay arm 追回 `4pp`，
> 与 source 仅差 `3pp`，但两组 paired 区间均跨零。因此它是当前**最简、最强的 teacher-free
> 零新增参数候选**，不是已经证明规划非劣的最终方法；显式 simulator matched pair 与两阶段
> warm start 仍是尚未消除的训练复杂性。
>
> §5.32 已直接删除两阶段 Motion warm start，并分离了 residual reset 的混淆。直接从标准 PushT
> baseline 训练 residual joint-pair，`1,024` step 可过全部 ICL 响应门，CEM 却仅 `20/100`；延长到
> `3,072` 又使 NRE 恶化到 `1.064`。保持原 LeWM absolute 坐标且不重置任何权重后，单阶段
> `2,048` step 达到 history/switch/worst=`0.609/0.918/0.184`、gain=`0.170`、NRE=`0.901`，
> CEM=`54/100`；九项检查仅 raw future=`0.533` 低于旧 `0.55` 门，而 paired-balanced lower
> 已为 `0.555`。因此当前最接近原始目标的是**原 LeWM、单阶段、无 teacher、零新增参数**版本；
> 它尚非正式全过候选，剩余核心简洁性缺口已收缩为显式 simulator matched pairs。
>
> §5.33 又检查了这个“显式 pair”究竟有多少 privilege。对完整 `8,192` 行，完全不读取 history、
> future、damping、pair/template id 或行顺序，只以模型可见的 `(query RGB, action bytes)` 为 key，
> 恰好恢复 `4,096` 个二元组；mined 与 explicit group-mapping SHA 均为 `d7c2866f…c0c9e`。
> canonical auxiliary 对 group permutation 与组内翻转严格不变。因此 hidden label 和 pair annotation
> 本身不是理论必需；剩余数据假设更准确地是 **conditional overlap**：训练集中必须存在相同
> `(Q,A)`、不同 `H` 及其真实 future。普通 unmatched replay 尚未满足或验证这一条件。
>
> §5.34 进一步把数据构造中的 named low/high enumeration 也删掉了。每次只取得一个不暴露
> damping 的随机环境 handle，用观测到的 query-state feedback 做一次黑盒 shooting，使轨迹从
> x0 连续自然到达预选 Q；独立重复抽样，仅在可见 history 完全重复时丢弃，再以可见 `(Q,A)`
> 分组。完整 `2,048` templates 上平均抽样 `3.025` 次、最多 `11` 次，公共 query 最大状态误差
> `3.24e-12`，所得 `8,192` 行在每个 template 内都与冻结训练资产逐字节同集合。因此当前结果
> 不依赖 hidden label、pair annotation 或按名字配出的 endpoint pair；但仍依赖可控环境随机化、
> 初始状态控制和主动 conditional-overlap 收集，不能冒充普通 unmatched offline replay。

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
3. full-gradient PCJA 能学会 ActionDelay，且相对 native 的 CEM 曾观察到 `-13.33pp`；但
   后补的同数据、同 sampler、PCJA-weight=`0` control 本身为 `-13.67pp`。PCJA 相对这个
   matched control 的增量只有 `+0.33pp`，区间跨零。因此该下降属于 full-F/data-sampler
   recipe，不能再归因于 PCJA 或 full-gradient routing。
4. predictor-only PCJA 完整 recipe 的 ActionDelay 冻结
   Private macro 达到 `0.9309`、正式 Public macro 达到 `0.9452`，累计 900 次配对 CEM
   的差值为 `-2.67` 个百分点，单侧 95% lower 为 `-3.67` 个百分点，通过 `-5` 个百分点
   的非劣界。这是有效配方正例，但 matched routing 对照没有分离出显著路由效应，故不能把
   成功单独归因于 predictor-only。
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
    ICL”解释，不否决 pairing、PCJR、SIGReg、LeWM 或 PLDM。后续 panel qualification 又发现
    窄面板到全 17-batch 面板发生符号翻转，且一阶量不能回顾性预测真实 optimizer trajectory；
    该 estimator 已按 `stop_this_estimator` 关闭，不再用更多 `beta` 项或 source routing 延长。
29. 零参数 causal transition basis 在 ActionDelay seed `3072`、1,024-step 上达到
    macro/worst/bootstrap-lower=`0.9767/0.9323/0.9715`，`960/960` query 对历史响应，且
    `0/2,880` target pair collapse。它只改变 Predictor 输入坐标，保留原生 MSE+SIGReg；这
    证明显式 pair loss 并非离散 delay 可学习性的理论必要条件，也把“原生目标本身只能学边缘”
    修正为“原生条件 MSE 可能因绝对坐标的优化几何而条件失明”。
30. 同一固定 basis 在 Motion 1,024-step 上虽相对 matched native snapshot 改善
    future/history/switch/worst=`+3.32/+12.30/+30.47/+12.89pp`，response gain 却为
    `-0.051`、NRE=`1.857`。消除 `19.2×` warm-start MSE 跳变的零参数 homotopy 在 step
    1,024 短暂改善，step 2,048 又回落，gain=`0.0031`、NRE=`1.571`。因此坐标暴露足以修复
    discrete assignment silence，但不足以学习连续 response operator；hard switch 与 schedule
    sweep 都停止。
31. 为检验 causal basis 中 `Δz_t` 与致因动作 `a_{t-1}` 错位是否是 Motion 的剩余根因，
    terminal-aligned v1 将 support token 改为 `(Δz_t,a_{t-1})`，末 token 保留当前状态与 query
    action，且只监督无泄漏的 terminal output。其 step-0 correct-future MSE 为 `0.04174`，仅为
    native 的 `1.017×`，但训练后 future/history/switch/worst 退化为
    `0.439/0.350/0.070/0.086`，gain=`-0.184`、NRE=`1.726`。
32. terminal-only 同时删除 standard early supervision，故又补了唯一的 confound-resolving v2：
    对每个 target 用独立 leakage-free prefix，恢复 native 的三个 standard transition MSE，hidden
    行仍只监督 terminal。该版本仍只有 future/history/switch/worst=
    `0.430/0.438/0.055/0.164`，gain=`-0.148`、NRE=`1.514`。因此固定 action alignment 的两个
    合法实现均失败；不续到 4,096、不做 seed/schedule sweep，也不在 ActionDelay 重训一个已被
    Motion kill test 否决的通用候选。
33. 当前 Motion 训练 episode 每个仅有一个可用 H3 query；所谓 ordinary episode-blocked
    multi-query sampling 在现有 release 上没有数据支撑。特殊 forward/reverse twin sampler 将
    完整 twin 从普通 epoch 的 `9` 个偶遇提高到每个 hidden batch `16` 个，同时保持一 epoch
    精确 row coverage；native future 仍为 `0.4512`，switch 从 `0.0781` 降至 `0.0703`、worst
    从 `0.1133` 降至 `0.0547`。row-separable MSE 不会因共批自动获得跨-row 条件约束。
34. 零参数 anchored context transfer 直接把同 damping、opposite-query 的 support history
    平移到 destination current latent，并保留 destination query action；相对 twin-only，
    future/history/switch 从 `0.451/0.361/0.070` 提高到 `0.494/0.439/0.133`，gain/NRE 从
    `-0.202/1.811` 改善到 `-0.088/1.324`。但首步 transfer MSE=`0.1195`，显著高于真实
    hidden MSE=`0.0662`，且 response 方向仍为负，说明 absolute image-latent 平移不是合法通解。
35. 最后一个 transition-context transfer 让 Predictor 直接看到
    `[Δz1_source,Δz2_source,z2_destination]` 与 `[a0_source,a1_source,a2_destination]`，消除了
    absolute latent 平移语义且仍为原 Predictor、零新增参数。结果继续改善到
    history/switch/worst=`0.473/0.148/0.090`、gain/NRE=`-0.058/1.188`，但 future 仅
    `0.494`，gain 未翻正。该线据此关闭；改善证明 direct coupling 有作用，却不足以把合成
    support 变成 population-consistent 连续 dynamics operator。
36. causal-transition + residual-output 的纯原生 LeWM 在 Motion 上接近但未通过六门：
    future/switch/worst=`0.537/0.719/0.172`、gain=`0.159`、NRE=`0.969`；同 checkpoint
    的标准 PushT CEM 为 `17/20`，与 native reference 相同。这排除了 residual 输出坐标和
    persistence 初始化本身造成 planning 损伤。
37. 在该结构上加入唯一的 paired normalized exact-future 辅助项后，权重 `0.09` 首次使
    Motion 六门全过（future=`0.582`、gain=`0.251`、NRE=`0.835`），但 CEM 跌到 `3/20`；
    预定的 `0.03` 内点只差 future 门 `0.0051`，CEM 仍只有 `6/20`。因此条件响应可学，当前
    失败已从“没有有效方法信号”收缩为 ICL–planning 优化冲突；继续扫标量权重不成立。
38. 同时对输入与输出做函数保持的 temporal homotopy 在 step 0 精确等于原生 absolute 模型，
    最终却得到 future/switch=`0.451/0.043`、gain=`-0.214`、NRE=`1.834`。hard persistence
    reset 不是无关实现细节，而是帮助模型离开 native negative-response basin 的 optimization
    bootstrap；该发现不等于允许通过重置牺牲原能力。
39. 从 `0.09` 六门成功 checkpoint 撤掉辅助项，保留 mixed ordinary + hidden-terminal 数据并
    用 fresh AdamW 运行 1,024-step 原生目标后，六门仍全部通过，CEM 从 `3/20` 恢复至
    `13/20`。这首次证明已学 coupling 可以在没有持续 paired loss 的阶段保留，并且 planning
    损伤可部分逆转；但 `13/20 < 17/20`，还不是同-checkpoint Pareto pass。
40. matched 的 ordinary-only consolidation 将 hidden rows 从 `64` 降到 `0`、ordinary rows
    从 `64` 增到 `128`，结果 future/switch/worst=`0.498/0.195/0.016`，发生明显 conditional
    forgetting，故不开 CEM。paired signal 可作为 bootstrap，但后续仍需要真实 hidden-terminal
    原生监督维持条件映射；简单地回到普通 PushT fine-tuning 不是解法。

因此当前主问题已从“如何创造连续条件响应”进一步收缩为：

> 在不新增独立 encoder/adapter/head 的条件下，如何把配对联合关系作为短暂而可靠的
> conditional bootstrap，并让原生 mixed-data 目标完成能力巩固，同时不把 Predictor 推离原始
> planning geometry？

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

predictor-only recipe 在 ActionDelay 上是有效正例，但路由的单因素归因已被后补 control
修正：full-gradient PCJA 相对 native 为 `-13.33pp`，其 matched full-F pair-sampler control
相对 native 为 `-13.67pp`，两者之差仅 `+0.33pp` 且区间跨零。因此合理的实现意图仍是把
辅助项限制为“教 Predictor 使用已经可用的条件信息”，但现有结果不能声称该路由本身解释了
CEM 保持；旧的 `-13.33pp` 不再作为 routing 因果证据。

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

随后以一个冻结的 300-query / 3,300-history-condition Private Development release 对
calibration seed 与两个未评分确认 seed 做了完整批次复现：

| 训练 seed | macro | 最差组 | paired bootstrap lower | 四门 |
|---:|---:|---:|---:|---|
| 3072 | 0.9404 | 0.9056 | 0.9231 | PASS |
| 4096 | 0.9356 | 0.9006 | 0.9185 | PASS |
| 5120 | 0.9418 | 0.9106 | 0.9252 | PASS |

三个 checkpoint 均为独立 1,024-step 训练终点；每格 target collapsed pair 为 `0`，评分前后
model-state SHA 不变。最终 receipt 明确记录 `cross_seed_pooling_performed=false`、
`cross_seed_averaging_or_rescue_used=false`。这将 ActionDelay 结论从单 seed 正例提升为
三 seed 可复现正例，但没有重新打开 Public 或 CEM，也不新增跨任务主张。

### 4.2 当前结论边界

可以确认：

- 原生 LeWM 架构能够学会 ActionDelay ICL；
- 直接、对称的 condition assignment 是目前第一个同时通过该任务 ICL 与 CEM 的简洁
  LeWM 辅助目标；
- 完整训练 recipe 的 ActionDelay ICL 结果在三个 seed 上稳定。

尚不能确认：

- 跨连续动力学、接触动力学或环境域的通用性；
- 对称列项、paired sampling 与 predictor-only routing 各自的单因素因果贡献；已有 matched
  对照对 routing 的效应区间跨零，因此不能把完整 recipe 的成功单独归因于路由；
- 显式 pair 能否被无 privileged metadata 的训练构造替代；
- PCJA 是否整体优于 PLDM，而不是互补或诊断性上界。

因此本阶段不把 ActionDelay 三 seed 正例写成通用方法级 SOTA，也不以它覆盖 Motion Damping
已经给出的连续响应反例。

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

> 本节记录的是早期 **binary-PCJA predictor-only** 候选的门控决定；后续 §5.22 出现新的
> function-anchor Pareto 候选后，Contact 迁移已按新候选重新开放并在 §5.23 完成。两者不是同一
> 配方，本节的历史停止记录不代表当前 Contact 尚未执行。

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

### 5.13 native-coordinate gradient-to-estimand 审计：PCJR trunk 方向正确，根因转向采样覆盖

§5.12 指定的那一个 kill test 已执行完毕，零训练步。审计在 parent step-256 与 K4 step-256 两个
冻结终点、各自 native latent 坐标里，把 predictor-only 的 PCJR 梯度投影到那份已打开的
256-pair Motion Development 面板上；native MSE 与 `pred_proj` 分列。`B1/B32/B128/B256` 的
anchor 批由一次 unchanged 256-step parent prefix replay 物化，终点 state 与盘上冻结 `.pt`
逐字节相同（`d04b220f…`），初始态 `c352c343…`。审计给两个端点各加 0 个 optimizer step。

每单位学习率的预测变化（predictor trunk-only 块，负号 = 该 estimand 下降）：

| 端点 | source | normalized error | gain | `beta` | `beta²` | margin |
|---|---|---:|---:|---:|---:|---:|
| parent-256 | `PCJR_B256`（本 recipe） | `-1.74e-3` | `+2.54e-3` | `+8.02e-3` | `+2.16e-2` | `+1.21e-5` |
| parent-256 | `MSE_B256` | `+5.36e-3` | `-9.20e-3` | `-4.01e-2` | `-1.21e-1` | `-4.38e-5` |
| parent-256 | `FULL_current` | `+3.62e-3` | `-6.67e-3` | `-3.21e-2` | `-9.96e-2` | `-3.17e-5` |
| K4-256 | `PCJR_K4`（本 recipe） | `-1.42e-3` | `+2.16e-3` | `-5.13e-4` | `+1.32e-3` | `+1.05e-5` |
| K4-256 | `MSE_B256` | `+1.35e-3` | `-8.94e-3` | `-3.07e-2` | `-1.04e-1` | `-4.35e-5` |
| K4-256 | `FULL_K4` | `-7.36e-5` | `-6.78e-3` | `-3.12e-2` | `-1.03e-1` | `-3.29e-5` |

判定落在 §5.12 的第二个分支：`objective_competition_or_sampling_coverage`。两个端点上
PCJR trunk 梯度对 held-out normalized response error 都是 corrective（parent `-1.74e-3`、
K4 `-1.42e-3`），对 gain 都是正向，交叉面板同号（parent 用 `PCJR_K4` 得 `-3.36e-3`，K4 用
`PCJR_B256` 得 `-4.93e-3`），因此**不满足**"PCJR trunk 本身方向错误"的停止条件；训练面板
交叉核对也与 kill test 同号（`center_loss` 投影 corrective，cosine `0.10–0.18`）。方向在
native MSE 汇合后才坏掉：parent 端 `FULL_current` 把 NRE 翻成 `+3.62e-3`，K4 端 `FULL_K4`
被压到 `-7.36e-5`（相对本 recipe 的 PCJR 衰减约 95%），trunk 块 cancellation ratio 分别
`0.532` 与 `0.483`。主因记为 sampling coverage：K4 的四个 anchor 批 NRE 投影符号分裂
（B1 `-1.09e-3`、B32 `+2.49e-3`、B128 `-2.17e-3`、B256 `-4.93e-3`），聚合后被削弱到
`-1.42e-3`；目标竞争分量同时记录。norm clipping 只按正标量整体缩放梯度，不改任何投影符号，
256 条 clip 回执只作 regime 描述。

据此保留条件关系目标本身，不再以"延长该 estimator"的方式推进：不增加 K、权重或训练预算，
不开 1024/Contact/Public/CEM/额外 seed，不授权 gradient surgery、EMA/lagged target、
margin/weight/cutoff 搜索，也不授权新候选训练。任何 minimal repair（目标竞争侧的重新配平，
或采样覆盖侧的 anchor 构造）都需要另行 append-only 预注册。这一结果不否决 pairing 这一更宽
方法族，也不否决 ActionDelay 侧那个冻结正例。按 §1.3，梯度方向是诊断量：以上全部数值都不
替代冻结终点评分。

实现走 append-only recovery：v1 执行器在完成同样的 256-step replay 后，停在自己过紧的
parity 门上——parent 端 512 个离散决策全部精确一致、聚合差 `1.074e-4` 落在冻结的 `2e-4`
envelope 内，但 v1 把该 envelope 也套在了原始 per-pair MSE 浮点上（§5.11 未对这一量类冻结
envelope，其跨 runtime 抖动经小 `E_t` 分母可达 `~9e-3`）。recovery-v2 只重校三个门（per-record
浮点改为描述性、硬门为决策精确；可微前向预测 parity `1e-6→1e-5`，实测一个 fp32 ulp；
float64 estimand 跨语义 parity `1e-8→1e-5`，实测 `2.3e-7`），不动数据、source/estimand 语义、
投影、判定规则与 authority，v1 文件与产物一律未改。K4 端冻结 JSON 与重算逐字节相同（差 0.0），
这把 parent 端的 per-pair 抖动干净地归因于跨 runtime 差异，正是 §5.11 记录的行为。

### 5.14 native loss 的来源分解：不利压力落在 hidden-actuation 行，以及一个采样覆盖的符号反例

§5.13 把根因指向"方向在 native MSE 汇合后才坏掉"，但没有回答 native MSE 的哪一部分在坏。
训练损失本身是两项的等权和 `L_native = 0.5·L_orig + 0.5·L_hidden`（由 trainer
`mixed_prediction_loss` 的 AST 逐字核对）：`L_orig` 是 64 条 original 行的全 horizon MSE，
`L_hidden` 是 64 条 hidden-actuation 行的**仅终点** MSE。因此可以把 §5.13 的
`FULL = MSE + PCJR` 再拆一层，问不利压力落在哪一项上。零训练步，同样两个冻结终点。

由于一阶投影 `π_E(g) = −⟨∇_θ E, g⟩` 对 source 线性，四个臂都是三个实测分量的闭式线性组合，
无需各自反传：`F_native = ½g_o + ½g_p + g_r`（本 recipe）、`S_strict_route = ½g_o + g_r`
（候选：predictor 侧只走 original 行）、`C_scale_control = ¼g_o + ¼g_p + g_r`（尺度匹配对照）、
`R_rescaled = g_o + g_r`（描述性，已明确否决）。代数上 `π_S − π_C = ¼(π_o − π_p)`，所以
"S 胜过 C"当且仅当 `π_o < π_p`——机制门与尺度门因此天然可分。`projector`/`pred_proj` 含
`BatchNorm1d`，三个分量必须来自同一次 128 行前向（`retain_graph=True`）。

预注册的机制预测是 `π_p > π_o`（理由：终点-only 监督不强迫 predictor 用 history 消歧，
最省力的下降方向是退回边缘均值，也就是 `beta` 分析器测到的 center bias），并预先写下了
被证伪时的含义。**预测成立**，两端点都是（`predictor_trunk_only`，主 estimand
`dev_normalized_response_error`）：

| 端点 | `π_o`（original 行） | `π_p`（hidden 行） | hidden 占不利压力 | 机制门 | 绝对门 |
|---|---:|---:|---:|---:|---:|
| parent-256 | `+1.81e-3` | `+1.076e-2` | 85.6% | `>5.359e-3` ✓ ×2.01 | `>7.234e-3` ✓ ×1.49 |
| K4-256 | `-1.94e-4` | `+2.91e-3` | 107.2% | `>1.351e-3` ✓ ×2.15 | 该端近乎空洞，机制门为紧约束 |

四个预注册数值阈值全部以约 2 倍余量越过；K4 端 `π_o` 已经是负的（即 original 行的一阶压力
本身是纠正性的），hidden 行占比因此超过 100%。四臂在主 estimand 上：parent 端
`S=-1.97e-3` 优于 `C=+2.70e-4` 与 `F=+3.41e-3`；K4 端 `S=-7.67e-3` 优于 `C=-6.89e-3` 与
`F=-6.21e-3`。

**一个反过来限制本节自身的发现。** 把 anchor 面板从 4 批加宽到 17 批（`[1,16,…,256]`，由同
一次 unchanged 256-step parent prefix replay 物化，终点仍逐字节等于 `d04b220f…`）后，K4 端
PCJR 对 `dev_response_gain` 的投影**符号翻转**：冻结 4 批面板上是 `+2.161e-3`，17 批面板上
是 `-2.790e-3`。逐批符号 7 正 / 10 负，而历史那 4 批是 3 正 / 1 负；穷举全部 C(17,4)=2380 个
四批子集（确定性，无 RNG），历史子集排第 65/2380（前 2.73%），panel 均值离零仅 0.46 个标准差。
parent 端不翻转（`+4.544e-3 → +3.062e-3`，排名 524/2380）。这独立复现了 §5.13 在 4 批上判下的
`sampling_coverage`，但它同时是对本节的警告：既然 4 批能翻转一个符号，任何有限面板上的量
——包括驱动本节结论的 17 批量——都带面板宽度风险。机制门与绝对门离阈值有 2 倍余量，
比增益对比稳健，但风险不为零。加宽捕获本身没有扰动任何东西：对冻结 recovery-v2 矩阵的
复现门在全部 source×estimand×block 上最差相对差为 `0.000e+00`。

判定与两次中止都保留在案。首轮判定为 `inconclusive_no_candidate_authorized`：机制门与绝对门
都过，唯独佐证门的第二个子句要求候选在 `dev_response_gain` 上**绝对为正**，而 K4 端两个臂都是
负的。该子句写下时可满足——当时唯一可见的证据（冻结 4 批矩阵）在两端都为正——它编码的先验
恰恰被加宽面板证伪。据此另立 append-only 判定修订，只删这一个子句、保留真正区分 routing 与
尺度的对比子句，不重算任何已测数字（`forward_passes: 0`、`gradients_taken: 0`、
`development_reads: 0`、`replay_performed: false`）：parent 端 `S=+1.515e-3` 对
`C=-2.472e-3`，K4 端 `S=-4.150e-3` 对 `C=-8.070e-3`，两端对比子句均成立，修订后判定为
`route_candidate_ready_for_preregistration`。原判定永久在案。修订预注册中另有一处自我更正：
先前用于支持修订的 cluster-bootstrap 置信区间被**撤回**，因为其上界距零仅 `~4e-5` 且随 RNG
消费顺序改变正负；替换为上述完全确定性的穷举统计，实质结论不变而更保守。

执行侧两次中止各留独立回执。第一次停在 reconstruction 恒等门（相对残差 `2.24e-3` >
容差 `1e-4`）：诊断证明分解在 float64 下代数精确（残差 `0.0`），`2.2e-3` 是 bf16 的 sub-ulp
量级（一个 ulp `3.9e-3`），同一代码在 fp32 下降到 `3.6e-7`，分量 cosine ≥ `0.99998`。但该噪声
在 K4 端已达机制门阈值的 7.16%，足以翻转紧对比，因此不放宽容差而是改为双精度设计：fp32 走
判定路径、bf16 走 native 语义与复现门。一个把残差降到 `1e-9` 的捷径（令 `g_p = 2g_n − g_o`）
被明确否决——它把一个检查变成定义，同时把两条路线在 `g_p` 上 `3.0e-3` 的分歧藏进判定量本身。
第二次中止是 fp32 路径上 detached 特征与参数 dtype 不匹配，修正后在真实 bf16 取值的特征上
复验，残差仍为 `3.6e-7`。

这一结果在当时**只授权起草**一次 predictor-source-routing 预注册；它没有授权训练。后续资格
审计保留了 K4 这个直接反例：PCJR source 在有限训练面板上 `9/9` 局部纠正，256 步 held-out
response 却更差；4→17 batch 又发生符号翻转。matched 单因素 routing 对照的效应区间也跨零。
因此 source-routing 训练最终没有执行，一阶 estimator 不再用于候选晋级。这个停止决定只否决
当前 estimator→candidate 的推断链，不否决 pairing 或条件干预方法族。

### 5.15 决定性收口：离散配对可复现，但 frozen residual 不是连续动力学解法

当前证据不是“所有方案都失败”，而是出现了一个稳定正例和一个机制明确的跨任务反例：

- ActionDelay 的完整 predictor-only PCJA recipe 在三个训练 seed 上独立通过冻结 Private
  Development（§4）；matched conditional assignment 对离散 mode identification 有稳定作用。
- Motion Damping 中，同一类 pair signal 能增加 switch/assignment，却不能稳定校准跨 query
  response。有限 panel 上的一阶方向又不能预测真实 held-out 轨迹，因此不再通过调权重、K、
  cutoff 或 source routing 延长这条 loss-only 路线。

为检验失败究竟来自 history system identification，还是来自 query→response 的表达能力，固定
seed-3073 LeWM 的全部 18,034,478 个参数，只训练零初始化的小 residual head；真实 damping
label 仅在 oracle 诊断中输入，不属于最终方法。目标 response 的 SVD 与严格 probe 先给出：

| 诊断 | Training | Development |
|---|---:|---:|
| target response `r95` | 49 | 44 |
| target response `r99` | 101 | 77 |
| query-only damping probe | — | 0.500 |
| history-only damping probe | — | 0.539 |
| full-history damping probe | — | 0.543 |

probe 使用 forward/reverse twin-group 隔离的 8-fold split；query-only 恰为 chance，说明评测没有
从当前 query 泄漏标签，但 frozen history 也只提供很弱的跨 query 可读信号。oracle 上限结果为：

| frozen-base oracle head | steps | correct future | worst mode | alignment | gain | NRE |
|---|---:|---:|---:|---:|---:|---:|
| rank-16 `W_o(φ(W_q q)⊙φ(W_c c))` | 1,024 | 0.521 | 0.074 | 0.315 | 0.139 | 0.916 |
| rank-64 factorized | 1,024 | 0.553 | 0.160 | 0.425 | 0.237 | 0.837 |
| rank-128 factorized | 4,096 | 0.582 | 0.254 | 0.435 | 0.258 | 0.836 |
| nonlinear MLP, hidden 256 | 4,096 | 0.602 | 0.305 | 0.411 | 0.251 | 0.871 |

rank-128 已覆盖实测 `r99` 的主要尺度，非线性 oracle MLP 具有 198,724 个参数；两者训练集
correct-future 也只有 `0.602/0.632`。因此不能把失败归因于 rank-16 或 learned context。被否定的
是更窄的实现假设：**在既有 frozen LeWM latent 后面加一个小输出 residual，就足以恢复连续
条件动力学。** 这与既有普通 history residual 在 8,192 步仍失去 calibration、并恶化标准 replay
MSE 的结果一致。

这些结果支持 factorization 作为机制解释：

\[
c=g(H_{\mathrm{support}}),\qquad
\hat z^+=F\bigl(s(H,Q),A;c\bigr).
\]

`c` 表示跨 query 共享的 episode dynamics，主 Predictor 负责把同一个 `c` 转换成当前 query
下的方向和幅值。但本项目的方法约束比“做一个小模块”更严格：最终候选不得增加独立 context
encoder、context token、FiLM/AdaLN adapter 或 residual head。§5.15--§5.16 中的 oracle
factorization 只用于拆分 system identification、query-state sufficiency、共享 trunk 冲突与
frozen-output capacity 四个瓶颈；它已经完成诊断使命，作为 architecture candidate 正式关闭。
这些负结果不会被继续扩展成 learned `g(H)`。

### 5.16 Context–response 最小因果阶梯：结构起效，但最简实现尚未闭环

§5.15 的 frozen-head 上限仍混合了三个设计因素：context 是否进入模型、hidden 数据是否通过
共享 Predictor 更新，以及 paired response 是否直接监督 `B(q)c`。因此追加一个有硬上限的
Development-only 阶梯；所有实验都用同一冻结 release，Public 与 CEM 均未打开。oracle arm
读取真实 damping identity 只用于表达上限，constant arm 保持参数量、数据、梯度路由和预算
一致但不携带 identity。

| 结构与训练路径 | steps | future | history | switch | gain | NRE | oracle–constant 主要差异 |
|---|---:|---:|---:|---:|---:|---:|---|
| additive context，native 全量联合训练 | 256 | 0.490 | 0.457 | 0.090 | -0.113 | 1.394 | 与 constant 数值相同 |
| rank-8 `B(q)c`，native 全量联合训练 | 256 | 0.490 | 0.459 | 0.098 | -0.112 | 1.390 | 仅有微弱分支效应 |
| rank-64 `B(q)c`，hidden terminal 只更新分支 | 256 | 0.500 | 0.516 | 0.648 | 0.0062 | 0.995 | gain `+0.0008` |
| 上行再加 paired normalized response | 256 | 0.502 | 0.523 | 0.664 | 0.0068 | 0.994 | history `+0.0215`、switch `+0.0234` |
| frozen native + 同一 paired response | 256 | 0.500 | 0.512 | 0.688 | 0.0135 | 0.994 | switch `+0.0664`、NRE `-0.0099` |
| frozen native，oracle 单臂轨迹 | 1,024 | 0.500 | 0.510 | 0.766 | 0.0282 | 0.974 | 相对同初始化 step 0 单调改善 |

这个阶梯给出四个可分离结论。

1. **context 可见不等于 context 被使用。** additive oracle 的 context adapter 已产生非零、
   类别不同的参数列，但 held-out 输出与 constant arm 一致；因此“再加一个 context token”本身
   不是机制。
2. **共享 trunk 的优化冲突真实存在。** 联合训练的 rank-8 分支在同一 checkpoint 的分解中
   弱正向，而 native trunk 的 response 为强负向；将 hidden-terminal 梯度路由到分支后，整体
   gain 从 `-0.112` 变为 `+0.006`、NRE 从 `1.390` 变为 `0.995`。routing 在这里是因果隔离，
   不是完整 ICL 方法，因为 oracle 与 constant 仍近似相同。
3. **pair signal 能进入正确结构。** paired-response 首步输出层梯度范数在 oracle/constant 为
   `0.522/0.0028`；冻结 target geometry 后为 `0.409/0.0037`。冻结版本的
   switch、alignment、gain 与 NRE 又随 `0→256→1024` 同向改善，排除了“256-step 偶然翻点”
   和“只有局部梯度好看”两种解释。
4. **当前最简训练实现仍不充分。** 1,024-step 只通过 rule-switch 与 NRE-bootstrap-upper
   子门；future 保持 `0.500`、worst condition 保持 `0.031`、gain 距 `0.1` 门仍有明显差距。
   因而不开放额外 seed、Public 或经验 CEM。标准 `c=0` 路径令分支严格为零，所以原 CEM
   函数在结构上保持不变；这不是同 checkpoint CEM 的实测通过声明。

这个结果也不能反过来否定 factorization 作为根因解释。§5.15 的 velocity-aware rank-64 oracle head 在同样
1,024 步达到 future `0.553`、switch `0.961`、gain `0.237`、NRE `0.837`；它使用
`q=[z_2,z_2-z_1,a_2]`、`1e-3` 学习率和 paired center+response，而本节最简分支只用
`q=[z_2,a_2]`、`5e-5` 和 terminal-MSE+response。前者证明 query→response operator 有可用
上限，后者证明受保护的分支能够沿正确方向学习；两者共同把剩余问题收窄到：

\[
c=g(H_{\mathrm{support}}),\qquad
\Delta \hat z^+=B(z_Q,\Delta z_H,A)c,
\]

其中 `c` 必须在 query/scene leave-out 下表示 episode dynamics，而不是 episode identity；
query 分支必须保留速度/transition 特征。这个式子说明原 Predictor 缺少了什么，不再授权新增
`g(H)` 或 `B(q)c`。下一 MVE 改为让原 Predictor 直接看到固定、可逆的 transition coordinates；
只有零参数变换仍不足时，才重新讨论是否需要直接的 paired relation loss，而不是先增加结构。

### 5.17 VISReg：边缘分布更强，但条件盲视仍在

为检验“新一代无额外参数的边缘正则是否已经足以替代条件配对信号”，我们在同一
ActionDelay H7 训练协议上接入 VISReg。没有增加 encoder、adapter 或可学习参数；采用论文
小数据设置 `lambda=0.6` 的等比例目标 `L_pred + 1.5 L_VISReg`，每 rank 使用 1,024 个投影。
只运行发现阶段的 seed `3072`，先做 256-step screen；由于同预算 PLDM 也尚处 chance，未据此
误杀，而是从头运行完整 1,024-step 轨迹。

| 终点 | macro | worst group | bootstrap lower | history-responsive queries | target pair collapse |
|---|---:|---:|---:|---:|---:|
| 256 | 0.3278 | 0.0000 | 0.3250 | — | 0 / 2,880 |
| 1,024 | 0.3316 | 0.0000 | 0.3299 | 8 / 960 | 0 / 2,880 |

1,024-step confusion 几乎全部落到 delay-4 物理组：三个真实组的 accuracy 为
`0.0146/0.9802/0.0000`；`952/960` 个 same-query family 对三种历史给出完全相同的选择。
与此同时，全部 2,880 个真实 target pair 仍非重合。这构成一个直接反例：VISReg 可以维持
健康的边缘 latent geometry，却没有识别 `history-action-future` 的联合条件关系。

因此 VISReg 保留为重要的 parameter-free marginal-regularizer baseline，但不投入权重扫描、
Motion 或 Contact 训练。这个 no-go 不否定 VISReg 在一般表征学习上的价值；它只否定
“当前 matched-budget discovery recipe 单独足以修复本 benchmark 的 conditional identifiability
failure”。VIS-WM 随后公开的 world-model 配方使用 `L_pred+4.5 L_VISReg`、`lr=1e-4`、
batch 128、10 epochs、三训练 seed；公开实现也明确按加权和而非 convex mix 计算。已有运行的
系数为 `1.5`、学习率为 `5e-5`、预算为四个逻辑 epoch，因此不能把当前结果冒充成 exact
published-recipe failure。为封住 recipe-mismatch 质疑，最多补一个 seed 的公开超参数、
matched-1,024-step confirmation；若仍保持近零 history response，再决定是否值得给它 10-epoch
超额预算。最小证据摘要见
[`artifacts/visreg_action_delay_discovery_v1/summary.json`](artifacts/visreg_action_delay_discovery_v1/summary.json)。
Public 与 CEM 均未打开；本结果是 Development-only discovery，不是正式 release claim。

### 5.18 零参数 transition basis：ActionDelay 强正例与 Motion 连续响应反例

Motion oracle 诊断最稳定的新信息是 velocity/transition 特征必要；满足项目约束的最小干预不是
训练 transition encoder，而是把现有 Predictor 的绝对历史坐标固定变换为：

\[
T(H)=[z_0,\ z_1-z_0,\ z_2-z_1,\ldots,z_{H-1}-z_{H-2}].
\]

该变换 token 数、维度和参数量均不变，可由 cumulative sum 精确恢复；任意长度为 `t` 的输出
前缀只依赖输入的前 `t` 帧，因此与 LeWM 的逐位置 causal MSE 兼容。最初提出的
`[Δz_1,Δz_2,...,z_{H-1}]` 虽整体可逆，却会在第一个位置直接暴露 `z_1-z_0`，而该位置正预测
`z_1`，构成 future leakage，不能执行。dense random orthogonal temporal placebo 同样会混入
未来帧；只有 causal、scale-matched 的下三角 placebo 才合法，并且只在正信号出现后补。

首轮没有运行形式上漂亮但不可辨识的 2×2。若只是把 loss 写成
`||(\hat z-z_t)-(z^+-z_t)||²`，它与 absolute MSE 完全相同，所谓 absolute/residual 两臂会得到
同一梯度；若改成让 Predictor 直接输出 `Δz`，又会改变预训练 H3 predictor 的函数初始化，不能
再声称“只改 loss”。因此第一阶段只有一个新训练臂：

- 复用已有 native LeWM A0 作为 control；
- seed `3072`，固定 1,024 optimizer steps；
- 原初始化 state dict、原 sampler、原 absolute-future MSE、`0.09` SIGReg 全部保留；
- 唯一变化是 `LeWM.predict` 内部的 causal transition basis；训练与 rollout 使用同一路径；
- 新增可学习参数 `0`，新增 loss `0`，Public/CEM 保持关闭。

#### ActionDelay：原生目标在正确坐标下可以学习条件联合关系

该候选从头完成 1,024-step 训练，冻结 stage-1 Development 结果为：

| 指标 | 结果 |
|---|---:|
| physical-group macro | **0.9767** |
| worst physical group | **0.9323** |
| paired-query bootstrap 95% lower | **0.9715** |
| history-responsive queries | **960 / 960** |
| target pair collapse | **0 / 2,880** |

三个真实物理组的 accuracy 为 `1.0000/0.9979/0.9323`；confusion 仅剩 delay-4 的 `2/960`
被选为 group-5，以及 delay-8 的 `65/960` 被选为 group-4。四个冻结 gate 全部通过。与 native
和 VISReg 的近 chance、近零 history response 不同，这不是由 target separation 或输出边缘比例
造成的假改善。

这项结果改变了此前过强的理论判断：**显式 privileged pair、额外 context encoder 和额外
joint loss 都不是离散 ActionDelay conditional identifiability 的理论必要条件。** 原生 MSE
本来就在样本层面优化 `p(O^+|H,Q,A)`；失败点可以来自绝对 latent 坐标让静态内容淹没历史
transition，而不一定来自 loss 只看到边缘分布。SIGReg 仍是边缘正则，但固定 transition basis
让 prediction MSE 获得了可优化的历史—动作—未来关系。当前边界仍是单 seed、Development-only；
在跨任务和 CEM 前不能升格成通用方法主张。

#### Motion：硬切换部分改善 assignment，但连续 response 仍错误

Motion 使用 absolute-basis PushT 预训练 checkpoint。直接切到 causal basis 后，step-0
correct-future MSE 从 native 的 `0.0410` 跳到 `0.7890`（`19.2×`），所以不能把该运行当成
无混淆的 warm-start 比较。即使如此，固定 basis 在同 seed、同 1,024-step 上相对 native
snapshot 的 future/history/switch/worst 分别改善 `+3.32/+12.30/+30.47/+12.89pp`；但绝对结果
仍只有 `0.484/0.475/0.383/0.242`，aggregate cosine=`-0.0588`、gain=`-0.0511`、
NRE=`1.857`。它让历史更能改变输出，却没有学对连续响应方向和幅值。

为只消除 warm-start 坐标突变，随后执行一次固定、无 sweep 的前缀可逆 homotopy：

\[
T_{\alpha}(H)=[z_0,z_1-\alpha z_0,\ldots,z_{H-1}-\alpha z_{H-2}],
\]

前 1,024 step 将 `alpha` 从 `0` 线性升到 `1`，再在最终 basis 保持 1,024 step；参数、loss、
数据和初始化均不变。`alpha=0` 的 step-0 MSE 恢复为 `0.041039`，说明初始化函数断点被消除。
轨迹却没有形成持续 onset：

| step | alpha | future | history | switch | worst | gain | NRE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 0 | 0.500 | 0.494 | 0.621 | 0.031 | 0.0056 | 1.004 |
| 1,024 | 1 | 0.521 | 0.525 | 0.504 | 0.281 | 0.0243 | 2.023 |
| 2,048 | 1 | 0.508 | 0.492 | 0.496 | 0.270 | 0.0031 | 1.571 |

因此当前判定不是“transition basis 无效”，也不是“再多训练即可”，而是：它足以消除离散 delay
的 conditional silence，并在 Motion 上改善一部分 assignment；但 **transition exposure 不足以
识别连续 conditional response operator**。homotopy 的短暂峰值随后回落，未通过任何 response
主门，故不延长到 4,096、不扫描 schedule、不开放 Public/CEM。下一方法若继续，必须直接解释
为什么同一 history code 在不同 query 上需要方向和幅值校准；不能再用新的边缘正则或仅靠更多
训练预算掩盖这个反例。

紧凑证据见
[`artifacts/causal_transition_basis_v1/summary.json`](artifacts/causal_transition_basis_v1/summary.json)；
完整 ActionDelay 结果、Motion hard-switch response 与 homotopy 轨迹分别保留在
`artifacts/action_delay_h7_causal_transition_basis_v1/`、
`artifacts/pusht_motion_damping_causal_transition_basis_v1/` 与
`artifacts/pusht_motion_damping_causal_transition_homotopy_v1/`。

### 5.19 Transition–causing-action 对齐：混淆已拆除，但 Motion 仍失败

§5.18 的 causal basis 在位置 `t` 输入 `Δz_t`，同位置 AdaLN 条件却是用于预测下一状态的
`a_t`；造成该 transition 的动作实际是 `a_{t-1}`。ActionDelay 只需辨认离散生效索引，可能容忍
这种错位；Motion 则要从 `(Δz_t,a_{t-1})` 识别连续衰减规律。为验证这个具体假设，只执行了
两个零参数、无隐藏标签的合法实现。

第一版将完整 H3 query 写成：

\[
X=[z_1-z_0,\ z_2-z_1,\ z_2],\qquad A=[a_0,a_1,a_2],
\]

其中前两格逐位对应 observed transition 与 causing action，最后一格对应 current state 与 query
action。只有最后输出预测未观测的 `z_3`；前两格已经含有各自 target transition，若继续监督会
发生泄漏，因此 v1 对 original/hidden 两类样本都只使用 terminal MSE。该表示保持参数量
`18,034,478 -> 18,034,478`，没有新 loss term、pair metadata 或 hidden label。更关键的是，它
保留最后 absolute query state，step-0 correct-future MSE=`0.041743`，对 native
`0.041040` 的比值仅 `1.017`，排除了 §5.18 hard basis 的初始化函数断点。

v1 仍可能因删去 standard early-transition supervision 而失败。为避免误伤 alignment 假设，v2
对每个 target `z_k` 分别构造只到 `z_{k-1}` 的 leakage-free prefix：

\[
[z_1-z_0,\ldots,z_{k-1}-z_{k-2},z_{k-1}],
\quad [a_0,\ldots,a_{k-1}],
\]

只取该 prefix 的最后输出预测 `z_k`。H3 因而调用三次同一 Predictor，但恢复原 recipe 的全部
三个 standard transition MSE；hidden 行仍只监督可辨识的 terminal transition。它增加训练计算，
却不增加参数、loss 成分或数据要求，是拆除 terminal-only 混淆所需的最小对照。

| candidate | standard supervision | future | history | switch | worst | cosine | gain | NRE |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| terminal-aligned v1 | terminal only | 0.439 | 0.350 | 0.070 | 0.086 | -0.308 | -0.184 | 1.726 |
| prefix-aligned v2 | native three transitions | 0.430 | 0.438 | 0.055 | 0.164 | -0.318 | -0.148 | 1.514 |

两者均为 seed `14321`、1,024-step、相同初始化、原 `0.09` SIGReg，且 target latent 的
`256/256` pair 全部分离。v2 保留 early supervision 后仍没有 onset，连续 response 方向反而稳定
为负。因此当前被否决的是：

> 仅靠固定、逐轨迹的 transition–causing-action 坐标对齐，原生 MSE+SIGReg 就足以学习连续
> episode dynamics。

该结论不外推到所有配对或联合关系方法。它说明 ActionDelay 的零参数成功主要是 discrete
history coupling 正例，不能自动推广成 system identification。后续若仍坚持零新增结构，唯一
有信息量的方法类应从“换坐标”转为“跨 query 的 population constraint”：在普通 episode
轨迹内联合采样多个 query，使同一 episode dynamics 的历史证据在不同状态/action query 上共同
约束现有 Predictor；可使用 episode/window 身份组织 batch，但不把 damping、delay 或 hidden
mode 输入模型。是否需要一个直接的 correct-history / swapped-history 辅助关系，应由该最小
sampler 对照决定，而不是继续设计边缘正则。

紧凑结果与身份见
[`artifacts/transition_action_alignment_v1/summary.json`](artifacts/transition_action_alignment_v1/summary.json)。
两条候选均未开放 Public/CEM，不续 4,096，不补多 seed，也不做 schedule/weight sweep。

### 5.20 跨-query population constraint：共批无效，直接 transfer 有梯度但仍未学对方向

§5.19 之后首先核对了数据边界，而不是直接实现 sampler。Motion 的每个 20-row episode 只在
`[0,5,10,15]` 形成一个 H3 query；因此现有 release 不存在“同一普通 episode 的多个不同
query”。唯一可用的受控 multi-query 结构是相邻 forward/reverse twins：每个 twin 含两种
damping、两个相反运动 query，共四行。该对应关系由 paired catalog 提供，不是模型在无标签
轨迹中自行发现的。

第一个单变量对照只改变 hidden sampler。普通 `PairedBatchStream` 在完整 256-batch epoch 中
只有 `9` 个完整 twin，`247/256` batch 一个也没有；`CompleteTwinPairedBatchStream` 则每批固定
`16` 个完整 twin，并保持 16,384 行恰好一次覆盖。模型、absolute input、native identifiable-
future MSE、`0.09` SIGReg、seed、初始化、optimizer 和 1,024-step 预算全部不变。结果没有任何
onset：future 与 IID native 精确相同，worst 反而更低。原因也由目标形式直接解释：native MSE
对每一行可分，四行是否同时出现不会创建跨行约束。

第二层首次让训练关系真正跨 query。对 destination 行 `d`，取 twin 中同 damping 的 opposite-
query source `s`，构造：

\[
H_{s\rightarrow d}=[z^s_0+\delta,z^s_1+\delta,z^d_2],\quad
\delta=z^d_2-z^s_2,\qquad
A_{s\rightarrow d}=[a^s_0,a^s_1,a^d_2].
\]

原 hidden 权重 `0.5` 被无超参地均分为 `0.25` real hidden terminal MSE 与 `0.25` transferred
terminal MSE；original full-horizon 权重仍为 `0.5`。没有新增参数、module、head、hidden-value
输入或推理分支。该 arm 明显改善 history、switch、gain 与 NRE，却保留负 gain。首步 transferred
MSE 几乎是真实 hidden MSE 的两倍，说明对 image latent 做 additive translation 产生了离流形
support，不能把这组改善误写成方法成功。

最后一层只修复这一明确缺陷。leakage-free prefix basis 使 transferred terminal 输入严格等于：

\[
[z^s_1-z^s_0,\ z^s_2-z^s_1,\ z^d_2],
\qquad [a^s_0,a^s_1,a^d_2],
\]

从而不再要求 absolute latent 具有平移语义，同时保留原 standard 三段监督。三条 seed `14321`、
step `1024` 的 frozen Development 结果如下；IID 行只列原 scorer 可直接比较的四项，连续 response
以 twin-only 为严格 sampler-matched control：

| arm | future | history | switch | worst | cosine | gain | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| IID native | 0.451 | 0.352 | 0.078 | 0.113 | — | — | — |
| complete-twin native | 0.451 | 0.361 | 0.070 | 0.055 | -0.316 | -0.202 | 1.811 |
| absolute anchored transfer | 0.494 | 0.439 | 0.133 | 0.059 | -0.228 | -0.088 | 1.324 |
| transition-context transfer | 0.494 | 0.473 | 0.148 | 0.090 | -0.216 | -0.058 | 1.188 |

这形成了一个有价值但必须克制的机制结论：**显式跨-query coupling 的确比 co-batching 更接近
目标，且去掉离流形 absolute translation 后连续误差继续下降；但两个 transfer arm 都只是在
收缩错误方向，未把 response gain 翻为正值。** 因而当前否决的是这两个具体 synthetic-support
实现和 sampler-only 解释，不是否决所有联合分布/干预信号。它们不是候选方法，不开放 Public、
CEM、4,096 step 或多 seed。

下一步也因此不再是 sampler、latent augmentation 或新 encoder/adapter/head。若继续方法探索，
必须二选一：生成真正无 hidden-label、同 episode 多 query 的长轨迹，使 correct/swapped history
干预落在真实 support 上；或在现有真实 rows 的 prediction/target 上预先定义一个跨-query
不变量，再做一次非可分联合目标。后者必须明确区别于已经失败的 PCJA/CCRM/PCJR estimand，
否则项目应转为 benchmark-first，而不是继续给同一 loss family 换名字。

紧凑判定见
[`artifacts/motion_cross_query_transfer_v1/summary.json`](artifacts/motion_cross_query_transfer_v1/summary.json)。

### 5.21 Residual-transition 配对 bootstrap：连续 ICL 首次过门，但 Pareto 尚未闭合

§5.20 的失败并不意味着真实 matched rows 无法提供有效联合信号；它否定的是 synthetic history
transfer。新的最小实现保持原 LeWM 参数量和推理接口，只做两项训练几何改变：

\[
T(H)=[z_0,z_1-z_0,\ldots],\qquad
\hat z_{t+1}=z_t+F_\theta(T(H),A),
\]

并在真实 matched group `g` 上加入：

\[
L_{\mathrm{pair}}
=\frac{\operatorname{MSE}(\hat z_g^+,\operatorname{sg}(z_g^+))}
{\operatorname{MSE}(\operatorname{sg}(z_g^+)-\bar z_g^+)}.
\]

它同时约束 centered response 与公共 prediction center，但不增加 encoder、adapter、head、可学习
参数或推理分支。residual 输出使用现有 `pred_proj`，从头训练时只将其最后线性层置零，使 step 0
严格为 persistence predictor。

固定 seed `14321`、相同 2,048-step 预算得到下表。CEM 五列均使用 seed `42` 的同一组 20 个
query，catalog SHA 均为 `a0ae49b…`；这里只是快速 discovery screen，不是正式非劣检验。

| arm | future | history | switch | worst | gain | NRE | 六门 | CEM |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| native CEM reference | — | — | — | — | — | — | — | 17/20 |
| residual-transition，no aux | 0.537 | 0.602 | 0.719 | 0.172 | 0.159 | 0.969 | 否 | **17/20** |
| `0.03 L_pair` | 0.545 | 0.664 | 0.859 | 0.207 | 0.178 | 0.893 | 否，仅 future | 6/20 |
| `0.09 L_pair` | **0.582** | **0.703** | **0.926** | **0.270** | **0.251** | **0.835** | **是** | 3/20 |
| function-preserving homotopy + `0.09 L_pair` | 0.451 | 0.322 | 0.043 | 0.039 | -0.214 | 1.834 | 否 | 未开 |

该矩阵给出三个直接结论。第一，原 LeWM 容量确实足够：`0.09` 是 Motion 上第一个通过全部
assignment、direction 与 magnitude 门的零新增参数 checkpoint。第二，规划损伤不是 residual
basis 或输出层 reset 造成，因为 no-aux arm 的 CEM 与 reference 完全相同；它来自能显著改变
条件响应的 paired optimization。第三，把配对权重降到唯一预定内点 `0.03` 没有得到 Pareto，
故不继续扫 `0.01/0.02/0.04`。函数保持 homotopy 的失败还表明 hard persistence reset 帮助模型
离开 native negative-response basin；但它不是可以单独发布的科学贡献，因为 full-time aux
仍破坏规划。

为检验 paired signal 是否只需负责 bootstrap，从六门成功的 `0.09` checkpoint 撤掉辅助项，
模型 state 和 causal/residual 函数均不重置，再以 fresh AdamW 运行 1,024 个原生步骤。该实验是
weight-only restart，总参数更新 exposure 为 `2,048+1,024`，不冒充 budget-matched 1,024-step
control。两个 consolidation 只改变第二阶段的数据组成：

| consolidation | 第二阶段 batch | future | history | switch | worst | gain | NRE | 六门 | CEM |
|---|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| mixed native | 64 ordinary + 64 hidden | **0.559** | **0.660** | **0.871** | **0.203** | **0.230** | **0.904** | **是** | **13/20** |
| ordinary-only native | 128 ordinary + 0 hidden | 0.498 | 0.479 | 0.195 | 0.016 | — | — | 否 | 未开 |

mixed consolidation 在同一 checkpoint 上保留六门并把 CEM 从 `3/20` 恢复到 `13/20`，首次证明
coupling 与 planning 并非不可兼得；逐 query 比较中，它相对 `0.09` 源终点恢复了十个成功案例，
但相对 17/20 reference 仍净少四个，因此不能声明非劣。ordinary-only arm 则远离直接门，按
硬止损不开 CEM；这说明单纯回到原任务 fine-tuning 会忘掉条件能力，mixed hidden-terminal
监督仍在巩固已学 response。

当前可保留、但尚未完成的方法假设因此是：**paired history–action–future 关系作为短暂的
conditional-response bootstrap，随后由无 paired auxiliary 的原生 mixed-data 目标完成能力
巩固。** 它符合零新增参数/模块/推理成本的约束，也比永久叠加 loss 更接近原 LeWM；但简单
`2048+1024` 两阶段配方仍未通过 CEM。本文据此停止 weight/cutoff/schedule sweep，不开 Public、
Contact 或额外 seed。只有出现同 checkpoint ICL 六门与正式 CEM 非劣，才把它升格为最终方法。

紧凑矩阵、checkpoint SHA 与证据哈希见
[`artifacts/pusht_motion_damping_residual_transition_tradeoff_v1/summary.json`](artifacts/pusht_motion_damping_residual_transition_tradeoff_v1/summary.json)。

### 5.22 函数锚定的 conditional bootstrap：首个同 checkpoint ICL–CEM Pareto 正例

§5.21 留下的唯一问题不是 Motion response 还能不能学，而是同一组 dynamics 参数在学习配对
关系时为何破坏标准 PushT 规划。为避免把损伤草率归给单个 module，先在 no-aux source `P`
与永久 `0.09 L_pair` 终点 `A` 之间做了零训练的 `action_encoder × Predictor × pred_proj`
八格互换。相同 seed42、20-query CEM 的结果为：

| 终点组合（action / Predictor / head） | PPP | PPA | PAP | PAA | APP | APA | AAP | AAA |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| success / 20 | **17** | 3 | 5 | 5 | 0 | 4 | 2 | 3 |

只有完整 source 保留规划；任意单块或多块 endpoint transplant 都没有恢复它。这不能证明三个
module 各自独立造成损伤，因为混合终点本身包含 basis/co-adaptation mismatch；它能严格否定的
是“只把 auxiliary 路由到某一个现成 module，或事后换回某一块，就会自动保住 CEM”。因此下一
实验不再做 predictor-only / head-only 路由，而直接约束 source 的输入输出函数。

固定候选从 CEM source SHA `520bfa…` 开始，不重置任何参数，以 fresh AdamW 再运行 1,024 步。
保存模型仍是原 LeWM；训练期间只做三件事：冻结已有 `action_encoder`，令 `pred_proj` 的
BatchNorm buffer 始终使用 inference mode，并保留一份不进入 checkpoint 的 frozen source
`Predictor+pred_proj`。在 matched hidden rows 上沿用已经验证的 `L_pair`，在 ordinary PushT
rows 上加入：

\[
L_{\mathrm{anchor}}
=\frac{\operatorname{MSE}(\hat z_{\theta},\operatorname{sg}(\hat z_{\mathrm{source}}))}
{\max\{\operatorname{MSE}(\operatorname{sg}(\hat z_{\mathrm{source}})-H),10^{-8}\}},
\]

\[
L=L_{\mathrm{native}}+0.09\,L_{\mathrm{SIGReg}}
+0.09\,(L_{\mathrm{pair}}+L_{\mathrm{anchor}}).
\]

这里 teacher 只在训练期提供函数约束，不是新 encoder、adapter、head 或 inference branch；最终
checkpoint 新增可学习参数、module 与推理计算均为零。step 1 前 anchor 严格为 `0`，训练末普通
rows 的归一化 source-function drift 为 `0.0168`。固定 seed `14321` 的结果是：

| arm | future | history | switch | worst | gain | NRE | 六门 | CEM 20 | CEM 100 |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|
| residual-transition source | 0.537 | 0.602 | 0.719 | 0.172 | 0.159 | 0.969 | 否 | **17/20** | **57/100** |
| 永久 `0.09 L_pair` | 0.582 | 0.703 | 0.926 | 0.270 | 0.251 | 0.835 | 是 | 3/20 | — |
| native mixed consolidation | 0.559 | 0.660 | 0.871 | 0.203 | 0.230 | 0.904 | 是 | 13/20 | — |
| **function-anchored bootstrap** | **0.588** | **0.670** | **0.879** | **0.305** | **0.256** | **0.862** | **是** | 15/20 | **57/100** |

20-query 小屏看似少两个 success，但在预先保持相同规则、扩大到同 seed 的 100 个 paired query
后，source 与候选均为 `57/100`。逐 query 列联为：共同成功 `47`、source-only `10`、
candidate-only `10`、共同失败 `33`，净差严格 `0pp`，exact two-sided McNemar `p=1.0`。
因此这是项目中第一个在**同一 checkpoint** 上同时通过 Motion assignment/direction/magnitude
六门，并在更稳定的标准 PushT CEM 面板上不下降的单-seed Pareto 正例。

机制结论也因此进一步收敛：paired joint relation 本身不是规划损伤的必然来源；损伤来自它在
无约束共享参数路径上改写原 planning function。直接保护 source function 后，原 Predictor 容量
可以同时容纳连续 history-conditioned response 与原任务规划。这个结果符合“LeWM 模型规模和
推理结构不变”的核心约束，但还不是最终最简配方：它仍使用显式 matched pair、两阶段 warm
start、训练期 frozen teacher 和第二个 auxiliary constraint。当前只把它升级为强方法候选，不
声称多 seed、跨任务、Public 或正式非劣已经完成；下一高信息量步骤是单 seed 迁移到另一个连续
隐藏动力学任务，而不是扫描 anchor/pair 权重。

紧凑结果、SHA、100-query 配对列联与 claim boundary 见
[`artifacts/pusht_motion_damping_residual_transition_function_anchor_v1/summary.json`](artifacts/pusht_motion_damping_residual_transition_function_anchor_v1/summary.json)。

### 5.23 Contact Friction 迁移：信号可迁移，但固定配方尚不充分

为区分 Motion 单任务特例与可迁移机制，§5.22 的配方未改权重、loss、模型或 sampler，直接迁移到
同为连续隐藏动力学的 Contact Friction。固定 seed `13313`，先从正式初始化训练 2,048-step
residual-transition source，再从同一 source 以 fresh AdamW 训练 function anchor 候选。各预算
终点是从同一 source 独立启动、scheduler horizon 与预算一致的 endpoint，并非一条轨迹的 exact
prefix；它们只用于快速判断学习趋势。最终 8,192 是该任务预定的正式预算，没有用 Development
选择 checkpoint。

| fresh auxiliary steps | future | history | switch | worst | gain | alignment | NRE | 全门 |
|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0（source） | 0.498 | 0.568 | 0.590 | 0.391 | 0.016 | 0.072 | 1.016 | 否 |
| 1,024 | 0.549 | 0.662 | 0.789 | 0.449 | 0.065 | 0.200 | 0.976 | 否 |
| 2,048 | 0.617 | 0.762 | 0.922 | 0.523 | 0.152 | 0.360 | 0.874 | 否 |
| 4,096 | 0.697 | 0.852 | **0.977** | 0.613 | 0.277 | 0.501 | **0.752** | 否 |
| **8,192** | **0.750** | **0.867** | **0.980** | **0.688** | **0.374** | **0.549** | **0.716** | **否** |

该曲线排除了“Motion 的成功只是任务局部噪声”：七项指标随预算整体同向改善，最终 switch 与
NRE 已过门；模型仍然没有新增可学习参数、保存 module 或推理计算，隐藏 friction 标签也未进入
模型或 loss。但正式 Contact 门要求 future/history/worst 至少 `0.95/0.95/0.90`、gain 至少
`0.50`；最终仍失败四项，故它只证明**跨任务有效信号**，没有证明**跨任务充分性**。按预定门控
不运行 Contact CEM、不打开 Public、不用额外 seed 或更长预算救援。

两个对照进一步收窄了剩余问题。第一，2,048-step 的 counterdirection-balanced sampler 相对
普通 matched sampler 在 future/history/switch/worst/gain 上为
`0.594/0.754/0.910/0.492/0.136` 对 `0.617/0.762/0.922/0.523/0.152`，所有主指标均略差，
所以缺口不是简单的 batch 对向配平。第二，用同一当前 runtime 重评已有 8,192-step 方法后，
不同目标呈现互补失败：

| 已有目标 | future | history | worst | gain | NRE | 失败形态 |
|---|---:|---:|---:|---:|---:|---|
| matching | **0.961** | 0.902 | **0.949** | 4.040 | 34.124 | assignment 强，但响应严重过冲 |
| paired fit | 0.871 | 0.871 | 0.824 | **0.666** | **0.701** | 校准较好，但 assignment 不足 |
| projected geometry | 0.928 | 0.895 | 0.922 | 1.166 | 1.891 | assignment 接近门，响应误差仍坏 |
| function anchor | 0.750 | 0.867 | 0.688 | 0.374 | 0.716 | 稳定、不过冲，但仍欠响应与绝对 future |

因此 Contact 把方法要求明确拆成三个必须同时满足的部分：matched condition assignment、proper
response calibration、ordinary planning-function preservation。现有目标分别优化其中一至两项，
尚无一个同时闭合三项。这个结果不否定 pairing/联合条件关系方法族，也不支持再扫 anchor 权重、
cutoff 或 sampler；它否定的是“§5.22 固定配方已经是通用修复”。下一候选若继续，必须在不增加
encoder/adapter/head 的约束下直接同时覆盖这三项，并先用单 seed 证明完整门，而不是靠更多 seed
放大一个明确失败的 endpoint。

完整 endpoint、checkpoint SHA、对照与 claim boundary 见
[`artifacts/pusht_contact_friction_residual_transition_function_anchor_v1/summary.json`](artifacts/pusht_contact_friction_residual_transition_function_anchor_v1/summary.json)。

### 5.24 Canonical-margin exact future：bootstrap 改善，但不能替代 exact fit

§5.23 表明 matching 能快速建立 assignment，却因持续非零的 softplus 压力把 response gain 推到
`4.04`；proper exact-future 不过冲，却在 8,192 step 仍只有 gain `0.374`。为只改变这一项，在
现有 pair-normalized exact-future 上加入 target-stationary barrier。对 binary pair 定义
`alpha` 为预测 response 沿真实 response 的 gain、`beta` 为沿同一轴的公共中心误差，两侧正确
assignment margin 为 `alpha/2-beta` 与 `alpha/2+beta`。真实 target 对应
`alpha=1,beta=0`，两侧 margin 恰为 `0.5`，因此使用：

\[
L_{\mathrm{canonical}}
=L_{\mathrm{exact}}
+\frac12\sum_{s\in\{-1,+1\}}
\operatorname{ReLU}\!\left(0.5-(\alpha/2+s\beta)\right)^2.
\]

无历史响应时该 barrier 有非零梯度；精确命中两条真实 future 时 loss 与梯度都为零，所以它
不同于会继续过冲的 matching，也不同于在 `alpha=beta=0` 不提供 assignment 梯度的 zero-margin
BC-CCRM。其余 source、数据、`0.09` 权重、optimizer、function anchor 与模型均不变；新增保存
参数、module 和推理计算仍为零。

2,048-step matched screen 的七项指标全部优于 exact control，因而晋级一次正式 8,192：

| endpoint | future | history | switch | worst | gain | alignment | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| exact + anchor，2,048 | 0.617 | 0.762 | 0.922 | 0.523 | 0.152 | 0.360 | 0.874 |
| canonical + anchor，2,048 | **0.654** | **0.809** | **0.957** | **0.570** | **0.208** | **0.425** | **0.824** |
| exact + anchor，8,192 | 0.750 | 0.867 | 0.980 | 0.688 | 0.374 | 0.549 | 0.716 |
| canonical + anchor，8,192 | **0.775** | **0.873** | **0.984** | **0.703** | **0.427** | **0.567** | **0.713** |

这是一个真实但有限的机制正例。短预算时 barrier 明显加快 assignment、gain 和 alignment；到
正式预算时优势缩小，且 future/history/worst/gain 仍未达到 `0.95/0.95/0.90/0.50`。终批
`alpha` 均值从 `0.032` 增至 `0.512`、`|beta|` 从 `0.229` 降至 `0.110`，但两侧 canonical
margin 满足率仍为 `0%`；更关键的是 exact-future residual 为 `1.939`，与无 barrier control 的
`1.956` 几乎相同。barrier 改变了 bootstrap，却没有解决剩余 absolute conditional fit。

冻结 Development 记录还能对这个 residual 作无需新模型的 exact identity 分解。按 256 个 pair
等权平均，canonical 终点为 `4.025 = 0.784 response + 3.241 common-center`，无 barrier control
为 `3.994 = 0.773 + 3.221`；两者约 `80.5%` 的 exact residual 都来自 query-dependent
common center，而不是 response vector。于是剩余断点进一步收窄：模型已能改变 history response，
但在不改写 ordinary planning function 的约束下，不能把隐藏 query 的公共 absolute future center
迁移到 held-out query。canonical barrier 对这一主导项基本无作用。

因此该终点不开 CEM/Public、不补 seed，也不搜索 margin、权重或 schedule。允许的结论是：
**target-stationary assignment pressure 比持续 matching 更合理，并能稳定改善优化；但 Contact
的主要剩余断点已不再是 assignment pressure 不足。** 下一方法必须直接改善跨 query exact
future operator，同时保持 §5.22 的 ordinary-function preservation；继续改边界项只会延长已经
回答的问题。

紧凑结果与冻结身份见
[`artifacts/pusht_contact_friction_canonical_margin_function_anchor_v1/summary.json`](artifacts/pusht_contact_friction_canonical_margin_function_anchor_v1/summary.json)。

### 5.25 Common-center oracle 与最小因果闭环

§5.24 的 residual identity 表明约 `80.5%` 的 held-out normalized exact error 来自公共中心，
但这仍可能只是一个相关分解。为验证其结果级充分性，在 canonical 8,192 checkpoint 上做了一个
零训练、privileged diagnostic：每个 pair 严格保留原 prediction response
`p_high-p_low`，只将 prediction common center 替换为 checkpoint-native target center。原分数被
逐项精确复现，response 保持误差最大仅 `2.38e-7`；替换后结果为：

| endpoint | future | history | switch | worst | gain | alignment | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical 8,192 | 0.775 | 0.873 | 0.984 | 0.703 | 0.427 | 0.567 | 0.713 |
| target-center oracle | **0.984** | **0.984** | 0.984 | **0.984** | 0.427 | 0.567 | 0.713 |

除 gain `0.427 < 0.50` 外全部正式门通过。它严格说明：在保持现有 response 的条件下，正确
common center 足以闭合全部 assignment 缺口；但它使用真实 future target，不是候选方法。

随后只运行四个与 canonical 2,048 同 source/seed/data/optimizer/budget 的零部署参数 MVE，避免
把 oracle 直接误写成 loss：

| 单因素 MVE，2,048 step | future | history | switch | worst | gain | alignment | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| canonical matched control | **0.654** | **0.809** | **0.957** | 0.570 | **0.208** | **0.425** | **0.824** |
| common-center metric weight `1→4` | 0.531 | 0.566 | 0.602 | 0.418 | 0.022 | 0.056 | 1.116 |
| endpoint response + pair-midpoint center | **0.654** | 0.807 | 0.953 | **0.574** | 0.205 | 0.424 | 0.824 |
| stop-grad + unfreeze existing Encoder/Projector | 0.537 | 0.611 | 0.754 | 0.461 | 0.052 | 0.163 | 0.997 |
| 上行改为 frozen source target copy | 0.521 | 0.551 | 0.906 | 0.469 | 0.100 | 0.308 | 0.905 |

第一项反而抹掉 conditional response，证明中心项与 response 在共享参数空间中不能靠标量放大
解耦。第二项将 center 梯度改走同一 Predictor 的 pair-barycenter forward，结果与 control 的最大
主指标差仅约 `0.004`，说明简单 latent mixup 没有改变 held-out operator。第三项允许原有表示
适配，却把 target response energy 从 `0.00988` 推到 `0.02374`，同时显著退化；第四项冻结
target 坐标后 switch 提高，但在线 deployed latent 与训练 target 坐标分离，correct-future MSE
升到 `0.418`，assignment 仍接近 chance。

因此当前能被证据支持的边界是：**common-center 确为 canonical Contact 的主导结果级瓶颈，
但不是一个可独立加权的 loss component；冻结表示、在线 stop-gradient 与固定 target teacher
分别只解决了联合问题的一侧。** 这四个实验共同停止 center-weight、midpoint-mixup、naive
unfreeze 和 frozen-target-copy 分支；不据此否决 matched conditional relation 或 Motion 已通过的
function-anchor 方法。下一候选若继续，必须在同一个 deployed latent 坐标中联合保持 ordinary
function、response 与 absolute center，不能再把其中一项当作可独立后处理的标量目标。

完整紧凑回执见
[`artifacts/pusht_contact_friction_common_center_followups_v1/summary.json`](artifacts/pusht_contact_friction_common_center_followups_v1/summary.json)。

### 5.26 直接条件响应足以学习连续 ICL，但不是完整的规划保持目标

前两节表明 Contact 的 direct common-center regression 会压制 conditional response，但删除它
是否仍能形成通用方法尚不清楚。为此固定一个更贴近研究初衷的 paired auxiliary：对同一
`(Q,A)` 的两段历史只拟合 normalized centered response，并保留在真实 target 处梯度为零的
canonical assignment barrier；native absolute MSE 与 `0.09` SIGReg 不变。它直接约束
`(H,Q,A,Z^+)` 的条件关系，不是 latent 边缘统计，且 checkpoint 不增加参数、module 或推理计算。

同一 Motion source、seed、数据与固定 `+1,024` fresh-AdamW continuation 得到：

| arm | future | history | switch | worst | gain | NRE | Motion 六门 | paired CEM100 |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| 从初始化训练 response-only 2,048 | 0.568 | 0.660 | 0.910 | 0.219 | 0.213 | 0.912 | 是 | 37 vs source 58 |
| 冻结 action encoder + pred-proj BN，无 teacher | **0.654** | 0.686 | **0.961** | **0.473** | **0.331** | **0.784** | 是 | 54 vs source 58 |
| response-only + ordinary function anchor | 0.652 | **0.689** | **0.961** | 0.465 | 0.330 | 0.785 | 是 | 52 vs source 57 |
| exact pair + ordinary function anchor（既有正例） | 0.588 | 0.670 | 0.879 | 0.305 | 0.256 | 0.862 | 是 | **57 vs source 57** |

无 teacher 冻结臂相对 source 的 CEM 配对列联为共同成功 `42`、source-only `16`、candidate-only
`12`、共同失败 `30`，净差 `-4pp`，exact McNemar `p=0.572`；95% paired bootstrap 区间约为
`[-14,+6]pp`。这不是显著劣化证据，但也没有建立预先要求的 `-5pp` formal non-inferiority，
更不能写成“规划不变”。function-anchor 合并臂则为 `46/11/6/37`，净差 `-5pp`，同样未通过
固定的 discovery Pareto 门。两次 CEM 使用相同 query catalog SHA `743a32df…`；source 在重复
runtime 中为 `57--58/100`，所以所有结论只使用各自同次 paired comparison。

该矩阵给出三个比继续调 loss 更重要的结论。第一，center-free 的直接配对关系本身已经足以让
原 LeWM 学会**连续** history-conditioned response；ActionDelay 正例不再只是离散 mode 选择。
第二，冻结已有 action 路径与 BN buffer 将规划缺口从 `-21pp` 缩到 `-4pp`，说明大部分损伤来自
不承载新 ICL 所必需的共享状态漂移。第三，剩余损伤不能由一个标量 source-function drift 判定：
response-only + anchor 的终点 drift 为 `0.00646`，比既有 exact-pair + anchor 的 `0.01679`
更小，CEM 却更差。函数变化的**方向与 query-dependent absolute center**比平均幅度更重要。

这也解释了 Contact 与 Motion 看似相反的现象。删除 center 项会改善 response，而正确 target
center 的 oracle 又足以闭合 Contact assignment；与此同时，保留 exact pair center 的 Motion
配方才守住 CEM。最终目标因此不是在 center 与 response 中二选一，而是联合学习：

\[
\Delta \hat Z^+(H_1,H_2;Q,A)\approx\Delta Z^+,
\qquad
\bar{\hat Z}^+(H_1,H_2;Q,A)\approx\bar Z^+,
\]

并限制 ordinary query--action function 的有害变化。response-only、freeze-only 与
response-only+anchor 的权重、步数、schedule 和 module-route 均据此停止；它们不否决 paired
conditional relation。当前最强方法仍是 exact conditional pair + training-only function anchor：
它满足零新增**部署**参数/模块/计算，在 Motion 上有同 checkpoint Pareto，并在 Contact 上有强
但未过正式门的迁移。最终简化若要去掉 teacher，必须同时保住 response、absolute center 与
ordinary function，不能再退回 SIGReg/VISReg 一类边缘正则，也不能只凭平均 drift 晋级。

紧凑跨实验判定见
[`artifacts/canonical_response_only_cross_task_v1/summary.json`](artifacts/canonical_response_only_cross_task_v1/summary.json)。

### 5.27 动作干预函数锚：规划缺口缩到 1pp，但严格 Pareto 门仍未闭合

§5.26 已证明平均 point-function drift 不能预测 CEM；因此只做了一次与部署边界一致的
action-conditioned 检验，而没有继续扫 response loss。ordinary rows 保留原 point anchor，另将
batch 内最后一个 query-action embedding 做固定 cycle-1 置换，support actions 不变，并令当前
Predictor 的 action response 匹配 frozen source：

\[
\frac{\left\|[F_\theta(H,A')-F_\theta(H,A)]-
\operatorname{sg}[F_0(H,A')-F_0(H,A)]\right\|_2^2}
{\operatorname{mean}\left\|F_0(H,A')-F_0(H,A)\right\|_2^2+\epsilon}.
\]

matched rows 仍只训练 canonical history response；native MSE、`0.09` SIGReg、source、seed、
fresh `+1,024` AdamW、action-encoder freeze 与 pred-proj buffer freeze 全部不变。checkpoint
仍是原 LeWM，新增部署参数、module 与推理计算均为零。

| candidate | future | history | switch | worst | gain | NRE | Motion 六门 | paired CEM100 |
|---|---:|---:|---:|---:|---:|---:|:---:|---:|
| response + point + action-function anchor | **0.650** | **0.688** | **0.961** | **0.461** | **0.329** | **0.786** | 是 | 56 vs source 57 |

CEM 逐 episode 列联为共同成功 `48`、source-only `9`、candidate-only `8`、共同失败 `35`；
净差 `-1pp`，exact McNemar `p=1.0`，固定 10,000 次 paired bootstrap 的 95% 区间为
`[-9,+7]pp`。因此没有证据表明存在实质规划劣化，但它没有满足事先固定的“候选成功数不得
低于同次 source”发现门，不能宣称同 checkpoint Pareto 已闭合，也不据此开放 Public、额外
seed 或跨任务扩展。

这个终点给出一个重要但克制的机制结论：**ordinary point prediction 不是正确的规划保持
充分统计量，action-conditioned future geometry 更接近 CEM 真正消费的对象。**加入一个
action intervention 后，CEM 缺口由 point-anchor 的 `-5pp` 缩到 `-1pp`，同时没有牺牲 Motion
六门；这支持 history intervention 与 action intervention 的联合关系视角。不过一次 1pp 差异
既不能被包装成成功，也不能被用来否决整个条件配对/函数保持家族。固定 cycle、权重、步数和
多 counterfactual 变体到此停止；当前最强严格正例仍是 §5.22 的 exact pair + ordinary function
anchor（`57/100 = 57/100`）。

这个现象还有一个直接的 identifiability 解释。ordinary replay 对给定 `(H,Q)` 通常只观测一个
执行动作 `A_i`；只要扰动函数 `g(H,Q,A)` 在这些观测点满足 `g(H_i,Q_i,A_i)=0`，那么
`F_\theta+g` 与 `F_\theta` 具有相同的 point training loss，却可以在 CEM 查询的未执行动作上
产生任意不同的 future 排序。因此 point anchor 即使几乎为零，也不能保证规划函数保持。matched
history pair 在固定 `(Q,A)` 下识别 hidden-dynamics response；matched action intervention 在固定
`(H,Q)` 下识别 control response。两条轴共同约束的才是本项目原始对象
`p(O^+\mid H,Q,A)`，而不是 latent marginal。该论证说明 frozen teacher 或真实 counterfactual
action pair 至少需要一种；它没有证明本次 cycle-1 teacher 实现已经是最终最简方法。

完整判定、checkpoint 与 paired CEM 身份见
[`artifacts/pusht_motion_damping_action_intervention_anchor_v1/summary.json`](artifacts/pusht_motion_damping_action_intervention_anchor_v1/summary.json)。

### 5.28 无 teacher 的真实 History×Action 四元组：ICL 成功，接触规划仍失配

现有 Motion、Contact 与 ActionDelay 训练 release 都只提供固定 query action 下的 hidden-condition
pair，真实 action-counterfactual 数量为零。为避免继续依赖 frozen source teacher，本节只改变
训练数据组织：对同一个 Motion causal template，在两种 damping history 下分别执行原零动作与
一个沿 query 速度方向的单位动作，得到四个 simulator-real future。模型仍只看到 RGB/action，
不接收 damping、physics state、pair id 或 action-branch id；checkpoint 参数、module 与推理计算
均不变。目标也没有增加 action loss：native MSE 消费四个真实 future，既有 canonical history
response 对每个 action 下的相邻 hidden pair 各执行一次。

先用 256 templates 做快速发现，随后只做一次有明确理由的覆盖确认：2,048 templates 对齐旧
legacy-prefix MVE 的规模，其他变量不变；它不是当前 v4 catalog 的完整 8,192 train pairs。

| real Cartesian coverage | future | history | switch | worst | gain | alignment | NRE | 六门 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|
| 256 templates | 0.582 | 0.633 | 0.848 | 0.301 | 0.284 | 0.312 | 1.258 | 否 |
| 2,048 templates | **0.625** | **0.670** | **0.926** | **0.398** | **0.315** | **0.447** | **0.866** | **是** |

小覆盖训练末 canonical loss 已很低、held-out NRE 却恶化，是典型 coverage overfit；扩大覆盖后
NRE 从 `1.258` 降到 `0.866` 并通过 bootstrap 上界门。因此在这个单 seed Motion 终点上可以
第一次直接写：**包含真实 history–action–future 联合配对的当前完整 recipe，在没有 frozen
teacher、没有新参数的情况下学会了连续条件 ICL。** 这不是局部梯度或 assignment proxy，而是
完整冻结六门终点；它仍包含 canonical response/assignment 辅助项并从 2,048-step source 续训，
所以四元组的单因素因果贡献、跨任务与多 seed 一般性均未验证。

同 checkpoint paired CEM100 仍为 candidate `51/100`、source `58/100`；列联为共同成功 `43`、
source-only `15`、candidate-only `8`、共同失败 `34`，95% paired bootstrap 为 `[-17,+2]pp`。
这里“同 checkpoint”只表示 candidate 的 Development 与 CEM 使用同一 SHA；source 只有
2,048-step exposure，candidate 是该 source 再续训 1,024 步，并非训练预算匹配对照。
严格非负差门失败，故不开 Public/seed/Contact 扩展。关键伴随现象是这 2,048 个 alternate
action branch 的 query contact step 全为 `0`：它们真实地约束了 action response，却没有覆盖
PushT CEM 主要消费的 contact-interaction action geometry。该事实与“intervention support 和
planner support 失配”解释一致，但当前单臂结果尚未完成因果证明；已能否定的更窄命题是：
**任意真实 counterfactual action 都足以同时保住规划。**

当前实现据此关闭：不增加 Motion 模板、不加第三动作、不延长训练、不扫 loss。它不否决真实
action pairing；若继续，唯一有信息量的下一项 falsification 是把 action intervention 放到
ordinary/contact-rich planning states 上，再与固定 `(Q,A)` 的 hidden-history intervention 组合。
该方向仍可保持原 LeWM 和同一个 conditional objective，不需要 encoder、adapter、head 或
marginal regularizer。

Development JSON 是先于 CEM 生成的不可变评估回执，其中 `cem_opened:false` 表示生成当时尚未
运行 CEM；随后执行的 paired CEM 由上面的独立 aggregate 记录。该回执 `claim_boundary` 内的
`optimizer_steps:0` 指评估器没有做参数更新，训练终点本身由顶层 `optimizer_step:1024` 记录。

完整紧凑判定见
[`artifacts/pusht_motion_damping_cartesian_action_pair_legacy_scale_v2/summary.json`](artifacts/pusht_motion_damping_cartesian_action_pair_legacy_scale_v2/summary.json)。

### 5.29 Contact-rich History×Action：接触不是 planner support 的充分统计量

§5.28 的唯一下一证伪问题是：CEM 缺口是否仅因为真实 action branch 没有进入物体接触区。本节
保持 source checkpoint、seed、2,048-template 规模、1,024-step 续训、native MSE+SIGReg 与
canonical history-response/assignment objective 不变，只把 alternate action 改成从 query agent
指向 query block、幅值 `0.45`、连续执行五步的真实 simulator action。该全局幅值是满足完整
2,048 templates 接触与边界硬门的最小固定构造；没有按 template 自适应，也没有过滤难例。

生成资产在训练前满足：

- `2,048/2,048` alternate rollouts 都发生 query contact，最少/平均/最多 contact step 为
  `1/2.146/3`；
- 所有 model-frame state 都在 playfield 内；
- `2,048/2,048` templates 的 History×Action interaction norm 非零，最小值 `1.097`；
- 两个 action branch 的 history/query RGB 逐像素相同，差异只出现在 action 与 simulator-real
  future；模型边界仍只有 RGB/action，没有 hidden label、pair id 或 branch id；
- saved LeWM 新增参数、模块和推理计算均为 `0`，训练期 frozen teacher 为 `false`。

因此这次不能再用“接触没有真正发生”解释终点。直接与 §5.28 的零接触构造比较：

| real 2×2 action support | contact steps | min action future gap | future | history | switch | worst | gain | alignment | NRE | 六门 | CEM100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| zero-contact velocity ray | `0` | `0.253` | **0.625** | 0.670 | 0.926 | **0.398** | **0.315** | **0.447** | **0.866** | 是 | `51` |
| toward-block contact ray | `1–3` | `1.187` | 0.602 | **0.680** | **0.938** | 0.328 | 0.292 | 0.420 | 0.900 | 是 | `40` |

contact checkpoint SHA 为 `dcf20ba…d47a82`。冻结 Development 的 paired-assignment lower 95%
为 `0.622`，NRE upper 95% 为 `0.940`，所以它是一个真实的连续 ICL 正例，而不是只过点估计。
同一 SHA 的正式 paired CEM100 则为 candidate `40/100`、source `58/100`；逐 episode 列联为
共同成功 `34`、source-only `24`、candidate-only `6`、共同失败 `36`，exact McNemar
`p=0.00143`，固定 10,000 次 bootstrap 的 95% 区间为 `[-28,-8]pp`。相对同 catalog 的
零接触 candidate，接触版为 `40 vs 51`，区间 `[-23,+1]pp`。source 只有 2,048-step exposure，
两个 candidate 均是该 source 再续训 1,024 步；因此前一个显著差异是规划晋级门结果，不是
contact 的预算匹配因果效应，后一个才是预算匹配的 action-construction 对照且统计区间跨零。

这个结果使当前固定实现未通过一个明确而窄的充分性检验：**让每个 alternate branch 发生接触，
并没有闭合零接触候选的 CEM 缺口。** 它不否定真实 History×Action×Future 配对；两个构造都
通过 Motion 六门，
说明这种联合数据对连续 conditional identifiability 的作用可重复。它也不证明“contact 导致
CEM 下降”：接触版的最小 action-induced pixel gap 约为零接触版的 `4.7×`，训练末 native
prediction loss 与 excluded center error 也更高，support 与 intervention energy 在这个构造中
共同变化。

因此 planner support 不能被压成一个 contact/no-contact bit。CEM 消费的是跨多方向、多步动作的
整个 `A ↦ O⁺` 排序几何；一个高能量、重复的 toward-block ray 即使在物理上位于接触区，也可能
只让 Predictor 拟合一条很窄的控制方向。当前实现据此关闭：不扫 action amplitude、loss weight、
训练步数、模板数或 seed。若继续探索，只允许一个单因素 MVE：保持同一 LeWM、同一 objective、
同一预算，把单射线替换为从 planner 所消费分布采样的多方向、多步 simulator-real action sequence。
先隔离 action-distribution coverage；在它得到终点前，不新增 factorial loss 或其他辅助项。

完整资产、checkpoint、Development、paired CEM 身份与停止判定见
[`artifacts/pusht_motion_damping_contact_cartesian_action_pair_v1/summary.json`](artifacts/pusht_motion_damping_contact_cartesian_action_pair_v1/summary.json)。

### 5.30 CEM 非劣重判：最强候选仍未定，而不是失败

§5.27 的 action-intervention function-anchor 在 Motion 六门上达到
`future/history/switch/worst/gain/NRE = 0.650/0.688/0.961/0.461/0.329/0.786`，CEM100
却因 `56 < 57` 被原 exact-count discovery gate 阻止晋级。这个门适合保守控制实验流程，却不适合
对有 GPU/CEM 数值波动的总体性能作推断。因此本节不重训、不改 checkpoint、不改 CEM，只在
打开新 eval seeds 前冻结统计问题：300 个 paired episode，实用非劣界 `-5pp`；在每个 seed 内
重抽 paired episode 的 100,000 次 bootstrap，一侧 95% lower `>-5pp` 才判 non-inferior，upper
`<-5pp` 才判 materially inferior，其余统一为 inconclusive。

| eval seed | source | candidate | paired difference |
|---:|---:|---:|---:|
| 42 | 58 | 55 | -3pp |
| 43 | 65 | 61 | -4pp |
| 44 | 64 | 64 | 0pp |
| **pooled** | **187/300** | **180/300** | **-2.33pp** |

pooled 列联为共同成功 `155`、source-only `32`、candidate-only `25`、共同失败 `88`；exact
McNemar `p=0.427`。固定 bootstrap 的一侧 95% lower/upper 为 `[-6.33,+1.67]pp`，双侧
95% 区间为 `[-7.33,+2.67]pp`。因此相对 `-5pp` margin 的正式标签是 **inconclusive**。
结果未显示超过 `5pp` 的实质劣化，但当前 300 对也不足以排除它。完全未参与设计的 seeds
`43/44` 单独为 `125 vs 129`、点差 `-2pp`、一侧区间 `[-7,+3]pp`，结论相同，排除了结论仅由
已观察 seed42 驱动的解释。

这次复现还直接验证了用户指出的门控问题。seed42 查询 catalog 的 episode/row/start-step 三组
身份逐项完全一致，但 source 的 100 位 outcome 中有 7 位翻转，计数由 `57` 变为 `58`；candidate
有 1 位翻转，计数由 `56` 变为 `55`。所以 hard gate 应保留给 checkpoint SHA、query identity、
标签边界与模型结构等不变量；随机性能必须使用 paired effect、预设实用 margin 与区间。新鲜
300 对的 source/candidate 同运行比较仍然有效，旧结果只用于记录 runtime variation。

对方法的结论也应精确：§5.27 仍是目前**综合指标最强、最接近成功**的版本，但没有建立规划
非劣，也没有被证明实质失败。按冻结规则不再事后增加 CEM episode 或修改 margin。下一研究问题
回到真正的简洁性缺口：如何去掉 frozen source teacher、两阶段 warm start 和 privileged matched
pair construction，同时保留其 history-response 与 action-function preservation；不会因为一次
统计未定结果重新扫 anchor 权重或发明边缘正则。

冻结设计与完整结果见
[`configs/pusht_motion_damping_action_intervention_anchor_cem300_noninferiority_v1.json`](configs/pusht_motion_damping_action_intervention_anchor_cem300_noninferiority_v1.json) 和
[`artifacts/pusht_motion_damping_action_intervention_anchor_v1/cem_seeds42_43_44_n100_v1/noninferiority_analysis_v1.json`](artifacts/pusht_motion_damping_action_intervention_anchor_v1/cem_seeds42_43_44_n100_v1/noninferiority_analysis_v1.json)。

### 5.31 经验 replay 动作支持：无 teacher 候选将规划缺口缩到 3pp

§5.29 已排除“只要 action branch 接触物体即可”的解释，但它仍只沿一个重复的 toward-block ray
训练。本节只改变这一项：从原始 PushT expert replay 的 action 列中，以 episode 内五步边界为
单位无放回抽取 `1,024` 个真实 action block；每个 block 恰好复用于一个 forward/reverse twin，
与零 action 组成相同的 `2 histories × 2 actions` 四元组。这里准确的分布名称是
**empirical replay marginal**，不是 CEM proposal distribution。模型仍只看 RGB 与 action，
不接收 damping、pair id、physics state 或 action-branch id；没有 frozen teacher，也没有增加
参数、module、loss family 或推理计算。

完整 `2,048`-template 资产包含 `4,096` 个 condition pair 和 `8,192` 行。八个角度扇区计数为
`686/666/638/541/685/625/620/659`，圆形 resultant 仅 `0.0254`，说明没有退化成单方向 ray；
五步序列 RMS 的 `q05/q50/q95` 为 `0.0717/0.2254/0.5326`。动作作用后的接触数、越界和
History×Action interaction 只作为**结果协变量**记录，没有参与筛样：真实 replay action 应在新
query state 上产生何种物理结果正是待测量，而不是数据进入资格。否则以“每条都接触、每条都
有非零 interaction”为门会把经验边缘改造成 outcome-conditioned 人工分布。实际资产有
`98.97%` model-frame state 在边界内、平均 contact step=`0.0735`，interaction 中位数为 `0`；
所有 action component 仍合法，历史与 query 前缀跨 action branch 逐像素相同。

在同一个 residual-transition source 上续训 `1,024` step 后，冻结 Development 为：

| teacher-free real `2×2` action support | future | history | switch | worst | gain | alignment | NRE | 六门 | CEM100 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| zero-contact velocity ray | **0.625** | **0.670** | 0.926 | **0.398** | **0.315** | **0.447** | **0.866** | 是 | `51` |
| toward-block contact ray | 0.602 | **0.680** | 0.938 | 0.328 | 0.292 | 0.420 | 0.900 | 是 | `40` |
| empirical replay five-step block | 0.600 | 0.660 | **0.941** | 0.328 | 0.307 | 0.425 | 0.908 | 是 | **`55`** |

replay checkpoint SHA 为 `1de63f3d…b94`。paired-assignment bootstrap lower 95%=`0.611`，
NRE bootstrap upper 95%=`0.951`，所以连续条件 ICL 的结论来自冻结终点而不是局部 proxy。
同一次 CEM 运行在相同 100 个 episode 上得到 source/zero-contact/replay=`58/51/55`。
replay 对 source 的逐 episode列联为共同成功/source-only/replay-only/共同失败=`41/17/14/28`，
差值 `-3pp`、95% paired bootstrap `[-14,+8]pp`；对预算匹配的 zero-contact arm 为
`37/14/18/31`，差值 `+4pp`、区间 `[-7,+15]pp`。因此多方向真实 replay support 呈现正确趋势：
它追回了零接触 arm 所失 `7pp` 中的 `4pp`，但当前样本量不能把该差异写成显著因果效应，也
不能声称规划非劣。

这一步仍带来一个实质收敛：**frozen teacher 不是 Motion 连续 ICL 与近源规划表现的必要条件，
单一 ray 也不是 teacher-free pairing 的合理 action support。** 在不改变 LeWM 的前提下，真实
History×Action×Future 联合数据已经同时给出 ICL 正例和接近 source 的 CEM；它与 §5.27 的
fresh seed42 CEM 同为 `55/100`，虽然后者的 ICL calibration 更强。当前 replay 版本因此是
最接近原始简洁目标的候选，而非综合数值上已经超越所有 teacher anchor 的候选。

尚未解决的复杂性必须如实保留：训练仍显式构造 hidden-condition matched simulator pairs，且从
`2,048`-step source 做两阶段续训；当前实验也没有把 empirical replay marginal 等同于 planner
distribution。下一步只允许消除其中一个复杂性并保持当前模型/目标不变，不再扫动作幅值、loss、
正则或新增 encoder/adapter/head。多 seed 与更大 CEM 留到最终方法选定后。

完整紧凑判定见
[`artifacts/pusht_motion_damping_replay_cartesian_action_pair_v1/summary.json`](artifacts/pusht_motion_damping_replay_cartesian_action_pair_v1/summary.json)。

### 5.32 单阶段坐标因果拆分：原 LeWM 已接近同时保住 ICL 与规划

§5.31 的 teacher-free replay 候选仍先训练 `2,048` step native Motion source，再续训 `1,024`
step joint pairs。为判断这一课程是否必要，本节保持同一 overlay、objective、seed、冻结模块和
部署边界，只从已发布的标准 PushT LeWM baseline 开始一次 Motion 适配。这里“单阶段”不表示
从随机初始化训练视觉模型；它准确表示删除额外的 native Motion source 阶段。实验又把另一个
混淆单独拆开：一种保留 §5.31 的 causal-transition/residual 坐标并归零既有输出投影，另一种
完整保留官方 LeWM 的 absolute 输入/输出坐标与全部初始权重。

| adaptation path | joint steps | future | history | switch | worst | gain | NRE | direct screen | CEM100 |
|---|---:|---:|---:|---:|---:|---:|---:|:---:|---:|
| two-stage residual reference | `2,048 + 1,024` | **0.600** | 0.660 | 0.941 | 0.328 | 0.307 | 0.908 | 是 | **55** |
| residual single-stage | `1,024` | 0.584 | **0.670** | **0.949** | 0.270 | 0.274 | 0.930 | **是** | 20 |
| residual single-stage | `3,072` | **0.600** | 0.666 | 0.934 | **0.371** | **0.307** | 1.064 | 否 | 未开 |
| absolute single-stage | `1,024` | 0.520 | 0.590 | 0.910 | 0.133 | 0.086 | 0.921 | 否 | 48 |
| absolute single-stage | `2,048` | 0.533 | 0.609 | 0.918 | 0.184 | 0.170 | **0.901** | 仅 future 未过 | **54** |

第一条结论很明确：**单独的 native Motion source 不是学会条件响应的必要条件。** residual
single-stage `1,024` 已在冻结 Development 上通过全部直接响应检查。然而它的 CEM 只有
`20/100`；这不能归因于 pairing 本身，因为同样的 pair recipe 在两阶段版本是 `55/100`。
真正的额外操作是 residual 重参数化将既有输出投影归零，前置 native stage 因而承担了重建普通
planning function 的职责。把 joint training 延长三倍也不是解：assignment 保持强，但 NRE 从
`0.930` 恶化到 `1.064`，bootstrap upper 达 `1.122`。因此 residual 单阶段与继续延长预算均关闭。

第二条结论更接近最终方法：**保留原 LeWM 坐标可以同时保住大部分规划，并逐步学出条件响应。**
absolute `1,024→2,048` 时 gain 从 `0.086→0.170`、NRE 从 `0.921→0.901`、CEM 从
`48→54`。`2,048` 终点的 paired-balanced macro lower 95%=`0.555`、NRE upper 95%=`0.931`，
九个 direct checks 中只有 raw two-future selection=`0.533` 低于历史固定阈值 `0.55`；约
`9/512` 个选择即可跨门，当前差距处在明显的统计波动尺度内。因此按冻结流程它仍是
`screen_passed=false`，不能事后改门宣称正式通过；按科学判断也不能因一个近门比例否掉已经同时
出现的 calibrated response 与 `54/100` CEM。

相同 query-catalog SHA 下，absolute `2,048` 相对 residual source 为 `54 vs 58`，列联
`both/candidate-only/source-only/neither=37/17/21/25`，paired interval `[-16,+8]pp`；相对两阶段
replay 为 `54 vs 55`，列联 `38/16/17/29`，区间 `[-12,+10]pp`。这些 arm 分别在独立 evaluator
进程中运行，虽 catalog 完全相同，仍可能有 GPU runtime outcome flip；所以它们只支持“接近且
未定”，不支持规划非劣声明，也不触发多 seed 或更大 CEM。

当前方法边界由此发生实质简化。最值得保留的候选不再需要 frozen teacher、额外 Motion source、
residual basis、权重重置、新 encoder/adapter/head 或推理计算；它只是**原 LeWM + 一次
History×Action×Future joint-pair 训练**。仍未满足最终目标的核心只剩显式 simulator-matched
pair construction，以及 raw future 门尚未正式闭合。到此不再扫坐标或训练预算；下一研究问题只
允许把显式/privileged pair annotation 换成可由可见条件或自然共现数据得到的配对规则，同时固定
本节 absolute `2,048` recipe。若做不到，无标签 conditional identifiability 可能需要承认额外
overlap/intervention data assumption，而不是再用 marginal regularizer 掩盖。

完整因果阶梯与身份见
[`artifacts/pusht_motion_damping_replay_cartesian_action_pair_single_stage_coordinate_ablation_v1/summary.json`](artifacts/pusht_motion_damping_replay_cartesian_action_pair_single_stage_coordinate_ablation_v1/summary.json)。

### 5.33 可见条件配对：privileged annotation 可被精确删除

§5.32 将模型与训练流程压回原 LeWM 单阶段后，剩余复杂性看起来是训练必须读取显式
hidden-condition pair。这里不再训练候选，而是先检验这个 metadata 是否真的提供额外信息。
对 replay Cartesian overlay 的每一行，仅计算：

\[
k_i=\operatorname{SHA256}(Q_i^{\mathrm{RGB}}\,\|\,A_i^{\mathrm{raw}}),
\]

其中 action 包含完整四个五步 block。key 明确排除 history、future、damping/hidden mode、pair id、
template id 与行顺序。然后把具有相同 key 的行无监督分组；若某个 key 不是恰好两行则 fail closed。

完整 `8,192` 行得到的结果是：

- 可见 key 数 `4,096`，每个 key 的 cardinality 都恰好为 `2`；
- 每组 history 与 future 均不同，但二者都没有参与 key；
- mined pair set 与旧显式 pair set 逐组完全相等；
- mined/explicit group-mapping SHA 同为 `d7c2866f911c…c0c9e`；
- canonical response/assignment loss 对 group permutation 和组内方向翻转数值严格不变。

因此当前成功 recipe **不需要 damping label，也不需要 pair annotation**。现有显式 row order 只是
一种可被 `(Q,A)` 可见条件精确重建的存储便利，不是监督来源；把 runtime sampler 的 group map
换成上述 miner 不改变被优化的 pair set。这个结论把“privileged pairing”进一步收窄为一个更
一般、也更诚实的数据条件：

\[
\text{same }(Q,A),\quad \text{different }H,\quad \text{observed real }O^+.
\]

也就是 conditional overlap / intervention coverage。它正是条件联合分布可辨识所需的信息，和
SIGReg/VISReg 只约束 latent marginal 有本质区别。

边界同样重要：当前 overlay 的 exact overlap 由 simulator 构造；普通 unmatched offline replay
中连续图像几乎不会出现逐字节相同的 `(Q,A)`。所以本节删除的是 hidden label 与 annotation，
不是数据覆盖假设，也没有证明 approximate nearest-neighbor matching 可行。下一步若继续追求
通用方法，应固定 §5.32 的 absolute `2,048` recipe，只比较一种无需 hidden label 的 overlap
获取方式，例如环境 reset 后重复同一 query action、或由自然多 episode 共现得到的可见条件组；
不再增加模型参数、context encoder、adapter 或 marginal regularizer。

零训练回执与可执行 miner 见
[`artifacts/pusht_motion_damping_visible_condition_pair_mining_v1/receipt.json`](artifacts/pusht_motion_damping_visible_condition_pair_mining_v1/receipt.json) 和
[`scripts/mine_visible_condition_pairs_v1.py`](scripts/mine_visible_condition_pairs_v1.py)。

### 5.34 无标签主动 overlap 收集：named endpoint matching 也不是必要输入

§5.33 删除了训练时的 hidden label 和 pair annotation，但原资产仍由 builder 明确遍历
`faster_decay/no_extra_decay` 两个名字，并使用各自解析反演出的 x0。这只能证明“annotation
可删”，尚不能证明数据收集本身不需要知道哪条轨迹属于哪个 dynamics endpoint。本节固定
§5.32 的模型、absolute 坐标、单阶段 `2,048` recipe、replay action support 与 joint auxiliary，
只替换数据获得协议。

新的 collector 对训练侧只暴露一个 opaque randomized-environment handle，协议为：

1. 从独立随机环境中取得一条隐藏动力学 realization，不读取其 damping 值或 mode 名；
2. 从预选 query state 开始作一次零动作 probe，仅用返回的 query-state transition 估计速度衰减
   与位移系数，再反推 x0；确认轨迹经过十个 history raw steps 后自然到达同一 Q；
3. 独立重复抽样；若两次得到的 **history RGB 与 history action** 完全相同才丢弃，future 从不参与
   接纳；
4. 从同一 x0 重放 zero 与 empirical-replay 两个 query-action branch，最后仍只按可见
   `(query RGB, raw action blocks)` 分组。

这里没有在 history/query 边界恢复状态：每条存储轨迹都从 x0 到 future 使用一个连续 simulator，
`state_installations_after_x0=0`。黑盒 shooting 的输入包含数据收集器可测的物理 query-state
feedback；这些量、随机环境 identity 与 post-hoc mode audit 均不进入保存给 LeWM 的
RGB/action、不进入分组，也不进入 loss。

全 `2,048`-template frozen prefix 的结果为：

- 独立 opaque draw 总数 `6,195`，每个 template 最少 `2`、平均 `3.0249`、最多 `11` 次；
- 所有接受的 context 在事后审计中确实来自不同 endpoint，但该 identity 没有参与接纳；
- 黑盒 shooting 的最大完整 query-state 误差为 `3.2401e-12`，远低于 `1e-8` 容差；
- 得到 `4,096` 个可见 condition group、`8,192` 行；每个 template 的四行与旧冻结 overlay
  作为无序集合逐字节相同，`2,048/2,048` 全部成立。

这项等价性有一个直接而节省算力的含义：§5.33 已证明 canonical objective 对 group 顺序与组内
方向翻转不变；本节又证明新 collector 提供完全相同的经验行集合。因此重新花 GPU 训练不会检验
新的方法变量，§5.32 的 absolute `2,048` checkpoint 已经是这一 label-blind collection recipe
的训练终点。当前最简候选可以准确写成：

> 原 LeWM + 单阶段训练 + visible-condition joint auxiliary + label-blind active overlap collection。

它删除了 teacher、额外 Motion warm start、residual basis/reset、新参数/模块/推理计算、hidden
label、pair annotation，以及按 named low/high endpoint 配对的 collector 逻辑。它**没有**删除
主动环境随机化、对初始/目标 query state 的控制，也没有让普通 unmatched replay 自动获得 exact
overlap。这是剩余的数据覆盖假设，不应再包装成模型监督。下一方法验证应固定这条 recipe，直接
迁移到另一个 LeWM/PLDM 失败的连续任务；若跨任务失败，应归因并修正 joint relation 的通用性，
而不是再增加 marginal regularizer 或部署模块。

全量执行回执与实现见
[`artifacts/pusht_motion_damping_label_blind_overlap_collection_v1/receipt_templates2048_v1.json`](artifacts/pusht_motion_damping_label_blind_overlap_collection_v1/receipt_templates2048_v1.json) 和
[`scripts/qualify_pusht_motion_damping_label_blind_overlap_collection_v1.py`](scripts/qualify_pusht_motion_damping_label_blind_overlap_collection_v1.py)。

### 5.35 Contact Friction：同一最简 joint relation 形成跨任务 Pareto 正例

§5.34 把 Motion 的剩余复杂性准确收缩为 active conditional-overlap 数据假设。本节不再改模型、
坐标或 loss family，而把同一个 center-free joint relation 直接迁移到当前 Contact Friction：

- 从 published standard PushT LeWM 开始一次训练，不经过 Motion/source warm start；
- 保持 absolute input/output、native MSE、`0.09` SIGReg 和原初始化；
- paired auxiliary 固定为 `0.09 × (centered response + canonical assignment 0.5)`；
- 只训练现有 Predictor/pred_proj，Encoder、Projector、action encoder 冻结；
- 不使用 teacher、hidden label，不增加参数、module 或推理计算。

这里的训练数据口径需要精确区分“batch 行数”和“每个 loss 的输入”。每个 optimizer step
固定拼接 `64` 条原始 PushT replay 与 `64` 条 current ContextWorld Contact 行（`32` 个完整
pair），所以**样本行数是严格 50/50**。原始数据的 optimizer rows 来自
`lewm_pusht.lance`；`pusht_expert_train.h5` 只提供 action normalization 与 provenance。
ContextWorld 的逻辑 component/release identity 仍为 `...contact_friction...v1`，但本实验实际
钉死的是 current `pusht_contact_friction_h3_release_v3` 内容，manifest SHA 为
`cbb9b1a1c030...`，不是旧 `h3_v1` 资产。loss 路由则不是“每项各占一半”：native prediction
是 `0.5 × original full-horizon MSE + 0.5 × Contact terminal MSE`；SIGReg 联合观察全部
`128` 行和全部 latent time；joint auxiliary 只观察 `64` 条 Contact 行的 terminal pair。

这一 50/50 行数协议与当前九项 benchmark reference component 中的八项一致：ActionDelay、
Speed、PortalExit、ActionStrength、MotionDamping、ContactFriction、ArmMass 与 GripperCarry
均为原环境/ContextWorld 各半；**DoorPassability 是明确登记的 synthetic-only 例外**，不能写成
九项全部 50/50。50/50 也始终只表示采样曝光，不表示 SIGReg、PLDM 或 paired auxiliary 的
非线性 loss contribution 可以拆成相等两半。

冻结 current Contact Development 的单 seed 结果为：

| arm | steps | future | history | switch | worst | gain | alignment | NRE | joint pairs |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| published source | 0 | 0.496 | 0.518 | 0.504 | 0.340 | -0.000 | -0.004 | 1.004 | 0.000 |
| matched native, **joint weight 0** | 4,096 | 0.521 | 0.594 | 0.668 | 0.406 | 0.015 | 0.128 | 0.984 | 0.027 |
| matched native, **joint weight 0** | 8,192 | 0.535 | 0.697 | 0.820 | 0.434 | 0.051 | 0.285 | 0.930 | 0.063 |
| shifted non-overlap joint | 2,048 | 0.518 | 0.545 | 0.633 | 0.410 | 0.009 | 0.088 | 0.993 | 0.020 |
| center-free joint | 2,048 | 0.732 | 0.826 | 0.996 | 0.707 | 0.401 | 0.632 | 0.600 | 0.383 |
| **center-free joint** | **4,096** | **0.771** | **0.850** | **1.000** | **0.734** | **0.447** | **0.650** | **0.579** | **0.430** |
| center-free joint | 8,192 | **0.809** | **0.859** | 0.996 | **0.773** | **0.491** | **0.655** | 0.580 | **0.488** |
| exact-center one-factor control | 2,048 | 0.525 | 0.623 | 0.742 | 0.422 | 0.030 | 0.182 | 0.967 | 0.043 |

这里最重要的不是某个比例是否越过历史高门，而是 source→candidate 的完整效应形态：原模型
基本没有连续 response；center-free joint training 同时创造了接近完整的 history switch、正向
且有幅值的 response、显著更低的 NRE，以及低/高 friction 两组都存在的 future preference。
正式冻结门在 8,192 仍未全过，故本节不打开 Public、不补 seed，也不改门；科学结论使用效应量
和配对区间，不把有采样波动的单个比例当作方法真假的唯一判据。

最关键的归因对照现已补齐。`matched native, joint weight 0` 从同一 published checkpoint
开始，使用相同 seed、current release、64/64 batch、pair 顺序、冻结模块、optimizer、scheduler、
4,096-step 预算，甚至保留同一个 deterministic joint forward；唯一变化是 auxiliary 的优化权重
从 `0.09` 置为 `0`，其数值只作诊断。它在终点仅得到 gain `0.015`、NRE `0.984`，而同轨迹
joint arm 为 gain `0.447`、NRE `0.579`。在冻结的 `256` 个 Development pair 上做描述性
paired bootstrap（50,000 次，seed `20260825`），joint-minus-native 的 future/history/switch/
joint-pair 差分别为 `+25.0pp [21.9,28.1]`、`+25.6pp [21.1,30.1]`、
`+33.2pp [27.3,39.1]`、`+40.2pp [34.4,46.5]`；gain/alignment/NRE 差为
`+0.432 [0.406,0.459]`、`+0.523 [0.488,0.557]`、`-0.405 [-0.433,-0.376]`。
这些区间是单 seed 的 paired effect 描述，不替代最终多 seed；但它们已明确否定“只要 50/50
mixed native continuation 就会出现同等 ICL”的解释，并确认联合条件关系梯度是当前完整
recipe 的活性成分。

第二个单因素对照进一步定位了数据假设。它保留相同的 `64` 条 Contact 行、`32` 个二元组、
low/high 行位、joint 公式、`0.09` 权重和 2,048-step 计算量，只把每组第二成员循环移到下一个
真实 pair，因此 `0/32` 个 auxiliary group 仍具有相同可见 query/action。终点几乎退回 native：
future/history/switch=`0.518/0.545/0.633`，gain=`0.009`，NRE=`0.993`。相对这一 shifted
control，正确 overlap pairing 的 paired bootstrap 差为 future `+21.5pp [18.4,24.6]`、
history `+28.1pp [23.8,32.4]`、switch `+36.3pp [30.5,42.2]`、gain
`+0.392 [0.366,0.418]`、alignment `+0.545 [0.509,0.580]`、NRE
`-0.392 [-0.420,-0.363]`。因此有效成分不是 generic binary contrast，也不是“任意两条历史
组成 joint”；它必须估计**同一 `(Q,A)` 下**的条件未来差分。这个结果不证明人工生成的 exact
pair 在理论上不可替代，但它排除了 random pairing：若要删除专用 pair，下一步必须从自然 replay
中用可见变量构造可靠的 exact/near overlap 或条件核，而不能把随机 negatives 当作条件干预。

common-center 对照又直接排除了一个很容易误走的方向。两臂除是否加入
`normalized_common_center_mse` 外完全相同；center-free 首 batch 的 response+assignment 为
`1.490`，被排除的 center 项却为 `14.509`。把它以同一 `0.09` 权重直接加入后，2,048-step
gain 从 `0.401` 降为 `0.030`，NRE 从 `0.600` 回到 `0.967`。因此 absolute future 剩余误差
虽然可被 oracle center 解释，却不能用 pair-scale-normalized center 回归直接训练；它会淹没真正
的条件差分。这一对照停止，不做 center 权重补救。

同 checkpoint 的 standard PushT CEM 必须用同数据、同预算的 native arm 做方法归因；published
source 只回答最终部署效用，不能隔离 auxiliary。修正后的同进程、同 query 结果为：

| checkpoint | matched native | exact joint | joint−native | 95% paired bootstrap | both / joint-only / native-only / neither |
|---|---:|---:|---:|---:|---:|
| exact-graph control 2,048 | `67` | `67` | `0pp` | `[-10,+10]pp` | `54/13/13/20` |
| center-free 4,096 | `73` | `72` | `-1pp` | `[-9,+7]pp` | `65/7/8/20` |
| center-free 8,192 | `69` | `63` | `-6pp` | `[-15,+3]pp` | `55/8/14/23` |

4,096 和 8,192 的同次 source 均为 `70/100`。旧 4,096 `75/69` 与 8,192 `63/70`
source-only 点差混入了训练数据与 evaluator/catalog 波动，现只保留为 deployment 观察，不再承担
方法因果归因。三个 matched 区间都包含零：不能声称 exact joint 显著提高规划，也不能把 8,192
写成确定劣化。结合 direct ICL，**2,048–4,096 是当前有直接正效应且未观察到规划代价的区间；
8,192 出现过训练后的不利趋势，但不足以否定方法族。** 发现阶段不再靠 noisy 单次成功数设置
逐字节硬门，也不继续做 budget sweep。

另一个动作覆盖诊断没有进入方法结论。旧 task-version 的 action-coverage v2 train 只有 2,048
pairs，future gap 约为当前任务的两倍；把它与 current Development 拼成 hybrid 后仅得到
future/history/switch/worst=`0.523/0.520/0.656/0.336`、NRE=`2.236`。这证明旧资产不能回收为
当前任务证据，不证明 action diversity 有害，也不反驳当前任务中的 within-query
History×Action Cartesian coverage；该跨版本 arm 停止。

这项跨任务结果把当前主方法假设进一步收紧为：

> 原 LeWM + 单阶段 center-free visible-condition joint training + conditional-overlap data。

这一 joint-pair 原理已在离散 ActionDelay 和连续 Motion/Contact 中给出强正信号，并在 Contact
首次同时改善 ICL
与标准规划；因此 PLDM/VISReg 式边缘或 target geometry 不是唯一可行路径。剩余未解决的核心不是
增加 encoder/adapter/head，而是把 active exact overlap 推广到自然共现或近似匹配数据，同时
保持 4,096 Pareto 点的 planning function。完整紧凑证据见
[`artifacts/pusht_contact_friction_visible_joint_transfer_v1/summary.json`](artifacts/pusht_contact_friction_visible_joint_transfer_v1/summary.json)。

### 5.36 公平 comparator 重审：救回 Contact，确认 Motion 的真实 Pareto 冲突

前述 CEM 记录混用了两类问题：候选是否优于 published source，以及 auxiliary 在同一训练
recipe 中的增量效应。这里统一拆成三项：

\[
\Delta_{\mathrm{data}}=\mathrm{native}_{\mathrm{mixed}}-\mathrm{source},\quad
\Delta_{\mathrm{method}}=\mathrm{joint}-\mathrm{native}_{\mathrm{mixed}},\quad
\Delta_{\mathrm{deploy}}=\mathrm{joint}-\mathrm{source}.
\]

只有第二项能归因 joint auxiliary。matched native 必须保持初始化、`64+64` 数据、sampler、
module freeze、optimizer/scheduler、训练步数和 CEM 进程完全相同，并只把 joint weight 从 `0.09`
置为 `0`。按此标准，Contact 2,048/4,096 的 method effect 分别为 `0pp [-10,+10]` 与
`-1pp [-9,+7]`，旧 source-only 下降不能再用来否定 exact joint；8,192 为
`-6pp [-15,+3]`，只支持过训练风险，不支持方法族硬失败。

Motion 的同一对照给出更有辨别力的结果：

| Motion 2,048 | future | history | switch | worst | gain | alignment | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched native，joint weight 0 | 0.449 | 0.385 | 0.098 | 0.070 | -0.161 | -0.261 | 1.701 |
| exact joint | 0.533 | 0.609 | 0.918 | 0.184 | 0.170 | 0.347 | 0.901 |
| joint−native | +0.084 | +0.225 | +0.820 | +0.113 | +0.331 | +0.607 | -0.801 |

这排除了“Motion 是 mixed native 自己学会”的解释：joint auxiliary 确实把响应从负方向翻到
正方向，并使 assignment、alignment 与 NRE 同时大幅改善。对应的同进程、同 100 query CEM 为：

| effect | successes | point | 95% paired bootstrap | both / first-only / second-only / neither |
|---|---:|---:|---:|---:|
| data：native−source | `66/70` | `-4pp` | `[-14,+6]pp` | `54/12/16/18` |
| method：joint−native | `55/66` | `-11pp` | `[-22,0]pp` | `45/10/21/24` |
| deployment：joint−source | `55/70` | `-15pp` | `[-27,-3]pp` | `43/12/27/18` |

因此 Motion 的旧 `54 vs 58` 并不是由不公平 source comparator 制造的假失败；在当前严格
matched catalog 上，数据 recipe 本身只有统计未定的 `-4pp`，joint 的增量 planning 信号反而更
不利。科学结论必须同时保留两面：**exact joint 是强 ICL 正因子，但在 Motion 上不是 Pareto
解。** Contact 表明 joint 与规划可以共存，Motion 表明 conditional overlap 本身还不充分；
剩余断点是 paired support 上学到的 response 如何外推到 planner 实际消费的 query/action
函数，而不是再怀疑 50/50 数据或模型容量。

这次重审没有机械重训全部历史候选，而按“结论是否可能翻转”分层：

| 历史候选 | 修正后状态 | 是否需重训 |
|---|---|---:|
| Contact exact visible-overlap joint | source-only 归因已撤销；强 ICL、2,048/4,096 planning-neutral | 否 |
| Motion absolute single-stage exact joint | 强 ICL 增量确认；matched-native planning 损伤也确认 | 否 |
| Motion function/action-function anchor | 原本就是 positive/inconclusive，不是 failed；但 teacher/额外约束更复杂 | 否，保留上界 |
| 旧 Motion response-only、replay、consolidation、两阶段 quartet | source-only CEM 不能作严格方法归因，但已被更简单单阶段或 function-anchor 证据支配 | 否，不为归档臂补算力 |
| ActionDelay full-gradient PCJA | `-13.33pp` 与 matched full-F control 的 `-13.67pp` 同源；PCJA 增量仅 `+0.33pp`、区间跨零 | 否，撤销错误机制归因 |
| Contact QA-only/RGB approximate graph | partial ICL 真实；相对 matched native 的 `-11pp/-8pp` planning 信号仍在 | 否 |
| VISReg、边缘正则、Motion PCJR、连续 transition-basis | 直接 conditional-response 终点失败；CEM comparator 无法救回 | 否 |

这一区分避免两种相反错误：既不因 source 数字漂移误杀真正的 joint 正效应，也不因发现 comparator
问题就把所有旧负例重新解释成成功。后续任何候选都必须同时报告
`data/method/deployment` 三个效应；方法发现期使用 paired effect 与区间，不再以一个波动比例的
硬越界替代判断。完整机器可读汇总见
[`artifacts/conditional_joint_comparator_validity_v2/summary.json`](artifacts/conditional_joint_comparator_validity_v2/summary.json)。

### 5.37 非冗余 Cartesian 关系与参数路径：规划保持来自 native adaptation，而非 action target 或简单缩步

§5.36 确认 Motion 的 joint auxiliary 既是强 ICL 正因子，也带来真实的 matched planning 冲突。
本节不再增加 encoder、adapter、head、teacher 或边缘正则，而依次回答两个更窄的问题：现有真实
`2 histories × 2 actions × 2 futures` 四元组是否缺少 action 关系，以及 joint 解与 native 解之间
是否存在不增加模型容量的 Pareto 路径。所有训练臂继续使用同一 published 初始化、seed `14321`、
`2,048` step、absolute LeWM 坐标、每 batch `64` 条原始 PushT 加 `64` 条 ContextWorld、原生
MSE+`0.09` SIGReg 和同一 replay overlay。

#### action target 关系的严格拆分

四条规范化边并不是四个独立约束。两条 history edge 已包含 history main effect 与
History×Action interaction；再加入两条 raw action edge 会重复施加 interaction。为区分这种冗余与
单纯减小 history 权重，实验加入了有效 history 系数同为 `0.045` 的 matched control；随后又只加
一个 history-averaged action-main contrast，避免重复 interaction。三者均为零新增参数、零新
module、零 teacher，复用完全相同的 canonical relation。

| Motion 2,048 | future | history | switch | worst | gain | alignment | NRE | CEM100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| history-only，weight `0.09` | **0.533** | **0.609** | **0.918** | **0.184** | **0.170** | **0.347** | **0.901** | `53–54` |
| history-only，weight `0.045` | 0.531 | 0.607 | 0.895 | 0.164 | 0.141 | 0.303 | 0.934 | `55` |
| history/action 四边均值 | 0.520 | 0.584 | 0.844 | 0.133 | 0.095 | 0.248 | 0.958 | `58` |
| history + 单一 action-main | 0.523 | 0.600 | 0.906 | 0.152 | 0.129 | 0.311 | 0.914 | `49` |

四边均值相对 half-history 的严格 action-edge CEM 效应只有 `+3pp [-8,+14]pp`，但所有直接 ICL
指标都更差；不能把它相对 full-history 的点数变化全部归因于 action edge。action-main 也没有解决
冲突：同进程 CEM 为 action-main/history-only/native=`49/53/66`，action-main−history-only 为
`-4pp [-14,+6]pp`，相对 native 为 `-17pp [-27,-7]pp`。因此当前证据关闭“再给 target action
geometry 加一条关系即可保住规划”的支线；不继续扫 action edge、mixing weight 或 margin。

#### matched native–joint 路径存在，但简单向初始化缩回并不等价

matched native 与 history-only joint 从同一初始化出发，并保持数据、seed、预算和 module freeze
完全相同。对两个终点作零训练权重插值

\[
\theta_\alpha=(1-\alpha)\theta_{\mathrm{native}}+
\alpha\theta_{\mathrm{joint}}
\]

可直接检验两种能力是否位于同一局部参数盆地。该实验只是机制诊断，不把两次训练的 model merge
包装成最终方法。

| joint 比例 `α` | future | history | switch | worst | gain | alignment | NRE | CEM100 |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.80 | 0.514 | 0.576 | 0.773 | 0.148 | 0.084 | 0.216 | 0.984 | 未开 |
| **0.90** | **0.525** | **0.605** | **0.887** | **0.164** | **0.127** | **0.295** | **0.931** | **59** |
| 0.95 | 0.529 | 0.609 | 0.902 | **0.176** | **0.149** | **0.324** | **0.913** | 未开 |

同进程 α=`0.90`/joint/native CEM=`59/54/66`。α=`0.90` 相对 joint 为 `+5pp
[-2,+13]pp`，相对 native 为 `-7pp [-17,+3]pp`。因此“conditional ICL 与 planning retention
只能二选一”过强：至少存在同时保留直接响应并追回约五个百分点规划表现的连续参数方向。但这个
结果仍是单 seed discovery，区间跨零，而且需要两个已训练终点；它不是最终简洁配方。

为检验第二个训练终点是否可删除，固定沿用 α=`0.90`，不再扫描，将 matched-native 端点替换为
joint 本来就使用的 published 初始化：

\[
\theta_{\mathrm{shrink}}=\theta_{0}+0.90
(\theta_{\mathrm{joint}}-\theta_{0}).
\]

这个单训练、零 teacher、零新增参数的版本仍保留很强的直接响应：
future/history/switch/worst=`0.523/0.619/0.918/0.148`，gain/alignment/NRE=
`0.136/0.332/0.896`。但同进程 source/native/joint/shrink CEM=`70/66/54/55`；shrink−joint
仅 `+1pp [-7,+9]pp`，shrink−native=`-11pp [-21,-1]pp`。所以成功的 α=`0.90` 路径不是普通
“少走 10% joint update”，而是 matched native 终点中已经学到的 mixed-data planning adaptation
具有实质作用。

#### 单轨迹顺序训练：native warmup 有效，固定参数 shrink 无效

最后把两终点诊断改成一条真实训练轨迹。从 matched native-2,048 checkpoint 出发，两臂都再看
相同的 `64+64` 数据并使用 fresh AdamW 训练 1,024 步；native-continuation 保持 joint weight
为零，joint-continuation 使用原 `0.09` center-free history relation。随后只对第二阶段增量作
预先固定的 `0.90` shrink。三臂总训练曝光均按 `3,072` 步解释；CEM 的方法效应只比较这组
同预算 continuation，不能拿 2,048-step source 替代。

| 3,072-step trajectory | future | history | switch | worst | gain | alignment | NRE | CEM100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| native continuation | 0.459 | 0.379 | 0.105 | 0.102 | -0.150 | -0.257 | 1.643 | **69** |
| **joint continuation** | **0.529** | **0.588** | **0.887** | **0.195** | **0.163** | **0.325** | **0.926** | **59** |
| joint-delta shrink `0.90` | 0.523 | 0.584 | 0.832 | 0.180 | 0.121 | 0.271 | 0.958 | 55 |

joint continuation 相对 native continuation 的 CEM 为 `-10pp [-20,0]pp`，列联
`both/joint-only/native-only/neither=50/9/19/22`。因此 native warmup 确实把单阶段 joint 的
同次约 `54` 提高到 `59`，并保留强连续响应；但它仍没有闭合严格 matched planning 差距。
固定 shrink 不仅没有进一步保护规划，反而相对未 shrink joint 为 `-4pp [-12,+4]pp`，同时降低
gain/alignment 并提高 NRE。由此 parameter interpolation、初始化 shrink 和 continuation-delta
shrink 三条权重空间补救均停止，不再扫描比例。

这一结果把“最简通用修复”的边界说得更清楚：**joint condition relation 负责学会 ICL；native
adaptation 能减轻但不能消除 planner-function forgetting；真正接近 Pareto 的旧正例仍额外保留
了 action-conditioned source function。** 因而下一步若仍要求 Motion 同 checkpoint Pareto，
需要直接验证最小 functional preservation 是否不可省，而不是继续造 target-action relation、
边缘正则或参数 shrink。若严格禁止任何 training-only function reference，则当前最诚实的结论是
simple joint-pair 方法已在 ActionDelay、Contact 和 Motion ICL 上成立，但 Motion planning
仍构成其通用性反例。

#### 严格 matched function anchor：CEM300 确认为部分规划修复，而非 Pareto 闭环

为避免再用旧 residual source 或不等训练曝光误判 function anchor，只增加一个决定性 continuation
臂。它与上表的 joint continuation 从同一 native-2,048 SHA `c1c48e…` 出发，使用相同 fresh
AdamW、`1,024` 步、seed `14321`、absolute LeWM、`64` 条原始 PushT + `64` 条 ContextWorld、
同一 replay overlay、原生 MSE + `0.09` SIGReg 和 `0.09` center-free history relation。唯一新增
训练信号是在普通 64 行上复用既有 normalized source-function anchor；teacher 是同一 source
Predictor + pred_proj 的冻结副本，不写入 checkpoint。保存模型仍为原 LeWM，新增部署参数、
module 与推理计算均为零。

直接冻结 Development 结果表明 anchor 没有通过牺牲 ICL 换规划：

| 3,072-step trajectory | future | history | switch | worst | gain | alignment | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| joint continuation | 0.529 | 0.588 | 0.887 | 0.195 | 0.163 | 0.325 | 0.926 |
| **joint + ordinary function anchor** | **0.531** | 0.584 | 0.883 | 0.191 | 0.161 | 0.325 | 0.928 |

100-query 同进程预览为 native/joint/anchored=`64/59/61`，三个点差的 paired 区间均跨零；它只
足以支持继续扩充同一 catalog，不能支持“近 Pareto”。随后在同一进程、seed42 和 300-query
catalog 上得到：

| CEM300 arm | successes | rate |
|---|---:|---:|
| native continuation | 208/300 | 69.33% |
| joint continuation | 181/300 | 60.33% |
| joint + ordinary function anchor | 191/300 | 63.67% |

100,000 次固定 seed `20260826` 的 paired-query bootstrap 与逐 query 配对检验为：

| paired effect | 点差 | paired 95% interval | both / first-only / second-only / neither | exact McNemar p |
|---|---:|---:|---:|---:|
| anchored − joint | +3.33pp | [-1.33, +8.00]pp | 160 / 31 / 21 / 88 | 0.212 |
| anchored − native | -5.67pp | [-11.67, +0.33]pp | 159 / 32 / 49 / 60 | 0.0748 |
| joint − native | -9.00pp | [-15.33, -2.67]pp | 145 / 36 / 63 / 56 | 0.00863 |

因此 300 次并非仍然“三者无法区分”：在本次冻结 catalog/runtime 上，**joint 相对 native 的
规划下降已经被配对区间分离**；anchor 位于两者之间。anchor 相对 joint 的 `+3.33pp` 是方向一致
的部分修复，但区间仍跨零；相对 native 仍有 `-5.67pp` 点缺口，零差区间也只是在上界轻微跨零，
更不能据此声称正式非劣。若采用 `-5pp` 非劣界，其区间下界明显越界。

这些区间的含义也需与“指标运行随机性”分开。固定 checkpoint、代码和 256-pair Development
catalog 时，future/history/switch/worst/gain/alignment/NRE 的点估计在 `eval/no_grad` 下不含
随机抽样；它们仍有**有限 catalog 的代表性不确定性**，paired bootstrap 表达的是换抽 query 后的
不确定性，而不是同一计算反复运行的噪声。CEM 还叠加了有限 query、迭代采样、GPU 数值与环境
rollout 的运行敏感性，所以方法比较必须优先使用本次同进程、同 query 的 paired 结果。训练 seed
变异则是第三层不确定性，单 checkpoint 的任何区间都不覆盖它。

严格结论因而更新为：ordinary function anchor 在几乎不改变 ICL 响应的前提下，呈现规划保持的
**部分修复**，但不是已闭环的近 Pareto 方法。继续把 CEM 从 300 增到更大只会更精确地测量当前
缺口，不能修复它；下一次训练仍只应复核旧 Pareto 正例中尚未进入严格 matched 路径的最小成分——
优先 exact pair center + ordinary function anchor，其次才是 action-conditioned function
retention——而不是扫描 anchor 权重、返回边缘正则或增加新模块。

完整机器可读结果见
[`artifacts/pusht_motion_damping_cartesian_parameter_path_v1/summary.json`](artifacts/pusht_motion_damping_cartesian_parameter_path_v1/summary.json)。
本次严格 matched function-anchor 回执、直接评估与 CEM 配对结果见
[`artifacts/pusht_motion_damping_native2048_joint_function_anchor_continuation1024_v1/summary.json`](artifacts/pusht_motion_damping_native2048_joint_function_anchor_continuation1024_v1/summary.json)。

### 5.38 短程筛查与完整预算：方向有时可预测，终点与规划不可外推

当前正证据并非都来自短跑。ActionDelay 的冻结完整 recipe 已有三个独立训练 seed；Contact
已有 `2,048/4,096/8,192` 步端点；本节又对当前最简 absolute joint relation 完成 Motion
`8,192` 步 joint/native 双臂，并在 TwoRoom Portal Exit 完成 `4,096` 步 joint/native 双臂。
四者均保持原 LeWM 参数量与推理结构，比较臂使用相同初始化、数据、sampler、冻结范围、
optimizer/scheduler 与预算，只把 joint auxiliary weight 从 `0.09` 置为 `0`。

完整预算并没有推翻 Motion 的早期机制方向，但显著改变了幅值和校准判断：

| Motion step | joint−native future | history | switch | worst |
|---:|---:|---:|---:|---:|
| 1,024 | +0.070 | +0.207 | +0.816 | +0.102 |
| 2,048 | +0.121 | +0.221 | +0.816 | +0.172 |
| 4,096 | +0.098 | +0.193 | +0.719 | +0.168 |
| 8,192 | +0.104 | +0.170 | +0.645 | +0.145 |

`8,192` 步 exact response evaluator 进一步给出 joint/native gain=`0.250/-0.072`、
NRE=`1.120/1.372`。因此 1,024 步已经正确识别“joint 会打开历史条件响应”，但不能外推
最终 switch、gain 或 NRE；尤其 NRE 仍高于无响应基线 `1.0`，正式 mechanism screen 仍未通过。

Portal 给出更强的反例：过短终点甚至会错排候选。下面是同一完整训练轨迹中 joint−native 的
差值：

| Portal step | future | history | worst |
|---:|---:|---:|---:|
| 128 | -0.006 | -0.061 | +0.008 |
| 512 | -0.016 | -0.051 | -0.047 |
| 1,024 | +0.027 | -0.062 | -0.008 |
| 2,048 | +0.055 | -0.002 | +0.141 |
| 4,096 | +0.135 | +0.006 | +0.207 |

冻结 256-pair evaluator 的最终 joint/native 为 future=`0.752/0.584`、worst=`0.746/0.512`、
gain=`0.604/0.366`、NRE=`0.284/0.471`。按 pair 重采样，future 差值 `+0.168` 的 95% 区间为
`[+0.133,+0.203]`，worst 差值 `+0.234` 的区间为 `[+0.148,+0.293]`；history 的
`+0.020` 区间跨零。也就是说，joint 在第二个环境域的完整预算收益主要是正确 future、最差组
和响应校准，而不是原本已较高的 history readout。两臂仍未通过 Portal 的全部高阈值，因此这是
跨域强正信号，不是“通用方法已完成”。

同一 Portal checkpoint 的 matched original-TwoRoom CEM 先只执行 seed42 的 50 条 frozen catalog：
joint/native=`47/46`，列联为 both/joint-only/native-only/neither=`46/1/0/3`，平均终距
`17.54/18.42`。paired bootstrap 点差为 `+2pp [0,+6]pp`，但只有一个 discordant query，不能
据此声称规划提升；它只说明当前样本没有出现 Motion 式 planning 损伤。由于 Portal 直接能力门
尚未全部通过，本轮不提前扩成 CEM300。Motion 的完整 `8,192`-step matched CEM300 则作为真正
finalist 的规划判据执行完成：joint/native=`202/219`，点差 `-5.67pp`，paired 95% interval
`[-11.67,+0.33]pp`，列联 both/joint-only/native-only/neither=`166/36/53/45`，exact McNemar
`p=0.0893`。零差异尚未被排除，但相对 `-5pp` margin 的单侧 95% 下界为 `-10.67pp`，因此仍是
**未证明非劣**，不能写成 Pareto。与短预算同类比较中约 `-11pp` 的点差相比，完整训练把 planning
缺口明显缩小；与此同时 NRE 却变差到 `1.120`。这进一步证明直接 ICL 校准与 CEM retention
不能互相代理，二者都必须在完整 checkpoint 上实测。

完整端点见 [Motion joint response](artifacts/pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_step8192_v1/s14321_baseline_plus8192_templates2048_v1/development_response_analysis_v1.json)、
[Motion native response](artifacts/pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_native_control_step8192_v1/s14321_baseline_plus8192_templates2048_v1/development_response_analysis_v1.json)、
[Portal joint ICL](artifacts/tworoom_portal_exit_visible_joint_full_budget_v1/joint_s15321_step4096_v1/development_score_current_runtime_v1.json)、
[Portal native ICL](artifacts/tworoom_portal_exit_visible_joint_full_budget_v1/native_s15321_step4096_v1/development_score_current_runtime_v1.json) 与
[Portal paired CEM50 summary](artifacts/tworoom_portal_exit_visible_joint_full_budget_v1/original_cem300_current_runtime_v1/paired_seed42_summary_v1.json)，以及
[Motion full-budget CEM300 paired analysis](artifacts/pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_full8192_cem300_v1/paired_analysis_v1.json)。

由此固定后续算力规则：短程只用于检查数值稳定、梯度路由以及是否出现非平凡条件响应，不再用
统一的 256/512-step accuracy 硬门否决候选；真正 finalist 必须在一次连续训练中保存约
`25%/50%/100%` 端点，并与 matched native 一起跑满。方法发现期每个代表任务先跑一个完整
seed；只有同一最终配方同时满足直接 ICL 与同 checkpoint CEM 后，才补多 seed 和更宽任务矩阵。
这避免把全部历史变体昂贵地重训，也避免用短程或 source-only comparator 误杀真实正方向。

### 5.39 Motion 全 query 覆盖：连续校准闭合，规划保持仍独立未解

§5.38 的 8,192-step Motion 终点出现了一个反常组合：训练继续改善，Development NRE 却升至
`1.120`。本节先做 checkpoint-only 分解，再执行一个不改变模型或 loss 的数据覆盖单因素实验。

旧 Cartesian 训练集只有 2,048 个 query 模板，8,192 个 optimizer step 会反复看到同一
query。逐终点结果显示：2,048 步时 training/Development NRE=`0.756/0.901`，差距只有
`+0.144`；8,192 步时变成 `0.207/1.120`，差距扩大到 `+0.913`。旧 8,192 checkpoint 在
training overlay 上的 gain/alignment=`0.766/0.891`，而 Development 只有
`0.250/0.317`。模块互换又表明完整 effect 随 Predictor trunk 转移，`pred_proj` 近似无关；
六层 progressive swap 才逐步把 gain 从负值提高到 `0.253`，没有一个单层可以单独解释。
这些结果支持跨 query response memorization，而不是训练不足或一个输出层故障。

新的 full-release 比较从相同 published PushT checkpoint 出发，固定 seed `14321`、absolute
LeWM、8,192 步、每步 `64` 条原始数据 + `64` 条 Motion 数据、原生 MSE+`0.09` SIGReg、
optimizer/scheduler 和 module freeze；joint/no-aux 之间只差 joint weight `0.09/0`。相对旧
Cartesian 版本，唯一方法相关变化是删除第二 action branch，并让训练直接覆盖冻结 release 的
全部 8,192 个 matched query；没有新增参数、模块、teacher 或推理计算。

| full-release 8,192 | future | history | switch | worst | gain | alignment | NRE |
|---|---:|---:|---:|---:|---:|---:|---:|
| matched no-aux | 0.494 | 0.500 | 0.441 | 0.230 | 0.0065 | 0.017 | 1.130 |
| **center-free joint** | **0.660** | **0.668** | **0.969** | **0.570** | **0.370** | **0.520** | **0.767** |

joint 的 target-energy-weighted NRE paired-bootstrap 95% interval 为 `[0.724,0.814]`，完整低于
零 response 参考 `1.0`。因此旧终点的 `NRE>1` 不是 joint relation 的能力上限；覆盖足够多的
query 后，原 LeWM Predictor 可以学到跨 query 的连续条件响应。

规划结果没有随 NRE 自动闭合。三套 evaluator catalog 各 100 条时，joint/native 分别为
`70/79`、`66/75`、`76/82`；等 seed 平均为 `70.67%/78.67%`，差值 `-8.0pp`，95%
paired interval `[-12.67,-3.67]pp`。随后按既有完整结果完全相同的 seed42×300 协议得到
`213/232`，差值 `-6.33pp`，95% interval `[-11.00,-1.67]pp`，paired discordance 为
joint-only/native-only=`18/37`，exact McNemar `p=0.0145`。前者检验跨 evaluator seed 的方向，
后者提供可与旧 CEM300 直接比较的估计；两者都不作为单阈值式晋级门。

这一结果把 Motion 根因拆成两个已经可分离的部分：

1. 有限 conditional-query 覆盖导致 Predictor 记忆，解释了旧 NRE 退化；full-release 已修复；
2. joint gradient 对 planner-consumed ordinary function 的改写仍存在，解释 CEM 缺口；覆盖增加
   本身不能修复。

当前 joint 终点的 scalar loss 中，native prediction 为 `0.0192`，加权 joint auxiliary 为
`0.0712`，后者约占 total 的 `79%`。这提示持续 relation exposure 可能主导优化，但 scalar
比例不等于参数梯度比例，不能直接据此调权重。下一项最小实验固定 full-release 数据和全部方法
变量，只重建 4,096-step joint/no-aux checkpoint：已有同一训练轨迹在该时点的
future/history/switch/worst=`0.643/0.668/0.973/0.508`。若其 NRE 仍明显改善而 CEM 缺口缩小，
主要问题是过长的 auxiliary exposure；若缺口不变，再以真实 original-vs-joint gradient conflict
为依据检验零参数的 native-safe gradient projection，而不是做任意 weight sweep。

本节同时更新统计口径：NRE 的 `1.0` 是可解释参考点，不是硬裁决线；CEM 报告 paired effect
与 50%/80%/95% intervals；suite 层按任务等权汇总 effect distribution，并将 assignment、
continuous calibration 和 planning 分轴呈现。完整论文式叙述见
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)。机器可读结果见
[full-release joint Development](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_step8192_v1/s14321_step8192_v1/development_response_analysis_v1.json)、
[matched no-aux Development](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_native_control_step8192_v1/s14321_step8192_v1/development_response_analysis_v1.json)、
[multi-catalog CEM](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_cem_seeds42_43_44_n100_runtimefix_v2/continuous_paired_effect_v1.json) 与
[seed42×300 CEM](artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_cem300_seed42_current_runtime_v1/continuous_paired_effect_v1.json)。

### 5.40 规划估计量纠正与历史候选重评

此前的 standard PushT CEM 使用没有 ContextWorld 隐藏动力学的原始环境。它回答的是
checkpoint 是否保留普通规划函数，不回答正确历史能否改善隐藏动力学下的动作选择。若 candidate
还使用了 `50/50` original/ContextWorld 混合数据，直接与 published source 比较又会把数据 recipe、
额外训练曝光和方法效应混在一起。后续统一报告三臂：published source、同数据同预算 native、
candidate；分别解释 `native-source`、`candidate-native` 和 `candidate-source`。隐藏规划还必须
在同 checkpoint、同物理条件下比较 correct 与 swapped history。

Contact Friction 的首个 256-query 三臂评测已经完成。规划使用五个 query-action block，物理 oracle
显示 `244/256` 个 pair 的 low/high acceptable scale region 不重叠，平均最优 scale gap=`0.358`。
因此这不是“一个默认动作同时适合两种动力学”的弱评测。结果为：

| arm | correct-history physical distance↓ | scale regret↓ | swapped−correct distance↑ | swapped−correct regret↑ |
|---|---:|---:|---:|---:|
| source | `87.08` | `0.3386` | `-0.37` | `-0.0023` |
| matched native | `94.72` | `0.4102` | `+0.28` | `+0.0001` |
| **COJA** | **`89.79`** | **`0.3783`** | **`+1.70 [0.80,2.68]`** | **`+0.0121 [0.0059,0.0192]`** |

COJA 相对 matched native 的方法效应为 physical distance 改善
`4.94 px [2.31,7.58]`、scale regret 改善 `0.0319 [0.0121,0.0519]`；其正确历史收益又比
native 多 `1.41 px [0.33,2.54]` 和 `0.0120 [0.0053,0.0198]`。这第一次证明 direct ICL
不仅改善 benchmark prediction，还转化成真实模拟器中的 history-conditioned planning benefit。
source 绝对值仍最好，说明 COJA 只追回了 mixed-data native 损失的一部分；当前一维
`64×6` CEM 也是机制 screen，不包装成完整部署成绩。

这一纠正要求重评旧候选，但不是全部重训。只恢复“direct ICL 已明显改善或仅因波动硬门近失，
且旧否决可能依赖错误 comparator”的 checkpoint：

- **COJA/早期 exact-overlap PCJA**：恢复为主候选；Contact 已得到 hidden-planning 正效应；
- **DynamicsResponseSIGReg**：恢复为 Action Strength 强正基线。它已有三 seed
  future=`0.966`、switch=`0.996`。同场 hidden planning 相对 matched native 将 mode
  classification 提高 `15.23pp [12.30,18.16]`，regret 降低 `0.0706 [0.0544,0.0872]`；但同一
  方法在 Motion 4,096-step 只有 future/history=`0.412/0.418`，所以不是通用最终方法；
- **target-JTCov**：错误 comparator 没有把它救回。同场 Action Strength hidden planning 相对
  matched native 的 classification 低 `8.79pp [5.86,11.91]`、regret 高
  `0.0725 [0.0530,0.0921]`、执行距离高 `2.66px [1.91,3.43]`；
- **terminal ConditionalSIGReg**：旧 1,024-step checkpoint 的 classification=`0.801`，相对
  4,096-step native 的差为 `-2.15pp [-5.47,1.17]`，regret 与距离也更差。由于训练 release 和
  预算不匹配，这只是不晋级的 checkpoint screen，不是严格否定方法族；
- **function/action-function anchor**：保留为较复杂的性能上界，不作为最终简洁方法；
- **VISReg、stop-gradient+SIGReg、Motion PCJR/CCRM、continuous transition basis**：直接
  conditional response 本身失败，规划 comparator 纠正不能救回。

历史 DynamicsResponseSIGReg 还提供了一个关键理论对照：它用 pair 构造 response contrast，
但随后只匹配跨 query 的 response population，没有逐 query 强制 prediction response 对齐自己的
target response。它能解决方向较一致的 Action Strength，却在 Motion 失败。COJA 的关键增量因此
不是“更强的边缘正则”，而是 matched `(Q,A)` 下逐实例的 conditional correspondence。

完整三臂结果见
[Contact hidden-dynamics CEM](artifacts/pusht_contact_friction_hidden_cem_h5_three_arm_development256_cpu_v1/summary.json)，
Action Strength 历史候选同场复评见
[historical candidate reevaluation](artifacts/historical_candidate_reevaluation_v1/summary.json)，
历史 comparator 判定见
[candidate reclassification](artifacts/conditional_joint_comparator_validity_v2/summary.json)，论文式叙述见
[TECHNICAL_REPORT.md](TECHNICAL_REPORT.md)。

### 5.41 单步可辨识仍会在 rollout 中丢失：RC-COJA

Motion 的四动作 planner-curve COJA checkpoint 已经具备明显的一步条件响应：
future/history/switch=`0.539/0.605/0.938`。但它在两步 hidden-planning 中的正确历史物理误差仍为
`103.17 px`，swapped−correct=`-0.54 px`。也就是说，第一步 Predictor 会随历史变化，不代表
规划器把预测重新送回模型后仍保持正确的 hidden dynamics。这个结果把此前“ICL 已学到但 CEM
为何不稳定”的问题收缩成新的具体断点：**one-step conditional identifiability 不蕴含
rollout conditional identifiability**。

为直接验证该断点，新 builder 对原 2,048 个 rollout templates 复用完全相同的 history、第一
action block 和模拟器状态，再执行一个固定零动作 block，得到真实 `x4` target。数据不按 hidden
label、future outcome、contact 或模型输出筛选；36 个两种 damping 恰好得到相同 `x4` 的 group
只退出 normalized relation，仍参加 native MSE。训练从同一个 curve4096 checkpoint 出发，固定
原始/ContextWorld=`64/64`、optimizer、batch stream 和 1,024 fresh steps。

RC-COJA 不增加 loss 家族。对 hidden rows 只将既有 native MSE 总权重 `0.5` 在一步和真实
自回归两步之间重分配；一步 COJA 总权重仍为 `0.09`。第二步使用

\[
\hat z_3=F(H,a_1),\qquad
\hat z_4=F(\operatorname{shift}(H,\hat z_3),a_2),
\]

并让 `x4` 误差穿过 `\hat z_3` 反传。保存的仍是原 LeWM state dict，新增参数、模块、hidden-label
输入、teacher 和 inference compute 均为零。

因子拆分结果如下。不同 horizon 的 physics oracle 保留 `34/156/134` 个 pair，只能在列内作
matched 比较，不能把绝对距离跨 horizon 汇总。

| continuation | 1-step distance↓ | 2-step distance↓ | 3-step distance↓ |
|---|---:|---:|---:|
| one-step placebo | `22.86` | `102.37` | `111.45` |
| rollout2 relation only | `23.18` | `86.17` | — |
| rollout2 native MSE，`ρ=0.25` | `25.90` | `58.54` | `80.53` |
| rollout2 native MSE，`ρ=0.50` | `29.56` | `44.40` | `71.21` |
| rollout2 MSE + relation，`0.5/0.5` | `32.67` | `43.18` | `69.69` |

`ρ=0.25` 是消融后唯一执行的折中候选，不是事后权重 sweep。它相对 placebo 的 1/2/3-step
改善为 `-3.04 [-4.21,-1.81]`、`+43.83 [39.57,48.01]` 和
`+30.92 [25.32,36.55]` px。第三步未进入训练，仍有大幅改善，说明候选学到的是可延续的
self-rollout 函数，而非只拟合第二步终点。correct-vs-swapped history 在两步/三步为
`+1.88/+0.70 px`，placebo 为 `+0.63/-0.89 px`。

主要活性成分是第二步 native MSE，不是第二步 paired relation：relation-only 对两步只有约
`16.20 px` 改善且 history benefit 为负；MSE-only `ρ=0.50` 改善约 `57.97 px` 并恢复正的
history benefit。故当前最小方法是“一步 COJA 建立条件对应 + 短自回归 native MSE 传递该对应”，
无需为每个 rollout horizon 增加新的 auxiliary。

标准 PushT retention 使用同一 100 episode 的 matched placebo，对 `ρ=0.25` 为 `60/100` 对
`57/100`，candidate−placebo=`+3pp [-6,+12]pp`；`ρ=0.50` 为 `53/100`。这不证明标准规划提升，
但也不支持此前“rollout 方案会损害 CEM”的判断。真正尚未闭合的是一步小幅退化和跨任务复现，
而不是参数复杂度或原任务 CEM。下一次训练只把固定 `ρ=0.25` 原则迁移到一个已有 hidden-planning
oracle 的连续任务，不再扫描 Motion 权重或增加新 loss。

紧凑机器结果见
[rollout-consistency summary](artifacts/pusht_motion_damping_rollout_consistency_mve_v1/summary.json)，
实现见
[rollout2 target builder](scripts/build_pusht_motion_damping_planner_curve_rollout2_targets_v1.py) 与
[RC-COJA continuation](scripts/run_pusht_motion_damping_planner_curve_rollout_consistent_continuation_v1.py)。

### 5.42 Contact 固定配方迁移：action support 恢复 Pareto

本实验只回答跨任务性，不在 Contact 上重新搜索 `ρ`。control 与 candidate 都从 SHA
`257638a7b546…` 的 4,096-step 一步 COJA checkpoint 开始，使用相同 seed `13313`、相同
`64+64` 数据、optimizer、batch streams 和 1,024 个 fresh steps。control 的 hidden 一步/两步
native 权重为 `0.5/0`；candidate 固定为 `0.375/0.125`。一步 COJA 保持 `0.09`，模型参数、模块、
保存格式和推理调用不变。

Contact release 只有一个 query-action block，因此用两个 RC arm 做单因素对照：repeated arm
再次执行该 query；empirical arm 从原始 PushT 训练总体的每个 episode 确定性抽取连续五步动作。
8,192 个模板全部保留，low/high friction 使用完全相同的第二动作；没有按未来、contact、模型
输出或标签筛选。

直接 Development 几乎重合：

| arm | future | history | switch | worst | NRE |
|---|---:|---:|---:|---:|---:|
| one-step continuation control | `0.781` | `0.854` | `1.000` | `0.758` | `0.617` |
| repeated-action RC | `0.779` | `0.846` | `1.000` | `0.758` | `0.619` |
| empirical-action RC | `0.783` | `0.850` | `1.000` | `0.762` | `0.618` |

隐藏动力学规划使用 256 个 matched query pairs，并对同一 checkpoint 比较 correct/swapped
history。h1 的 oracle acceptable regions 为 `0/256` 可分，只作弱诊断；h2 为 `256/256`，未训练
h5 为 `244/256`，后两者才是关键：

| horizon | control 误差↓ | repeated RC↓ | empirical RC↓ | empirical 相对 control 改善 | empirical 历史收益 difference-in-differences |
|---|---:|---:|---:|---:|---:|
| h2 | `29.31` | `24.84` | **`24.92`** | **`4.39 [2.93,5.90]`** | **`1.26 [0.67,1.87]`** |
| h5 | `93.41` | `91.87` | **`89.04`** | **`4.37 [1.87,6.84]`** | **`2.58 [0.86,4.34]`** |

h2 的 empirical 与 repeated 误差不可区分（差 `-0.08 [-1.26,1.04] px`）；h5 empirical 相对
repeated 点估计再改善 `2.82 px`，区间 `[-0.79,6.49]`。empirical 的 history benefit 小于
repeated，但相对 control 在 h2/h5 都明确为正，因此它没有靠忽略历史换取更好的绝对误差。

标准无隐藏摩擦 PushT 的 300 个共同 queries 为 control/repeated/empirical=`212/197/216`。
repeated−control=`-5.00 [-9.33,-0.67]pp`，empirical−control=`+1.33 [-3.33,+6.00]pp`，
empirical−repeated=`+6.33 [1.67,11.00]pp`。三臂共享起点、数据、`ρ`、预算和 episode，只有第二
action support 不同，故这不是相关性解释：重复动作的窄 support 是 retention 损伤的因果来源。

当前 empirical-action RC 在一个 training seed 上同时保持 direct ICL、改善 h2/h5 hidden
planning，并未检测到 standard retention 损伤，形成 Contact Pareto 正例。随后冻结此配方并执行
从标准初始化开始的完整 no-aux/COJA/RC 三臂训练；没有重新开放边缘正则、margin 或新模块搜索。

该完整训练现已完成。RC 从公开初始化直接单阶段训练 4,096 steps，不依赖先取得一步 COJA
checkpoint。direct Development 的一步 COJA/RC 为：future=`0.771/0.775`、
history=`0.850/0.844`、switch=`1.000/1.000`、worst=`0.734/0.738`、NRE=`0.617/0.619`，
即一步能力基本保持。

| horizon | 公开原始数据参考↓ | 同 mixture 一步 COJA↓ | 同 mixture RC↓ | RC 相对一步 COJA改善 | RC 历史收益 DID |
|---|---:|---:|---:|---:|---:|
| h2 | `25.99` | `28.45` | **`25.29`** | **`3.16 [1.58,4.79] px`** | **`1.89 [1.28,2.52] px`** |
| h5（未训练） | `94.73` | `89.82` | **`86.72`** | **`3.09 [0.65,5.46] px`** | **`3.61 [1.87,5.40] px`** |

标准 PushT 的相同 300 queries 为公开参考/一步 COJA/RC=`237/206/207`。RC−一步 COJA 为
`+0.33 [-4.33,+5.00]pp`，discordant wins/losses=`25/24`，没有检测到 RC 方法代价；
一步 COJA−公开参考已为 `-10.33 [-15.33,-5.67]pp`，RC−公开参考为
`-10.00 [-15.00,-5.00]pp`。因此约 `10pp` 是共享 mixture/适配路径的独立 estimand，不能再次
误归因给 rollout 方法。Motion action-support 复核与完整单阶段训练已在 §5.43 完成；发现期
改动据此停止，后续只补 training seeds 和 publication-level 跨任务统计。

机器汇总见
[Contact rollout-consistency transfer](artifacts/pusht_contact_friction_rollout_consistency_transfer_v1/summary.json)，
训练实现见
[Contact RC continuation](scripts/run_pusht_contact_friction_rollout_consistent_continuation_v1.py)，
真实 target 构建见
[Contact rollout2 builder](scripts/build_pusht_contact_friction_rollout2_targets_v1.py) 与
[empirical-action builder](scripts/build_pusht_contact_friction_rollout2_empirical_action_targets_v1.py)；
完整单阶段证据见
[h2 hidden planning](artifacts/pusht_contact_friction_empirical_action_rc_full4096_hidden_cem_h2_dev256_v1/summary.json)、
[h5 hidden planning](artifacts/pusht_contact_friction_empirical_action_rc_full4096_hidden_cem_h5_dev256_v1/summary.json) 与
[RC standard CEM300](artifacts/pusht_contact_friction_empirical_action_rc_full4096_standard_cem300_v1/aggregate.json)；
两 seed 的紧凑汇总见
[Contact replication summary](artifacts/pusht_contact_friction_rc_coja_full4096_replication_v1/replication_summary_v1.json)，
独立 seed 实现见
[frozen replication runner](scripts/run_pusht_contact_friction_rc_coja_full4096_replication_v1.py)。

### 5.43 Motion action-support 资格检验与单阶段闭环

Contact 的 empirical-action 正例只证明重复 action 不是普适 continuation。为检验“无条件 action
diversity 是否足够”，Motion 保持 curve4096 起点、`ρ=0.25`、loss、batch stream、seed 和
1,024-step budget 不变，只把 zero-hold 第二 action 换成从普通 PushT replay 每个 episode
确定性抽取的五步 block。该数据不按 future、contact、模型输出或 hidden condition 筛选。

结果不是新候选失败，而是对 action support 理论的限定：empirical arm 相对一步 placebo 在
h2/h3 仍改善 `21.17 [17.80,24.64]` 与 `13.28 [9.08,17.99] px`，证明第二步 native MSE 仍
有效；但它相对 zero hold 分别差 `23.36 [20.33,26.40]` 与
`16.70 [12.28,21.11] px`。更关键的是，empirical 的 swapped−correct history benefit 为
`-0.64/-1.95 px`，而 zero hold 为 `+1.93/+0.63 px`。所以 rollout action 必须与 query/
deployment support 相关；普通 replay marginal 的多样性不能替代这种相关性。

随后从公开 PushT 初始化直接执行一次 4,096-step zero-hold RC-COJA，删除此前额外的一步 COJA
warm start。matched no-aux、一步 COJA 和 RC 共享初始化、`64+64` 数据、optimizer、预算及
评测 query；RC 保存的仍是原 LeWM state dict。

| horizon | matched no-aux↓ | 一步 COJA↓ | 单阶段 RC↓ | RC−COJA 改善 | correct-history benefit DID |
|---|---:|---:|---:|---:|---:|
| h1 | `23.28` | **`23.23`** | `26.24` | `-3.01 [-4.13,-1.86]` | `-0.24 [-0.46,-0.03]` |
| h2（训练） | `100.20` | `103.17` | **`45.78`** | **`57.39 [52.19,62.42]`** | **`4.12 [2.90,5.34]`** |
| h3（未训练） | `106.23` | `108.34` | **`69.26`** | **`39.08 [33.90,44.21]`** | **`2.28 [0.28,4.01]`** |

h1/h2/h3 分别有 `34/156/134` 个物理 oracle 可辨识 pair，只在 horizon 内作 paired comparison。
RC direct future/history/switch/worst=`0.553/0.617/0.938/0.258`。h2/h3 同时改善绝对误差和
history intervention，不能由“模型忽略历史”解释；h1 的负效应则是需要在多 seed 层级统计中
继续报告的真实 horizon tradeoff。

标准无隐藏 damping PushT 使用同一 300-query catalog：

| arm | success | paired comparison |
|---|---:|---:|
| matched no-aux | `203/300` | — |
| one-step COJA | `188/300` | `-5.00 [-11.00,+0.67]pp` vs no-aux |
| RC-COJA | `194/300` | `+2.00 [-2.67,+6.67]pp` vs COJA；`-3.00 [-8.67,+2.67]pp` vs no-aux |

这 300 次结果不支持把 2–3pp 点差用作硬门。它排除了“RC 自身已被证明导致大幅标准规划损伤”，
但没有证明严格非劣或提升。与 h2/h3 数十像素且区间远离零的 hidden-planning 主效应相比，标准
retention 是独立且仍带不确定性的副作用估计。

冻结配方随后只改变 training seed 为 `14322`，one-step COJA 与 RC 同时从公开初始化重训
4,096 steps。精确 31 点 action-grid 复现为：

| horizon | seed14321 RC−COJA | seed14322 RC−COJA | seed14322 history-benefit DID |
|---|---:|---:|---:|
| h1 | `-3.01 [-4.13,-1.86]` | `-3.30 [-4.53,-2.14]` | `-0.06 [-0.28,+0.21]` |
| h2 | `+57.39 [52.19,62.42]` | `+57.60 [52.55,62.55]` | `+4.24 [3.16,5.37]` |
| h3 | `+39.08 [33.90,44.21]` | `+41.56 [36.02,47.02]` | `+2.91 [0.90,4.95]` |

seed14322 的 direct RC/COJA future=`0.555/0.543`、history=`0.621/0.592`、
switch=`0.930/0.926`，没有以破坏一步 ICL 换取多步收益。标准 CEM300 为 RC/COJA=`192/196`，
即 `-1.33 [-6.67,+4.00]pp`；与 seed14321 的 `+2.00pp` 合并时，先重采样 training seed、再在
seed 内重采样 paired query，得到层级均值 `+0.33pp`，95% 区间 `[-4.00,+4.50]pp`。这使“不能
用单个 CEM 点数硬杀候选”从统计原则变成直接证据。

这一闭环排除了两阶段 schedule、额外 horizon relation 和无条件 action diversity 三个不必要
组成。发现期方法固定为：**一步 COJA 建立 matched history–action–future 对应，再以部署相关
action support 上的短自回归原生 MSE保持该对应**。它没有新增参数、encoder、adapter、head、
loss family 或推理计算；仍需显式 conditional-overlap pairs 与短 trajectory targets。

机器结果见
[Motion rollout-consistency summary](artifacts/pusht_motion_damping_rollout_consistency_mve_v1/summary.json)、
[h1](artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected34_blocks1_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json)、
[h2](artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected156_blocks2_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json)、
[h3](artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected134_blocks3_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json) 与
[standard CEM300 paired analysis](artifacts/pusht_motion_damping_full4096_standard_cem300_paired_v1/paired_analysis_v1.json)，
两 seed 汇总见
[replication summary](artifacts/pusht_motion_damping_rc_coja_full4096_replication_v1/replication_summary_v1.json)；实现见
[empirical target builder](scripts/build_pusht_motion_damping_planner_curve_rollout2_empirical_action_targets_v1.py)、
[empirical continuation](scripts/run_pusht_motion_damping_planner_curve_rollout_consistent_empirical_action_continuation_v1.py) 与
[single-stage RC training](scripts/run_pusht_motion_damping_rollout_consistent_zero_hold_full4096_v1.py)，独立 seed 使用
[frozen replication runner](scripts/run_pusht_motion_damping_rc_coja_full4096_replication_v1.py)。

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

### 6.1 一类反复出现的合同缺陷：位置代理冒充内容不变量

同一形状的缺陷至今出现了**五次**（前三次见下表，第四次是"输出目录必须为空"，
第五次在冻结的 builder 里，见 §6.1.1），值得单独记下来，因为它既产生假警报，
又会**掩盖真正的检查**。
模式是：合同想保证的是某个**内容不变量**（评分代码没变、训练确实跑满、源闭包逐字节相同），
但断言写成了一个更容易取到的**位置或身份代理**（一个 git commit、一个模块路径集合、一个
step 计数器）。代理与不变量在写下的那一刻等价，之后就开始漂移。

| 实例 | 断言的代理 | 想保证的不变量 | 被什么无关变化打破 |
|---|---|---|---|
| ContextWorld import 闭包 | 模块路径集合精确相等 | 评分时装入的实现不变 | 一次 lazy-import 重构，行为未变 |
| loss trace 终点行 | 最后一行 == `trainer.global_step` | 训练确实跑到授权步数 | 冻结 recipe 自身的有意提前停止，该断言不可满足 |
| source-rebind HEAD 钉 | `HEAD == 66761639…` | 45 文件评分源闭包不变 | 12 个无关提交；起草期间又来了 1 个（宣告许可证） |

**为什么这不只是"严格"。** 位置代理在审计器里通常排在内容检查**前面**。predictor-only 的 v2
审计器把 HEAD 比较放在 `:280`，45 文件逐字节比较放在 `:283-308`——HEAD 一动，审计停在 280，
那个真正承载科学含义的比较**永远不会执行**。于是最需要保证的时候，保证反而消失了。反过来，
一个被改写的评分 kernel 只要恰好坐在被钉的 commit 上，就能通过先跑的那道门。代理不是更严，
是**又松又吵**。

三次的处理方式相同，也是不变的规矩：**用满足内容不变量来消除告警，绝不靠放宽断言。**
import 闭包那次是显式恢复冻结的 import 状态，不是把集合相等改成子集；loss trace 那次是让
冻结的 post-fit recovery runner 依据落盘证据补写 phase report，不是插补缺失行，并把它记为
`infrastructure_NOGO_not_method_failure`；source rebind 这次是把判定权交给 45 文件内容闭包，
四个确有差异的文件逐一书面裁定并**按其被裁定时的观测哈希钉死**（再变一次即作废，需重新裁定），
HEAD 降级为**记录项**——但 HEAD 在审计**过程中**必须稳定（`head_before == head_after`），因为
读到一半还在动的树从来就不是一个状态。结果是比 v2 更紧：内容比较现在总会运行。

这次还留下一个当场的自证。v3 起草期间 ContextWorld HEAD 又前进了一个提交
（`c0a542f4`，"declare the source and generated-data licenses"），它触碰的 3 个文件与 45 文件闭包
交集为**空**。在 v2 的门下，这个纯许可证提交会第二次拒绝审计，并第二次阻止内容比较运行；
在 v3 下比较照常执行：41 个文件逐字节相同，4 个裁定文件恰好落在各自被钉的哈希上，未裁定差异
为 0。同一份 v3 审计器在实现里也踩到了同一个模式的第四个实例——它原本要求输出目录**为空**，
而"目录为空"同样是"没有预置回执"的位置代理；改为把允许预先存在的文件**按哈希钉住**。

写新合同时的判据很简单：先问"我真正要保证的量是什么"，再问"我写下的断言在什么无关变化下
会与它分离"。如果答案是"上游随便提交一次就会"，那写的就是代理。仍然适用的边界是 §1.3——
以上都属于审计层，诊断量不替代冻结终点。

相关记录：v3 预注册 `configs/…_source_rebind_addendum_v3.yaml`、v3 回执
`artifacts/…_source_rebind_addendum_v3/source_rebind_receipt.json`
（`content_sha256 = eefbc58f4fb06fe899ac37122bcb98d785fc9c3ec517fdfe4b44f8d13d96439f`）、
自我更正 `artifacts/…_multi_seed_v1/receipts/evaluation_blocked_v1_correction_v1.json`。
v1/v2 审计器与其回执均未改动，v2 回执仍逐字节等于 `7d3fc96f…`。

#### 6.1.1 第五个实例：同一缺陷在冻结的 *builder* 里（2026-08-20）

前四个实例都在审计器里。第五个在**建 release 的代码**里，而且是同一形状的最干净版本：

```python
require(_git_head(contextworld_root) == str(config["contextworld_commit"]),
        "ContextWorld commit changed")
```

这行排在**同一个函数内**四个逐文件 sha256 循环（`source_hashes`、`renderer_dependency_hashes`、
`scoring_runtime_dependency_hashes`、`overlap_sources`，合计 45 个文件）**之前**。HEAD 一动，
四个循环全部够不到。**建 release 时最需要保证源码没变，而那一刻保证恰好失效。**

实测状态：ContextWorld HEAD 已推进到 `6a82f670`，距 v1 记录的 `0fb72e1d` 有 16 个提交；
v2 source-rebind 钉的 `66761639` **不是 HEAD 的祖先**——它在一条废弃的历史上，
所以"把 HEAD 切回去"不是恢复状态，是丢弃 16 个提交。加上 v3 已经确认的那件事
（闭包里有一个文件当时是**未跟踪**的，任何 commit 都从未等于被审计的那个状态），
位置代理在这里从来就没有描述过它声称描述的东西。

处理与前四次相同：`…_v2_build_source_gate_addendum_v1.yaml` 把 commit 降为记录项，
判定权交给内容闭包；四个已裁定文件**继承** v3 的裁定并按其哈希钉死，不重新裁定；
保留"读取期间树必须稳定"这一条（这是真的内容属性）。
29 个反例全部拦下（22 个配置层 + 7 个运行时层）。
回执 `artifacts/…_v2_build_source_gate_addendum_v1/build_source_gate_receipt.json`，
`content_sha256 = f485cdc32c2a4035ea48ed4ed80801f120238dd2206dd63187095b8fb43da5e6`。

**一个不是论证而是观测的旁证。** 做这件事期间，内容门**真的报警了**：
`contextworld/benchmarks/adapters.py` 的哈希与裁定值不符，审计拒绝；几分钟后重跑又通过。
查清原因是别人在工作区加 PreJEPA adapter 家族的改动，未提交，随后回退——**全程没有任何 commit**。
HEAD 钉对这类改动完全无话可说，内容门抓到了并拒绝执行。

> **自我更正（保留而非删除）：** 第一次比对用的是正则，报告说 `StableWorldModelLeWMAdapter` 变了。
> 那是正则的假象——模式依赖尾部 `class` 边界，而改动后的文件没有。用 AST 重做：
> 21 个顶层定义对 21 个，共同定义 0 处改动，模块级代码相同，评分路径上的
> `LatentWorldModelAdapter` / `StableWorldModelLeWMAdapter` / `StableWorldModelLeWMHistory7Adapter`
> 三个类逐字节相同。错误的中间结论已写入回执，没有悄悄丢掉。

#### 6.1.2 一条新规矩：预注册必须**执行**冻结校验器（2026-08-20）

第五个实例是在准备 v2 评测时**跑出来的**，不是读出来的。同一轮还跑出了第二个阻断：
v2 预注册照着 v1 写，看起来完备，实际调用冻结校验器时缺三样东西——
`training_contract`、`runtime_rebind`、`exclusion_contract.binding_chain`。
缺的都是**继承链上游要求的**，不在 v1 预注册的显眼位置，肉眼没看出来。

由此立一条规矩（AUDIT_CONTRACT §3.5）：**预注册冻结前必须实际调用它将来要通过的冻结校验函数，
把报错贴进回执。** 几十秒的执行，省一轮返工。

附带的推论值得记：**冻结的校验器不一定适用于它的后代。** v1 的 `_validate_config_binding`
钉死了 v1 自己的父辈和自己的 9-release 列表，因此**结构上无法**校验一个以 v1 为父辈的 release。
这不是 bug，是它被正确地钉死了。后代写自己的等价物，**不去改冻结件**。
补齐方式为 append-only：`…_v2_contract_sections_addendum_v1.yaml`，
仅在内存中与预注册合并，且合并逻辑拒绝覆盖预注册已声明的任何字段——addendum 只能**增**。
回执 `content_sha256 = 0787fd2a5d90a7c9d176d7fe66910ee7e2419d662968c42b12eb90383c006e31`。
两处 addendum 之后，冻结链的 `_contract_sections` 齐全（11 节）、`_validate_runtime_rebind` 通过、
v2 版 binding chain 通过；builder preflight 通过，**仍未建任何资产、未开任何 checkpoint、未算任何分数**。

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

PCJR estimator 的最后资格审计已经结束，结论是 `stop_this_estimator`，不是再开一个
source-routing candidate。理由不是一次小样本不显著，而是预测链本身没有 population
consistency：有限 panel 的符号随宽度翻转，局部梯度方向不能可靠预测真实 optimizer trajectory，
而 routing 的 matched 单因素效应区间跨零。§5.12–§5.14 保留完整反例与数值，但不再承担候选
选择职能。

因此方法主线已从“为共享 Predictor 再造一个标量 loss”转到 §5.15–§5.16 的结构性根因，随后又
由项目的零新增结构约束收缩到 §5.18 的固定时序坐标干预。context 直接相加无效，shared-trunk
联合更新会淹没弱正向分支，source routing 只能消除负漂移；velocity-aware oracle operator 则
说明 transition state 是当前原 Predictor 缺失的具体信息。encoder/adapter/head 路线已经关闭，
不再以“参数很少”为理由重开。

该零参数实现已经完成 §5.18 的两级证伪。ActionDelay 单 seed 1,024-step 出现强 history
coupling 并通过全部冻结 Development 门，说明原生 conditional MSE 在 transition-oriented 坐标下
可以学习离散 delay，不必理论上强制 privileged pairing。Motion hard switch 与 function-preserving
homotopy 则都未学到连续 response 的方向和幅值；短暂 assignment 改善没有转化为正 gain 或合格
NRE。因此 transition basis 保留为离散 ActionDelay 正例和新的机制基线，不升级为通用候选，
不开放 Public/CEM，不用 4,096-step、schedule sweep 或更多 seed 延长其在 Motion 上的生命。

§5.19 又拆除了 action 错位与 terminal-only supervision 两个具体混淆。保留当前 absolute query
state 后，step-0 MSE 已与 native 基本连续；恢复全部 standard transition supervision 后，Motion
的 response gain 仍为负，assignment 也明显低于 chance。故固定时序重参数化这一候选类到此
停止；不会把 ActionDelay 的离散正例外推成连续 dynamics 方法，也不会用更多预算或 seed 救援。

§5.20 又完成了 multi-query sampling/transfer 的三级证伪。数据核对首先发现当前每个普通 episode
只有一个 H3 query，不能合法执行原计划的 episode-blocked multi-query；特殊 twin co-batching
随后被 matched native 对照否决。直接 cross-query transfer 的确连续改善 history/switch/NRE，
transition-space 实现也比 absolute-latent 平移更好，但 gain 仍为负，证明它们只收缩错误响应。
因此 sampler 与 synthetic latent transfer 同时关闭，不用更多预算或 seed 救援。

§5.21 随后第一次越过了连续 Motion 的六项机制门。residual-transition + paired normalized
exact-future 不增加参数或推理结构，证明现有 Predictor 能学习正确方向和幅值；no-aux CEM
对照又排除了 residual basis 与 persistence reset 对规划的直接损伤。真正剩余的是同一参数路径上
的 ICL–planning 优化冲突：永久 paired auxiliary 使 CEM 从 `17/20` 降至 `3/20`，撤掉辅助项的
mixed native consolidation 在保留六门的同时恢复到 `13/20`，ordinary-only consolidation 则
发生 conditional forgetting。因此下一问题不再是提出另一个 estimand，而是验证 paired signal
能否作为有限 bootstrap、由原生 mixed-data 目标稳定巩固且达到正式 CEM 非劣。当前固定配方尚未
闭环，故不做更多权重/cutoff/schedule 后验搜索，也不提前打开 Public、跨任务或额外 seed。

§5.22 已在单 seed discovery 层面闭合这个冲突。八格 module-swap 表明规划能力不能靠单块 endpoint
移植恢复，因而改为在 ordinary rows 上直接锚定 CEM source 的 frozen prediction function，同时在
matched hidden rows 上做 conditional bootstrap。最终 checkpoint 仍是零新增参数/模块/推理计算的
原 LeWM：Motion 六门全过，100-query paired PushT CEM 与 source 同为 `57/100`，逐 query 的
source-only 与 candidate-only 又恰好各 `10`。这证明 ICL–planning 不是容量上的必然 Pareto 冲突，
也把根因从“pair loss 有害”收窄为“无保护的共享参数更新会改写原规划函数”。不过显式 pairing、
两阶段 warm start 与训练期 frozen teacher 仍使其属于强候选而非最终最简配方；跨任务、额外 seed、
Public 与正式非劣仍保持关闭。

§5.23 完成了固定配方的单 seed Contact Friction 证伪。8,192-step 终点相对 source 在
future/history/switch/worst、gain 与 NRE 上均大幅改善，证明 conditional bootstrap 可以跨连续
动力学任务传递；但正式门仍失败四项，因而不开 CEM/Public，也不补 seed。counterdirection
co-batching 未改善结果；已有 matching、paired-fit 与 projected-geometry 终点又分别暴露过冲、
assignment 不足和 response error。这把通用方法的必要条件收缩为：必须同时解决 condition
assignment、proper response calibration 与 ordinary-function preservation。当前 function-anchor
配方只保留为 Motion Pareto 正例和 Contact 部分机制正例，不再用更多预算、权重或 sampler 延长。

§5.24 又分离了 assignment bootstrap 与最终 exact fit。canonical `0.5` margin 在精确 target 处
loss/梯度均为零，却在 history-independent prediction 处提供非零梯度；它在 2,048 和 8,192
两个 matched endpoint 上均一致改善 assignment、gain、alignment 与 NRE，排除了“所有额外
assignment pressure 都必然导致过冲”。但正式终点仍失败四门，且 exact residual 与无 barrier
control 几乎相同。因此 margin 只修复学习速度的一部分，不能解释或解决跨 query absolute future
残差；逐 pair exact identity 又显示约 `80.5%` 的 held-out residual 来自 common-center error，
把下一断点定位为 query-dependent absolute center 与 ordinary-function preservation 的冲突。该候选
不做 CEM/Public/多 seed，margin-family 到此停止。

§5.25 完成了 common-center 假设的最小因果闭环。privileged target-center oracle 在 response
完全不变时把 future/history/worst 全部提高到 `0.984`，证明该中心误差对当前 endpoint 的
assignment 缺口具有结果级充分性；但 matched 2,048-step 对照显示，center 权重 `4×` 会抹掉
response，pair-midpoint forward 与 control 等价，stop-gradient 解冻在线表示以及冻结 target copy
也都明显失败。因此不能把 oracle 结果翻译成“加大 center loss”或“加 target encoder”处方。
当前方法边界进一步收敛为：必须在同一 deployed latent 坐标中学习 joint response 与 absolute
center，同时保护 ordinary function；上述四个近邻分支停止，Motion Pareto 正例不受影响。

§5.26–§5.27 随后把“ordinary function”进一步拆成 point prediction 与 action-conditioned
geometry。center-free response 本身足以让 Motion 六门全过，但 point anchor 即使平均 drift 更小，
CEM 仍为 `52 vs 57`；固定 action intervention 将它恢复到 `56 vs 57`，同时保持六门。这是当前
最直接的证据：planner preservation 不能由平均 prediction MSE 代理，必须关注同一历史/当前状态
下随 action 改变的 future geometry。由于严格 exact Pareto 门仍差一个 episode，该实现不晋级；
但 `48/9/8/35` 的配对列联和跨零区间也不支持否决整个 action-conditioned preservation 思路。
后续若发展新方法，应以 **history intervention 学条件响应、action intervention 保普通控制几何**
为理论对象，而不是回到 marginal regularizer、额外 context 模块或这条实现的超参数延长。

§5.28 又第一次移除了 frozen teacher。当前完整 recipe 使用 simulator-real `2×2`
History×Action 网格，在不增加任何部署结构的情况下于单 seed Motion 通过全部六门；这证明在
该任务与配置中 frozen teacher 不是必要条件，但尚未把收益隔离归因于四元组单因素。candidate
同一 SHA 的 CEM 为 `51/100`，source 为 `58/100`；contact-free action branches 是一个重要伴随
差异，与 counterfactual 未覆盖 planner 交互支持的假设一致，但不能替代 contact-matched 因果
对照。当前最接近通用且仍满足简洁约束的假设因此收敛为 **support-matched dual intervention**：
固定 `(Q,A)` 改历史以学习隐藏动力学，固定 `(H,Q)` 在普通/接触状态改动作以保持控制几何。
这个假设尚未训练验证，不能写成最终方法；本次零接触实现停止，不用更多 Motion 规模救援。

§5.29 已直接执行上述 contact-rich sufficiency test，并推翻了把 support 简化为 contact bit 的版本。
全部 alternate rollouts 都接触、都在边界内、都有非零 History×Action interaction；同一无 teacher、
零新增部署结构的 recipe 仍通过 Motion 六门，说明连续 conditional ICL 正信号没有消失。可是
paired CEM 从 source 的 `58/100` 降到 `40/100`，其 `[-28,-8]pp` 区间已排除零差；相对零接触
候选也从 `51` 降到 `40`，但该预算匹配差异的区间跨零。source 与 candidate 累计训练曝光又
不同，所以不能归因成“contact 导致退化”。能够确认的是：支持“接触覆盖不足”并不等于证明
“补一个接触 action 即可”。
真正需要保持的是 planner action distribution 上的整条 action-conditioned future function，而非
一个物理事件标签或单一高能量 action ray。该实现停止，joint pairing 家族保留；任何后续简洁
方法仍不得增加 encoder、adapter、head 或推理计算。唯一下一 MVE 先保持 objective 不变，只把
action branch 换成 planner-distributed 的多方向、多步真实动作，以单因素检验 distribution coverage。

§5.30 又纠正了方法判定本身。对 §5.27 frozen checkpoint 的 300-query paired CEM 扩评得到
`180 vs 187`、点差 `-2.33pp`，但相对预先固定的 `-5pp` margin，一侧 95% 区间
`[-6.33,+1.67]pp` 仍跨界，故标签是 inconclusive 而非 fail。seed42 相同 query 的重跑计数与
outcome bits 发生变化，进一步证明 exact success-count sign 不是适合随机 CEM 的科学门。当前
action-function anchor 仍是最接近成功的候选，但既不晋级也不被否决；后续所有规划保持结论改用
paired effect、固定 practical margin 与 confidence bound，hard gate 只留给身份和结构不变量。

§5.31 随后把 teacher-free `2×2` 的单 action ray 换成经验 replay 中真实、多方向、五步 action
block。该零新增参数候选仍通过 Motion 六门，并把同 catalog CEM 从 zero-contact 的 `51/100`
提高到 `55/100`，距 source `58/100` 为 `-3pp`；paired 区间仍跨零，所以结论是有利趋势而非
已证明非劣。它已经与最强 teacher action-function anchor 的 fresh seed42 CEM 点数相同，且去掉
teacher；因而当前最值得保留的简洁候选是 **empirical-action-support joint pairing**。剩余方法
缺口不再是模型容量或边缘正则，而是如何去掉显式 simulator matched pairs 和两阶段 warm start，
同时保留同一 History×Action×Future 关系。多 seed、Public 与额外 CEM 不在方法发现阶段展开。

§5.32 已进一步删除两阶段 warm start 并找到了此前被混在一起的变量。residual 单阶段能在
`1,024` step 学会 ICL，却因输出投影重置使 CEM 降到 `20/100`；延长训练只会破坏 NRE。
完整保留原 LeWM absolute 坐标后，单阶段 `2,048` 达到 gain=`0.170`、NRE=`0.901`、CEM
`54/100`，且 paired-balanced lower=`0.555`；仅 raw future=`0.533` 未越过旧 `0.55` 门。
因此它不被包装成正式全过或非劣，但已成为最接近原始简洁目标的候选：标准 LeWM、一次训练、
无 teacher、零新增部署结构。后续不再扫 basis/budget，唯一核心方法问题是怎样去掉显式
simulator matched pair annotation，同时保留这条真正的联合条件监督。

§5.33 已证明最后一句需要更精确。完整训练资产的 `4,096` 个显式二元组可以只用可见
`(query RGB, action)` 精确恢复；hidden label、pair/template id、history、future 和行序均不参与，
且 canonical loss 对组排序/方向不敏感。因此 privileged annotation 已可删除，当前真正不能删除的
是 conditional-overlap data assumption。最终方法应被表述为 **visible-condition joint pairing**，
而不是 hidden-label supervision；下一验证必须考察这种 overlap 能否由无标签数据收集或近似可见
matching 获得，不能再把普通 unmatched replay 与已满足 overlap 的数据混为一谈。

§5.34 已完成前一种验证。一个不向 collector 暴露 damping identity 的随机环境，通过一次
query-state feedback shooting 自行找到连续到达公共 Q 的 x0；独立抽样只按可见 history 去重，
最终仍按可见 `(Q,A)` 分组。全 `2,048` templates 的 `8,192` 行逐 template 与旧训练资产精确
同集合，最大 query-state 误差仅 `3.24e-12`。因此 named endpoint matching 也不是训练 recipe
的必要输入，旧 absolute `2,048` checkpoint 无需重复训练即可代表该收集协议。剩余假设已经不能
再含糊称为“privileged labels”：它是主动环境随机化与可控 reset 所提供的 conditional overlap；
普通 unmatched offline replay 仍未被解决。下一步转向同一无新增参数 joint relation 的跨任务
能力验证，不再在 Motion 上改 loss、模型、坐标或预算。

§5.35 已完成这次跨任务验证，并把“只是 Motion 特例”的解释明显削弱。相对 published LeWM，
同一 center-free visible-condition joint relation 在 current Contact 把 future/history/switch/worst
从 `0.496/0.518/0.504/0.340` 提高到 4,096-step 的 `0.771/0.850/1.000/0.734`，gain 从约零
升至 `0.447`、NRE 降至 `0.579`。§5.36 的 matched-native 修正后，2,048/4,096 的 joint−native
CEM 为 `0pp/-1pp`，而不是旧 source-only 表面的 `75 vs 69`；8,192 为 `-6pp [-15,+3]pp`，
只作为过训练风险。Motion 2,048 的同类 matched 对照则为 joint−native `-11pp [-22,0]pp`：
它确认 joint 是 direct ICL 的强正因子，也确认当前最简配方在 Motion 上仍改坏 planner function。
因此 2,048–4,096 Contact 被保留为 discovery Pareto 区间，更多预算停止。严格 exact-center
单因素对照将 gain 压回 `0.030`，说明 oracle
common-center 瓶颈不能被直接归一化 center regression 转化成处方。当前可以支持的最简方法主张是：
**在不改变 LeWM 参数或推理结构时，visible conditional overlap 上的 center-free 联合关系监督能
跨离散与连续隐藏动力学创造条件可辨识性，并在 Contact 上与原规划能力共存；但 exact overlap
并不自动保证跨任务 planning preservation。** 仍不能主张普通 unmatched offline data 已足够，
也不能以单 seed 取代最终多 seed/Public 确认。

## 8. 证据入口

- [ContextWorld ICL Suite v2 完整性重封判定](../../../ContextWorld/configs/benchmark/contextworld_icl_suite_v2_integrity_reseal_decision_v2.json)
- [常设审计契约（规则文件，非某次实验记录）](AUDIT_CONTRACT.md)
- [v2 build 源身份门 addendum：commit 降为记录项](configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_build_source_gate_addendum_v1.yaml)
- [v2 build 源身份门回执（45 文件 / 41 逐字节 / 0 未裁定）](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_build_source_gate_addendum_v1/build_source_gate_receipt.json)
- [v2 build 源身份门审计器（闭包取自 v3 权威，不复述）](scripts/audit_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_build_source_gate_v1.py)
- [v2 契约小节补全 addendum（append-only，只增不改）](configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_contract_sections_addendum_v1.yaml)
- [v2 契约小节补全回执（执行冻结校验器的前后对照）](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_contract_sections_addendum_v1/contract_sections_receipt.json)
- [v2 多 seed 评测预注册（三格：calibration + s4096 + s5120）](configs/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_development_v2.yaml)
- [v2 release builder（300-query release 已构建并消费）](scripts/build_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2.py)
- [v2 阻断留痕：两个阻断如何被执行发现](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_preflight/build_blocked_by_frozen_builder_v1.json)
- [v2 recorded-HEAD 非门控恢复附录](configs/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_head_record_recovery_addendum_v1.json)
- [v2 恢复评估器](scripts/eval_action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2_head_record_recovery_v1.py)
- [v2 三 seed 最终消费回执](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multiseed_private_development_v2/consumption_receipt.json)
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
- [Motion frozen-base factorization/oracle MVE 实现](scripts/run_pusht_motion_damping_factorized_residual_mve_v1.py)
- [Motion rank-128 oracle factorization 结果](artifacts/pusht_motion_damping_factorized_residual_mve_v1/oracle_rank128_s3073_step4096_v1/report.json)
- [Motion oracle-conditioned MLP 结果](artifacts/pusht_motion_damping_factorized_residual_mve_v1/oracle_mlp_h256_s3073_step4096_v1/report.json)
- [Motion context–response 因果阶梯：additive context 实现](scripts/run_pusht_motion_damping_oracle_context_predictor_mve_v1.py)
- [Motion context–response 因果阶梯：source-routed factorized 实现](scripts/run_pusht_motion_damping_routed_factorized_response_mve_v1.py)
- [Motion context–response 因果阶梯：paired-response 实现](scripts/run_pusht_motion_damping_factorized_paired_response_mve_v1.py)
- [Motion 跨-query sampler/transfer 紧凑判定](artifacts/motion_cross_query_transfer_v1/summary.json)
- [Motion function-anchor ICL–CEM Pareto 摘要](artifacts/pusht_motion_damping_residual_transition_function_anchor_v1/summary.json)
- [Contact function-anchor 跨任务迁移与停止判定](artifacts/pusht_contact_friction_residual_transition_function_anchor_v1/summary.json)
- [Contact 8,192-step 冻结 Development 终点](artifacts/pusht_contact_friction_residual_transition_function_anchor_v1/s13313_source2048_plus8192_v1/development_score_current_runtime_v1.json)
- [Canonical-margin exact-future 实现](scripts/canonical_margin_exact_future_v1.py)
- [Contact canonical-margin 单因素验证摘要](artifacts/pusht_contact_friction_canonical_margin_function_anchor_v1/summary.json)
- [Contact canonical-margin 8,192-step 冻结 Development 终点](artifacts/pusht_contact_friction_canonical_margin_function_anchor_v1/s13313_source2048_plus8192_v1/development_score_current_runtime_v1.json)
- [Contact common-center oracle 与四个 matched MVE 汇总](artifacts/pusht_contact_friction_common_center_followups_v1/summary.json)
- [center-free conditional-response 跨 Motion/Contact 判定](artifacts/canonical_response_only_cross_task_v1/summary.json)
- [Motion 无 teacher freeze-only 训练报告](artifacts/pusht_motion_damping_canonical_response_only_freeze_v1/s14321_source2048_plus1024_v1/training_report.json)
- [Motion 无 teacher freeze-only CEM100](artifacts/pusht_motion_damping_canonical_response_only_freeze_v1/cem_seed42_n100_v1/aggregate.json)
- [Motion response-only + function-anchor CEM100](artifacts/pusht_motion_damping_canonical_response_function_anchor_v1/cem_seed42_n100_v1/aggregate.json)
- [Motion action-intervention function-anchor 紧凑判定](artifacts/pusht_motion_damping_action_intervention_anchor_v1/summary.json)
- [Motion action-intervention 六门终点](artifacts/pusht_motion_damping_action_intervention_anchor_v1/s14321_source2048_plus1024_v1/development_response_analysis_v1.json)
- [Motion action-intervention paired CEM100](artifacts/pusht_motion_damping_action_intervention_anchor_v1/cem_seed42_n100_v1/aggregate.json)
- [Motion action-intervention CEM300 非劣冻结设计](configs/pusht_motion_damping_action_intervention_anchor_cem300_noninferiority_v1.json)
- [Motion action-intervention paired CEM300](artifacts/pusht_motion_damping_action_intervention_anchor_v1/cem_seeds42_43_44_n100_v1/aggregate.json)
- [Motion action-intervention CEM300 非劣分析](artifacts/pusht_motion_damping_action_intervention_anchor_v1/cem_seeds42_43_44_n100_v1/noninferiority_analysis_v1.json)
- [无 teacher Cartesian History×Action 紧凑判定](artifacts/pusht_motion_damping_cartesian_action_pair_legacy_scale_v2/summary.json)
- [2,048-template Cartesian Motion 六门终点](artifacts/pusht_motion_damping_cartesian_action_pair_legacy_scale_v2/s14321_source2048_plus1024_templates2048_v1/development_response_analysis_v1.json)
- [2,048-template Cartesian paired CEM100](artifacts/pusht_motion_damping_cartesian_action_pair_legacy_scale_v2/cem_seed42_n100_v1/aggregate.json)
- [Contact-rich Cartesian History×Action 紧凑判定](artifacts/pusht_motion_damping_contact_cartesian_action_pair_v1/summary.json)
- [Contact-rich 2,048-template overlay 回执](artifacts/pusht_motion_damping_contact_cartesian_action_overlay_v1/train_templates2048_v1.pt.json)
- [Contact-rich Cartesian Motion 六门终点](artifacts/pusht_motion_damping_contact_cartesian_action_pair_v1/s14321_source2048_plus1024_templates2048_v1/development_response_analysis_v1.json)
- [Contact-rich Cartesian paired CEM100](artifacts/pusht_motion_damping_contact_cartesian_action_pair_v1/cem_seed42_n100_v2/aggregate.json)
- [common-center privileged oracle 原始结果](artifacts/pusht_contact_friction_common_center_oracle_v1/canonical_margin_8192_result.json)
- [Motion native complete-twin sampler](scripts/run_pusht_motion_damping_native_twin_sampler_v1.py)
- [Motion absolute anchored context transfer](scripts/run_pusht_motion_damping_anchored_context_transfer_v1.py)
- [Motion transition-context transfer](scripts/run_pusht_motion_damping_transition_context_transfer_v1.py)
- [Motion frozen-base paired-response 预注册](configs/pusht_motion_damping_frozen_factorized_paired_response_mve_v1.yaml)
- [Motion frozen-base oracle 256-step 结果](artifacts/pusht_motion_damping_frozen_factorized_paired_response_mve_v1/oracle_s14321_step256_v1/training_report.json)
- [Motion frozen-base oracle 1,024-step 轨迹](artifacts/pusht_motion_damping_frozen_factorized_paired_response_mve_v1/oracle_s14321_step1024_trajectory_v1/training_report.json)
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
- [gradient-to-estimand 审计预注册](configs/pusht_motion_damping_pcjr_cv_gradient_to_estimand_audit_v1.yaml)
- [gradient-to-estimand 审计执行器](scripts/run_pusht_motion_damping_pcjr_cv_gradient_to_estimand_audit_v1.py)
- [v1 parity 门失败留痕](artifacts/pusht_motion_damping_pcjr_cv_gradient_to_estimand_audit_v1/training/s14321_step256_v1/gradient_to_estimand_audit_v1_phase2_failure_receipt_v1.json)
- [parity 容差重校恢复边界](configs/pusht_motion_damping_pcjr_cv_gradient_to_estimand_audit_recovery_v2.yaml)
- [gradient-to-estimand 审计结果与三态判定](artifacts/pusht_motion_damping_pcjr_cv_gradient_to_estimand_audit_recovery_v2/development/s14321_step256_endpoints_v1/gradient_to_estimand_audit_recovery_v2.json)
- [unchanged 256-step parent prefix replay 回执](artifacts/pusht_motion_damping_pcjr_cv_gradient_to_estimand_audit_recovery_v2/training/s14321_step256_v1/audit_prefix_replay_receipt_v1.json)
- [容差重校执行回执](artifacts/pusht_motion_damping_pcjr_cv_gradient_to_estimand_audit_recovery_v2/development/s14321_step256_endpoints_v1/tolerance_recovery_execution_receipt_v2.json)
- [native loss 来源分解预注册（含机制预测与双精度修正案）](configs/pusht_motion_damping_pcjr_cv_source_decomposition_audit_v1.yaml)
- [来源分解审计执行器](scripts/run_pusht_motion_damping_pcjr_cv_source_decomposition_audit_v1.py)
- [来源分解结果、四臂矩阵与首轮三态判定](artifacts/pusht_motion_damping_pcjr_cv_source_decomposition_audit_v1_attempt3/development/s14321_step256_endpoints_v1/source_decomposition_audit_v1.json)
- [来源分解执行回执](artifacts/pusht_motion_damping_pcjr_cv_source_decomposition_audit_v1_attempt3/development/s14321_step256_endpoints_v1/source_decomposition_execution_receipt_v1.json)
- [bf16 sub-ulp 中止留痕与双精度修正案](artifacts/pusht_motion_damping_pcjr_cv_source_decomposition_audit_v1/training/s14321_step256_v1/attempt1_reconstruction_gate_abort_receipt_v1.json)
- [佐证门判定修订预注册（含 RNG 敏感性撤回）](configs/pusht_motion_damping_pcjr_cv_source_decomposition_corroboration_revision_v1.yaml)
- [判定修订执行器（零重算，仅读冻结 JSON）](scripts/run_pusht_motion_damping_pcjr_cv_source_decomposition_corroboration_revision_v1.py)
- [修订后判定与 C(17,4) 穷举采样覆盖证据](artifacts/pusht_motion_damping_pcjr_cv_source_decomposition_corroboration_revision_v1/development/s14321_step256_endpoints_v1/corroboration_revision_v1.json)
- [预注册注释勘误与哈希对账](artifacts/pusht_motion_damping_pcjr_cv_source_decomposition_corroboration_revision_v1/development/s14321_step256_endpoints_v1/preregistration_comment_errata_v1.json)
- [target stop-gradient V8 反证](artifacts/paired_terminal_target_stopgrad_sigreg_v8_validation/development/action_delay_stage1_s3072_step256_v1_gate_decision.json)
- [ActionDelay 多 seed 复现训练完成回执（2,048 步，0 评分）](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1/receipts/multi_seed_training_completion_v1.json)
- [多 seed 评估受阻记录（原始，含被本人更正的 blocker 2）](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1/receipts/evaluation_blocked_v1.json)
- [对上述 blocker 2 的自我更正与 45 文件内容证据](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_multi_seed_v1/receipts/evaluation_blocked_v1_correction_v1.json)
- [第三次 source-rebind 预注册（HEAD 降为记录项，内容闭包为判定门）](configs/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_source_rebind_addendum_v3.yaml)
- [第三次 source-rebind 审计器（41 相同 + 4 按哈希钉死的裁定）](scripts/audit_action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_source_rebind_addendum_v3.py)
- [第三次 source-rebind 回执](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_source_rebind_addendum_v3/source_rebind_receipt.json)
- [v1 审计器排除集拷贝缺陷与 stale marker 结案](artifacts/action_delay_h7_a0_aux_pcja_predictor_only_private_development_v1_source_rebind_addendum_v3/v1_auditor_exclusion_set_defect_v1.json)
- [Encoder-only / history-value 阶段报告](results/history_value_encoder_only_stage_report_v1.md)
- [target-JTCov 任务广度报告](results/joint_temporal_covariance_sigreg_task_breadth_report_v1.md)
- [conditional identifiability 理论说明](results/identifiability_corrected_mse_theory_v1.md)
- [Contact visible-condition joint 跨任务紧凑结论](artifacts/pusht_contact_friction_visible_joint_transfer_v1/summary.json)
- [Contact CEM published-source 精确序列化回执](artifacts/pusht_contact_friction_visible_joint_transfer_v1/cem_source_materialization_v1.json)
- [Contact center-free 单阶段 2,048-step 执行器](scripts/run_pusht_contact_friction_visible_joint_absolute_single_stage_v1.py)
- [Contact exact-center 单因素执行器](scripts/run_pusht_contact_friction_visible_joint_exact_future_single_stage_v1.py)
- [Contact center-free 4,096-step Pareto 执行器](scripts/run_pusht_contact_friction_visible_joint_absolute_single_stage_step4096_v1.py)
- [Contact center-free 8,192-step预算边界执行器](scripts/run_pusht_contact_friction_visible_joint_absolute_single_stage_step8192_v1.py)
- [Contact 4,096-step matched native-no-aux 执行器](scripts/run_pusht_contact_friction_visible_joint_native_control_step4096_v1.py)
- [Contact joint-vs-native paired effect 回执](artifacts/pusht_contact_friction_visible_joint_native_control_step4096_v1/matched_control_comparison_v1.json)
- [Contact 8,192-step matched native-no-aux 执行器](scripts/run_pusht_contact_friction_visible_joint_native_control_step8192_v1.py)
- [Contact 8,192-step matched native Development](artifacts/pusht_contact_friction_visible_joint_native_control_step8192_v1/s13313_step8192_v1/development_score_current_runtime_v1.json)
- [Contact 8,192-step source/native/joint 同进程 CEM](artifacts/pusht_contact_friction_visible_joint_native_control_step8192_v1/cem_matched_comparison_seed42_n100_v1/aggregate.json)
- [Contact shifted-pair 单因素执行器](scripts/run_pusht_contact_friction_visible_joint_shifted_pair_control_v1.py)
- [Contact exact-overlap vs shifted-pair 回执](artifacts/pusht_contact_friction_visible_joint_shifted_pair_control_v1/shifted_pair_comparison_v1.json)
- [Contact 2,048-step native/exact/QA/RGB 四臂 CEM](artifacts/pusht_contact_friction_visible_joint_rgb_qa_only_graph_v1/cem_four_arm_seed42_n100_v1/aggregate.json)
- [Contact exact/approximate comparator 汇总](artifacts/pusht_contact_friction_visible_joint_rgb_qa_only_graph_v1/comparator_validity_summary_v1.json)
- [Motion 2,048-step matched native-no-aux 执行器](scripts/run_pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_native_control_step2048_v1.py)
- [Motion 2,048-step matched native focused tests](tests/test_motion_absolute_single_stage_native_control_step2048_v1.py)
- [Motion 2,048-step matched native Development](artifacts/pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_native_control_step2048_v1/s14321_baseline_plus2048_templates2048_v1/development_response_analysis_v1.json)
- [Motion 2,048-step source/native/joint 同进程 CEM](artifacts/pusht_motion_damping_replay_cartesian_action_pair_absolute_single_stage_native_control_step2048_v1/cem_matched_comparison_seed42_n100_v1/aggregate.json)
- [历史候选 comparator 公平性总判定](artifacts/conditional_joint_comparator_validity_v2/summary.json)

机器回执保留完整路径、输入哈希、checkpoint 身份和 gate 字段；本文只呈现支撑当前研究
判断与下一步证伪协议所需的结果，避免把 recovery/version 执行流水写成方法叙事。
