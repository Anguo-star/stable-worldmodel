# ContextWorld / COJA：世界模型能否从历史中识别当前回合的动力学

本目录研究一个具体问题：世界模型（world model）在预测未来时，会不会**读懂前面发生了什么**。
本文件是新读者入口，只讲问题、方法、已确立的结论和仍然开放的部分；不含实验日志与逐日记录。

## 1. 一个最小例子

设想同一个推物任务的两段回合。两段回合在**当前这一刻完全一样**：同样的画面、同样的物体位置、
同样的目标，接下来也执行**同一串动作**。唯一的差别在更早的历史里：

- 回合 A 的前几步显示，物体一推就滑很远（阻尼小）；
- 回合 B 的前几步显示，同样的力只能推动一点点（阻尼大）。

于是真实的未来必然不同：同一个动作，在 A 里把物体推得更远。一个真正"从历史识别动力学"的
模型，应当对这两段历史给出两个不同、且方向和幅度都对的预测。

我们观察到的却是：模型的隐空间表征健康、不坍缩，历史信息也确实能被探针（probe）读出来，
但预测器（Predictor）对 A 和 B 输出了几乎相同的未来。它学会了"这类情况下未来平均是什么样"，
没有学会"这段历史对应哪个未来"。这两件事完全不同，而常规评测看不出差别。

## 2. ContextWorld 测什么

ContextWorld 是为这个问题造的受控评测：固定当前观测 $Q$ 与查询动作 $A$，**只干预历史**，
看模型预测是否随之改变。它把三类互不替代的目标量分开报告：

| 目标量 | 回答的问题 | 主要指标 |
|---|---|---|
| 直接条件响应 | 预测是否随历史改变，方向和幅值对不对 | future / history / switch / worst；gain、alignment、NRE |
| 隐藏动力学规划 | 正确历史是否让规划器选出更好的动作 | 正确历史 vs 交换历史的规划误差 |
| 原任务保持 | 训练后是否还会做原来的标准任务 | 标准 PushT / TwoRoom 的 CEM 成功率 |

其中 `NRE = 1` 有物理含义：预测完全不随历史变化。离散的分配准确率**不能**替代连续量——
分配可以在 NRE 很差时虚高，这正是下文 ActionDelay 的情况。标准环境的 CEM 成功率只回答第三项，
不是 ICL 指标。

## 3. COJA 改变了什么

COJA（Conditional-Overlap Joint Alignment，条件重叠联合对齐）不改模型结构，也不改推理：
它在**共享同一可见条件 $(Q,A)$、但历史与真实未来不同**的样本组内，直接监督"哪段历史对应
哪个未来"。只在训练时多走一次已有 Predictor，部署参数与推理开销增量为零。

它的理论动机是：任何只依赖目标表征**经验集合**的正则项对样本置换不变，因而在原理上无法编码
历史—未来的对应关系；并且在缺少条件重叠时，这个反事实响应在非参数意义下不可识别（技术报告
命题 1）。条件重叠是充分条件之一，不是唯一形式。

## 4. 已经确立的

**COJA 明显改善了历史条件分配。** 在 Motion Damping 相同的 Development pairs 上，原生 LeWM
的 gain/NRE 为 `0.0065 / 1.1298`，逐 query 的 $G_{\mathrm{swap}}$ 只有 `44.1%` 为正（与随机
无异）；COJA 为 `0.3696 / 0.7665`，正号比例 `96.9%`。这证明数据里确实存在可学习的条件信号，
问题不在"任务不可学"。

**ActionDelay：分配已学会，校准发生漂移。** 单 training seed 的完整 10-epoch 终点为
future/history/switch/worst = `0.882/0.918/1.000/0.763`，即模型已经能按历史把未来分配对；
但同一 checkpoint 的 gain/NRE = `6.877/70.995`，响应幅值被严重过放大。逐 epoch 重评分把漂移
起点定位在 epoch 3 到 4 之间（epoch 3 的 NRE 最低为 `0.595`），epoch 5 起明显过响应。
准确的表述是**assignment 已建立、response calibration 未闭合**，不是 COJA 失败。

**两个更简单的解释已被否定。** 在同 seed、同完整预算的 Development 机制实验中：把历史窗口从
滑动改为动态扩展，没有救回原生 ICL；只用原生损失做 `K=3` 多步 rollout，也没有产生足够的条件
辨识压力，无法替代 COJA。三者的 NRE 为 sliding `1.0018`、expanding `1.0089`、单步 COJA
`0.9579`——前两者本质上就是"预测不随历史变化"。这是单训练种子的 Development 机制证据，不是
正式发布分数。

**Motion 根因解释：原生配对 MSE 里的条件项很弱，但不为零。** 对共享 $(Q,A)$ 的二元组，配对平方误差
精确分解为组中心项加响应项，正确历史与交换历史的损失差恰好等于 $\langle\Delta p,\Delta t\rangle$。
也就是说，**原生目标在数学上确实包含条件响应项**，native 失败不能解释成"训练信号里根本没有
这个量"。实测的是它的相对能量与梯度信噪比太低：冻结 latent 中条件能量只占目标方差的
`0.23%--0.29%`；response 项占终点配对 MSE 的 `13.62%`，但映射到真实 optimizer 参数后，
response 梯度相对非条件梯度的范数只有 `5.57%`，平方能量占比 `0.309%`，32-pair 的 SNR 约
`1.10`——刚好在噪声边缘。同期训练把中心误差从 `0.04532` 降到 `0.02033`，response 误差却没有
下降。这是一个**有证据支持的工作假说**，来自 Motion 的冻结 Training batch 与 Development，
不是跨任务的普遍证明。ActionDelay 与 Action Strength 还出现了 LeWM/PLDM 成败方向相反的历史
结果，进一步说明不能把所有失败归成一个低 `rho_phys` 标量；通用候选应写成数据、表示和梯度聚合
共同决定的“有效条件可见性”。

**D1-MS50 证明了数据有因果作用，但当前操作不充分。** 在 Motion 的同一 Training twin 池内，
D1 把三个局部口径的 `rho_phys` 提高 `3.87%--4.95%`，冻结初始化的整体 response 梯度范数和
`SNR(16)` 分别提高约 `6.9%/4.6%`。完整 native 训练后，自然 Development 上的 gain、NRE 和
$G_{\mathrm{swap}}$ 都有小幅正确方向变化，但 D1 仍未通过历史使用门。准确结论是
**single-seed directional data effect=true, history-use positive=false**，不能写成数据路线已经解决。

**像素差异弱是部分原因，不是统一答案。** 四任务 Training-only 原始像素审计中，Motion 的
future 条件差异和相对份额最低（`C_pixel=1.08e-4, rho_pixel=0.093`），Speed 为
`3.29e-4/0.154`，Action Strength 为 `8.35e-4/0.290`。这支持 Motion 的 native loss 里历史导致的
future 差异太不显著。但 ActionDelay 的 `rho_pixel=0.360` 仍出现 LeWM 失败、PLDM 成功；Action
Strength 则 LeWM 成功、PLDM 失败。因此更一般的根因是：历史条件信号在**数据 target、模型表示、
实际梯度或跨 query 泛化**的某一层失去有效可见性，具体限制层依赖任务和模型。Speed 没有 observed
same-query cross-speed twins，其数值来自 Training query 上经逐像素校验的模拟器反事实，不能当成
observed history 证据。

**模型侧反转进一步否定了单一数据解释。** ActionDelay 的 A0/A3/A4 使用相同 Training probe、
shared-core 初始化和 logical batches；同一 LeWM 实现上的 A3 只换 PLDM-active objective，step-0
total-gradient 范数和有利 response-residual 一阶变化分别达到 A0 的 `2.801x/48.583x`，并近似
native PLDM A4。首个局部分离因此在 objective/Jacobian 路由，不在 raw data。Action Strength 的
exact-batch 反向检查却没有形成确认：PLDM 在 Training batch 终点已有 gain/NRE=`0.7337/0.2203`，
仍对应冻结 held-out 负标签，说明 coverage/迁移/校准也可能限制结果。两者都是机制边界，不是统一
跨模型阈值。

## 5. 仍然开放的

- **校准闭合**：如何让长训练在保住分配的同时不放大响应幅值。ActionDelay 的漂移已按 NRE 分解
  为约 `34.54` 的尺度误差与约 `36.46` 的正交残差，两者都要处理，不是纯尺度问题。
- **训练种子方差**：多数结论目前是单 training seed。逐 query 的 bootstrap / sign-flip 只刻画
  评测 query 抽样不确定性，**不能**冒充训练种子重复。
- **跨模型族机制的完整性**：ActionDelay 已有共同 probe/shared-core 的目标路由分离，Action
  Strength exact-batch 反向确认未成立；仍缺把多个 task-model 单元的模型侧量与自然 held-out
  `G_swap` 严格闭合的证据，不能把局部门写成普遍根因。
- **原任务保持代价**：一步 COJA 在部分任务上相对同数据 matched native 有可辨认的保持性下降，
  需与方法增量分开报告。
- **公开 Test**：本轮根因与数据实验不打开或重跑原始 Public 数据；既有发布 scoreboard 只用于
  标记历史 outcome，不参与配方选择。

## 6. 接下来的两条路线

两条路线共享同一把评测尺子，不互相替代：

- **路线 A（benchmark + COJA 论文）**：冻结 benchmark v1，补齐 matched native / COJA /
  expanding / rollout 的方法结论。COJA 在这条线上既是主方法，也是"该能力可以被诱导"的正对照。
- **路线 B（面向未来预训练的数据分布原则）**：不动 native loss，只改训练分布，看什么样的数据
  能让原生目标自己产生使用历史的压力，并把结果沉淀为可复用的数据构造原则。

关键的最小因子表是 {D0 自然分布, D1 高辨识分布} × {native, COJA}。`D1 + native` 已先于
`D1 + COJA` 完成，并只得到小幅方向效应，因此 D1 冻结、不再做救援式调权。`REL50/ABS50/HASH50`
构造、schedule、冻结 latent/gradient 门和唯一 D1 native 训练也都已完成：相对量优于只看绝对
能量，但同池操作幅度仍不够。

上述模型侧门已经完成。下一步只在 Motion 上规划 D2-0：构造更高 relative target separation、
非零 state-dependent action leverage 和新 query coverage 的 paired 轨迹，同时保留自然幅值锚点；
构造门未通过前不训练。评测协议保持不变，所以仍是 training-distribution track，**不构成
benchmark v2**；COJA 继续作为论文方法与可学习性正对照。

## 7. 该读哪份文件

| 文件 | 用途 |
|---|---|
| [paper_zh/main.tex](paper_zh/main.tex) | **论文正文的唯一公开来源**（中文工作稿；在 `paper_zh/` 运行 `make pdf` 生成 PDF） |
| [TECHNICAL_REPORT.md](TECHNICAL_REPORT.md) | 专家证据链：问题定义、可识别性命题、方法、跨任务结果、消融、局限 |
| [ROOT_CAUSE_DATA_STRATEGY_ZH.md](ROOT_CAUSE_DATA_STRATEGY_ZH.md) | 通用根因五层因果链、冻结指标、数据构造流水线与两路线优先级 |
| [ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md](ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md) | 四任务反转矩阵、claim 边界、matched comparator 与最小 pass/kill 条件 |
| [D1_CONSTRUCTION_PLAN_ZH.md](D1_CONSTRUCTION_PLAN_ZH.md) | D1 高辨识训练分布的预注册构建方案与训练前门控 |
| [PAPER_OUTLINE_ZH.md](PAPER_OUTLINE_ZH.md) | 内部编辑规划：章节结构、图表位次与投稿前实验优先级 |
| [EXPERIMENT_LOG_ZH.md](EXPERIMENT_LOG_ZH.md) | 完整实验档案：逐日结果、运行身份、历史证据索引 |

## 8. 读结果时的四组口径

这四组区分贯穿全部文档，混用会直接改变结论：

1. **Development vs 公开 Test**：Development 用于方法开发与配方选择，Test 只用于最终报告，
   不反馈到调参。
2. **训练种子 vs 评测回合**：多个评测 seed × 每 seed 若干 episode 仍然只是**一个**训练种子；
   它刻画评测抽样，不刻画训练方差。
3. **未运行 vs 失败**：没跑的格子写"—"，不写成负结果。
4. **绝对保持 vs matched 方法效应**：标准环境的 CEM 成功率是该 checkpoint 的绝对保持量；
   只有在同初始化、同数据、同预算的 native 对照存在时，才能谈方法增量。

最后一条通用约束：不要把离散分配、连续校准、隐藏动力学规划和原任务保持压成一个"总 ICL 分数"。
它们是四个不同的估计量，且已经出现过彼此背离的实例。
