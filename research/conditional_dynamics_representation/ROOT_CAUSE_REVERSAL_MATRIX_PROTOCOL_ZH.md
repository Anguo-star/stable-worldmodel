# 条件 ICL 机制边界：反转矩阵与最小因果验证协议

状态：2026-09-02，**第一轮机制边界界定完成；统一低-`rho` 充分解释已证伪；通用根因未闭合。**
协议、现有证据矩阵、四任务 Training-only 物理/原始像素上游审计，以及
ActionDelay/Action Strength 的只读模型侧反转汇总均已冻结；本轮新增审计的 optimizer step 为
`0`。本轮不打开或重跑公开 Test。既有公开 scoreboard 只用于标记历史上的正、负单元，不作为
根因证据。承重 claim 的 scope 与降级条件见 [`CLAIMS_LEDGER_ZH.md`](CLAIMS_LEDGER_ZH.md)。

## 1. 要回答什么

数据中出现“不同历史对应不同 future”，只说明条件信号存在。它不保证 native world-model loss
会优先使用历史，因为逐样本 MSE 也可以主要学习不同条件下的平均 future。

本轮不寻找一个跨模型通用的神奇阈值，而检验下面这条可证伪机制：

> 历史条件必须依次穿过物理数据、训练表示、参数梯度和 held-out 行为四层。任何一层把条件分量
> 压小或让不同 query 的梯度互相抵消，native 训练都可能忽略历史。数据分布可以操作前两层并改变
> 梯度聚合，但它是否是 binding cause 必须在具体任务-模型单元内由匹配干预证明。

这句话包含三个不同层级，不能互相替代：

| 层级 | 当前状态 | 允许的表述 |
|---|---|---|
| 普遍脆弱性 | 已由 paired MSE 恒等式成立 | native loss 没有单独保护历史条件项 |
| 一般机制候选 | 已获两类匹配证据，仍非普遍定理；**通用根因未闭合** | 有效条件可见性/可迁移性可在数据、表示、目标或梯度聚合中衰减 |
| 单元 binding cause | Motion 数据因素与 ActionDelay 局部目标路由已有证据 | 哪一层真正限制某个任务与模型，仍须逐单元判断 |

因此，“所有失败都是 `rho_phys` 低”不是当前 claim，且该充分解释已被 §4.1 的两组反转证伪；
“数据分布永远能单独解决”也不是。第一轮完成的是**机制边界界定**，不是根因收口。

## 2. 为什么同时看四个任务

四个任务承担不同的反事实角色，不做对称铺量：

| 任务 | 冻结历史结果 | 作用 | 当前根因资料 |
|---|---|---|---|
| ActionDelay | LeWM 负，PLDM 正 | 同数据上的跨模型反转 | 上游已测；共同 core/probe 的 A0/A3/A4 目标路由门已完成 |
| Action Strength | LeWM 正，PLDM 负 | 反方向的跨模型反转 | 上游已测；exact-batch 模型侧反向确认未成立 |
| Door | LeWM、PLDM 均正 | native 正对照候选 | 尚须证明交换历史会损害结果，否则不能当历史使用正例 |
| Motion Damping | LeWM native 负，COJA 正 | 四层深挖与首个因果干预单元 | 数据、表示、梯度和 Development 行为均已有资料 |

**这是 factorial / triangulation 设计，不是独立复制。** 每个 cell 承担不同的反事实角色，
cell 之间不构成彼此的重复实验，也不能按“多数 cell 一致”累积统计置信度——本项目**没有、也从未
预注册过“3/4 单元一致即可确认”这类规则**。矩阵的作用只有两个：否定过强的统一 claim，以及把
binding layer 的搜索范围三角定位到具体单元。

正式清单
[`artifacts/icl_root_cause_reversal_matrix_v1/inventory_v1.json`](artifacts/icl_root_cause_reversal_matrix_v1/inventory_v1.json)
覆盖 `8` 个 task-model cells，其中 **four-layer-complete 的只有 `1` 个**
（`motion_damping_lewm_native_data_path`），`all_cells_four_layer_complete=false`。任何跨单元的
概括都必须显式引用这个事实。

这组反转已经否定“某个模型族天生不会 ICL”以及“同一物理数据能量单独决定成败”。它只定位
共同机制的候选范围；不能用历史结果标签代替新的根因测量。

Cube Carry、Reacher 等结果保留为现象层外部检查，不进入首轮承重干预。这样可以避免为了矩阵
对称而补大量不改变结论的审计。

## 3. 四层证据契约

每个承重单元必须区分以下四层：

| 层 | 问题 | 主量 |
|---|---|---|
| 数据 | 同一或近邻 `(Q,A)` 下，条件差异相对普通 future 变化有多大 | exact overlap、balance、`rho_phys`、coverage |
| 表示 | 物理条件差异是否进入冻结训练坐标 | `rho_lat`、target separation、center/response risk |
| 优化 | 历史响应梯度是否到达实际更新参数且方向一致 | `E||g_i||`、`||Eg_i||`、coherence、`Bcrit/SNR`、参数组分解 |
| 行为 | 模型在自然 held-out query 上是否真的使用正确历史 | on-support `G_swap`、gain、alignment、NRE、cross-query null |

主行为证据必须是同一 query/action/target 下的 correct-vs-swapped history。Removed-history 会改变
输入支持，只保留为辅助诊断，不能替代 `G_swap`。

跨模型族不比较 latent 或梯度的绝对数值，也不设统一数值阈值。只允许三种比较：同单元相对变化、
配对不确定性，以及预注册干预的方向一致性。

## 4. 现有证据给出的边界

Motion native 的 Development gain/NRE 为 `0.0065/1.1298`，平均 `G_swap=4.67e-5`，正号比例
`44.1%`；同数据上的 COJA 为 `0.3696/0.7665`、`G_swap=2.666e-3`、正号比例 `96.9%`。
这证明条件信号可学，也证明 native 当前没有把它变成一致响应。

`D1-MS50` 把三个局部口径的 `rho_phys` 提高 `3.87%--4.95%`，冻结初始化的整体 response
梯度范数和 `SNR(16)` 分别提高约 `6.9%/4.6%`，但 `pred_proj` 路径几乎不动。完整训练后，
自然 Development 的 gain、NRE 和 `G_swap` 均有小幅正确方向变化，仍未通过历史使用门。该结果的
正式名称是**方向性数据干预证据 / upstream-to-gradient directional transmission**，不是 native
behavior reversal。

所以目前能够确定的是：

1. 数据分布是 Motion 因果链中的一个因素；
2. 当前同池 soft reweighting 的操作强度不足；
3. 不能把小幅改善外推成数据路线已经充分；
4. 不能从 Motion 的低 `rho` 外推所有 LeWM/PLDM 负单元；
5. COJA 是显式条件学习正对照，不是失败方案，也不因数据路线而被提前替换。

### 4.1 四任务上游审计对像素假说的裁决

同一冻结 Training source 上的结果如下。`C_pixel` 是 `[0,1]` RGB 的 per-pixel-channel MSE，
`rho_pixel,64=C/(C+B_64)`；每任务 raw-pixel 样本数为 `256`。

| 任务 | `C_phys` | `rho_phys,64` | `C_pixel` | `rho_pixel,64` |
|---|---:|---:|---:|---:|
| Motion Damping | `2.1459` | `0.1036` | `1.0756e-4` | `0.0929` |
| Action Strength | `107.6017` | `0.4962` | `8.3454e-4` | `0.2903` |
| ActionDelay | `35.7292` | 退化，不可用 | `5.1673e-4` | `0.3596` |
| Speed | `23.1554` | `0.1693` | `3.2859e-4` | `0.1538` |

ActionDelay 的 `rho_phys,64` 之所以在原始产物中记为 `1.0000`，是因为该任务的 `B_32/B_64/B_128`
在全部 `5,120` 个 query 上恒为 `0`，`C_phys` 也是常数 `35.7292`（std `7.1e-15`）：这是**退化分母**
的构造结果，不表示条件份额高。该数值**退出跨任务强弱比较**；ActionDelay 仍可用的是 `C_phys` 的
描述性数值与 raw-pixel `rho_pixel,64=0.3596`，后者同样不设跨任务阈值。

Motion 的 raw future 条件差异与相对份额最低，和“像素 target 差异弱会降低 native 条件压力”一致。
但这只是 task-specific contributor：ActionDelay 的 raw-pixel 份额并不低却出现 LeWM 负/PLDM 正；
Action Strength 上游更强却出现 LeWM 正/PLDM 负。两组同数据跨模型反转直接否定 universal data-only
充分解释。Motion 的 observed `H_pixel` 还高于 Action Strength，说明历史 cue 是否可见与它对
future loss 的影响强弱是两件事。

Speed 必须单列证据类型：它没有 observed same-query cross-speed twins，表中来自 Training query
上的 simulator-rendered counterfactual；物理回放与 observed future PNG 的最大残差均为 `0`。
物理量的任务坐标不同，像素量虽同单位但 renderer、条件基数与邻域不同，因此表格只支持方向性
反例和层级定位，不支持跨任务统一阈值或“Speed 因像素更大所以必然学会”的因果句。

本轮据此把一般机制**候选**表述冻结为：**effective conditional visibility bottleneck**。数据 target
separation 决定上游可用压力，表示、loss/Jacobian 与梯度聚合决定该压力能否到达参数；binding cause
必须按任务-模型单元判定。这是候选机制陈述，不是已闭合的通用根因。

### 4.2 同数据模型侧反转门

现有 `recovery_v7` 已经包含可复用的 Training-only 机制资产，因此无需重新训练。主门使用三 seed、
相同 ActionDelay probe、相同 shared-core 初始化和每 seed 完全相同的 8-rank logical batches：

- `A0`：native LeWM；
- `A3`：同一 LeWM 实现路径，换成 PLDM-active objective；
- `A4`：native PLDM 实现参照。

三臂 step-0 的 `delta_history/delta_target/delta_prediction` 逐元素相同。相对 A0，A3 的 step-0
沿 response-residual 有利方向的一阶变化绝对值从 `-135.214` 变为 `-6569.057`（倍数 `48.583`），
全参数 total-gradient 范数比为 `2.801`，因此**按总梯度范数归一化后的方向效率约 `17.35x`**；
A3 与 A4 在该一阶量上的相对差只有 `7.47e-6`。这把**首个局部分离**定位到 objective/梯度路由，
即 **objective/regularizer-induced gradient routing** 的强局部支持。

必须同时保留的三条边界：

1. **scale caveat**：一阶量是局部线性化，其绝对尺度依赖各 objective 的整体 loss scale；`48.583x`
   不能读成“48 倍学习速度”，归一化后的 `17.35x` 才是方向质量口径，且两个 objective 的 loss 单位
   不同，这不是 step-size 受控的因果幅度；
2. **不是唯一瓶颈、不是行为反转**：三臂 step-256 的 signed gain 仍全为负，这是 bounded local
   mechanism gate；不得称目标路由是 ActionDelay 的唯一瓶颈，也不得称观察到 native behavior reversal；
3. **raw data / initial latent 是本门中的固定量**：三臂的数据与 step-0 latent 逐元素相同，属于受控
   常量，因此它们不可能是本门内的分离来源——这**不**等于数据在 ActionDelay 或其他任务中被普遍否定。

这里的共同初始化只指 shared core；不假设 A0/A4 的完整参数集合完全同构，也不使用跨模型绝对阈值。

到 step 256，A3/A0 的 target-latent matched-to-unrelated ratio 为 `1.702x`；三 seed 均值为：

| arm | target ratio | signed gain | cosine | NRE |
|---|---:|---:|---:|---:|
| A0 native LeWM | `0.0798` | `-0.0100` | `-0.7325` | `1.0102` |
| A3 LeWM + PLDM objective | `0.1359` | `-0.0033` | `-0.2374` | `1.0035` |
| A4 native PLDM | `0.1402` | `-0.0030` | `-0.1170` | `1.0031` |

三个 arm 的 signed gain 仍为负，所以这只是 bounded local mechanism gate，不能冒充完整训练后的
history-use 或泛化成功。它支持“ActionDelay 的 objective/Jacobian 路由可成为限制层”，不支持
“PLDM 在所有任务都更好”。本汇总的复现层级是 `L1` 聚合可重复（`30` 个冻结输入下 `summary.json`
与 `per_cell.jsonl` 字节一致，`model_loaded=false`、`optimizer_steps=0`），没有重跑任何上游训练，
因此不是训练复现、统计复现或因果复现；配套单元测试属于**软件合同证据**，不是科学结论证据。

三 seed 的原生 1,024-step LeWM/PLDM endpoints 已完成只读盘点，但其外部 `config.json` 没有直接
记录 initialization checkpoint，且当前没有同 probe 的 endpoint latent/gradient 回执。因此它们
不进入本门的共同初始化 claim。若以后需要把局部路由一直闭合到原生终点，应另建 endpoint identity
gate；当前 D2-0 决策不以这项可选扩展为前置条件。

Action Strength 的 exact-batch 反向检查没有形成确认，该单元当前是 **unlocalized downstream
bottleneck**：LeWM/PLDM 使用各自预训练初始化，condition-pair prediction coherence 分别为
`0.2365/0.2420`，两者 native total gradient 在初始化都局部提高 signed gain；PLDM 终点在该 Training
batch 上甚至达到 gain/NRE=`0.7337/0.2203`，却对应冻结的 held-out 负标签。因此未解释部分位于该
Training batch 的**下游**；`coverage / 迁移 / 校准` 只是**候选集合**，没有证据把限制层指向其中任何
一项。不能为追求对称而事后选择跨模型阈值。

## 5. 两个不能混写的数据检验

### 5.1 曝光分布检验：选择哪些样本

在 Motion 的同一 Training twin 池内构造三种相同权重形状的 soft exposure。三者都保持完整
forward/reverse x hidden-mode twin、每个 coverage cell 的总质量、所有样本正权重、总 exposure、
有效样本量和训练预算不变，只重新分配每格内同一组 rank weights：

| arm | rank 依据 | 回答的问题 |
|---|---|---|
| `REL50` | 多尺度局部相对条件份额 | 提高相对条件份额能否进入 latent 和参数梯度 |
| `ABS50` | 绝对条件 gap energy | 只追求大响应是否同时抬高背景变化而失去相对优势 |
| `HASH50` | 不读取任何物理量的冻结哈希 | 收益是否只是 soft 重复、权重集中或 coverage 模板造成 |

已有自然等权 `D0` 只作锚点，不重跑数据构造。`ABS50` 是机制 comparator，不称 placebo；
`HASH50` 才是 exposure placebo。数据选择只使用 Training 物理量，禁止使用模型梯度、Development
结果或隐藏标签泄漏特征调 rank。哈希并不因定义就保证有限样本中与物理量零相关；必须报告它与
`C_phys`、`B_loc` 和相对分数的秩相关，若偶然相关过强则判 comparator 无效，不能另搜 seed。

有效 comparator 必须在运行前满足：同一 rank-weight multiset、每格质量、pair/twin 完整性、行数、
总步数和正权重下限完全一致。运行后还要报告而非事后修补 query/action、绝对条件能量、背景变化、
非条件梯度和参数组构成的差异。若某个非处理量超出预注册等价界限，该 comparator 失效，不能用
“效果不显著”掩盖失衡，也不能依据梯度重新搜索哈希 seed。

#### 5.1.1 Training-only 构造结果

三臂已按冻结 seed `20260901` 构建，全部 multiset、coverage、质量、ESS、熵和正支持不变量通过。
`HASH50` 与条件能量秩的全局 Spearman 相关为 `0.0140`，与 `REL50` 秩相关为 `0.0219`，未出现
明显偶然对齐。三个尺度的 `rho_phys` 如下：

| arm | `k=32` | `k=64` | `k=128` | 加权 `C_phys` |
|---|---:|---:|---:|---:|
| `D0` | 0.13580 | 0.10358 | 0.07736 | 2.14589 |
| `HASH50` | 0.13598 | 0.10367 | 0.07746 | 2.14674 |
| `ABS50` | 0.13866 | 0.10593 | 0.07920 | **2.18794** |
| `REL50` | **0.14251** | **0.10830** | **0.08036** | 2.16826 |

`ABS50` 的绝对 `C_phys` 比 `REL50` 高 `0.91%`，但三个尺度的相对份额反而低
`1.46%--2.78%`；`REL50` 相对 `HASH50` 的 `rho_phys` 高 `3.75%--4.80%`。因此构造阶段已经
验证“绝对高能不等于相对条件可见性”，并支持将相对份额作为下一门的处理变量。它仍只是物理数据
层结果；是否穿过 latent 与梯度，必须由零步审计回答。

四臂整数 schedule 也已冻结。三处理臂共享每个 batch 的抽象 `(coverage cell, rank-weight slot)`、
自然半批、步数和随机种子；只把同一权重 rank 映射到各臂定义的 twin。每臂均为 `8,192` batches、
每 batch `16` twins，全部 69 个完整性门通过；三处理臂的 full-distribution TV integerization error
均为 `0.004009`。随后冻结零步门通过：D1 相对 D0 的 local `rho_lat` 提高
`1.15%--1.52%`，all-parameter response mean gradient norm 提高 `6.94%`，`SNR(16)` 提高
`4.56%`。唯一 D1 native 完整训练只产生小幅正确方向效应，仍不是历史使用正例。因此同池 D1
已经完成并冻结，不能继续把增加重复率当作下一解法。

### 5.2 Batch 组织检验：如何组成一次更新

第二个检验固定完全相同的样本多重集，只改变 batch partition：相似 query/response signature 的
`COHERENT`、自然顺序 `NATURAL`、最大异质 `DIVERSE` 和冻结哈希 `HASH`。

这里有一个重要不变性：若模型没有 BatchNorm、随机增强等跨样本耦合，且逐样本梯度在同一参数点
独立计算，那么重排不会改变全体 `E[g]`、全局 coherence 或全局 `Bcrit`。它只能改变每批的平均
梯度、批内相干性和后续有限步优化轨迹。LeWM 的 `pred_proj` 含 BatchNorm，train-mode 结果还可能
包含真实 batch coupling；必须用共同 RNG 多次复算，并以 eval-mode 不变性检查区分它与随机噪声。

因此取消原先 `A- < A0 < A+` 的**全局** `Bcrit` 单调门。该检验只报告：

- 每批 `c_b=||mean_i g_i|| / mean_i ||g_i||`；
- 每批条件梯度均值范数及批间分布；
- 批间方向余弦和训练顺序自相关；
- eval-mode 全局矩量的不变性；
- train-mode BatchNorm/Dropout 引入的差值及共同 RNG 不确定性。

它最多支持“batch 组织改变了有限步优化动力学”，不能支持“样本曝光分布变好了”。首轮不为该
schedule 单独开训练网格；它只决定 D2 训练是否采用预注册的 coherent batching。

## 6. 幅值弱与梯度抵消

两个表型按同单元、同参数组判别：

| 观测 | 解释 |
|---|---|
| `E||g_i||` 小，coherence 未明显更低 | 单样本条件幅值/Jacobian 弱 |
| `E||g_i||` 尚可，但 `||Eg_i||`、`c` 和 SNR 很低 | 不同 pair/query 的梯度抵消 |
| `rho_phys` 升、`rho_lat` 不升 | target 表示压缩 |
| `rho_lat` 升、单样本梯度不升 | loss route/Jacobian 衰减 |
| 单样本梯度升、批内相干不升 | 数据更强但方向仍互相抵消 |
| 零步梯度改善、held-out `G_swap` 不改善 | coverage、训练动力学或校准仍是 binding cause |

`Bcrit` 和 SNR 是描述同单元梯度总体的量，不是跨模型门槛。至少同时看 `E||g_i||` 与
`||Eg_i||`，否则无法区分信号本身弱和强信号相互抵消。

## 7. 预注册 pass/kill 条件

### 7.1 Motion 零步门

`REL50` 只有同时满足以下条件才允许进入一次匹配短训：

1. 构造有效：相对 `ABS50` 和 `HASH50`，`rho_phys` 按预期提高；全部 balance/integrity 门通过；
2. 表示传输：同一冻结 encoder 下 `rho_lat` 相对 comparator 同向提高；
3. 梯度传输：`predictor` 或全参数的条件均值梯度、coherence/SNR 至少一组给出配对正方向，且
   `pred_proj` 不出现方向相反的明确证据；
4. 特异性：`REL50` 必须优于 `HASH50`；若只与 `D0` 不同，不能排除重复和权重集中；
5. 稳健性：结论不由单一 batch、单一 RNG 或少数 top-mass twins 决定。

任一完整性或 placebo 有效性门失败，先修构造，不训练。构造有效但数据到梯度传输未通过，则 kill
当前同池数据路线，直接转向 D2 新轨迹或表示瓶颈，不搜索 mixing weight。

### 7.2 Motion 短训门

最多训练 `REL50` 与一个预先冻结的 comparator，保持 checkpoint、optimizer、总 transition、步数、
自然数据锚点与自然 Development 完全一致。只有 on-support `G_swap`、gain/alignment 和 NRE 的配对
方向共同改善，且原始任务保持性没有明确下降，才能称数据干预对 history-use 有因果贡献。

零步通过但短训不通过，结论是“梯度几何可操纵但不足以形成可迁移行为”，不是数据方案成功。

### 7.3 反转单元模型侧门

该门现已完成。ActionDelay 在共同 probe/shared core 上形成同实现 objective-route 分离；Action
Strength 的反向 exact-batch 检查没有给出 outcome-aligned 确认，仍是 unlocalized downstream
bottleneck。因此当前结论不是一个跨任务梯度阈值，而是：**限制层会随 task/model/init 改变**。
Motion 的低上游份额与 D1 方向性数据干预证据足以授权一个 D2 数据干预；ActionDelay 证明不能把该数据
结论推广为普遍充分解释。

D2 只在 Motion 上先做 Training-only 新轨迹构造门，不立即训练：必须实际提高 relative target
separation、引入非零 state-dependent action leverage 和新 query coverage，并保留自然幅值锚点。
**D2 获授权不等于预设成功**：构造门未通过即停止；通过后至多运行一个 native 短训格，其结果同样可能
为负。不得增加第三任务训练、loss 权重搜索或 COJA+rollout。

Door 只做一次 on-support swapped-history necessity 检查。若交换历史不损害正确 future，它从 native
正对照中移除，不据此改写 Motion 或 Action Strength 的结果。

## 8. Claim 分级

| 最终结果 | 最强允许 claim |
|---|---|
| 只有 Motion 零步通过 | 在 Motion 支持域内，条件梯度几何可由数据构造操纵 |
| Motion 零步与短训通过，Action Strength 迁移失败 | 数据构造在 Motion 中是条件充分因素；不能称普遍根因或通用解决方案 |
| Motion 与 Action Strength 均通过零步和短训 | 有效条件可见性在两个预注册模型-任务单元中具有可迁移的因果贡献；仍不证明必要性或单一标量根因 |
| 两者均未通过数据到梯度门 | 数据优先假设被削弱，应转向表示/Jacobian 或显式条件目标；COJA 正对照仍成立 |

即使两个单元都通过，也只能说一套**诊断与数据构造原则**具有迁移证据，不能说所有历史忽略都由
低 `rho_phys` 引起。通用性来自相同的因果检查方法和可迁移干预方向，不来自共享一个绝对阈值，
更不来自“多数 cell 结果一致”——矩阵是 factorial/triangulation，不是独立复制（§2）。
逐条 claim 的当前状态见 [`CLAIMS_LEDGER_ZH.md`](CLAIMS_LEDGER_ZH.md)。

## 9. 最小执行顺序

1. **已完成**：冻结反转矩阵与四任务 Training-only physical/raw-pixel 上游量；
2. **已完成**：Motion `REL50/ABS50/HASH50`、零步门和唯一 D1 native 因果格；结论为数据因素有
   小幅方向作用，但同池重加权不充分；
3. **已完成**：ActionDelay A0/A3/A4 共同 probe/shared-core 的 latent + objective-gradient 门；
4. **已完成**：Action Strength exact-batch 反向检查；结果为不确认统一零步阈值，该单元记为
   unlocalized downstream bottleneck；
5. **下一步**：只规划并构建 Motion D2-0 新轨迹候选（已获授权，未预设成功），提高 relative target
   separation、非零 action leverage 与 query coverage，同时保留自然幅值锚点；具体合同见
   [`D2_CONSTRUCTION_PLAN_ZH.md`](D2_CONSTRUCTION_PLAN_ZH.md)；
6. D2-0 与 D2-1 均通过后最多运行一个 D2 native 匹配预算训练格，随后冻结结论，再更新论文与最终技术文档。

现阶段不访问公开 Test，不增加 COJA loss 变体，不做跨任务全量对称审计，也不继续调 D1 重复率。

## 10. 冻结资产

- 承重 claim 台账：[`CLAIMS_LEDGER_ZH.md`](CLAIMS_LEDGER_ZH.md)
- D2 新轨迹预注册：[`D2_CONSTRUCTION_PLAN_ZH.md`](D2_CONSTRUCTION_PLAN_ZH.md)
- D2 机器可读合同：[`configs/pusht_motion_damping_d2_preregistration_v1.yaml`](configs/pusht_motion_damping_d2_preregistration_v1.yaml)
- 反转矩阵配置：[`configs/icl_root_cause_reversal_matrix_v1.yaml`](configs/icl_root_cause_reversal_matrix_v1.yaml)
- 只读审计器：[`scripts/audit_icl_root_cause_reversal_matrix_v1.py`](scripts/audit_icl_root_cause_reversal_matrix_v1.py)
- 矩阵清单（`8` cells，four-layer-complete `1` 个，`all_cells_four_layer_complete=false`）：[`artifacts/icl_root_cause_reversal_matrix_v1/inventory_v1.json`](artifacts/icl_root_cause_reversal_matrix_v1/inventory_v1.json)
- Motion 曝光 comparator：[`artifacts/pusht_motion_damping_root_cause_comparators_v1/comparators_v1_final/summary.json`](artifacts/pusht_motion_damping_root_cause_comparators_v1/comparators_v1_final/summary.json)
- comparator 构建器：[`scripts/build_pusht_motion_damping_root_cause_comparators_v1.py`](scripts/build_pusht_motion_damping_root_cause_comparators_v1.py)
- 四臂 schedule：[`artifacts/pusht_motion_damping_root_cause_comparator_schedules_v1/comparator_schedules_v1_final/summary.json`](artifacts/pusht_motion_damping_root_cause_comparator_schedules_v1/comparator_schedules_v1_final/summary.json)
- schedule 构建器：[`scripts/build_pusht_motion_damping_root_cause_comparator_schedules_v1.py`](scripts/build_pusht_motion_damping_root_cause_comparator_schedules_v1.py)
- 四臂零步分析器：[`scripts/analyze_pusht_motion_damping_root_cause_zero_step_v1.py`](scripts/analyze_pusht_motion_damping_root_cause_zero_step_v1.py)
- native 反转门配置：[`configs/icl_native_reversal_mechanism_v1.yaml`](configs/icl_native_reversal_mechanism_v1.yaml)
- native 反转门汇总器：[`scripts/summarize_icl_native_reversal_mechanism_v1.py`](scripts/summarize_icl_native_reversal_mechanism_v1.py)
- native 反转门结果（复现层级 `L1`：`30` 个冻结输入，`summary/per_cell` 字节一致）：[`artifacts/icl_native_reversal_mechanism_v1/training_only_v3/summary.json`](artifacts/icl_native_reversal_mechanism_v1/training_only_v3/summary.json)
- 通用指标：[`scripts/conditional_signal_metrics.py`](scripts/conditional_signal_metrics.py)
- D1 完整记录：[`D1_CONSTRUCTION_PLAN_ZH.md`](D1_CONSTRUCTION_PLAN_ZH.md)

所有新增 artifact 必须拒绝覆盖，记录源码与输入 SHA、optimizer step、split read、checkpoint 前后状态和
RNG 回执。历史公开 scoreboard 可以复核标签，但不得进入根因统计或配方选择。
