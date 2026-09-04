# 条件 ICL 承重 claim 台账（claims ledger）

状态：2026-09-04。本文件是本研究**承重 claim 的唯一台账**：每条 claim 写清它说什么、在哪个域
成立、由哪个 artifact 支撑、复现到哪一层、还有哪些替代解释没有被排除，以及什么证据会把它降级。
README、`ROOT_CAUSE_DATA_STRATEGY_ZH.md`、`ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md`、
`D1_CONSTRUCTION_PLAN_ZH.md` 与 `EXPERIMENT_LOG_ZH.md` 中的措辞必须与本表一致；出现分歧时以本表
为准并同步修正对应文档。

当前总体状态：**第一轮机制边界界定完成；统一低-`rho` 充分解释已证伪；Motion D2-v1 已在
P1b calibration 结构门有效失败并关闭；通用根因未闭合。**

## 0. 使用规则

**字段。** 每条 claim 固定十项：`statement`、`scope`、`evidence type/artifact`、`seed/eval unit`、
`uncertainty`、`gate`、`reproduction level`、`alternatives not excluded`、`downgrade trigger`、
`stated_in`。字段缺失即视为未通过审阅，不允许写“同上”。

**状态。**

| 状态 | 含义 |
|---|---|
| `CLOSED` | 在写明的 scope 内已可依赖；再做同类实验不改变结论，只可能扩大 scope |
| `PARTIAL` | 方向或局部机制有匹配证据，但预注册门未全通过，或只在一个单元/一个训练种子上成立 |
| `OPEN` | 只有现象层标签或候选集合，尚无定位到某一层的匹配证据 |

**复现层级。** 只用下面四级，不得混写：

| 级别 | 含义 |
|---|---|
| `L1` | **聚合可重复**：在冻结输入上重跑同一汇总/分析代码，产物字节一致。不含重跑上游训练或评测 |
| `L2` | **计算/评测可重复**：同 checkpoint、同评测集独立重跑，产物字节一致 |
| `L3` | **训练可复现**：多个 training seed 重跑训练本身并给出同向结论 |
| `L4` | **跨单元复现**：在另一个 task-model 单元上按同一预注册协议复现 |

**维护规则。** 新证据只修改对应字段并更新日期；`downgrade trigger` 命中时必须先降级本表，再改其他
文档；claim 被证伪时保留条目并标注证伪证据，不删除历史。逐日记录仍在 `EXPERIMENT_LOG_ZH.md`，
本表不复述过程。

## 1. 索引

| ID | 状态 | 一句话 |
|---|---|---|
| C1 | CLOSED | native paired MSE 在数学上包含条件项，但不单独保护它 |
| C2 | CLOSED | COJA 是有效方法与可学习性正对照：Motion matched 对照上把 gain/NRE 从 `0.0065/1.1298` 改到 `0.3696/0.7665` |
| C3 | CLOSED | 扩展历史窗口与 `K=3` native rollout 都不能替代 COJA |
| C4 | CLOSED | “统一低-`rho` 充分解释”已被两组同数据跨模型反转证伪 |
| C5 | OPEN | 通用根因未闭合；`effective conditional visibility` 只是候选机制陈述 |
| C6 | PARTIAL | Motion-LeWM 的上游条件份额弱且条件梯度可见性低 |
| C7 | PARTIAL | D1-MS50 给出方向性数据干预证据（upstream-to-gradient directional transmission），不是 native behavior reversal |
| C8 | PARTIAL | ActionDelay：objective/regularizer-induced gradient routing 的强局部支持 |
| C9 | CLOSED | ActionDelay physical `rho=1` 是退化分母，退出跨任务强弱比较 |
| C10 | OPEN | Action Strength 是 unlocalized downstream bottleneck |
| C11 | PARTIAL | ActionDelay + COJA：assignment 已学会，response calibration 未闭合 |
| C12 | CLOSED | 反转矩阵是不同角色的 factorial/triangulation，其模型侧汇总只到 `L1` |
| C13 | PARTIAL | Motion D2 的受控接触物理路由在双方向 P1a 上局部可行；该局部结果没有外推到 P1b 全覆盖 |
| C14 | CLOSED | Motion D2-v1 在 P1b calibration 结构门有效失败：`low_approach` 的 64 个 coverage slots 全部缺失，当前 revision 关闭 |

## 2. claim 明细

### C1 native paired MSE 不单独保护条件响应（CLOSED）

- **statement**：对共享 `(Q,A)` 的二元条件组，配对平方误差精确分解为组中心项与响应项，
  `G_swap = L_swapped - L_correct = <Delta p, Delta t>`。条件响应项**在数学上存在于 native
  目标中**，但没有任何机制保证优化器优先降低它。
- **scope**：所有使用逐样本平方误差、且训练集含共享 `(Q,A)` 条件组的世界模型训练；与具体任务无关。
- **evidence type/artifact**：解析恒等式，见 `ROOT_CAUSE_DATA_STRATEGY_ZH.md` §3；数值一致性由
  `scripts/conditional_signal_metrics.py` 在 Motion 上核对。
- **seed/eval unit**：不适用（恒等式）。
- **uncertainty**：无统计不确定性；仅受“共享 `(Q,A)` 条件组存在”这一前提约束。
- **gate**：无需门控；它是其他 claim 的记账基础。
- **reproduction level**：`L1`（解析结论，数值核对可重跑）。
- **alternatives not excluded**：不排除其他损失族（非平方、rollout、对比式）有不同的分解结构。
- **downgrade trigger**：发现实际训练目标不等于所写的 paired MSE（例如额外归一化改变了分解）。
- **stated_in**：README §4；`ROOT_CAUSE_DATA_STRATEGY_ZH.md` §3；`EXPERIMENT_LOG_ZH.md` §5.47。

### C2 COJA 是有效方法与可学习性正对照（CLOSED）

- **statement**：在 Motion Damping、同数据同预算 matched 对照（8,192 steps）上，native 的
  Development gain/NRE 为 `0.0065/1.1298`、mean `G_swap=4.67e-5`、正号比例 `44.1%`；COJA 为
  `0.3696/0.7665`、`2.666e-3`、`96.9%`。因此数据中的条件信号**可学习**，失败不是“任务不可学”。
  COJA 在全部文档中是有效方法与正对照，**不得**被描述为失败方案。
- **scope**：Motion Damping + LeWM，冻结 benchmark v1 自然 Development；同一 target encoder。
  跨任务另有 Portal Exit 等 matched 正例，但不由本条承担。
- **evidence type/artifact**：matched 训练对照 + 冻结 Development 评测；
  `artifacts/native_conditional_signal_root_cause_v1/motion_damping/{native,coja}_s14321_step8192_development_v1.json`。
- **seed/eval unit**：单 training seed `14321`；评测单元为 256 个冻结 Development query pairs。
- **uncertainty**：逐 query sign-flip：native `p=0.420`、COJA `p=0.000488`（Monte Carlo 下限）；
  cross-query null：native `p=0.177`、COJA `p=0.000488`。这些只刻画 query 抽样，不刻画训练种子。
- **gate**：已通过“matched 方法效应存在”的门；未通过“COJA 全任务达硬门”的门（见 C11）。
- **reproduction level**：`L2`（同 checkpoint 评测可复现）；training seed 维度仍为 `L3` 未达成。
- **alternatives not excluded**：不排除 COJA 收益部分来自训练期额外前向带来的正则/优化副作用，而非
  条件监督本身；未做逐组件消融。
- **downgrade trigger**：matched native 对照在同一评测器下复算出显著更高的 gain/NRE，或发现
  Development pairs 泄漏条件标签。
- **stated_in**：README §3--§4；`ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1、§4.2；
  `ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §4。

### C3 扩展历史与 native rollout 不能替代 COJA（CLOSED）

- **statement**：同 seed、同完整预算的 Motion Development 机制实验中，把历史窗口由滑动改为动态扩展
  （NRE `1.0089`）、只用原生损失做 `K=3` 多步 rollout（NRE `1.0018`），都没有形成可用条件响应；
  单步 COJA 为 `0.9579`。`NRE≈1` 的物理含义是预测几乎不随历史变化。
- **scope**：Motion Damping + LeWM，单训练种子、Development-only 机制实验。
- **evidence type/artifact**：matched 机制对照，见 `EXPERIMENT_LOG_ZH.md` §5.46 与 2026-08-30 条目。
- **seed/eval unit**：单 training seed；Development query pairs + 6 个标准 PushT CEM seed × 50。
- **uncertainty**：未做逐 query bootstrap；结论依赖 `NRE≈1` 与随机水平 future/history rate 的量级差。
- **gate**：作为“更简单解释”的 kill 门，已通过。
- **reproduction level**：`L2`。
- **alternatives not excluded**：不排除更长 rollout、不同 `K`、不同 history 编码方案有不同结果；本条
  只否定这两个具体替代方案。
- **downgrade trigger**：在同预算 matched 条件下出现 expanding 或 native rollout 达到 COJA 量级的
  gain/NRE。
- **stated_in**：README §4；`ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1；`EXPERIMENT_LOG_ZH.md` §5.46。

### C4 统一低-`rho` 充分解释已证伪（CLOSED）

- **statement**：“所有 native ICL 失败都由原始数据条件份额 `rho` 低造成”这一充分解释**已被证伪**：
  同一物理数据上出现两组方向相反的跨模型反转——ActionDelay 为 LeWM 负/PLDM 正，Action Strength 为
  LeWM 正/PLDM 负；Door 两者均正。上游条件份额不能同时解释这两组结果。
- **scope**：对“单一上游标量充分解释”的证伪；不否定 `rho` 在 Motion 等单元中是可信上游因素（见 C6）。
- **evidence type/artifact**：冻结历史 outcome 标签 +
  `artifacts/icl_root_cause_reversal_matrix_v1/inventory_v1.json`；上游量见
  `artifacts/icl_training_conditional_visibility_v1/training_only_v2/` 与
  `artifacts/icl_training_raw_pixel_visibility_v1/training_only_v1/`。
- **seed/eval unit**：outcome 标签为既有发布 scoreboard 的冻结行（只作历史标签）；上游审计为
  Training-only，optimizer step=`0`。
- **uncertainty**：反转的方向性由离散 outcome 标签支撑，不依赖阈值；上游量的跨任务绝对值不可比。
- **gate**：证伪型 claim，无需通过正向门。
- **reproduction level**：`L1`（上游审计聚合可重复，独立 CPU 复跑 `per_task.jsonl` 字节一致）。
- **alternatives not excluded**：未排除“存在另一个尚未测量的统一上游量”；本条只否定当前 `rho` 口径的
  充分性，不证明数据层与结果无关。
- **downgrade trigger**：发现某组反转的 outcome 标签本身有配置/评测错误，使反转消失。
- **stated_in**：README §4；`ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1.1--§1.2；
  `ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §4.1；`EXPERIMENT_LOG_ZH.md` §5.62。

### C5 通用根因未闭合（OPEN）

- **statement**：当前**没有**已确立的通用根因。可写的最强表述是候选机制陈述：历史条件在优化中的
  **有效可见性**可能在 raw target、训练表示、loss/Jacobian 路由、跨 pair/query 梯度聚合、query 覆盖
  或响应校准的任一层衰减，且 binding layer 依赖具体 task/model/init。第一轮完成的是**机制边界界定**，
  不是根因收口。
- **scope**：全项目层面的元 claim；约束其他所有文档的措辞。
- **evidence type/artifact**：由 C4（证伪）、C6/C7（Motion 数据段）、C8（ActionDelay 目标路由段）、
  C10（Action Strength 未定位）共同界定；无单一 artifact。
- **seed/eval unit**：不适用。
- **uncertainty**：候选机制陈述可证伪但当前未被检验为定理；层间权重未知。
- **gate**：闭合条件为：至少两个 task-model 单元上，`rho_phys -> rho_lat -> V_grad -> 自然 held-out
  G_swap` 全链由预注册匹配干预闭合，且多 training seed 复现。当前 `0/2` 通过。
- **reproduction level**：不适用（元 claim）。
- **alternatives not excluded**：单层充分解释（纯数据、纯目标、纯覆盖）均未被逐一排除，只被证明
  不能单独覆盖全部单元。
- **downgrade trigger**（此处为升级条件）：上述闭合条件达成后，本条改写为具体机制 claim。
- **stated_in**：README §1、§4--§5；`ROOT_CAUSE_DATA_STRATEGY_ZH.md` 标题与 §1.1；
  `ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §1、§8。

### C6 Motion-LeWM 上游条件份额弱、条件梯度可见性低（PARTIAL）

- **statement**：Motion Damping 上，条件能量在冻结 latent 的无条件全局背景口径下只占目标方差的
  `0.23%--0.29%`；Training 原始像素局部口径 `rho_pixel,64=0.0929 [0.0896,0.0962]` 为四任务最低；
  native 终点 response 项占配对 MSE `13.62%`，但映射到真实 optimizer 参数后 response/非条件梯度
  范数比只有 `5.57%`、平方能量占比 `0.309%`、32-pair SNR 约 `1.10`。
- **scope**：Motion Damping + LeWM 的冻结 Training batch 与冻结 Development；口径不跨任务比较。
- **evidence type/artifact**：冻结 batch 零步梯度审计 +
  `artifacts/native_conditional_signal_root_cause_v1/motion_damping/*_train_gradient_snr_v1.json`；
  像素量见 `artifacts/icl_training_raw_pixel_visibility_v1/training_only_v1/`。
- **seed/eval unit**：单初始化/单 native 终点 checkpoint；梯度统计单元为 32 个 condition pairs（同
  twin 内两 pair 非独立，`Bcrit` 为该 batch 的 population estimate）。
- **uncertainty**：`rho_pixel` 有 cluster-bootstrap 区间；梯度 SNR 无 twin-cluster 区间，只作线索。
- **gate**：支持“Motion 上游弱”的描述性门已通过；**不**支持把它写成 Motion 失败的唯一 binding cause。
- **reproduction level**：`L2`。
- **alternatives not excluded**：表示压缩、Jacobian 路由、query 覆盖与校准都可能同时是限制层；
  已知 `pred_proj` 路径最弱（终点 `r_grad=1.60%`）。
- **downgrade trigger**：在保持 `rho_phys/rho_lat` 不变的模型侧干预下 Motion 学会条件响应，则
  上游份额在该单元不是 binding constraint。
- **stated_in**：README §4；`ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1、§4.1--§4.2；
  `ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §4。

### C7 D1-MS50：方向性数据干预证据（PARTIAL）

- **statement**：在同一 Motion Training twin 池内，D1-MS50 把三尺度 `rho_phys` 提高 `3.87%--4.95%`，
  冻结初始化上 local `rho_lat` 提高 `1.15%--1.52%`、all-parameter response mean gradient norm 提高
  `6.94%`、twin-cluster `SNR(16)` 由 `0.3244` 提高到 `0.3392`（`+4.56%`）；完整 native 训练后自然
  Development 的 `Delta G_swap=+6.43e-5`（95% CI `[+3.79e-5,+9.16e-5]`、sign-flip `p=1.00e-5`）、
  `Delta gain=+0.00739`、`Delta NRE=-0.01362`。准确命名是**方向性数据干预证据 /
  upstream-to-gradient directional transmission**；它**不是** native behavior reversal，也不是
  history-use 正例。
- **scope**：Motion Damping + LeWM，同池 50% soft 重加权，单训练格；不涵盖新轨迹、动作 leverage
  或其他任务/模型。
- **evidence type/artifact**：预注册数据干预（构造门 → 零步门 → 唯一 native 训练格 → 冻结自然
  Development）；`artifacts/pusht_motion_damping_d1_multiscale_soft_v2/` 及
  `D1_CONSTRUCTION_PLAN_ZH.md` §3.6/§6.2.1/§7.2.1 中的 SHA256。
- **seed/eval unit**：单 training seed `14321`；256 个冻结 Development query pairs（paired 比较）与
  `4,096` twins 等权 Training panel；零步门为 16 个预注册 batch、twin 为 cluster。
- **uncertainty**：所有区间均为 query/twin 抽样 bootstrap，不跨训练种子；CEM300 配对效应
  `-1.33pp [-5.67,+3.00]pp`，既不支持保持性受损也未证明非劣。
- **gate**：状态冻结为 `single-seed directional data effect=true, history-use positive=false`。D1 自身
  正号比例 `0.449`、sign-flip `p=0.0756`、NRE `1.1140>1`，三项主门未过。
- **reproduction level**：`L2`（各阶段独立复跑字节一致）；`L3` 未达成。
- **alternatives not excluded**：不排除该方向效应来自曝光带来的背景分母下降而非条件压力上升的一部分；
  也不排除同池重加权的效应上限本身就低于 history-use 门。D2 已获授权，但**不预设成功**。
- **downgrade trigger**：第二个 training seed 的 paired `Delta G_swap` 区间跨零，或复算发现
  Development 指标受 exposure 构造性放大污染。
- **stated_in**：README §4、§6；`D1_CONSTRUCTION_PLAN_ZH.md` §1、§7.2.1、§8；
  `ROOT_CAUSE_DATA_STRATEGY_ZH.md` §4.6。

### C8 ActionDelay：objective/regularizer-induced gradient routing 的强局部支持（PARTIAL）

- **statement**：在同一 ActionDelay Training probe、同一 shared-core 初始化、每 seed 逐 rank 完全
  相同的 logical batches 下（step-0 `delta_history/delta_target/delta_prediction` 逐元素相同），把
  A0 的 native LeWM objective 换成 PLDM-active objective（A3，同一 LeWM 实现路径）后，step-0 沿
  response-residual 有利方向的一阶变化**绝对值**从 `-135.214` 变为 `-6569.057`（**倍数** `48.583x`），
  同时 total-gradient 范数比为 `2.801`；因此**按总梯度范数归一化后的方向效率约 `17.35x`**。A3 与
  native PLDM A4 在该一阶量上的相对差仅 `7.47e-6`。
- **scope**：ActionDelay 单元的 step-0/256-step 局部机制门；bounded local mechanism evidence。
- **evidence type/artifact**：只读 Training-only 机制汇总，
  `artifacts/icl_native_reversal_mechanism_v1/training_only_v3/`（`summary.json` SHA256
  `85cf6989c158316f263043646ce1ee2e9d802fff295e0b67e77137bfe337331b`）。
- **seed/eval unit**：三 training seeds `3072/4096/5120`；每 seed 固定 8-rank logical batches，
  记录 steps `0/1/4/16/64/256`；optimizer step 在本审计中为 `0`（数值取自冻结训练侧机制资产）。
- **uncertainty**：报告 min/mean/max 而非区间；一阶量是局部线性化，**其绝对尺度依赖各 objective 的
  整体 loss scale**，故 `48.583x` 不能读成“48 倍学习速度”，归一化后的 `17.35x` 才是方向质量口径，
  且两个 objective 的 loss 单位不同，不是 step-size 受控的因果幅度。
- **gate**：`model_side_objective_route_separation_supported`；行为 claim 明确为
  `not_established_by_256_step_training_probe`——三臂 step-256 signed gain 仍全为负
  （A0 `-0.0100`、A3 `-0.0033`、A4 `-0.0030`）。
- **reproduction level**：`L1`（见 C12）。
- **alternatives not excluded**：**不称目标路由是 ActionDelay 的唯一瓶颈**，也不称观察到行为反转；
  coverage、表示与校准仍可能是并行限制层。raw data 与 initial latent 在本门中是**逐元素固定的受控量**
  （因此不可能是本门内的分离来源），这不等于说数据在 ActionDelay 或其他任务中被普遍否定。共同初始化
  只指 shared core，不假设 A0/A4 完整参数集合同构。
- **downgrade trigger**：换用 step-size 或 loss-scale 受控的对照后归一化方向效率落回 `~1x`；或原生
  1,024-step endpoint 身份门显示 A0/A3 并非共同初始化。
- **stated_in**：README §4；`ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §4.2；
  `ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1.3；`EXPERIMENT_LOG_ZH.md` §5.63。

### C9 ActionDelay physical `rho=1` 是退化分母（CLOSED）

- **statement**：四任务上游审计中 ActionDelay 的 `rho_phys,64=1.0000` 来自**退化分母**：该任务的
  `B_32/B_64/B_128` 在全部 `5,120` 个 query 上恒为 `0`，且 `C_phys` 为常数 `35.7292`
  （std `7.1e-15`），故 `rho=C/(C+0)=1` 是构造结果而非“条件份额极高”。该数值**退出**跨任务强弱比较。
  保留可用的是 `C_phys` 的描述性数值与 raw-pixel `rho_pixel,64=0.3596 [0.35945,0.35969]`；
  raw-pixel 量同样**不设跨任务阈值**（renderer、条件基数与邻域均不同）。
- **scope**：ActionDelay 上游审计的度量口径；影响所有引用该表的段落。
- **evidence type/artifact**：`artifacts/icl_training_conditional_visibility_v1/training_only_v2/per_task.jsonl`
  （ActionDelay 的三个 `B_k` 分布 mean/std/分位数全为 `0`）。
- **seed/eval unit**：Training-only，`5,120` 个 observed 11-condition group queries；raw-pixel 子样本
  `256` queries。
- **uncertainty**：`rho_pixel` 有 cluster-bootstrap 区间；`rho_phys` 无意义区间（分母恒零）。
- **gate**：度量有效性门失败 → 该指标在本任务上不可用于强弱排序；不阻塞其他 claim。
- **reproduction level**：`L1`（独立 CPU 复跑 `per_task.jsonl` 字节一致）。
- **alternatives not excluded**：不排除换一个非退化的局部背景定义后 ActionDelay 能重新进入比较；本条
  只否定当前口径下的可比性。
- **downgrade trigger**：重新定义 `B_loc` 后得到非零分母，且新口径通过稳定性门。
- **stated_in**：`ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1.2；
  `ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §4.1；`EXPERIMENT_LOG_ZH.md` §5.61。

### C10 Action Strength 是 unlocalized downstream bottleneck（OPEN）

- **statement**：Action Strength 的 LeWM 正/PLDM 负反转**尚未定位到任何一层**。exact-batch 零步反向
  检查没有形成 outcome-aligned 分离：LeWM/PLDM 的 condition-pair prediction coherence 为
  `0.2365/0.2420`，两者 native total gradient 在各自初始化上都局部提高 signed gain；PLDM 终点在该
  Training batch 上甚至达到 gain/NRE=`0.7337/0.2203`，却仍对应冻结 held-out 负标签。因此未解释部分
  位于 Training batch **下游**；`coverage / transfer / calibration` 只是**候选集合**，不是已确认原因。
- **scope**：Action Strength 单元；不外推到其他任务。
- **evidence type/artifact**：`artifacts/icl_native_reversal_mechanism_v1/training_only_v3/`
  的 Action Strength 部分（gate `reason` 字段记录同一判断）。
- **seed/eval unit**：单 exact Training batch，两个**不同**预训练初始化（非共同初始化）；
  optimizer step=`0`。
- **uncertainty**：单 batch 零步量，无区间；跨模型绝对数值不可比。
- **gate**：反向确认门**未通过**；禁止为追求对称而事后选择跨模型阈值。
- **reproduction level**：`L1`。
- **alternatives not excluded**：coverage、跨 query 迁移、响应校准、初始化差异本身，以及尚未列出的
  第五种解释；三者并列，没有优先级证据。
- **downgrade trigger**（此处为定位条件）：出现共同初始化 + held-out `G_swap` 的匹配干预把限制层
  指向其中一个候选。
- **stated_in**：README §4--§5；`ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §4.2、§7.3；
  `ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1.3。

### C11 ActionDelay + COJA：assignment 已学会，calibration 未闭合（PARTIAL）

- **statement**：LeWM+COJA、training seed `3072` 的完整 10-epoch 终点 future/history/switch/worst 为
  `0.882/0.918/1.000/0.763`，即已能按历史把未来分配对；同 checkpoint 的 gain/NRE=`6.877/70.995`，
  响应幅值严重过放大，NRE 分解为约 `34.54` 的尺度误差与约 `36.46` 的正交残差。逐 epoch 重评分把漂移
  起点定位在 epoch 3 到 4 之间（epoch 3 的 NRE 最低 `0.595`），epoch 5 起明显过响应。准确表述是
  **assignment 已建立、response calibration 未闭合**，不是 COJA 失败。
- **scope**：ActionDelay + LeWM + COJA，单训练种子，Development-only。
- **evidence type/artifact**：`artifacts/action_delay_lewm_coja_seed3072_full10_trajectory/summary.json`
  与 `artifacts/contextworld_joint_scratch_full_single_seed_v1/summary.json` 系列。
- **seed/eval unit**：单 training seed `3072`；10 个保存 epoch 在同一 Development 上重评分；原始
  TwoRoom CEM 为 6 个评测 seed × 50。
- **uncertainty**：无训练种子重复；CEM `238/300` 是绝对保持量，非方法增量。
- **gate**：分配门通过、校准门未通过；这是当前“仍然开放”的第一项。
- **reproduction level**：`L2`。
- **alternatives not excluded**：漂移可能来自训练分布响应幅值、损失权重或长训练动力学；未做单因素分离。
- **downgrade trigger**：重评分显示早期 epoch 的分配率也接近随机（即分配从未真正建立）。
- **stated_in**：README §4--§5；`ROOT_CAUSE_DATA_STRATEGY_ZH.md` §7；`EXPERIMENT_LOG_ZH.md` §5.45。

### C12 反转矩阵是 factorial/triangulation，其模型侧汇总只到 `L1`（CLOSED）

- **statement**：四任务反转矩阵中的各 cell 承担**不同反事实角色**（跨模型反转、反方向反转、native 正
  对照候选、四层深挖单元），因此它是 **factorial/triangulation 设计，不是独立复制**；cell 之间不构成
  彼此的重复实验。正式 inventory 中只有 `1` 个 four-layer-complete cell
  （`motion_damping_lewm_native_data_path`），`all_cells_four_layer_complete=false`；
  **不存在“3/4 单元一致即可获得统计置信”这类规则**，本项目也从未预注册过这种规则。
  与之配套，`icl_native_reversal_mechanism_v1/training_only_v3` 的复现层级是 **`L1` 聚合可重复**：
  在 `30` 个冻结输入上重跑汇总器得到字节一致的 `summary.json` 与 `per_cell.jsonl`
  （`model_loaded=false`、`optimizer_steps=0`）；它**没有重跑任何上游训练**，因此不是训练复现、
  统计复现或因果复现。配套单元测试是**软件合同证据**（汇总器行为符合契约），不是科学结论证据。
- **scope**：反转矩阵的证据结构与模型侧汇总的复现层级。
- **evidence type/artifact**：`artifacts/icl_root_cause_reversal_matrix_v1/inventory_v1.json`；
  `artifacts/icl_native_reversal_mechanism_v1/training_only_v3/{summary.json,per_cell.jsonl,receipt.json}`。
- **seed/eval unit**：inventory 覆盖 8 个 task-model cells；汇总输入为 30 个冻结 artifact 文件
  （另加 config 与汇总器源码 SHA）。
- **uncertainty**：inventory 是身份/完整性盘点，不含统计量；`L1` 复现不提供任何抽样不确定性信息。
- **gate**：矩阵只用于**界定 claim 范围与三角定位**；禁止用 cell 计数或多数票支撑通用结论。
- **reproduction level**：`L1`。
- **alternatives not excluded**：不排除补齐更多 cell 的四层资料后可以做更强推断；当前尚未补齐。
- **downgrade trigger**：任一被引用的冻结输入 SHA 变化，或 inventory 的 `four_layer_complete` 判定
  规则被修改。
- **stated_in**：`ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md` §2、§4.2、§10；
  `ROOT_CAUSE_DATA_STRATEGY_ZH.md` §1.3、§9；`EXPERIMENT_LOG_ZH.md` §5.63。

### C13 Motion D2 受控接触物理路由局部可行（PARTIAL）

- **statement**：在冻结 Motion base forward/reverse 两个方向上，planner-support scale `0.625` 的
  query action 保持每步 `L2<=1`，两种 damping 均在同一 raw step/physics substep 首次接触，并产生
  非零 block-response `Gamma`；scale `0.25` 的 contact-free `Gamma` 与 same-damping `Gamma-null`
  均为逐分量精确零。因此“在 canonical `x2` 后通过受控接触产生 state-dependent action leverage”
  这条**物理路由可行**。
- **scope**：Motion Damping P1a、base forward/reverse、Training-only CPU；不涵盖 384-group P1b、
  完整 query/action coverage、D2-0、表示/梯度门、训练或 Development 行为。
- **evidence type/artifact**：确定性 simulator intervention + research-local substep instrumentation；
  `artifacts/pusht_motion_damping_d2_p1a_v1/training_only_cpu_probe_v2.json`（SHA256
  `e2453321c4b76f43ffc65a287d7df08094396be114f927bbb5793d0f812dbd42`）。
- **seed/eval unit**：两个冻结 base directions；每方向两种 hidden modes、`a_ref/contact-free/contact`
  三类动作及 same-damping repeat；无模型、无 optimizer step。
- **uncertainty**：这是结构/数值可行性门，没有抽样区间；仅两个 base directions，不能估计 coverage
  成功率或自然数据分布中的效应量。
- **gate**：P1a `passed_p1a_physical_route`；随后 P1b calibration 的全覆盖结构门已有效失败，
  因而这条局部 route-existence 证据不再许可 D2-v1 继续到 holdout/D2-0/D2-1/训练。
- **reproduction level**：`L1`（独立输出目录复跑 JSON 逐字节一致；探针 CPU 测试 `4 passed`）。
- **alternatives not excluded**：P1b 已确认 `low_approach` 无法在冻结 64-cell coverage 上同时保持
  双方向双 condition 接触与 separation；仍不排除新的、独立预注册的 action/generator 假设可行，
  也未检验本 revision 的信号能否传入 latent/gradient。
- **downgrade trigger**：若 P1a artifact/hash 或 native-step parity 失效，则局部 route-existence 本身降级；
  P1b 失败已经触发的是**禁止向全覆盖和训练外推**，不反向抹除 base forward/reverse 的观测事实。
- **stated_in**：README §4、§6；`EXPERIMENT_LOG_ZH.md` §7.1–§7.2；D2 预注册合同保持执行前冻结，
  不反向改写。

### C14 Motion D2-v1 在 P1b calibration 结构门有效失败（CLOSED）

- **statement**：在预先冻结的 `64 coverage cells × 3 action strata`、每槽取递增 candidate index
  中首个结构通过者的协议下，calibration 只填满 `128/192` slots。`mid_approach` 与
  `mid_tangent_assisted` 各覆盖 `64/64` cells；`low_approach` 的 `2751` 个冻结候选全部耗尽，
  `64/64` slots 均缺失。因此 D2-v1 是有效结构失败并关闭，不冻结 effect/equivalence threshold，
  不打开 sealed holdout，也不进入 D2-0、D2-1 或 native 训练。
- **scope**：Motion Damping D2-v1 generator、LeWM 数据路线的 Training-only P1b calibration；结论只否定
  这一具体 action/geometry/coverage recipe，不否定所有条件可辨识性数据设计，也不改写 COJA 贡献线。
- **evidence type/artifact**：冻结候选窗口的确定性 simulator census + 完整接受/拒绝/跳过账本；
  `artifacts/pusht_motion_damping_d2_p1b_v1/calibration_open1_20260904/calibration_receipt.json`
  （SHA256 `64f069604bbc6c79b831a33bfaca727107ffd27fa3eb50523e79e0314bdf37fe`）。
  对应 accepted/rejected/skip 三文件哈希分别为 `9c1c7e45...`, `02851a22...`, `b9652563...`，
  `128` 个 accepted sidecar 均逐文件哈希通过。
- **seed/eval unit**：pilot catalog seed `2026090401`；冻结 calibration group indices `0..8191`；
  统计/选择单元为完整 `forward/reverse × faster/no-extra damping` group，配额单元为 cell×action stratum。
- **uncertainty**：这是对冻结有限候选窗口的完整枚举，不使用抽样置信区间。`low_approach` 中仅 `8/2751`
  候选满足整组双 condition 接触与同 raw-step，且这 8 个全部未通过冻结的 history/future separation；
  该比例不能外推到尚未预注册的新 generator。
- **gate**：命中 addendum 的 `valid_negative_p1b_or_d2_0: close_d2_v1`；无 threshold relaxation、
  seed/action route shopping、困难 cell 删除或 holdout 开启。
- **reproduction level**：`L1`（不重开一次性 simulator outcome；对冻结 receipt、三份 JSONL 与全部
  sidecar 复算哈希和计数闭合）。不是独立 simulator 重跑或训练复现。
- **alternatives not excluded**：不排除 D2-v2 以新的预注册假设改变 action family/generator 后可行；
  不排除 `mid_*` 子分布本身具有可用效应，但按当前合同不得删除失败的 `low_approach` cells 后继续。
- **downgrade trigger**：发现 runner 将 Gamma/robustness 用于准入、槽内顺序不等价于递增全局扫描、
  native/substep parity 失效，或任一冻结输入/输出哈希不匹配，则本次结果改记 invalid execution。
- **stated_in**：README §6；`EXPERIMENT_LOG_ZH.md` §7.2；
  `configs/pusht_motion_damping_d2_v1_pre_p1b_execution_addendum_v1.yaml` 的 revision failure semantics。

## 3. 台账外的明确非-claim

下列说法**当前不成立**，任何文档不得写出：

1. “统一/通用根因已冻结、已收口”——见 C5；正确表述是第一轮机制边界界定完成。
2. “COJA 失败”或“COJA 已被数据路线替代”——见 C2、C11。
3. “D1 实现了 native behavior reversal / 数据路线已解决 ICL”——见 C7。
4. “ActionDelay 的目标路由是唯一瓶颈”或“已观察到行为反转”——见 C8。
5. “ActionDelay 上游条件份额最高”——见 C9（退化分母）。
6. “Action Strength 的瓶颈是 coverage”（或 transfer / calibration 中任一项）——见 C10，仅为候选集合。
7. “反转矩阵中多数 cell 一致即可作为统计证据”——见 C12。
8. “`training_only_v3` 复现了反转结果”——见 C12，它只是 `L1` 聚合可重复。
9. “P1a 已证明 D2 数据分布可训练或 native ICL 会反转”——见 C13；P1a 只证明物理路由可行。
