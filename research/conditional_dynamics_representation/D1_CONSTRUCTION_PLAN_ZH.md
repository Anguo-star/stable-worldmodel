# D1 数据分布构建与 native 因果验证计划

状态：2026-08-31，首任务与 50/50 曝光框架已定；**D1 配方尚未冻结，也尚未构建**。下一步先做
§3.3 的 `D1-0` Training-only 指标可行性审计，确认相对条件份额能被稳定定义和实际提高后，才生成
训练 schedule。尚未启动 `D1 + native`；ContextWorld-v1 Development 保持不变，公开 Test 不访问。

## 1. 当前决策

首个数据实验固定为 **Motion Damping `D1 + native`**：从同一冻结 Training 候选池提高高**相对**
条件响应样本的曝光，同时保留 50% 自然曝光锚点。它只改变训练样本分布，不改变模型、loss、
初始化、batch 行数、optimizer step 或评测分布。

曝光比例框架（50% 自然锚点、50% 高辨识臂，§4）保持不变，但**具体入池规则尚未冻结**。存在两个
候选：

| 候选名 | 排序量 | 状态 |
|---|---|---|
| `D1-E50`（绝对） | `E_gap(u)=4*C_phys(u)`：条件 future gap 能量本身 | 只作零训练对照，**不得**替代相对候选进入首轮训练 |
| `D1-R50`（相对） | `s_rel(u)=C_phys(u)/(C_phys(u)+B_loc(u)+tau)`，其中 `B_loc` 为 Training-only 局部非条件 future 变化 | 推荐候选，需通过 §3.3 门 |

修订原因：绝对 `E_gap` 排序会连带抬高 query speed 与整体 future 方差，分母可能与分子同步上升，
于是 native risk 中的条件占比不变甚至下降，参数梯度可见性也不必改善。候选贡献的正确陈述是
**聚合相对条件份额上升 + 梯度可见/一致性上升**，而非绝对响应幅度上升。

本轮不做三件事：

- 不追加 COJA 训练、COJA+rollout 或辅助权重搜索；
- 不把 latent energy 或梯度统计写进选择规则，避免数据配方绑定 LeWM 的当前 target encoder 与
  初始化；两者只作构造后审计；
- 不声称验证了 action leverage。现有 Motion synthetic query action 全为零，同池重采样无法识别
  独立的动作杠杆效应；该问题属于后续 D2 新数据采集。

因此，D1 回答的单一因果问题是：**在同一候选池、相同 native 训练预算下，提高条件 future 相对
背景变化的物理响应能量并保留自然覆盖，是否足以让模型在未改动的自然 Development queries 上
更多使用历史？**

这里的干预单位是完整的 **high-identifiability exposure schedule**，不是被完全隔离的单一标量。
任何 schedule 都会连带改变 speed、geometry 和 response-amplitude 边际；首轮只检验这种可实施
训练分布是否充分，不能声称识别了“相对条件能量相对所有协变量的唯一因果效应”。builder 必须
报告这些边际变化，后续只有在确有必要区分机制时才增加 speed-matched sham，而不把它设为首个
数据正例的前置条件。

本文件描述的是**预注册假设与构建方案**，不是结果。D1 未构建、未训练，没有任何 `D1 + native`
证据。COJA 保持为已通过的条件学习正对照，不被描述为失败或被替代。

### 1.1 读者约定

| 术语 | 本文含义 |
|---|---|
| `D0` | 当前自然训练分布，不做辨识度重加权 |
| `D1` | 从同一 D0 候选池生成的重加权索引 schedule；不是新评测集，也不复制图像数据 |
| `D1-0` | D1 构建前的 Training-only 指标审计；不训练模型，也不生成 schedule |
| `native` | LeWM 原有训练目标，不加入 COJA 条件辅助项 |
| `COJA` | 已验证能诱导历史使用的专项目标，本阶段只作正对照 |
| `twin group` | 同一几何下 forward/reverse 各含两个隐藏 mode 的四行最小采样单元 |
| `natural anchor` | D1 中仍按 D0 规则等权覆盖全部 twins 的 50% 曝光，用于保留覆盖与自然响应幅值 |

### 1.2 构造前的 D1 任务复核

在构建前重新评估“首个数据实验是否仍应是 Motion Damping”，结论是**维持**，理由为：

- 已有与 D0 严格 matched 的 native 与 COJA 终点，四格中只缺 `D1 + native`；
- 存在精确的 forward/reverse × 两 mode twin 分组，重采样时可保持 mode/direction 平衡；
- 隐藏动力学（damping）产生**连续**物理响应，可定义分子 `C_phys` 与分母 `B_loc`，而不是只有
  离散标签差；
- 已有 `rho_cond`、`G_swap`、参数梯度/SNR 诊断链路，构造后审计不需要新工具。

**Motion Damping 无法检验的部分**必须显式写明：该 release 的 query action 全为零，因此
`action_leverage` 在本任务中不可识别，D1 的任何正结果都**不能**推广为“动作杠杆有效”。动作
杠杆需要 D2 的新采集或非零 action 的任务，不能靠同池重采样或 action-norm 代理量补上。

## 2. 冻结身份与计算单位

### 2.1 数据与模型身份

| 项目 | 冻结值 |
|---|---|
| Motion release | `contextworld_pusht_motion_damping_icl_history3_v1` |
| Training source | `pusht_motion_damping_h3_release_v4/train.lance` |
| release manifest SHA256 | `48246aa4ae4a13d5b1c9677ba37a92fe114129027745f8e258137a016899563b` |
| release config SHA256 | `1795717d8bfa1d1cfcc01a69931b6241b0e7759dcf43553a6c4cca225ec9326b` |
| Training condition pairs | `8,192` |
| forward/reverse twin groups | `4,096` |
| 初始化 checkpoint SHA256 | `9f13b2c28bb909338047e08d62bf6eb16ea3616cb53e57cfa9b5072b43db1c59` |
| training seed | `14321` |
| optimizer steps | `8,192` |
| 每批原始/隐藏行 | `64/64` |
| native objective | 原生 MSE + 既有 `0.09*SIGReg`，COJA/条件辅助权重为 `0` |
| 模型可见字段 | `pixels, action` |

D0 matched native 仍是
`pusht_motion_damping_full_release_visible_joint_absolute_single_stage_native_control_step8192_v1`。
D1 必须继承该运行的初始化、冻结模块、optimizer、scheduler、精度与评测入口；不得借 D1 名义顺带
修正其他训练细节。

### 2.2 为什么按 twin group 构造

每个最小训练单元不是单条 episode，也不是一个二元 damping pair，而是：

```text
同一几何的 forward query  × {faster_decay, no_extra_decay}
同一几何的 reverse query  × {faster_decay, no_extra_decay}
```

即一个 twin group 含两个方向、两个 mode、四条 condition rows。当前训练流已经用
`CompleteTwinPairedBatchStream` 保证四条记录同批出现，抵消仅靠初始外观或方向猜 condition 的
捷径。D1 必须完整保留这个四元关系。按单行或单 pair 重采样会同时改变 mode/direction 平衡，无法
把结果归因于条件能量。

## 3. 候选打分

所有统计量与分箱边界只由冻结 Training release 计算。Development/Test 路径在 audit/builder 中
列入禁止读取集。

### 3.1 首轮选择量

对 twin `u`、方向 `r in {forward, reverse}` 和隐藏条件 `c in {0,1}`，令 `y_{u,r,c}` 为 native
`identifiable_future_only` loader 所监督终点的 block 位置（px），`x_{u,r}` 为共享 query 时刻的
block 位置。它们从冻结 `train.lance` 的 `physics_state` 读取；manifest 的 `future_gap` 只用于
交叉核验，不能代替带方向的 endpoint。先定义条件均值位移与条件能量：

```text
m_{u,r}   = 0.5 * [(y_{u,r,0}-x_{u,r}) + (y_{u,r,1}-x_{u,r})]
C_phys(u) = mean_r [ ||y_{u,r,1}-y_{u,r,0}||^2 / 4 ]   # paired MSE 中的条件分量
E_gap(u)  = 4 * C_phys(u)                               # 仅作绝对能量对照
```

Motion 的所有 `future_gap.block_angle_rad` 已核验为 `0`，所以首轮主物理量只用 block position px。
角度和其他状态分量继续单独记录，不能在没有预注册单位换算时与 px 平方相加。每个方向仍须通过
release 已冻结的 `||y_1-y_0|| >= 2.0 px` separation 门，并单列 `min_r`；因此方向平均不能掩盖
某个方向不可辨识。本轮指标只覆盖 native 实际监督的 endpoint，不冒充完整轨迹 leverage。

`B_loc` 衡量相近 query 的**非条件** future 位移变化。对每个有方向的 query，用 Training-only
query descriptor（block 位置、速度、朝向和 goal-relative geometry）找近邻。候选全集是 `8,192`
条 directed pair records `(v,s)`；先排除当前 twin `u` 的 forward/reverse 两条记录，再按标准化
Euclidean distance 取恰好 `k` 条，同距时按 `pair_index` 升序打破并列。随后计算邻居条件均值位移
`m_{v,s}` 的方差：

```text
bar_m_k(u,r) = (1/k) * sum_{(v,s) in N_k(u,r)} m_{v,s}
B_k(u)       = mean_r [ (1/k) * sum_{(v,s) in N_k(u,r)} ||m_{v,s}-bar_m_k(u,r)||^2 ]
B_loc(u)     = B_64(u)                                 # k=64 为预注册主定义
s_rel(u)     = C_phys(u) / (C_phys(u) + B_loc(u) + tau)
rho_phys(pi) = E_{u~pi}[C_phys(u)] /
               (E_{u~pi}[C_phys(u)] + E_{u~pi}[B_loc(u)])
```

descriptor 固定为 query block 的 `(x,y,vx,vy,sin(theta),cos(theta))` 与 goal-relative block
位移 `(dx_goal,dy_goal)`；各连续维度只用 Training 中位数与 IQR 缩放。`k=32/128` 只作灵敏度
检查，不能看结果后替换主 `k=64`；IQR 为零的常量维度距离贡献固定为零并写入回执。数值下限固定
为 `tau=1e-8 px^2`，并报告其命中率及 `tau=0` 时的排名一致性。现有 release 的 absolute
future-separation 门继续保留，故 `s_rel` 不能靠“分子和分母都接近零”成为高分。

这里必须区分两个量：`s_rel(u)` 是逐 twin 的有界排序分数；`rho_phys(pi)` 是一个 schedule 的
ratio-of-means 操作量。禁止用 `mean_pi[s_rel(u)]` 冒充聚合 `rho_phys`，否则少数极小分母样本会
把整体效果虚高。`B_loc` 只对条件均值 `m` 求变异，不能把同一 pair 的条件差 `Delta y` 再算进
背景分母。所有 `B_k` 都在等权 D0 Training reference 上计算一次并冻结；比较 D0、E50 与 R50 时
只改变曝光权重，不按候选分布重新估计分母。每个候选都必须满足 `C_phys`、`B_loc`、`s_rel` 与
聚合分子/分母全部 finite，且 `E_pi[C_phys]+E_pi[B_loc] > 0`；否则 fail closed。由于每个方向已
通过 `>=2.0 px` separation，合法候选的聚合条件分子本身严格为正；`tau` 不加入聚合式。

同时保存、但不压成一个总分：

- `conditional_energy_physical`：上述 `C_phys` 与 `E_gap`，并保留位置与角度原分量；
- `background_future_variation`：`B_32/B_64/B_128`、邻居身份与留一稳定性诊断；
- `relative_conditional_score_physical`：局部 `s_rel`，即首轮排序量；
- `aggregate_relative_conditional_energy_physical`：曝光分布的 `rho_phys(pi)`，即 schedule 主操作量；
- `history_clue_physical`：两方向 `history_visible_response_gap` 的均值与最小值；
- `conditional_energy_latent`：共同冻结初始化下的相对 target-response energy（`rho_lat`），仅用于
  验证 D1 操作是否也提高当前表示中的条件占比，**不参与排序**；
- `coverage_cell`：§3.2 的 orientation×speed×goal-distance joint cell，另记录响应幅值分位数；
- `pair_quality`：query pixels/action identity、future separation、无泄漏、forward/reverse twin 完整性；
- `action_leverage`：本轮写为 `not_identifiable_query_action_zero`，不得用 action norm 伪造代理量。

首轮使用 physical-only（不含 latent、不含梯度）排序是刻意的。若 physical 条件份额明显提高而
frozen-latent `rho_lat` 不提高，应解释为表示压缩；若两者都提高而 native 仍不学，才把原因推进到
参数 Jacobian、梯度一致性或跨-query 覆盖。直接用 LeWM latent 或梯度统计选数据会混淆这两种情况，
也不利于主模型预训练复用。`C_phys`/`B_loc` 使用的是 synthetic paired generator 已知的训练
target，是数据策展 oracle 而非模型输入；正结果证明“这种分布可以解决”，不自动证明普通未配对
预训练语料可用同一分数直接挖掘。迁移到主模型时还需把它落成 paired simulation、主动采集或
可观测 proxy。

### 3.2 高辨识池

高辨识池必须在匹配覆盖后比较，而不是从全局榜单直接取 top-25%。先只用 Training descriptor
建立 `64` 个 joint coverage cells：四个 `orientation_bin`；每个 orientation 内按 forward/reverse
平均 query speed 稳定排序切成四个等频箱；每个 orientation×speed 箱再按 forward/reverse 平均
query-to-goal block distance 切成四个等频箱。现有 `4,096` twins 因而每格恰有 `64` 个。

每格按选定分数排序，以稳定 `twin_id` 打破并列，取前 `16` 个。最终 high-ID pool 仍是
`64*16=1,024` 个 twin groups，但 orientation、speed 和 goal-distance joint-cell 配额与 D0 精确
匹配。`D1-E50` 在每格按 `E_gap` 取 16 个，`D1-R50` 在同一格按 `s_rel` 取 16 个；两者的曝光
模板和 coverage 完全相同，差异只来自格内排序量。

旧方案按 orientation 内绝对 `E_gap` 排序时，pool 的 physical future gap RMS 为约
`2.153--3.690 px`，四分位点为
`2.512/2.878/3.277 px`；query speed 为约 `14.001--23.998`。这些数字描述的是**绝对候选**
的旧草案，不是当前 `D1-E50`/`D1-R50` 配方结果。新候选必须按上述 joint-cell 配额重算；结果必须
写成“受覆盖约束的 exposure schedule 效应”，不能写成某个标量相对所有协变量的唯一因果效应。

### 3.3 D1-0 指标可行性审计（必须在 schedule 构建前通过）

`D1-0` 不是数据集，也不是训练：它只读取冻结 Training manifest/`physics_state`，输出 per-twin
诊断表和两个候选的**投影曝光统计**，不写训练 schedule、不访问 Development/Test、不运行
optimizer。它回答“这个数据池中能否稳定定义并提高相对条件份额”，按下列顺序执行：

1. **验证物理量**：`train.lance` endpoint 与 manifest `future_gap` 逐 pair 一致；Motion 主分数只用
   position px，angle 单列为零；每个方向的 gap 均 `>=2.0 px`；所有输入、局部统计和聚合分母均
   finite/positive；当前 twin 及其反向记录在取恰好 `k` 个近邻之前同时排除。
2. **验证局部定义**：主 `k=64` 与 `k=32/128` 的 `s_rel` 排名必须稳定。两次全局 Spearman
   均须 `>=0.90`，按 §3.2 各格取 top-16 后形成的两个完整 high-ID pools 与主 pool 的 Jaccard
   均须 `>=0.80`。同时报告 `B_k` 分布、近邻距离、`tau` 命中率和每个 coverage cell 的有效样本数；
   任一门失败即 no-go，不能事后挑最有利的 `k`。
3. **投影两个候选**：按 §3.2 的相同 joint-cell 配额和 §4 的相同 50/50 曝光权重，计算
   `D1-E50` 与 `D1-R50` 的投影统计，但暂不生成 schedule。
4. **同表比较**（Training-only、零 optimizer step）：

   | 指标 | 要求 |
   |---|---|
   | ratio-of-means `rho_phys(pi)` | 相对 D0 严格上升；这是主操作判据，禁止用 `mean(s_rel)` 代替 |
   | exposure-weighted `C_phys` 与 `B_loc` | 分别报告，识别“分子分母同步上升”的伪改善 |
   | 64 个 orientation×speed×goal cells | 候选配额精确匹配，全部原始 twins 仍由 natural anchor 曝光 |
   | 连续 speed/goal-distance 分布 | 另报 SMD、KS 与 1-Wasserstein；用于限定结论为 schedule 效应，不把等频箱误写成连续变量完全匹配 |
   | 自然 response-amplitude 锚点分位数 | 部署幅值分位数未被截断 |
   | twin/mode/direction 平衡、总 transition 数 | 精确不变 |

5. **冻结相对候选**：只有 `D1-R50` 稳定且相对 D0 提高 `rho_phys(pi)`，才允许它成为唯一首轮训练
   配方。`D1-E50` 始终只作零训练参照；若它也提高条件份额，只能说明 `E_gap` 在 Motion 中恰好是
   有效 proxy，不能据此声称绝对能量一般充分。把排序量、邻域定义、cell 边界、并列规则和构造 seed
   一次写入 config；禁止使用 Development 或短跑结果改选。
6. **no-go**：若 `B_loc`/排名不稳定，或 `D1-R50` 不能在精确覆盖约束下提高聚合 `rho_phys(pi)`，
   则停止。先修改物理分母、query descriptor 或改为 D2 新采集，不得训练 `D1-E50`，也不得退回
   旧的全局绝对 top-25%。

`rho_lat` 与参数梯度可见性（`r_grad`、余弦一致性、`SNR`）**不进入本门的选择判据**，只在 §6.2
的构造后零步审计中报告，用于确认操作是否进入当前表示与 optimizer 路径。

## 4. 50% 曝光模板的精确不变量

本节的曝光模板对 `D1-E50` 与 `D1-R50` **完全相同**，两者只在 64 个 coverage cells 内的入池
排序量上不同。这正是 §3.3 同表比较成立的前提。

现有 stream 每 `256` 个 optimizer steps 完整访问 `4,096` 个 twin groups，每批 `16` 个 twin。
`8,192` steps 恰好是 `32` 个完整 cycle。D1 保持 cycle、batch 与总曝光数不变：

| 每 cycle 的 4,096 个 twin slots | 数量 | 规则 |
|---|---:|---|
| natural anchor | `2,048` | 每个 orientation 取 `512`：其中 high-ID `128`、ordinary `384`；相邻两个 cycle 使用互补半分并覆盖全池一次 |
| high-ID exposure | `2,048` | 四个 orientation 各 `512`，从本箱 high-ID pool 做两次无遗漏置换 |

每个 cycle 的 `256` 个 batch 都固定为 `8` 个 high-ID-arm slots 加 `8` 个 natural-anchor slots；
natural 部分再固定为 `2` 个 high-ID 与 `6` 个 ordinary，因此每批共有 `10` 个 high-ID、`6` 个
ordinary twins。high-ID arm 与 natural arm 在同批不得出现相同 `twin_id`。cycle 内仍做带 seed 的
确定性置换，不设置 per-batch orientation quota；这样既避免把高能样本堆在 cycle 前后形成隐式
curriculum，也不额外把方向比例钉死到每个梯度步。

由此在整个 8,192-step 运行中：

- 每个 twin 都获得恰好 `16` 次 natural-anchor 曝光；
- 每个 high-ID twin 额外获得 `64` 次曝光，总计 `80` 次；其他 twin 为 `16` 次；
- high-ID pool 占隐藏数据曝光的 `62.5%`，D0 为 `25%`；
- 每个 joint coverage cell 的总曝光为 `64*16 + 16*64 = 2,048`，与 D0 的 `64*32=2,048`
  精确相同；重加权发生在 cell 内，不改变三个覆盖变量的联合直方图；
- 每 cycle 四个 orientation 各 `1,024` slots；每个 slot 仍展开为完整四行 twin group；
- 总 twin 曝光 `131,072`，隐藏行 `524,288`；原始 PushT 行仍为 `524,288`，保持严格 50/50；
- 每批 arm/high-status 配额和 `twin_id` 唯一性严格固定；orientation 只在 cycle 总量上平衡。

构造 seed 固定为 `20260831`，训练 seed 仍为 `14321`。builder 可额外生成 E25/E75 比例以及绝对/
相对两种排序量的**零训练分布审计**；但训练候选的比例预先固定为 50%，排序量由 §3.3 的
Training-only 比较一次性冻结，之后不依据 Development 或短跑结果改选。

## 5. 最小实现边界

D1 不修改 ContextWorld-v1 release，也不复制约 1.9 GB 的 Lance pixels。只新增一个训练期索引层：

1. D1-0 audit 从冻结 Training manifest/`physics_state` 生成 per-twin catalog（含 `C_phys`、
   `B_32/B_64/B_128`、`s_rel`）、两个候选的投影曝光统计与 §3.3 比较表，不写训练 schedule；
2. D1-0 通过后，builder 只为被冻结的唯一候选生成 schedule，并把排序量写入 config 与 sidecar；
3. `EnergyStratifiedTwinBatchStream` 读取 schedule，把每个 `twin_id` 映射回现有四条 condition rows；
4. D1 runner 继承 matched native runner，只临时替换
   `CompleteTwinPairedBatchStream`；模型、loss 和 materialized arrays 不变；
5. sidecar 记录 manifest、source code、checkpoint、排序量标识与实际消费 schedule 的 SHA256。

计划新增文件（名称沿用 `energy_stratified`，具体排序量以 config 中的 `selection_score` 字段为准，
不得从文件名推断）：

- `configs/pusht_motion_damping_d1_energy_stratified_native_v1.yaml`：冻结本文件中的身份和比例；
- `scripts/audit_pusht_motion_damping_d1_metric_v1.py`：Training-only D1-0 catalog 与候选投影审计；
- `scripts/build_pusht_motion_damping_d1_schedule_v1.py`：只为 D1-0 选中的候选生成 schedule；
- `scripts/run_pusht_motion_damping_d1_energy_stratified_native_v1.py`：继承 matched native 的单因素 runner；
- `scripts/audit_pusht_motion_damping_d1_schedule_v1.py`：零 optimizer-step 数据与 batch 审计；
- `tests/test_pusht_motion_damping_d1_schedule_v1.py`：确定性、覆盖、模式平衡和禁止 split 测试；
- `tests/test_run_pusht_motion_damping_d1_energy_stratified_native_v1.py`：runner 单因素身份测试。

首轮验证前不把 sampler 抽象进 `stable_worldmodel/` 核心包。D1 若给出跨 query 正信号，再把
“带 provenance 的 group reweighting”提炼成通用预训练数据接口，避免为一个尚未成立的假设提前
增加核心复杂度。

## 6. 训练前门控

### 6.1 必须通过的 CPU 门

- 输入只包含冻结 `train.lance`/manifest，Development/Test read count 均为 `0`；
- §3.3 的 D1-0 门已通过：`B_loc`/`s_rel` 稳定性、绝对/相对候选同表比较、冻结的
  `selection_score` 规则均已归档；ratio-of-means `rho_phys(pi)` 相对 D0 上升且 joint-cell 配额精确，
  否则记为 no-go 并停止构造；
- release、config、builder、schedule 与 source checkpoint SHA 全部匹配；
- 4,096 个 source twins 全部至少曝光一次，high/ordinary multiplicity 精确为 `80/16`；
- forward/reverse、两个 damping mode、四个 condition rows 永不拆组；
- 8,192 steps、每批 16 twins、每批隐藏 64 行、原始/隐藏 64/64 全部精确；
- 每批恰有 `8` 个 high-ID-arm 与 `8` 个 natural-anchor slots，按实际 twin 身份为 `10 high/6 ordinary`，
  且 `twin_id` 无重复；
- orientation 总量精确匹配，所有原始 query-speed/goal support cell 仍有曝光；
- 模型与 loss 边界仍只收到 `pixels, action`，任何 score、mode、pair/twin id 均不进入前向；
- `E0` identity schedule 与原 `CompleteTwinPairedBatchStream` 逐 batch 比对完整索引 tensor，包含
  pair 相邻性与 `[2p,2p+1]` 行展开顺序；不能只比较集合或计数。

这些构建与审计均为 CPU/I/O 工作，不需要 GPU。

### 6.2 最小表示/梯度检查

GPU 空闲后，仅做冻结初始化上的零步检查。统计单位是完整 twin group，不把同 twin 的两个方向
当作两个独立样本；固定至少 `16` 个 D0 batch 与相同 twin 数的 D1 batch，并保持 loss reduction、
参数集合和 BF16/FP32 路径与训练一致：

- 报告 ratio-of-means `rho_phys(pi)` 与局部 `rho_lat(pi)` 相对 D0 的变化，并分别给出条件分子与
  背景分母；latent 口径复用 D1-0 冻结的 physical neighbor graph，只替换成 latent pair center 与
  response，不能在 latent 空间重新找最有利邻居；既有 global `rho_lat` 另列，不能混写；
- 分别对 `predictor`、`pred_proj` 和全体训练参数报告 `||E g_response||`、
  `||E g_nonconditional||`、两者比值，以及 raw-gradient 口径；分母为零时 fail closed，不报无穷大；
- 以 twin 为 cluster 估计 response 梯度到其跨 twin 均值的余弦、协方差、`Bcrit` 与目标 batch size
  下的 `SNR(B)`；同时报告有效 twin 数，不能把“32 pairs”写成 32 个 iid 样本；
- D1 若既没有提高 response 平均梯度，也没有改善其跨 twin `SNR`，停止完整训练并回到数据定义。

该检查确认物理数据操作实际进入当前表示与 optimizer 路径。`rho_lat` 与 `V_grad` 是构造后的
机制门，不反向用于选择排序量、邻域大小或曝光比例，也不访问 Development。

## 7. 训练与评测

### 7.1 唯一新增训练格

首轮只运行**一个** `D1 + native` 格（§3.3 冻结的排序量 + 50% 曝光模板），使用一个 GPU、
seed `14321`、8,192 steps。保存 step `256/1024/2048/4096/8192`，但训练不中途按 Development
早停或换配方。落选候选不进入训练，只保留其零训练分布审计。DINO-WM、PLDM 及额外 LeWM seed
等该格给出方向后再决定。

### 7.2 冻结自然评测

训练完成后统一评分所有保存点；评测仍用 benchmark v1 的自然 Development，不构造 D1 专用
Development。D1 exposure-weighted Training 指标只作为 manipulation check，因为重权高
`||Delta t||` 会构造性放大未归一化 `G_swap`；改用相对量选择会削弱但不会消除这种构造性膨胀，
因此它仍不能作为学习成功判据。训练侧模型比较统一在全部
`4,096` twins 等权的冻结 D0 audit panel 上完成，并同时报告 response-normalized alignment。
主比较为：

- D0-weighted Training audit panel 与 natural Development 的 `G_swap` 均值/中位数/正号比例、
  sign-flip 检验及 cross-query null；
- gain、alignment、NRE、响应尺度/正交残差分解；
- correct / swapped / removed-history 三臂差，即 history-ablation drop；
- final checkpoint 的标准原始 PushT CEM300 保持性。

这是单 training seed 的机制筛查。逐 query bootstrap/sign-flip/cross-query null 只能量化冻结
query population 上的不确定性，不能冒充训练 seed 重复；单 seed 正信号只触发复现，不形成论文
方法结论，也不能据此声称 D1 已替代 COJA。公开 Test 继续锁定。

## 8. 结果分流

| 结果 | 结论 | 下一步 |
|---|---|---|
| D0-weighted train panel 与自然 Development 的 `G_swap`、gain/history drop 同时改善，NRE 不恶化 | 同池数据分布可以让 native 更使用历史 | 先补至少两个 native training seeds；确认后补 `D1+COJA` 判断替代/互补 |
| 仅 D1-weighted train 改善，D0-weighted train/Development 近零 | 重加权的构造性增益或高能模板记忆，主要瓶颈是 query/action coverage | 进入 D2 新采集，不继续提高同池重复率 |
| assignment 改善但 gain/NRE 过放大 | 数据已建立条件分配，后段缺自然幅值校准 | 做 D3 high-ID→natural 调度，不加专项 loss |
| physical/latent 条件份额均提高但参数梯度和 Development 都不动 | 同池数据重分布不足，Jacobian/表示或 objective 成为主要瓶颈 | 再考虑表示或 COJA 类显式条件目标 |
| 原始 CEM 明显下降 | support shift/过度重复 | 保持 D1 机制结论但不作为主模型配方，增加自然混合或新覆盖 |

在 `D1+native` 给出自然 Development 正信号前，COJA 保持论文方法和可学习性正对照；D1 不替换
COJA。若 D1 后续接近或超过 COJA，数据原则可升为主贡献，COJA 降为存在性证明或互补工具。
“接近或超过 COJA”必须由至少三个 D1 native training seeds 支持，不能由首个 seed 触发。
整个阶段只新增 training-distribution track，不改 benchmark v1；只有新增评测 query/action/split
或指标时才讨论 benchmark v2。

## 9. 执行顺序与资源

1. CPU 执行 D1-0：构建 per-twin 指标 catalog、绝对/相对候选的投影曝光统计与不可变回执；
   no-go 时停在此步并修订指标，**不生成 D1 schedule**；
2. D1-0 通过后，CPU 只构建唯一入选 schedule，并执行 schedule/batch/forbidden-split 全部门；
3. 单卡做冻结 latent 与至少 16 个 batch 的梯度零步检查；
4. 机制门通过后，单卡运行唯一的 `D1 + native` 8,192-step 训练；
5. CPU/单卡完成自然 Development 轨迹与 final CEM300；
6. 根据 §8 决定 D2、D3 或 `D1+COJA`，不并行铺开。

因此当前可开始的是 **D1-0 指标可行性审计，不是 D1 构建**。D1-0 只占 CPU/I/O；它通过后才
生成训练 schedule。真正需要 GPU 的只有一次小规模零步检查和随后唯一的 native 训练格；它们都
不与当前云侧 D0 native 队列构成逻辑前置关系。
