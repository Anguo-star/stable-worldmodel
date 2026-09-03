# D2 新轨迹构造与 native 因果验证预注册

状态：2026-09-02，**`structural_preregistration_frozen_effect_bounds_pending_pilot`**。本文件与
[`configs/pusht_motion_damping_d2_preregistration_v1.yaml`](configs/pusht_motion_damping_d2_preregistration_v1.yaml)
是同一份合同的散文层与机器可读层；两者冲突时以 YAML 的字段与 `status` 为准。
这里记录的是执行前冻结状态；P1a 及后续 live 状态只写入独立结果收据与实验日志，禁止反向改写本合同。

本 D2 P0 子阶段**不实现 generator、不生成数据、不训练、不访问 Development/Public Test**，也不产生
任何 artifact；D2 新增资产只有这两个文件，既有文档只增加入口链接与边界记录。D1、
`REL50/ABS50/HASH50`、反转单元模型侧门与既有 COJA
资产保持冻结，不被本文件改写。上游依据见
[`ROOT_CAUSE_DATA_STRATEGY_ZH.md`](ROOT_CAUSE_DATA_STRATEGY_ZH.md) §8.2/§10、
[`D1_CONSTRUCTION_PLAN_ZH.md`](D1_CONSTRUCTION_PLAN_ZH.md) §10 与
[`ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md`](ROOT_CAUSE_REVERSAL_MATRIX_PROTOCOL_ZH.md) §7.3。

**最短执行路径只有四步：**

1. P1a 用 CPU 验证受控 query contact 能稳定产生非零 `Gamma`；失败就停；
2. 通过后才生成 `192 + 192` 个 calibration/holdout groups，冻结并验证数据门；
3. D2-0 通过后才占用 1 张 GPU 做零 optimizer-step 的 D2-1；
4. D2-1 通过后只跑一个匹配预算 native 训练格。此前不写完整数据集、不训练、不扩任务。

## 0. 两类门的强制区分

本预注册把所有判据分成两类，任何文档、脚本或后续汇报都不得混写：

| 门类 | 含义 | 何时冻结 | 本轮状态 |
|---|---|---|---|
| **结构硬门** `structural_hard` | 由构造、身份、不变量或已冻结常量决定，与效应大小无关（例如 group 完整性、canonical query state 容差、CRN、tensor boundary、provenance/hash、支撑保持） | **现在即可冻结**，不需要任何 pilot | 已在 YAML 中写死数值或断言 |
| **效应/等价界限** `effect_pilot_frozen` / `equivalence_pilot_frozen` | 需要估计测量噪声或自然分布范围才能定标的量（例如 `rho_phys/rho_pixel` 的最小可检测改善、ESS/duplicate/top-mass、amplitude `W1`、probe 可识别下限） | 只用预先划分的 **Training-only calibration pilot** 与冻结 D0/D1 基线定标一次，再在未参与定标的 holdout pilot 上验门 | 全部为 `pending_pilot_freeze`，只写规则不写数字 |

因此本文件**不出现** `+20%`、`+30%` 这类未经数据支持的阈值。凡是当前无法诚实给出的数值，YAML
里写成显式 `status` 加 `rule`，**禁止用 `null`、`0` 或占位数字冒充已完成**。pilot 只允许冻结一次；
冻结后不得因为 D2-0/D2-1/训练结果不利而重开、放宽或另选口径。

## 1. 单一 claim 与 scope

D2 检验的唯一 claim 是：

> **loss-native, data-designed**：在 **Motion Damping + LeWM** 这一个单元内，保持模型、native
> objective（原生 MSE + 既有 `0.09*SIGReg`，COJA/条件辅助权重为 `0`）、初始化、精度、优化器与
> 训练预算完全不变，只更换训练数据的**内容**（新采集的 matched dynamics groups），能否让 native
> 训练在冻结自然 Development 上更多使用历史。

必须同时保留的 scope 边界：

1. **仅限 Motion-LeWM**。ActionDelay 的 objective-route 分离与 Action Strength 的 unlocalized
   downstream bottleneck 已经证伪"低 `rho` 是跨任务充分解释"；D2 的任何结果都不得外推到 PLDM、
   DINO-WM 或其他任务，也不得写成通用根因收口。
2. **不是新 loss**。D2 不引入 COJA 项、不引入 pair-level 监督、不做 mixing weight 搜索。
3. **不是 benchmark 变更**。D2 不进入 ContextWorld registry / scoreboard，不是 benchmark v2，
   评测仍是冻结的 benchmark v1 自然 Development；Public Test 全程锁定。
4. **privileged 信息只用于生成与审计**：hidden damping 标签、`physics_state`、`pair_id`、
   `hidden_mode`、`Gamma`、coverage cell、generator/action-program 家族标识、shadow 划分标识等，全部只出现在
   generator、curation 与离线审计通道。它们**不进入 model tensor、不进入 native loss、不进入任何
   batch 级监督**；模型可见字段仍严格只有 `pixels, action`。该边界由 §5.4 的 tensor boundary
   receipt 以 fail-closed 方式强制，而不是靠约定。

"数据策展 oracle" 的定位不变：`C_phys`、`Gamma` 依赖 paired simulator 真值，是构造侧 oracle，
不是可直接用于任意未配对预训练语料的现成分数。

## 2. 继承身份与新增身份

D2 **继承并不得修改**的冻结身份（来自 D1 §2.1）：初始化 checkpoint SHA256
`9f13b2c2...`、training seed `14321`、optimizer steps `8,192`、每批原始/隐藏行 `64/64`、
native objective 定义、模型可见字段、评测入口与冻结自然 Development。

D2 **新增**的是数据身份：新的 generator（research-local）、新的 catalog seeds、新的 release
候选目录、新的 manifest 与全部 SHA256。为遵守所有权边界，新 generator 只能**只读导入**
`contextworld.evaluation.pusht_motion_damping_h3` 与 `pusht_contact_friction_h3` 并在本仓库内
组合新模板；不得修改 `../ContextWorld` 的任何文件，也不得写入 ContextWorld 的 registry。

继承的物理结构常量（来自 `pusht_motion_damping_h3.py`，全部为结构硬门）：

| 常量 | 值 | 作用 |
|---|---:|---|
| `DAMPING_VALUES` | `faster_decay=0.2` / `no_extra_decay=1.0` | 唯一隐藏变量 |
| `HISTORY_RAW_STEPS` / `QUERY_RAW_STEPS` / `ACTION_BLOCK` | `10 / 5 / 5` | 历史与 query 的 raw 步数、动作块 |
| `MODEL_FRAME_ROWS` | `(0,5,10,15)` | 模型帧位置，预测 horizon 不变 |
| `QUERY_STATE_TOLERANCE` | `1e-8` | canonical query 全 12 维状态匹配容差 |
| `MINIMUM_HISTORY_GAP_PX` | `3.0` | 历史必须有可测物理响应 |
| `MINIMUM_FUTURE_GAP_PX` | `2.0` | 绝对 future separation 下限 |
| `EFFECTIVE_CONTACT_FRICTION` | `0.25` | 接触摩擦固定，不作为隐藏变量 |

### 2.1 首版物理路由已经收缩

D2-v1 固定为：**零 prefix action + query 阶段接触介导的非零动作 + 全 12 维 canonical query
state 匹配**。不在首版加入 history excitation，也不写数值反解器。这样现有自由衰减反解仍负责让
两种 damping 自然汇合到同一 `x2`；新变量只在 `x2` 之后生效。

动作幅值**不从 D0/D1 Motion 训练集估计**，因为两者的 query action 全为零。P1a 唯一允许的外部
动作参考是冻结的 planner-curve receipt
[`train_templates4096_v1.pt.json`](artifacts/pusht_motion_damping_planner_curve_cartesian_action_overlay_half4096_v1/train_templates4096_v1.pt.json)
（SHA256 `747c2909...1450`，其 payload SHA256 `aa8ac1f0...ddc7`、raw action blocks SHA256
`f2a15416...76a86`）。候选仍需满足每个 raw step `||A_t||_2 <= 1`；该单位球在 rigid rotation 下不变，
旋转后的动作必须**无需 clip** 即通过 `[-1,1]^2`，且旋转前后逐步 L2 范数相等。

这个选择不是偏好判断。2026-09-02 的只读 CPU 探针在现有 base template 上得到：接触自由的
query actions（轴向幅值最高测到 `0.5`）对两种 damping 的 condition gap 给出逐分量
`Gamma=0`；朝向 block 的接触动作则产生非零 block position/velocity `Gamma`，且两种 condition
首次接触发生在同一 raw step。该探针只用于 P0 路由裁决，不作为正式 artifact；P1a 必须用独立
脚本、更多 Training-only geometry 与完整回执复现后，才允许生成 D2 池。

接触不构成新的隐藏因子：contact friction 在全部 condition、动作与 comparator 中固定为 `0.25`，
唯一干预仍是 motion damping。为防止把接触分布变化误写成 damping 证据，D2-0 必须记录首次接触
的 physics substep（不能只记 raw step）、逐步接触/arbiter 轨迹、接触前状态与首次 contact
normal/point/impulse，并拒绝 grazing-contact：在预注册的小幅 geometry/action 扰动下，两种 condition
都必须保持接触且 `Gamma` 方向稳定。另设置同动作、同 canonical `x2`、相同 damping 的
`Gamma-null` 模拟器 comparator。该 comparator 是 audit-only 的 `x2` replay，可在新 simulator
起点安装一次 `x2`；它不写训练行，也不受训练轨迹“`x0` 后不安装状态”的约束。

D2 必须使用 **research-local 新校验器**。现有 `evaluate_template` / `validate_motion_damping_pair`
把 query contact-free 和整个 query 无 arbiter 当作成功条件，与本路线相互否定，不能整段复用。
新校验器把 `history_arbiter_counts[-1]` 定义为首个 query action 前的 arbiter 数；first-contact raw step
从 `rows['n_contacts'][10:15]` 的首个正值派生，但该量只是 raw step post-solve 计数，physics substep
必须新增 instrumentation。每个 active、`a_ref` 和 `Gamma-null` rollout 还要断言：agent/block shape
friction 的乘积为 `0.25`、wall friction 为 `0`、`space.damping` 等于当前 hidden mode。

## 3. D2 的设计单位：matched dynamics group 与 `Gamma`

### 3.1 完整 matched dynamics group

D2 的最小构造、审计与曝光单位是**完整 matched dynamics group**，沿用 D1 的四行结构并新增动作
程序维度：

```text
group(u) = 同一 base geometry g(u) × 固定全零 prefix P0 × 同一语义 action cell alpha(u)
           × A_{u,r}=equivariant_action(g(u), r, alpha(u))
           × {forward, reverse} × {faster_decay, no_extra_decay}
```

即每个 group 含 `2 × 2 = 4` 条 condition rows。Cartesian action **只要求在同一方向的两个
condition 间逐元素相同**；forward/reverse 共享 action cell 的方向/幅值语义，但实际向量按各自
agent-to-block query geometry 协变。强行让两个方向使用同一 Cartesian 向量会造成一边接触、一边
不接触。forward/reverse 镜像结构必须保留，因为它是
"仅凭 `x0` 判定隐藏模式 = 50%" 的构造性来源。生成、curation、曝光计数与统计聚类都必须保留
完整四行；只有 arm C 的 **sample-to-batch assignment** 可以把四个完整 model examples 分到不同
batches，且 group metadata 不进入 model/loss。group-block 与 global sample shuffle 的区别仅在这一
层（见 §9）。

### 3.2 canonical query state 匹配（结构硬门）

同一 group 内两个 hidden condition 的轨迹必须在 query 时刻 `x2` 汇合到同一个 canonical state。
匹配范围是 `body_snapshot` 的完整 12 维（agent 位置/线速度/角度/角速度 + block 位置/线速度/
角度/角速度），角度维用 wrapped difference，容差 `1e-8`；**唯一允许不同的是目标 hidden damping
本身及其在历史中造成的可见响应**。除状态外还必须同时匹配：

- `query_pixels_identical`：query 画面逐字节相同；
- `query_actions_identical_within_direction`：同一 forward 或 reverse 下，两种 damping 的 query
  动作程序逐元素相同；两个方向的 Cartesian 向量只需满足预注册 equivariance 与 action-cell 身份；
- 每个 query raw step 的动作 L2 范数 `<= 1`；rigid transform 前后逐步范数相等，禁止靠 clip 修复越界；
- prefix action program 固定为全零，`history_actions_identical`；
- history 全程 contact-free，query 起点没有 contact/arbiter；query action 必须使两个 condition
  都接触 block，且首次接触 raw step 相同。首次接触后的接触次数允许因 damping 产生差异，但必须
  完整记录，不能再作为构造匹配量；
- 两种 damping 的 history 最小 agent-block clearance 必须逐 mode 为正并完整报告；同时记录候选
  接受/拒绝率，避免缩短 clearance 后只保留一侧无历史接触的幸存样本；
- `goal_pixels_identical`；
- 每条轨迹从 `x0` 到 `x3` 单一模拟器连续推进，`state_installations_after_x0 == 0`、
  `query_simulator_recreated == false`。

### 3.3 common random numbers

同一 group、同一方向内的两个 condition，以及 §3.4 的 `a_ref` / `Gamma-null` 反事实 rollout，
必须使用**同一个 `simulator_seed` 与同一条 RNG 派生链**；forward/reverse 使用由 group seed
确定性派生、彼此独立的子流。CRN 是 `Gamma` 可解释性的前提：没有 CRN，条件差
与随机数差无法分离。shadow manifest 与 pilot 使用**不同**的 catalog seed，但同 group 内部仍是
CRN。RNG 回执必须逐 group 写入 provenance。

### 3.4 `Gamma`：state-action interaction 的 difference-of-differences

对 group `u`、方向 `r`、hidden condition `c ∈ {0,1}`，令 `y^phys_{u,r,c}(a)` 与
`y^pix_{u,r,c}(a)` 分别为 query 动作程序 `a` 下 native 监督终点（`x3`）的完整 simulator state 与
`[0,1]` RGB target。参考动作程序固定为 `a_ref = 全零 query action`。它与 D0/D1 使用相同动作
语义，但新 query geometry 未必属于 D0/D1 分布，因此只能称**同 geometry 的零动作反事实**。定义

```text
Delta^s_c(u,r,a)   = y^s_{u,r,1}(a) - y^s_{u,r,0}(a), s in {phys,pix}
Gamma^s(u,r)       = Delta^s_c(u,r,A(u)) - Delta^s_c(u,r,a_ref)
Gamma^phys_comp    = agent/block 的 position、velocity、angle、angular-velocity 分量分别报告
Gamma^pix_energy   = mean_{h,w,c} (Delta^pix(A) - Delta^pix(a_ref))^2
Gamma_axis         = 先对齐每个 r 的 query approach / tangent 局部坐标，再聚合带符号 block response
```

像素口径固定为**模型实际可见的后编码再解码 RGB**，分辨率固定 `224`：使用 release builder 的
同一 encoder/decoder，将 decoded `uint8` **先**转为 `float64 / 255`，再分别计算
`Delta^pix(A)`、`Delta^pix(a_ref)` 和 `mean_{H,W,C}(Gamma^pix ** 2)`。禁止 `uint8` 直接相减，禁止额外
resize/crop/normalize；四个 `{condition0, condition1} × {A, a_ref}` rollout 共用 CRN。renderer 原始
`uint8` 口径只作编码敏感性检查。render 函数、分辨率、通道顺序、压缩参数、encoder/decoder 版本与
转换前后 hash 全部进入 receipt。

`Gamma` 是**向量组**，必须同时报告带符号物理分量与像素能量，禁止把 px、px/s、rad、rad/s 未经
定义地平方相加成一个物理范数。主排序量使用共同单位的 `Gamma^pix_energy`；物理量按分量验证方向，
角度如需转成表面位移必须使用预注册 body radius。它衡量"动作在当前状态与隐藏动力学下实际放大的
条件效应"，因此：

- `A(u) != 0` **不构成** `Gamma != 0`。若动作在该状态下不与隐藏 damping 交互（例如接触自由且
  动作只驱动 agent 的不可观测维度），`Gamma` 可以精确为零。审计必须显式报告
  `fraction_of_groups_with_negligible_gamma`，并按 §3.5 判定池是否退化。
- `||A(u)||` **不得**作为 `Gamma` 的代理量。必须报告 `Gamma^pix_energy` 与 `||A||` 的秩相关，以及
  给定 `||A||` 分箱后的 `Gamma` 条件离散度；若 `Gamma` 近似是 `||A||` 的确定性函数，则该池退化为
  "动作幅值"设计，不满足合同（Motion Development 的 action-norm 分层全为零，已经说明幅值不是
  leverage 的合法定义）。
- 还必须分别报告 `E[Delta^pix(A)^2]` 与 `E[Delta^pix(a_ref)^2]`，并检查 `Gamma^pix_energy` 与
  `a_ref` 条件能量的秩相关，以及在 `a_ref` energy 分箱内的 Gamma 离散度。否则高 Gamma 可能只是
  旧零动作条件差在像素支撑上的重排，不能被解释为独立的 action leverage。

`Gamma` 的计算需要 `a_ref` 反事实 rollout。**默认**：`a_ref` rollout 是 **audit-only**，只在
generation/curation 阶段用 CRN 复算，不写入训练行（避免抬高近重复质量）。P0 已冻结这一选择，
不再把 `a_ref` 是否入池作为可搜索分支。

### 3.5 `Gamma` 非退化分布要求

`Gamma` 池必须是"多方向、多幅值、多 action cell 的非退化分布"，且不被单一尾部支配。结构层面
现在冻结的是**规则**，数值配额在 pilot 后一次冻结：

1. **方向多样性**：按 block response 相对 query approach 轴与 tangent 轴的投影分箱，多个方向箱都必须
   有 group；`Gamma_axis` 的正负两侧都必须有非平凡份额（避免单一系统性方向）。
2. **幅值多样性**：`Gamma^pix_energy` 与物理分量的多个分位区间都必须有 group，不允许只有一个极端层。
3. **action cell 多样性**：在 `(query action 方向 bin × 幅值 bin × query 状态几何 cell)` 的联合
   网格上，每个被声明支持的 cell 都必须有最小 group 数。
4. **不被尾部支配**：`sum_u Gamma^pix_energy(u)` 的 top-`q%` 份额有上界；`Gamma` 加权曝光的 ESS 有
   下界；并且 **leave-top-`q%`-out 稳健性**——删除 top-`q%` 的 `Gamma` group 后，D2 相对 D1 的
   `rho_phys/rho_pixel` 改善方向必须保持（这是结构规则，`q` 与上下界在 pilot 后冻结）。
5. **非退化于动作幅值**：§3.4 的秩相关与条件离散度检查通过。
6. **不被参考条件差支配**：`Gamma` 与 `Delta^pix(a_ref)` energy 的秩相关有上界，且在相同 reference
   energy 分箱内仍有足够 Gamma 离散度；界限只在 calibration 冻结一次。

任一条不满足即 D2-0 **no-go**，回到 generator 设计，不允许通过调权、换 seed 或改口径补成正例。

## 4. 目标与约束：多目标向量，不压成单标量

D2 沿用"受约束的分布设计问题"框架（策略文 §6.0），但目标与约束都是**向量**，不允许合成一个
总分：

```text
目标（同时、非替代）：
  ↑ 相对量   rho_phys@{32,64,128}, rho_pixel@{32,64,128}   （ratio-of-means，局部背景口径）
  ↑ 交互量   Gamma^pix_energy 与物理分量的非退化分布（§3.5）
约束（硬门 + pilot 冻结界限）：
  absolute separation：每方向 future gap >= 2.0 px，future pixels 可区分（结构硬门）
  diversity：ESS 下界、近重复率上界、top-mass 份额上界（pilot 冻结）
  support：D0 的全部 query/state/action/goal-distance 支撑 cell 仍有正曝光；
           action 在原 PushT 动作范围、5-step block 格式及冻结 natural/planner 经验支持内
  amplitude：response amplitude 分位数与相对 D0 的 1-Wasserstein 在等价界限内（pilot 冻结）
  natural anchor：保留 D1 冻结的 50% 自然锚点曝光（结构硬门，避免新增可搜索超参数）
  预算：总 transition 数、batch 行数、optimizer steps 与 D0 精确相同（结构硬门）
```

**两层改善判定规则**（现在冻结，不含数字）：

- `B_loc` 邻居图必须在**相同 query-action cell** 内匹配 canonical query descriptor，禁止把不同
  action program 的普通 future 方差混进背景后冒充相对条件份额提高。
- **层 I，D2 内精确配对**：active action 与 audit-only `a_ref` 共享同一 base geometry、direction、
  hidden conditions 与 CRN；以 group 为 cluster 比较条件分子并报告 `Gamma`。它回答动作是否改变
  条件效应；由于 active/ref 属于不同 action cell，**不得**把两边不同的 `B_loc` 冒充同一背景后做
  `rho` 配对门。
- **层 II，跨分布标准化**：D1 只有零 query action，不能与 D2 contact-action 行冒充同 action-cell
  配对。D2 与 D0/D1 只在共同的 orientation × speed × goal-distance coverage cells 上做质量标准化
  后的 cluster-bootstrap 分布差；各池的 `B_loc` 都只在本池相同 query-action cell 内计算。它回答新
  训练分布是否整体更强，不承担“只改变 action”的单因素 claim。
- 两层都同时报告分子 `E[C]` 与分母 `E[B]`，以识别"分子分母同步上升"的伪改善。
- `rho` 主总体固定为**实际训练曝光多重集**，包括 50% natural anchor；只看 designed groups 的结果
  只能作为次要描述。六个 `rho_phys/rho_pixel@{32,64,128}` 在 D2−D0 与 D2−D1 的 simultaneous
  cluster-bootstrap 95% 下置信界都必须大于零，禁止事后挑一个尺度。
- `Gamma` 只由 §3.5 的非退化门判断，不与 D0/D1 做“改善”门：D0/D1 的 active action 就是
  `a_ref`，其 Gamma 恒为零，任何非零值都会自动通过，属于空对照。
- **不劣类**约束（diversity、support、amplitude、planner 保持）用 **equivalence/TOST 规则**：
  差值的 CI 必须落在预注册等价边界内；等价边界在 pilot 后一次冻结。
- **禁止跨任务阈值**：不引用 Action Strength / ActionDelay / Speed 的 `rho` 数值做门槛，也不设
  跨模型的绝对梯度阈值。所有比较都限定在同单元、同口径，并按上述层级报告不确定性。

## 5. identifiability 与 leakage 契约

### 5.1 必须可识别的对象

D2 的正例前提是：**完整 observation-action history**（三帧图像 + 全部已执行动作）在
**query-family held-out** 与 **action-program-family held-out** 两种划分下都能识别隐藏 condition。
可识别性由离线 probe 在 Training-only 数据上评估，probe 的训练/评估划分按 query 家族与 action
program 家族切分，不按随机行切分。当前 generator 是确定性 open-loop 构造，**不存在 collector
policy**；不得为了满足一个名词而额外发明 policy。若未来引入闭环 collector，才追加
collector-policy-held-out 门。可识别下限属于效应门，pilot 后冻结。

### 5.2 必须接近 chance 的对象

- `query_only`：仅 query 帧（+ goal 帧）→ condition；
- `prefix_action_only`：仅 prefix 动作程序（含 query 动作程序）→ condition；
- `x0_static`：仅 `x0` 帧 → condition（继承 benchmark v1 的 50% 上限与几何分类器 55% 上限口径）。

### 5.3 结构断言优先

若某个通道在每个 direction 的两个 condition 间**逐字节相同**（D2 的 canonical 构造保证 query
帧、query 动作、prefix 动作在两种 damping 下字节相同，且全池 action multiset 按 mode 相同），
则该通道的 leakage 判定**优先使用结构断言**（字节/哈希相等、
多重集相等、镜像标签交换），probe 只作为冗余检查。理由：结构断言是精确、可复现、不依赖 probe
容量与超参数的证据；训练 probe 反而引入"probe 太弱"这一无法排除的替代解释。对无法结构断言的
通道（例如跨 group 的动作程序分布是否与 condition 相关），才使用 held-out probe，并同时报告
permutation null。

### 5.4 tensor boundary

模型/loss 边界必须由 receipt 保证只收到 `pixels, action`：

- 训练与审计入口对每个 batch 断言输出 tensor 的 dtype/shape/字段集合；索引流只允许 `int64`
  行索引，`Gamma`、coverage cell、arm、`hidden_mode`、`pair_id`、shadow 标识一律不得跨界；
- 违反即 fail closed（抛错终止），不写 warning 继续；
- receipt 记录：Development/Public Test read count `= 0`、optimizer step 计数、checkpoint 前后
  参数/buffer/module mode/RNG 状态、以及模型可见字段清单。

## 6. D2-0：Training-only CPU 构造门

D2-0 只读新生成的 Training 候选池与冻结的 D0/D1 资产；不加载模型、不占 GPU、optimizer step 为
`0`、不访问 Development/Public Test。它的通过条件分两组。

**结构硬门（现在冻结）**

1. 完整 matched dynamics group：每个 group 恰含 4 条 condition rows，forward/reverse × 两 mode
   完整，无孤儿行、无跨 group 混装；
2. canonical query state 匹配（§3.2）全部通过，包括 12 维状态 `1e-8`、query 像素/动作字节相等、
   全零 prefix、history contact-free、query 起点无 arbiter、两个 condition 首次接触 raw step 相同、
   goal 像素相等、`x0` 后零状态安装、模拟器不重建；
3. CRN（§3.3）逐 group 校验通过；
4. prefix 全零；同一方向内 query action 逐元素相同；forward/reverse action 满足冻结 equivariance；
   每步 L2 范数 `<=1`、旋转前后范数相等且不裁剪；horizon 相同（`MODEL_FRAME_ROWS` 不变）；
5. 使用 research-local D2 validator；逐 rollout 验证 query-start arbiter 定义、raw-step/substep contact、
   固定 contact friction 与当前 mode 的 `space.damping`；两种 mode 的 history 最小 clearance 都为正；
6. 绝对 separation：每个方向 future gap `>= 2.0 px`，且 future pixels 可区分；历史可见响应
   `>= 3.0 px`；
7. leakage 结构断言（§5.3）全部通过；
8. 支撑结构：D0 的全部 coverage cell 仍有正曝光；动作在原始 PushT 范围与 action-block 格式内；
9. 预算等同：总 transition 数、batch 行数、optimizer steps 与 D0 精确相同；自然锚点 50% 结构保留；
10. 与 D2-shadow、以及 ContextWorld 四个冻结 split 的 template-id / query-hash / 几何 hash 重叠
   计数为 `0`；
11. **完整 provenance/hash**：generator 源码 SHA256、config SHA256、catalog seeds、每条 clip 的
    pixels/actions/physics hash、per-group receipt、release manifest SHA256、审计脚本 SHA256、
    输入目录 SHA256，全部写入不可覆盖的 artifact；独立目录复跑必须逐字节一致。

**效应门（pilot 后一次冻结，本轮只写规则）**

- 实际训练曝光多重集上的六个 `rho_phys/rho_pixel@{32,64,128}`，相对 D1/D0 的 coverage-standardized
  simultaneous cluster-bootstrap 改善；六项全部通过，不称 paired；
- `Gamma` 非退化配额（§3.5 的方向/幅值/action cell 最小份额、top-mass 上界、ESS 下界、
  leave-top-out 稳健性）；
- contact robustness：非 grazing；预注册小幅 geometry/action 扰动不改变两支接触存在性，且
  `Gamma` 的符号/主方向稳定；
- first-contact substep 差异落在 calibration 冻结的等价界限内；
- 报告 first-contact raw-step 门的剔除率、被剔除 group 的 Gamma 分布、两侧首次接触 substep、冲量与
  penetration depth，避免按 damping 下游结果静默筛样；
- natural/planner 经验 support overlap 的等价门；
- diversity：ESS 下界、近重复率上界、top-mass 上界；
- amplitude：分位数与 `W1` 等价界限；
- probe 可识别下限与 near-chance 上界（对无法结构断言的通道）。

任一结构硬门失败 → 先修 generator，不进入 pilot 冻结、不训练。效应门失败 → D2-0 **no-go**，
回到 generator 设计层，**不得**降低门槛、换 `k`、换 seed 或用重加权补救。

## 7. Training-only feasibility pilot

pilot 分两段，二者都**不看 Development、不训练模型**：

1. **P1a 物理可行性探针**：只验证 query-contact 路由。固定全零 prefix，从 §2.1 已冻结的
   planner-curve receipt 派生候选幅值和 approach/tangent 方向；D0/D1 的零 query action 不作为幅值源。
   优先调整 canonical agent-block clearance 让 support 内动作产生接触，而不是把动作推到分布尾部；
   同时记录两种 mode 的 history 最小 clearance 与候选接受/拒绝率。必须复现“无接触 `Gamma=0`、
   接触后 `Gamma` 非零”，并通过动作 L2/旋转门、新 D2 validator、同首次接触 step、固定 friction 与
   `Gamma-null` comparator。
2. **P1b calibration/holdout pilot**：预先按 coverage/action cell 分层成 `192 + 192 = 384`
   groups，全部以模型训练分辨率渲染。calibration 只用于冻结测量噪声、自然等价边界与 action grid；
   holdout 在冻结后一次性验门。两部分及其 `a_ref` rollout 都不进入训练池。

- 输入：上述 `384` 个新 group，CPU/渲染预算；
- 输出：`rho_phys/rho_pixel` 的可达区间、`Gamma` 分布形状、diversity/duplicate/top-mass 的实际
  分布、amplitude 分位数与 `W1`、probe 可识别性的量级；
- 冻结规则：只用 **D0/D1 冻结基线 + calibration 的测量噪声/自然参考分布**设最小可检测改善与
  等价边界，禁止用 calibration 中已经看到的 D2-D1 效应大小把门放到它下方；写入 YAML 后，
  holdout 才能开启一次。holdout 通过/失败都不得改门；
- **一次性**：冻结后不得重开。pilot 数据**不进入** D2 训练池（避免选择性偏倚），也不得用于挑选
  loss、mixing weight、曝光比例或 Development checkpoint。若 holdout 失败，当前 holdout 即烧毁；
  任何 generator redesign 必须升级 protocol/config 版本、递增 `generator_revision_index` 并使用未生成
  的新 seeds；已经冻结的数值门限和等价边界必须逐字节复用，不能重新估计。receipt 记录每版
  `holdout_open_count`。sealed holdout 只控制 calibration 后的抽样噪声，**不**声称控制 generator
  选择偏差；若估计量改变到旧门限不适用，必须终止本研究并另立新假设，不能称为续跑。

## 8. D2-1：单卡零 optimizer-step 机制门

D2-1 在冻结初始化 checkpoint 上运行，**optimizer step 严格为 `0`**，不访问 Development/Public
Test，统计单位是完整 group（不把同 group 的两个方向当独立样本）。必须同时报告以下六组量：

1. **V1 `rho_lat`**：复用 D2-0 冻结的 physical neighbor graph，在冻结 target encoder 坐标下计算
   local `rho_lat@{32,64,128}`，并分别给出 latent 条件分子与背景分母；不得在 latent 空间重新
   选择最有利邻居；既有 global 口径另列，不与 local 口径横向比较。
2. **objective-route 的 scale-aware 与 scale-invariant 两套量**：沿 response-residual 有利方向的
   一阶变化（scale-aware，依赖 loss scale，不得读成学习速度倍数）与按总梯度范数归一化后的方向
   效率（scale-invariant）。这是 ActionDelay 门的直接教训：`48.583x` 与 `17.35x` 是两个不可互换的
   口径。
3. **response 梯度 / coherence / SNR**：按 `predictor`、`pred_proj`、全参数分别报告
   `||E g_resp||`、`E||g_resp||`、相对非条件梯度的范数比、coherence `c=||Eg||/E||g||`、
   group-cluster `Bcrit` 与目标 batch size 下的 `SNR(B)`；分母为零时 fail closed，不报无穷大。
4. **center-response interference**：`cos(g_center, g_response)`、`g_response` 在 `g_center` 张成
   方向上的投影份额，以及正交剩余。用于区分"条件梯度弱"与"条件梯度被 center 更新抵消"。
5. **virtual global-sample-shuffle 与 group-block 比较**：在**同一完整 model-example 多重集、同一
   冻结参数、零步**下，
   用两种虚拟 batch 划分计算每批 `c_b`、批内条件梯度均值、批间余弦与全局矩量；必须做 eval-mode
   不变性检查以把 `pred_proj` 的 BatchNorm 耦合与随机噪声分开，并用共同 RNG 多次复算。该比较
   只回答"batching 是否改变有限步优化输入"，不回答"曝光分布是否更好"。
6. **`Gamma`-high / `Gamma`-low 传输**：按 `Gamma^pix_energy` 预注册分层（边界在 pilot 后冻结），
   比较两层的 `rho_lat`、response 梯度与 SNR；同时在 coverage cell 与
   `Delta^pix(a_ref)` energy bin 内置换完整 group 的 Gamma 分层标签，形成预注册一侧 null。观测到的
   联合传输统计量必须超过 calibration 冻结的 null 分位；否则 Gamma 只是构造排序或 reference-energy
   代理，不能称为进入了表示与梯度。

**通过规则**：`rho_lat` 三尺度满足 pilot 后冻结的联合方向门；response 梯度可见性向量
（`||E g_resp||`、coherence、`SNR(B)`）满足预注册的联合判据，而不是事后挑其中最有利的一项；
`Gamma`-high/low 传输呈预注册方向并超过上述置换 null，且没有 `pred_proj` 明确反向证据。**失败即停止**：不进入训练、
不搜索 mixing weight、不调曝光比例、不换 batching 补救；
回到 generator/设计层，并把该失败写成"数据操作未进入表示或 optimizer 路径"的机制结论。

## 9. 第一轮训练矩阵：顺序最小设计

| arm | 内容 | 类型 | 触发条件 |
|---|---|---|---|
| **A** | `D0 + native` 冻结终点 | **复用**，不重训 | 已存在 |
| **B** | `D1-MS50 + native` 冻结终点 | **复用**，不重训 | 已存在 |
| **C** | `D2 + native`，**global sample shuffle** sampler | **唯一首发训练格**（1 GPU, seed `14321`, 8,192 steps） | D2-0、D2-1 与 native additivity 零步门均通过 |
| **D** | `D2 + native`，**group-block** batching（与 D0/D1 匹配的 batching） | 条件启动的第二训练格 | **仅当** C 未达门 **且** D2-1 的 §8.5 比较显示明确 batching 依赖 |
| **E** | 自然 Development 终点 history 干预：correct / swapped / removed；`Gamma` 分层只在一次性 shadow 报告 | **终点评测干预，不是训练臂** | C（或 D）完成后 |
| **F** | 终点校准与保持性干预：amplitude/NRE 分解、原任务 CEM300 | **终点评测干预，不是训练臂** | C（或 D）完成后 |
| **G** | `D0 + COJA` 冻结正对照 | **复用**，不重训 | 已存在 |

这里的 `global sample shuffle` 精确定义为 **完整 model training example / clip 级全局随机化**。
绝不打乱 Lance table 内一个 clip 的时间行，绝不打乱 `MODEL_FRAME_ROWS`，也不把
pair/group metadata 送入 model 或 loss。

arm C 先冻结一个 `16,384` 个完整 model examples 的**曝光多重集**：`2,048` 个完整 natural groups
和 `2,048` 个完整 D2 groups，各展开为 `8,192` 个 examples；每个 example 恰好重复 `32` 次，共
`524,288 = 8,192 steps × 64` 个训练槽位。natural groups 在每个 coverage cell 内用确定性等质量 hash
选择，保持所有 cell 正曝光；总 unique example 数与 D0 相同。sampler 对该多重集运行 `32` 个 seeded
permutation cycles，每个 cycle 恰好曝光每个 example 一次，
只按全局 sample index 置换，不按 hidden mode、pair 或 group 做 batch stratification。实现沿用 D1 的
注入点，替换 `CompleteTwinPairedBatchStream` 但保持构造签名；运行时向 loader 交付的唯一策展信息是
长度 `64` 的 `int64` sample indices，且曝光多重集 hash 写入 receipt。

**为什么 C 用 global sample shuffle，而这最直接检验 data content。** D0/D1 使用
`CompleteTwinPairedBatchStream`：同一 group 的四条 condition rows 必然同批出现。即使 native loss
是逐样本 MSE，这仍是一种 **batch 级隐式配对结构**——`pred_proj` 含 BatchNorm，且有限步优化轨迹
可以利用同批内匹配条件的对比。如果 D2 直接沿用 group-block，一旦出现正结果，就无法区分"新数据
内容起作用"与"同批配对提供了隐式对比监督"。global sample shuffle 把 group 结构从 batch 组成中移除
（每批从冻结曝光多重集做确定性置换，匹配条件同批出现只是偶然事件），因此 C 的正结果支持
`loss-native, data-designed`，并排除 group 邻接所提供的 pair-level/同批对比监督；它不把一般随机
batch 优化动力学宣称为已被消除。这也与既有 twin co-batching 负对照一致：同批出现本身不足以诱导
历史使用。

必须同时写明的边界：C 相对 B **同时**改变了数据内容与 batching，因此 C-vs-B 不是单因素比较；
只有在 D 启动后，C/D 一起才构成 batching 维度的分解。此外，group 仍是构造与统计聚类单位（§3.1），
sample shuffle 只作用于 batch 组成层。

arm C 启动前还有一个**零步 native additivity 硬门**：源码审计确认当前 native objective 路径不存在
依赖 batch 邻接的 `[0::2]/[1::2]` 或等价 pair 索引；在 eval mode、共同 RNG、同一参数与同一完整
example 曝光多重集下，只改变 batch 分组/顺序时，曝光加权总 native loss 与每个参数的总梯度必须在
按重复计算噪声冻结的容差内相等。train-mode BatchNorm 差异另列描述，不作为 objective 可加性的证据。
该门失败则 arm C 合同无效，先修 loss/sampler 口径，禁止训练。

首轮**不做**：多 seed 铺量（正信号只触发复现计划）、`D2 + COJA`、COJA+rollout、第三个任务、
loss 权重搜索、Development 上的 checkpoint 挑选。

## 10. 冻结内部 shadow manifest

shadow manifest 是**内部**的一次性泛化探针，与 benchmark 完全隔离：

- 三个 held-out 轴：**leave-query-family-out**、**leave-action-program-out**、
  **leave-generator-template-family-out**；每轴的划分键、group 清单与 SHA256 在**训练开始前**冻结；
- **不用于调参**：不参与 generator 选择、曝光设计、超参数、checkpoint 选择或任何门的定标；
- **只在 recipe 与 checkpoint 全部冻结后开启一次**；开启记录（时间、脚本 SHA、读取计数）写入
  receipt；重复开启视为协议违规，结果作废；
- **不进入 ContextWorld registry / scoreboard**，不产生 benchmark 判定，**不是 benchmark v2**；
  其结果只允许写成"内部泛化方向证据"，不得与 benchmark v1 的 Development/Test 数值并列成绩表；
- 与 D2 训练池、pilot 池以及 ContextWorld 四个冻结 split 的重叠计数必须为 `0`（结构硬门）。

Public Test 在全部阶段保持锁定；shadow manifest 不是 Test 的替代品，也不解锁 Test。

## 11. 阶段 P0--P5：资源、go/kill 与分流

| 阶段 | 内容 | 资源 | go 条件 | kill 条件 |
|---|---|---|---|---|
| **P0** | 冻结本预注册（本文件 + YAML） | CPU，无 GPU | 主线程回答 §14 的阻塞项并批准 | 未批准即停在 P0 |
| **P1** | P1a query-contact 物理探针；P1b `192/192` calibration/holdout pilot；一次性冻结效应/等价界限 | CPU + 渲染 | holdout 在冻结门下通过 | 接触路由无法在冻结 planner support 内产生非退化 `Gamma`，或需破坏 canonical/history 门 → 停止 D2，转表示/目标路线 |
| **P2** | 全量 D2 生成 + D2-0 构造门 + 独立复跑 | CPU/IO（渲染与存储为主） | §6 结构硬门与效应门全部通过 | 任一门失败 → 回 generator，不训练 |
| **P3** | D2-1 单卡零步机制门 | 1 GPU，短 | §8 通过规则成立 | 失败 → 停止训练，不搜 mixing weight |
| **P4** | native additivity 零步硬门；arm C 唯一训练格；条件 arm D | 1 GPU，8,192 steps | additivity 门与 P3 通过 | additivity 门失败则不训练；训练契约任一失败则作废重跑而非改判据 |
| **P5** | 冻结自然 Development 终点 + E/F 干预 + CEM300；随后一次性开启 shadow manifest | 1 GPU + CPU | recipe/checkpoint 已冻结 | — |

**结果分流**（预注册规则，不是预测）：

| P5 观测 | 结论 | 下一步 |
|---|---|---|
| 自然 Development 达到历史使用正例（on-support `G_swap` 正、sign-flip 显著、gain 与 NRE 同向改善且 NRE < 1），CEM 保持 | data-designed native ICL 在 Motion-LeWM 成立（单 seed） | 先补 `>=3` 个 native training seeds 复现；之后才讨论 `D2+COJA` 与跨任务 |
| 仅方向效应（配对 CI 不跨零但未过历史使用门） | 与 D1 同类结论：数据内容有因果贡献但不充分 | 进入 binding-bottleneck 判据（§12）定位下一层，不追加同池重复或救援式调权 |
| assignment 改善但 gain/NRE 过放大（校准失败） | 条件分配已建立，后段缺自然幅值校准 | 走预注册的 high-`Gamma` → natural 数据调度（D3 类），不加专项 loss |
| 原任务 CEM300 明显下降 | support shift / 过度重复 | 保留机制结论但不作为主模型配方；提高自然混合或扩充覆盖，不改 ICL 判据 |
| D2-1 通过但 C 未达门且 D2-1 显示 batching 依赖 | 有限步优化动力学可能是限制层 | 启动 arm D，仅此一格 |

## 12. V0--V5 向量化 stage contract 与 binding bottleneck

**stage contract 是向量，不是标量。** 禁止跨空间构造单一标量 `T_k`（例如把 `rho_phys`、
`rho_lat`、梯度范数与行为分数压成一个"传输分"）：这些量单位不同、依赖不同（物理/encoder/
参数化/batch 组成），压成一个数会把"数据弱"与"表示或 Jacobian 衰减"混为一谈。

| stage | 层 | 向量分量（同层、同单位、可配对比较） |
|---|---|---|
| **V0** | 数据构造 | `C_phys`、`B_loc@{32,64,128}`、`rho_phys@k`、`C_pixel`、`rho_pixel@k`、每方向最小 separation、`Gamma^pix_energy` 与物理分量（方向/action cell/top-mass/ESS）、duplicate 率、amplitude 分位数与 `W1`、coverage 配额 |
| **V1** | 冻结表示 | local `rho_lat@{32,64,128}`、latent 条件分子与背景分母、latent target separation、center/response risk 分解 |
| **V2** | 目标路由 | scale-aware 一阶变化、scale-invariant 方向效率、response/native risk 比、route cosine |
| **V3** | 参数梯度 | `\|\|E g_resp\|\|`、`E\|\|g_resp\|\|`、相对非条件范数比、coherence、`Bcrit`、`SNR(B)`、参数组分解、center-response interference |
| **V4** | batching / 有限步动力学 | 每批 `c_b`、批内条件梯度均值、批间余弦与顺序自相关、eval-mode 不变量、train-mode BN 差值与共同 RNG 不确定性、shuffle vs group-block 对比 |
| **V5** | held-out 行为 | on-support `G_swap`（均值/中位数/正号比例/sign-flip/cross-query null）、gain、alignment、`NRE` 的尺度与正交残差分解、`Gamma` 分层表现、CEM300 保持、removed-history（仅辅助） |

比较规则：只允许**同 stage、同口径、同单元**的配对比较与不确定性；跨 stage 只允许方向性陈述
（"上游升、下游不升"），不允许比值或代数合成。

**binding bottleneck 的五个必要条件（必须同时满足）**：

1. **上游可用**：该层的上游向量在干预后确实提高（不是仅内部自洽）；
2. **下游衰减**：在同一单元、同一口径下，下游 stage 相对上游存在可测量的衰减；
3. **匹配干预**：存在一个只改变该层的干预，其余受控量逐元素或统计等价（并有 receipt 证明）；
4. **下游及后续改善**：该干预同时改善**该层与其后续层**的预注册量，而不是只改善本层；
5. **替代解释受控**：placebo/comparator（曝光 placebo、`Gamma`-null comparator、batching 不变性
   检查、CRN）与已知混杂（loss scale、coverage、duplicate、RNG、初始化）已排除，且未使用跨任务
   或跨模型阈值。

五条不全满足时，只能写 **bounded local mechanism evidence**，不得写 binding cause；下游表型
（弱梯度、`G_swap` 近零、NRE 漂移）永远不得倒写成上游原因。

## 13. P0 所有权与后续计划资产

P0 冻结时只新增本文件与 YAML（既有文档允许增加入口链接）。后续阶段的资产均在本仓库
`research/conditional_dynamics_representation/` 下新增，不触碰 `../ContextWorld`、`paper_zh`、
`TECHNICAL_REPORT.md` 与既有 artifacts/scripts/tests：

- `scripts/probe_pusht_motion_damping_d2_p1a_v1.py`（P1a Training-only CPU 物理路由门）
- `scripts/build_pusht_motion_damping_d2_groups_v1.py`（research-local generator，只读导入 ContextWorld）
- `scripts/audit_pusht_motion_damping_d2_construction_v1.py`（D2-0 Training-only 门）
- `scripts/analyze_pusht_motion_damping_d2_zero_step_v1.py`（D2-1 零步门）
- `scripts/run_pusht_motion_damping_d2_native_v1.py`（arm C，条件 arm D）
- `tests/test_pusht_motion_damping_d2_*.py`（确定性、group 完整性、tensor boundary、禁止 split）
- `artifacts/pusht_motion_damping_d2_*/`（拒绝覆盖，含完整 receipt）

## 14. P0 主线程裁决

三个原阻塞项已经裁决：采用 query-contact 路由；完整 12 维只分量报告、不混合单位；pilot 固定为
`192` calibration + `192` holdout groups，由主线程按本合同审阅并一次冻结。YAML 中只保留不阻塞的
执行默认项。

---

**本文件是执行前合同，不承载实验结果。** 状态为
`structural_preregistration_frozen_effect_bounds_pending_pilot`；所有效应与等价界限仍待 §7 的 pilot
一次性冻结。运行结果只进入独立 receipt 与实验日志，不反向修改本合同；Public Test 保持锁定。
