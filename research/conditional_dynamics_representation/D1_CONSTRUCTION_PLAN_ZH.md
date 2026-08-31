# D1 数据分布构建与 native 因果验证计划

状态：2026-08-31，构建方案经独立只读审查后冻结，可进入 CPU schedule 构建；尚未启动
`D1 + native` 训练。本文只定义训练数据臂；ContextWorld-v1 Development 保持不变，公开 Test
不访问。

## 1. 当前决策

首个数据实验固定为 **Motion Damping `D1-E50 + native`**：从同一冻结 Training 候选池提高
高物理条件响应样本的曝光，同时保留 50% 自然曝光锚点。它只改变训练样本分布，不改变模型、
loss、初始化、batch 行数、optimizer step 或评测分布。

本轮不做三件事：

- 不追加 COJA 训练、COJA+rollout 或辅助权重搜索；
- 不把 latent energy 直接写进首轮选择规则，避免数据配方绑定 LeWM 的当前 target encoder；
- 不声称验证了 action leverage。现有 Motion synthetic query action 全为零，同池重采样无法识别
  独立的动作杠杆效应；该问题属于后续 D2 新数据采集。

因此，D1 回答的单一因果问题是：**在同一候选池、相同 native 训练预算下，提高条件 future
物理响应能量并保留自然覆盖，是否足以让模型在未改动的自然 Development queries 上更多使用历史？**

这里的干预单位是完整的 **high-identifiability exposure schedule**，不是已被完全隔离的单一
`E_phys` 标量。top-`E_phys` 会连带改变 speed、geometry 和 response-amplitude 边际；首轮只检验
这种可实施训练分布是否充分，不能声称识别了“条件能量相对所有协变量的唯一因果效应”。builder
必须报告这些边际变化，后续只有在确有必要区分机制时才增加 speed-matched sham，而不把它设为
首个数据正例的前置条件。

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

所有分箱边界只由 Training manifest 计算。Development/Test 路径在 builder 中列入禁止读取集。

### 3.1 首轮选择量

对 twin `u` 的 forward/reverse 两个 pair，令 manifest 中未来 block 位置差为 `d_uf,d_ur`，定义：

```text
E_phys(u) = 0.5 * (d_uf^2 + d_ur^2)
```

同时保存、但不压成一个总分：

- `conditional_energy_physical`：上述 `E_phys`，并保留位置与角度原分量；
- `history_clue_physical`：两方向 `history_visible_response_gap` 的均值与最小值；
- `conditional_energy_latent`：共同冻结初始化下的 target-response energy，仅用于验证 D1 操作
  是否也提高当前表示中的 `rho_cond`，不参与 `D1-E50` 排序；
- `coverage_cell`：首轮用 `orientation_bin`，另记录 query speed、goal geometry 与响应幅值分位数；
- `pair_quality`：query pixels/action identity、future separation、无泄漏、forward/reverse twin 完整性；
- `action_leverage`：本轮写为 `not_identifiable_query_action_zero`，不得用 action norm 伪造代理量。

首轮使用 physical-only 排序是刻意的。若 physical exposure 明显提高而 frozen-latent `rho_cond`
不提高，应解释为表示压缩；若两者都提高而 native 仍不学，才把原因推进到参数 Jacobian、梯度
一致性或跨-query 覆盖。直接用 LeWM latent 选数据会混淆这两种情况，也不利于主模型预训练复用。
`E_phys` 使用的是 synthetic paired generator 已知的训练 target，是数据策展 oracle 而非模型输入；
正结果证明“这种分布可以解决”，不自动证明普通未配对预训练语料可用同一分数直接挖掘。迁移到
主模型时还需把它落成 paired simulation、主动采集或可观测 proxy。

### 3.2 高辨识池

在四个 `orientation_bin` 内分别按 `E_phys` 排序，以稳定 `twin_id` 打破并列。每个方向箱的最高
25% 构成 high-ID pool：manifest 已核验每箱恰有 `1,024` 个 twins，因此每箱取 `256` 个，合计
`1,024/4,096` 个 twin groups。分箱内选择避免高能曝光顺带改变整体 orientation 比例。

当前 pool 的 physical future gap RMS 为约 `2.153--3.690 px`，四分位点为
`2.512/2.878/3.277 px`；query speed 为约 `14.001--23.998`。高响应与 state leverage 的相关性
是 D1 配方的一部分，因此只要求完整 speed support 与自然锚点，不要求 speed 边际逐点不变；结果
必须写成“D1-E50 配方效应”，不能写成纯 `E_phys` 因果效应。

## 4. `D1-E50` 精确曝光表

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
- 每 cycle 四个 orientation 各 `1,024` slots；每个 slot 仍展开为完整四行 twin group；
- 总 twin 曝光 `131,072`，隐藏行 `524,288`；原始 PushT 行仍为 `524,288`，保持严格 50/50；
- 每批 arm/high-status 配额和 `twin_id` 唯一性严格固定；orientation 只在 cycle 总量上平衡。

构造 seed 固定为 `20260831`，训练 seed 仍为 `14321`。builder 可额外生成 E25/E75 的
**零训练分布审计**，但首个训练候选预先固定为 E50，不依据 Development 或短跑结果改选比例。

## 5. 最小实现边界

D1 不修改 ContextWorld-v1 release，也不复制约 1.9 GB 的 Lance pixels。只新增一个训练期索引层：

1. builder 从冻结 Training manifest 生成 per-twin catalog 和完整 8,192-step schedule；
2. `EnergyStratifiedTwinBatchStream` 读取 schedule，把每个 `twin_id` 映射回现有四条 condition rows；
3. D1 runner 继承 matched native runner，只临时替换
   `CompleteTwinPairedBatchStream`；模型、loss 和 materialized arrays 不变；
4. sidecar 记录 manifest、source code、checkpoint 与实际消费 schedule 的 SHA256。

计划新增文件：

- `configs/pusht_motion_damping_d1_energy_stratified_native_v1.yaml`：冻结本文件中的身份和比例；
- `scripts/build_pusht_motion_damping_d1_schedule_v1.py`：Training-only catalog/schedule builder；
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

GPU 空闲后，仅做冻结初始化上的零步检查：

- 报告 exposure-weighted physical/latent `rho_cond` 相对 D0 的变化；
- 在固定 32-pair batch 上复算 response/nonconditional 梯度范数与 SNR；
- 可加少量固定 batch 看方向是否一致，但不把全面 SNR 扫描设为 D1 训练前置审计。

该检查确认数据操作实际进入当前表示与 optimizer 路径，不用于选择 E25/E50/E75，也不访问
Development。

## 7. 训练与评测

### 7.1 唯一新增训练格

首轮只运行 `D1-E50 + native`，使用一个 GPU、seed `14321`、8,192 steps。保存
step `256/1024/2048/4096/8192`，但训练不中途按 Development 早停或换配方。DINO-WM、PLDM
及额外 LeWM seed 等该格给出方向后再决定。

### 7.2 冻结自然评测

训练完成后统一评分所有保存点；评测仍用 benchmark v1 的自然 Development，不构造 D1 专用
Development。D1 exposure-weighted Training 指标只作为 manipulation check，因为重权高
`||Delta t||` 会构造性放大未归一化 `G_swap`，不能作为学习成功判据。训练侧模型比较统一在全部
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
| physical/latent 能量均提高但参数梯度和 Development 都不动 | 同池数据重分布不足，Jacobian/表示或 objective 成为主要瓶颈 | 再考虑表示或 COJA 类显式条件目标 |
| 原始 CEM 明显下降 | support shift/过度重复 | 保持 D1 机制结论但不作为主模型配方，增加自然混合或新覆盖 |

在 `D1+native` 给出自然 Development 正信号前，COJA 保持论文方法和可学习性正对照；D1 不替换
COJA。若 D1 后续接近或超过 COJA，数据原则可升为主贡献，COJA 降为存在性证明或互补工具。
“接近或超过 COJA”必须由至少三个 D1 native training seeds 支持，不能由首个 seed 触发。
整个阶段只新增 training-distribution track，不改 benchmark v1；只有新增评测 query/action/split
或指标时才讨论 benchmark v2。

## 9. 执行顺序与资源

1. CPU 构建 per-twin catalog、E50 schedule 和不可变回执；
2. CPU 执行 schedule/batch/forbidden-split 全部门；
3. 单卡做冻结 latent 与最小梯度零步检查；
4. 单卡运行唯一的 `D1-E50 + native` 8,192-step 训练；
5. CPU/单卡完成自然 Development 轨迹与 final CEM300；
6. 根据 §8 决定 D2、D3 或 `D1+COJA`，不并行铺开。

因此当前即可开始 D1 构建，前两步不占 GPU。真正需要 GPU 的只有一次轻量零步检查和随后唯一的
native 训练格；它们都不与当前云侧 D0 native 队列构成逻辑前置关系。
