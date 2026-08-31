# 世界模型条件 ICL：通用根因与数据构造路线

状态：2026-08-31，第一轮零训练诊断已完成。本文只使用训练数据和 Development；公开 Test
尚未访问。

## 1. 当前结论

当前证据不支持“COJA 失败”。恰恰相反，COJA 是已经通过的条件学习正对照：在相同 Motion
Development pairs 上，native 的 gain/NRE 为 `0.0065/1.1298`，COJA 为
`0.3696/0.7665`；COJA 的逐 query `G_swap` 有 `96.9%` 为正，而 native 只有 `44.1%`。
它证明现有历史、查询与 future 中存在可学习的条件信号。

同时，现有 rollout 对照已经否定两个更简单的解释：动态扩展历史窗口没有救回 ICL，多步
native rollout 也不能替代 COJA。它们增加了可见上下文或预测视界，却没有提高正确历史相对
交换历史的训练压力。

第一轮 Motion 诊断支持一个三部分、仍需跨任务验证的根因：

1. **数据能量弱**：冻结 latent 中的条件能量只占总 target variance 的约 `0.23%--0.29%`；
2. **参数可见性更弱**：response loss 在终点训练 pair MSE 中占 `13.6%`，但映射到真实
   optimizer 参数后，response 梯度相对非条件梯度的范数只有 `5.57%`，两项平方能量占比
   只有 `0.309%`；
3. **覆盖不足**：native 在冻结训练 batch 上把平均 `G_swap` 从负值推到正值，但在
   Development 上仍接近零且随机化检验不显著，说明仅提高已有 query 的曝光可能继续得到训练
   条件记忆，而不是可迁移的历史使用。

因此下一步应优先研究**哪些样本让 native 条件梯度更强、更一致且能跨 query 迁移**，再决定
数据重采样、定向扩充和自然幅值校准混合。专项 loss 暂不扩展；COJA 保留为正对照和机制探针。

## 2. 五层因果链

不能用一个 NRE 或一个 loss gap 概括根因。后续统一按五层检查：

| 层 | 问题 | 主要量 |
|---|---|---|
| 数据可辨识性 | 相同或近邻 `(Q,A)` 下，不同历史是否真的对应不同 future | overlap、target separation、物理状态与 latent 的 `rho_cond` |
| 原生目标显著性 | 条件差异在 native risk 中占多少 | center/response 精确分解、response/native risk |
| SGD 可见性 | 条件梯度到真实 optimizer 参数后是否够强、够一致 | 梯度范数比、余弦、`Bcrit`、SNR、block 分解 |
| 跨 query 泛化 | 学到的是历史规则还是训练 query/template | train/Development `G_swap` 分布、cross-query null |
| 响应校准 | 已分配正确后，方向、尺度和残差是否正确 | gain、`q`、`NRE=(q-gain^2)+(gain-1)^2` |

`target separation` 与 `rho_cond` 是独立问题。前者回答“两条 future 能否区分”，后者回答这部分
差异相对所有可预测变化是否足够大。`rho_cond` 也不是脱离表示的物理常数：必须同时报告物理状态
或任务参考表示和冻结训练 latent，不能只在一个 encoder 坐标中下结论。

## 3. 冻结指标

对共享 `(Q,A)` 的二元条件组，令 `Delta p=p_1-p_0`、`Delta t=t_1-t_0`。成对 MSE 精确满足

```text
L_correct = ||p_bar-t_bar||^2 + 1/4 ||Delta p-Delta t||^2
G_swap   = L_swapped-L_correct = <Delta p, Delta t>
```

固定报告：

- `G_swap`：均值、中位数、正号比例、绝对抵消率、top-10% 质量、sign-flip 和 cross-query null；
- `rho_cond = E||Delta t||^2 / (4 E||t-E[t]||^2)`：物理/参考表示与冻结 latent 两套；
- `r_grad = ||g_response|| / ||g_nonconditional||`：按 `predictor`、`pred_proj` 和整体报告；
- `Bcrit = E||g-Eg||^2 / ||Eg||^2`，`SNR(B)=sqrt(B/Bcrit)`；
- gain、prediction/target response energy ratio `q`、NRE 的正交残差与尺度误差分解。

Development 上只能测输出行为，不能替代训练梯度 SNR。当前 `Bcrit` 是一个真实训练 batch 内
32 个完整 pair 的 population estimate；下一轮仍需冻结至少 16 个训练 batch，分离 pair 内噪声和
batch 间漂移。

## 4. 第一轮实测

### 4.1 冻结训练 batch：初始化与 native 终点

同一训练 batch 含 `32` 个完整 condition pairs，审计参数与训练报告一致，仅包含
`predictor/pred_proj` 共 `11,584,128` 个参数。两次运行均为零 optimizer step。

| 快照 | latent `rho_cond` | response/native pair risk | gain | NRE | `r_grad` | 梯度能量占比 | response `Bcrit` | SNR(32) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 共同初始化 | 0.291% | 6.37% | -0.0419 | 1.440 | 3.92% | 0.153% | 29.59 | 1.04 |
| native step 8192 | 0.291% | 13.62% | 0.0840 | 1.498 | 5.57% | 0.309% | 26.56 | 1.10 |

训练把 hidden center MSE 从 `0.04532` 降到 `0.02033`，但 response MSE 从 `0.003083` 微升到
`0.003207`。也就是说总 native objective 明显下降主要来自 center；条件方向出现少量正 gain，
但 response error 并没有随之下降。`pred_proj` 最弱：终点 response/nonconditional 梯度范数比
仅 `1.60%`，平方能量占比 `0.0257%`。

这组结果排除了“只是长训练后才把条件梯度弄丢”的单一解释：初始化时条件参数梯度已经很弱且
SNR 只在 batch-size 32 附近达到约 1；训练后略有改善，但仍被非条件更新主导。

### 4.2 Motion Development：native 与 COJA

| checkpoint | latent `rho_cond` | `G_swap` mean/median | 正号比例 | gain | NRE | `G_swap Bcrit` | SNR(32) |
|---|---:|---:|---:|---:|---:|---:|---:|
| native step 8192 | 0.228% | `4.67e-5 / -1.09e-4` | 44.1% | 0.0065 | 1.1298 | 376.94 | 0.29 |
| COJA step 8192 | 0.228% | `2.666e-3 / 2.225e-3` | 96.9% | 0.3696 | 0.7665 | 0.689 | 6.81 |

native 的 sign-flip `p=0.420`、cross-query `p=0.177`；COJA 两者均为 Monte Carlo 下限
`p=0.000488`。同一 target encoder 下 `rho_cond` 完全相同，变化发生在模型是否把已有条件信息
变成一致的 query-specific response，而不是 COJA 改变了数据能量。

同一 Development 集在逐维标准化物理状态中的 `rho_cond` 只有 `0.00535%`，比 frozen latent
还低。这两个数不能直接按绝对值互换，但方向一致：自然条件效应相对全部状态/视觉变化很小，
“target 可区分”并不等于“native objective 中条件能量足够”。

native 在冻结训练 batch 的 `G_swap` 已为 `7.19e-4`、正号比例 `59.4%`，但 Development 上
退回近零。这要求后续把“高辨识度”与“query 覆盖”同时纳入数据设计；只重复高能量训练模板不是
充分方案。

### 4.3 分层线索

按 Development latent target-response energy 分四层，最低到最高四分位的 `rho_cond` 从
`0.087%` 增至 `0.473%`，conditional target energy/native risk 从 `7.49%` 增至 `16.58%`，
native `G_swap` 均值从 `-2.56e-4` 变为 `+4.24e-4`。这是“高辨识样本可能改善 native 压力”
的直接线索，但仍是 post-hoc Development 分层，不是训练因果结果。

Motion Development 的 query action norm 四层全部为零，因此**动作幅值不能作为通用 leverage
定义**。leverage 必须衡量动作在当前状态与隐藏动力学下实际放大的条件效应，例如

```text
L_action = ||Delta_c f(x,a) - Delta_c f(x,a_ref)||
```

或 simulator 中的 counterfactual state/latent response、局部有限差分、可验证物理 work/impulse。
高动作幅值最多是候选 proxy，不能直接用来筛数据。

## 5. 什么情况下 native 能学到

native 更可能学到条件 ICL，需要以下条件同时成立：

1. 历史中存在可恢复的隐藏动力学充分统计量，且 preprocessing/augmentation 没有破坏它；
2. 训练数据包含足够的 exact 或近邻 `(Q,A)` overlap，使“预测条件平均 future”不再是便宜解；
3. query/action 对隐藏动力学有实际 leverage，条件 response 相对背景、静态外观和中心运动足够大；
4. 动力学模式组内平衡，不能靠 query、动作或采集模板旁路预测 condition；
5. 高条件梯度跨 query 在共享参数中大体同向，或者 batch/采样覆盖足以降低 `Bcrit`；
6. 训练后段仍包含部署分布的自然响应幅值和完整 query/action support，防止只学会夸大响应。

相反，target separation 通过但 `rho_cond` 极低时，模型仍可用条件平均 future 获得低 native
loss；latent `rho_cond` 尚可但参数 `r_grad` 很低时，共享 Jacobian 或冻结表示会进一步衰减条件
更新；train `G_swap` 改善而 Development 不动时，主要问题转向覆盖而不是继续重复同一 pair。

## 6. 数据构造流水线

### 6.1 候选组构造

先按 exact `(Q,A)` 建组；连续域再使用预注册的 state/query/action 距离构造近邻组，并保存距离
与匹配质量。每组要求动力学 mode 平衡、历史不同、future 可验证不同，且所有 privileged label
只用于离线构造和审计，不进入模型输入。

### 6.2 每组离线打分

每个候选组至少保存五个相互独立的字段：

- `conditional_energy_physical`：物理状态或任务参考表示的 response energy；
- `conditional_energy_latent`：冻结预训练 target encoder 的 response energy；
- `action_leverage`：状态依赖的 counterfactual effect，而非原始 action norm；
- `coverage_cell`：query/state/action/response-magnitude 的部署支撑分箱；
- `pair_quality`：overlap 距离、mode 平衡、可见历史差异和泄漏检查。

不能把这五项压成一个未经验证的总分。先分层报告，再用受约束采样保证高辨识度与覆盖同时成立。

### 6.3 采样与混合

第一轮数据臂固定总 transition 数、更新步数、初始化和 optimizer：

| arm | 数据变化 | 要回答的问题 |
|---|---|---|
| D0 natural native | 当前自然混合 | 基准 |
| D1 energy-stratified | 同一候选池提高高 physical response 层曝光，保留自然锚点与 coverage 配额；frozen latent 只作操作审计 | 仅提高辨识度是否增强 native 条件梯度 |
| D2 leverage+coverage | 新增高 counterfactual-leverage 样本，并匹配 query/action coverage | 收益能否跨 query 迁移 |
| D3 high-ID then natural | 前段高辨识，后段退火到自然 response 幅值与部署 support | 能否兼顾 onset 与尺度校准 |
| D4 high-ID only | 全程高辨识，仅作诊断上界 | 检验过放大与 support collapse，不作为最终配方 |

已有 twin co-batching 负对照不重跑。它说明同一 pair 同批出现本身不够；D1/D2 必须实际改变
条件效应或跨 query 覆盖。COJA 只作为正对照，不参与数据 arm 选择。

D3 的自然校准不是额外 loss。它是训练数据分布调度：先让 native objective 看见稳定、可辨识的
历史依赖，再逐步恢复部署分布的 response magnitude、弱 leverage query 和完整 action support。
比例与切换点不凭经验拍定，先用 `25/50/75%` 高辨识曝光的小矩阵定位趋势，再冻结一个配方进入
完整训练。

## 7. 决策路由

| 观测 | 根因解释 | 下一动作 |
|---|---|---|
| physical 与 latent `rho_cond` 都低 | 数据本身条件效应弱 | 采集高 leverage query/action，不能只调 sampler |
| physical 高、latent 低 | target 表示压扁了条件差异 | 调整预训练 target/data 表示；数据重采样可能不够 |
| latent 高、参数 `r_grad` 低 | Jacobian/参数路由衰减 | 先换数据验证能否提高参数比；仍不动再考虑结构 |
| `r_grad` 尚可、`Bcrit` 很高 | pair/query 梯度相互抵消或覆盖不足 | 分层 batch、扩展 query coverage，测多 batch SNR |
| train `G_swap` 升、Development 不升 | 模板记忆/支持不足 | 定向扩充而不是继续重复已有高能 pair |
| gain 先正、NRE 后爆炸 | 分配已学到，响应尺度/正交残差失控 | 自然幅值校准混合或数据调度 |
| 数据臂改善 `G_swap` 但损害原环境 CEM | support shift | 提高自然数据比例并约束 action/query 边际 |
| overlap、能量、SNR、覆盖都充分仍无响应 | 数据优先假设被削弱 | 才进入显式条件 loss 或结构修改 |

ActionDelay full-10 的 gain=`6.877`、NRE=`70.995` 不应只写成“纯尺度问题”。按分解，尺度误差
约 `34.54`，剩余正交误差约 `36.46`；自然校准混合必须同时检查幅值和方向残差。十个 checkpoint
已在同一 Development 上完成重评分：epoch 2 已形成强 assignment（future/history=
`0.913/0.918`，joint-pair=`0.547`），epoch 3 的 NRE 最低为 `0.595`，epoch 4 首次明确退化且以
正交残差为主，epoch 5 起 gain/NRE 增至 `3.496/18.693`、joint-pair 归零。因而漂移起点已定位为
epoch 3 到 4 之间，严重过响应起于 epoch 5；这证明 COJA 先学会了 ICL 分配，不能解释为方案失败。
完整轨迹见
[`action_delay_lewm_coja_seed3072_full10_trajectory/summary.json`](artifacts/action_delay_lewm_coja_seed3072_full10_trajectory/summary.json)。

## 8. 两条路线与执行优先级

### 8.1 路线定位

| 路线 | 当前目标 | COJA 的位置 |
|---|---|---|
| A：Benchmark/顶会论文 | 证明现有世界模型会忽略历史，且扩大上下文与 native rollout 不足以解决；给出有效干预 | 当前主方法，也是“该能力可以被诱导”的正对照 |
| B：主模型预训练数据 | 提炼什么训练分布能让 native loss 自然使用历史，并形成可扩展的数据经验 | 机制探针和比较上界，不预设必须进入最终预训练配方 |

两条路线不应先被强行合并或互相替换。决定它们关系的最小因子表是：

| | 自然训练分布 `D0` | 高辨识重构分布 `D1` |
|---|---:|---:|
| native | 已有基线 | **下一项必须先补的纯数据效应格** |
| COJA | 已有方法结果 | 仅在 `D1 + native` 给出正信号后补 |

四格始终使用同一冻结自然 Development/Test 分布。不能用 `D1` 同时修改训练和评测，否则只会
证明新分布内部更容易，而不能证明模型获得了可迁移的历史使用能力。

### 8.2 实际执行顺序

1. **冻结而非调优 `D0`**：固定 benchmark v1 的评测 split、指标、seed、CEM 配置，同时冻结
   当前自然训练数据、sampler、transition 数与训练预算。归档已有 `D0 + native`、
   `D0 + COJA` checkpoints 和结果。此后不得为了改善 COJA 再改 `D0`。
2. **完成 COJA 的最低闭环**：补齐已有结果的机器身份、matched native 对照和 ActionDelay
   短预算到 full-10 的校准轨迹。这里是整理既有证据，不新增 COJA loss、COJA+rollout 或救援式
   数据调参。
3. **立即构造 `D1`，先只跑 native**：优先从同一原始候选池按完整 twin group 重采样高物理
   响应、mode-balanced 样本，同时保留自然曝光锚点；总样本数、optimizer step 和主要 coverage
   与 `D0` 匹配。frozen latent energy 只验证操作是否进入当前表示，不参与首轮选择。先补
   `D1 + native`，因为这是判断数据路线是否成立所缺的唯一关键格。
4. **使用同一自然 Development 判定**：除 gain/NRE 和原环境保持外，必须比较正确历史、交换或
   删除历史后的性能差，即 history-ablation drop。训练集改善而自然 Development 不改善，按覆盖
   失败处理，不能宣布数据路线成立。
5. **有条件地补第四格**：只有 `D1 + native` 出现 held-out 历史依赖后，才运行
   `D1 + COJA`，判断两者替代还是互补；随后才决定多 seed、跨任务和跨模型族的算力投向。

因此，当前不应“先完善 COJA 的训练数据”。必须先冻结 COJA 所使用的自然数据，随后用 native
单独检验新的训练分布。否则 COJA 改善究竟来自目标还是数据将永久无法归因。

**2026-08-31 执行门状态。** LeWM 的四个原生困难任务均已有 `D0 + COJA` 完整训练终点，
因此不再追加 COJA 优化任务；DINO-WM、LeWM 与 PLDM 的 `D0 + native` 任务已在云侧排队，
也不作为 D1 数据构造的前置阻塞。ActionDelay full-10 的 checkpoint/代码/数据哈希、Development
原始结果、六个 CEM cell 与十个保存 epoch 的 Development 轨迹均已归档，其余任务保留现有
matched 证据。故当前状态是**COJA 最低闭环完成、D1 立即启动**。在
`D1 + native` 给出 held-out 正信号前，不运行 `D1 + COJA`、COJA+rollout、额外 COJA seeds
或救援式权重调参。

### 8.3 结果分流

| `D1` 结果 | 论文与后续路线 |
|---|---|
| `D1 + native` 接近/超过 `D0 + COJA`，且 `D1 + COJA` 无额外收益 | 数据原则升为主贡献；COJA 保留为存在性证明、机制对照或次要方法 |
| `D1 + native` 与 COJA 都改善，组合进一步改善 | 同一篇统一为“数据决定条件压力，COJA 提高有限数据利用效率” |
| `D1 + native` 仅局部改善或不稳定 | 当前论文继续以 benchmark+COJA 为主；数据经验进入后续主模型预训练工作 |

无论哪种结果，当前 expanding/rollout 负对照和 COJA 正对照都不会失效。变化的只是 COJA 在最终
叙事中是主方法、互补方法，还是证明 benchmark 可被解决的诊断工具。

### 8.4 Benchmark 版本边界

只改变训练数据时不发布 benchmark v2。保持 benchmark v1 的评测协议和可比性，新增
`training-distribution track`，包含 `D0 natural`、`D1 high-identifiability` 和后续
`calibration-mix` 配方。只有新增未见 query/action、校准 split、跨域任务或改变指标与评测契约
时，才升级为 benchmark v2。

数据优先路线最终仍需在一个 native 正例、一个 native 负例和一个尺度漂移例上验证；但这些完整
扩展应放在最小 `D1 + native` 决策之后，而不是作为启动数据实验的前置审计。公开 Test 继续锁定，
配方与论文主线冻结后只访问一次。

## 9. 已落地资产

- D1 构建与 native 因果验证预注册：[`D1_CONSTRUCTION_PLAN_ZH.md`](D1_CONSTRUCTION_PLAN_ZH.md)
- 通用 paired 指标与梯度统计：[`scripts/conditional_signal_metrics.py`](scripts/conditional_signal_metrics.py)
- Motion Development runner：[`scripts/run_motion_damping_conditional_signal_diagnostic_v1.py`](scripts/run_motion_damping_conditional_signal_diagnostic_v1.py)
- 训练 batch 梯度 runner：[`scripts/run_motion_damping_native_gradient_snr_diagnostic_v1.py`](scripts/run_motion_damping_native_gradient_snr_diagnostic_v1.py)
- 初始化训练梯度结果：[`initialization_seed3073_train_gradient_snr_v1.json`](artifacts/native_conditional_signal_root_cause_v1/motion_damping/initialization_seed3073_train_gradient_snr_v1.json)
- native 终点训练梯度结果：[`native_s14321_step8192_train_gradient_snr_v1.json`](artifacts/native_conditional_signal_root_cause_v1/motion_damping/native_s14321_step8192_train_gradient_snr_v1.json)
- native Development 结果：[`native_s14321_step8192_development_v1.json`](artifacts/native_conditional_signal_root_cause_v1/motion_damping/native_s14321_step8192_development_v1.json)
- COJA Development 结果：[`coja_s14321_step8192_development_v1.json`](artifacts/native_conditional_signal_root_cause_v1/motion_damping/coja_s14321_step8192_development_v1.json)

所有 runner 都拒绝覆盖已有输出。训练梯度结果中的 guard receipt 为 optimizer constructor/step、
Development scorer、Public scorer 和非训练 benchmark read 全部 `0`，唯一允许的 training table
read 为 `1`；checkpoint 参数、buffer、module mode 和 RNG 均在退出前恢复。
