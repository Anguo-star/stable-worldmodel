# 条件动力学 ICL：从边缘表示保护到条件响应识别

> 阶段状态（2026-08-11）：最新的 **target stop-gradient + 原始 SIGReg** 先未通过正式
> step-0 proxy gate，随后在明确授权的 exploratory 256-step actual-learning falsification
> 中再次失败。冻结的是 ActionDelay Development 的数据、评分和阈值，不代表候选获得了
> 正式晋级资格。本阶段不续训、不运行该 checkpoint 的 CEM、不打开 Public Test，也不
> 自动切换候选；下一步先完成 LeWM–PLDM 匹配机制审计。

## 摘要

本研究关注的不是表示的边缘分布是否“看起来健康”，而是世界模型能否从历史和动作中
识别 episode 内隐藏的动力学，并据此预测同一当前状态下不同的未来：

\[
p(O^+\mid H,Q,A),
\]

其中 `H` 是动作—观测历史，`Q` 是当前观测，`A` 是 query action。目标因此是保持并
学习 **历史、动作与未来之间的条件联合关系**，而不是只约束 `p(Z_t)` 或轨迹表示的全局
边缘统计。

现有结果给出四点阶段结论：

1. Door 与 PushT 的早期实验确认了原生 LeWM/SIGReg 的条件动力学盲区；pair-dependent
   response SIGReg 能在特定构造上学会能力并保持 CEM，是能力上界而非通用解。
2. 时间中心化、轨迹联合协方差、预测侧联合协方差、Encoder-only 和多种 paired/global
   统计改造均只在局部任务有效，未同时满足跨任务 ICL 与原能力保持。
3. ActionDelay 上，已审计的 LeWM checkpoints（共享主体架构、主要为单 seed）都停留在
   约三分类随机水平，输出随历史改变的 query 不超过 2%；1,024 步的 native LeWM 与
   PLDM 同预算对照中，只有 PLDM 明确学会了条件映射。
4. stop-gradient 能关闭在线 target 通过 MSE 向条件中点收缩的一条捷径，但最新完整实验
   表明它不会自动创建 Predictor 的 history/action-to-future coupling；“stop-gradient +
   原始 SIGReg 足够”已被当前冻结实验否定。

这使研究问题从“再设计一个更强的边缘正则”收束为一个可判别的机制问题：条件信号究竟
在 history representation、future target、Predictor coupling，还是优化梯度的哪一环
中断？在回答这个问题前继续扫权重、seed 或新统计量，不构成有效进展。

## 1. 问题与评价对象

### 1.1 隐藏动力学任务

Door Rule、ActionDelay 和 PushT Hidden Actuation 共享如下结构：

\[
p(O^+\mid Q,A,M_1)\ne p(O^+\mid Q,A,M_2),
\qquad M\text{ 可由 }H\text{ 识别，但不能由当前 }Q\text{ 单独识别}.
\]

评测中的 matched query 固定当前可见状态和 query action，只改变能够揭示隐藏机制 `M`
的历史。一个真正学到 ICL 的模型必须对历史变化作出方向正确的未来响应。仅仅保持 target
表示不坍缩、提高整体方差或改变三类输出的边缘比例，都不等价于学会该映射。

### 1.2 三个必要环节

为避免把现象误写成唯一根因，本文把系统拆成三个可独立检验的环节：

1. **历史可读（history readable）**：Encoder/context aggregation 保留了区分隐藏机制
   所需的历史信息；
2. **条件未来可用（future usable）**：相同 `Q,A` 下，不同机制的 target future 不仅
   距离非零，而且具有可供任务读取的尺度、方向和投影几何；
3. **预测耦合（predictor coupled）**：Predictor 使用 `H,A`，产生与 target response
   方向和尺度一致的预测变化。

前两项是第三项的必要条件，但都不是充分条件。ActionDelay 的最新终点 pair-distance
审计只排除了 target 完全重合；它不能单独证明 target 已达到任务可用的尺度和方向，而
prediction 此时仍几乎完全不随历史变化。

### 1.3 证据合同

ContextWorld 负责冻结任务、数据、评分门和参考可学习性；本目录负责训练目标、机制假设
和候选结果。候选必须按以下顺序晋级：

1. 零步身份、runtime、sampler 和 proxy gate；未通过者不获得正式训练资格，额外训练
   必须另行声明为 exploratory falsification；
2. 冻结 Development ICL 门；
3. Development 通过后，才授权在 **同一 checkpoint SHA** 上执行 Public ICL 与原任务
   CEM；
4. Public ICL 与 CEM 都通过后，才进入任务广度和多 seed。

因此，局部梯度方向、训练 loss、表示方差和单个分层准确率只能用于诊断，不能替代实际
条件响应门。

## 2. 从任务局部正例到通用主张失败

### 2.1 早期证据确认了问题存在

TwoRoom Door 的匹配对照首先排除了“数据或模型容量根本不可学”的解释：

| 训练目标 | 正确未来 | 正确历史 | 最差分层 |
|---|---:|---:|---:|
| 原始 LeWM 联合训练 | 50.33% | 53.83% | 4.00% |
| LeWM 固定图像表示 | 100.00% | 100.00% | 100.00% |
| PLDM 联合训练 | 99.33% | 99.67% | 92.00% |

PushT Hidden Actuation 又给出了一个 task-local 正例。pair-dependent response SIGReg 在
独立确认集上把正确未来从 `90.04%` 提高到 `97.07%`，最差增益从 `80.86%` 提高到
`94.92%`；相同原始 replay 曝光下，标准 PushT CEM 为 `224/300`，几乎等于
standard-only 控制的 `225/300`。

这个结果证明“学会隐藏动力学必然损害规划”的说法不成立，也说明对条件 response 施加
约束可以起作用。但它依赖 condition-matched pair，跨数据构造迁移仍接近随机，因此只
能作为能力上界，不能被写成通用修复。

### 2.2 候选谱系应按假设而不是版本号理解

| 方法族 | 它直接约束什么 | 最重要的正证据 | 冻结反证与结论 |
|---|---|---|---|
| 原生 SIGReg | 每个时刻的表示边缘 `p(Z_t)` | 保持一般表示和规划能力 | 边缘健康不推出条件未来可辨识 |
| response SIGReg | matched pair 的 target/prediction response | PushT Hidden Actuation ICL 与 CEM 同时通过 | pair-dependent、跨构造不泛化；保留为能力上界 |
| Temporal-centered SIGReg | 轨迹内时间残差边缘 | Door 达到 `100%` | PushT 未过门，Door CEM 降至 `73–77%`；拒绝通用化 |
| target JTCov | target 轨迹的联合时间协方差 | Contact Friction ICL `95.12%`，Door `90%` | 六项 ICL 仅 `3/6`、原任务 CEM 非劣仅 `2/9` |
| predictive JTCov | prediction 轨迹的联合时间协方差 | 直接梯度确实进入 Predictor | 条件 prediction response 反而收缩，ICL 与 CEM 均失败 |
| Encoder-only / history-value | 历史可读性和 target response | Door 可明显改善 | Action Strength 资源门失败；仅修 target 侧不充分 |
| ActionDelay paired/global 统计 | sampler balance、assignment 或全局联合形状 | 部分改变边缘类别偏好 | 代表性候选均约随机，历史响应不超过 2% |
| target stop-gradient + 原始 SIGReg | 阻断 MSE 对 target 的梯度，同时保持原边缘正则 | 终点 target pair-distance anti-collapse 通过 | step-0 gate 先失败；exploratory 256 步仍无条件响应 |

这些失败并不相互矛盾。多数候选虽然公式不同，却仍在调节同一层面的全局或边缘统计；它们
没有直接保证相同 `Q,A` 下，历史变化会引起方向正确的 prediction response。即使一个
统计量作用于整条轨迹的“联合分布”，只要它未按 history/action 条件化，无关的状态变化
仍可满足全局矩约束，而稀少的条件响应继续被忽略。

## 3. ActionDelay：决定性的条件响应压力测试

### 3.1 为什么采用 ActionDelay

ActionDelay H7 Stage-1 为每个 query 提供三个物理组（delay `0/4/8`）。同一 query 的
当前输入和 action 相同，历史指示实际 delay。Development 含 960 个 query、2,880 个
history condition；三分类随机基线约为 `1/3`。

它比“整体表示有没有坍缩”更直接：如果三个 history condition 下模型总选择同一未来
组，即使三组在全数据上的边缘比例很均衡，条件 ICL 仍然是失败的。

### 3.2 最新候选的精确定义

最新候选在内部实验身份中记为 V8。正式 BF16 step-0 offset gate 先给出
`failed_v8_step0_guard_zero_candidate`：36 个算法条件中 32 个通过，且相对 native SIGReg
control 的 18 个比较全部达到 `8/8`；但四个 normalized absolute-improvement 条件只有
`6/8` offsets 为正，使 joint condition 低于预注册的 `7/8`，live-target-dispersion
guard 也为 `0/8`。这解释了局部方向为何一度显得积极，却不改变正式失败。该结果被永久
保留，不能追溯性改写成通过。由于零步 proxy 不能回答实际学习问题，随后另行冻结并授权
了一次 **exploratory** 256-step falsification；它不是正式 gate 晋级。

方法本身只有：

\[
L=\operatorname{MSE}\bigl(\hat z^+,\operatorname{sg}(z^+)\bigr)
  +0.09\operatorname{SIGReg}(z_{\mathrm{live}}).
\]

也就是 **terminal target stop-gradient + 原始 native SIGReg**。它没有新增 head、参数或
loss 项，也没有把 pair/delay metadata 传入模型或 loss。训练 batch 为 50% 原始 replay
和 50% ActionDelay synthetic；每个 rank 的 64 条 synthetic 中有 63 条组成 21 个
triplet、1 条不配对。配对身份不参与目标计算。

256-step checkpoint：

```text
SHA256 0401bb123ecc1cb2b63cb30c1e72775016466df10aebb6f058b833be3c4336ff
```

8-rank runtime/sampler 审计确认 optimizer 恰好执行 256 步，全部 297 个 trainable tensor
（18,035,246 个参数）进入优化，实际 loss route 与上式一致。

### 3.3 Exploratory 的冻结 Development 结果

| Stage-1 硬门 | 阈值 | V8 结果 | 通过 |
|---|---:|---:|---:|
| physical-group macro accuracy | `≥0.55` | `0.33368` | 否 |
| minimum physical-group accuracy | `≥0.40` | `0.25208` | 否 |
| paired-query bootstrap 95% lower bound | `≥0.50` | `0.33333` | 否 |
| target latent anti-collapse | 必须通过 | 通过 | 是 |

三个真实物理组到模型选择组的 confusion rows 如下；数组列顺序均为 selected group
`[0, 4, 5]`：

```text
true delay 0: [242, 313, 405]
true delay 4: [242, 313, 405]
true delay 8: [241, 313, 406]
```

三行几乎完全相同。959/960 个 query 在三种历史下选择同一组，只有 1 个 query 对历史
有任何响应；没有 query 同时覆盖三种输出，也没有 query 三种历史全部选对。经验离散
selected-group CMI `I(H; G_hat | Q)`（确定性映射下等于每个 query 的输出熵再取均值）
仅约 `0.00096 bit`；这不是连续 latent CMI。

终点 anti-collapse 审计同时确认 2,880 个 target physical-group pair 按冻结距离判据均未
完全重合，最小 target pair MSE 为 `0.0530`。这排除了该判据下的完全 target-pair
collapse，但没有排除 target-side 尺度、方向或投影可用性不足；尤其 step-0 的 live
dispersion guard 是 `0/8`。可以确定的是，终点 Predictor 没有把历史变成条件 prediction
response。

### 3.4 跨候选与 PLDM 对照

下表只保留能区分假设的代表性行；256 与 1,024 步预算明确分列，不把较短 prefix 当作
最终同预算排名。

| 模型 / 假设 | 步数 | macro | 最差组 | bootstrap lower | 历史响应 query | `I(H;G_hat|Q)` |
|---|---:|---:|---:|---:|---:|---:|
| LeWM native release control | 1,024 | 0.3333 | 0.0000 | 0.3333 | 2/960 | 0.0019 bit |
| LeWM tail-mass joint SWReg | 256 | 0.3302 | 0.0781 | 0.3281 | 19/960 | 0.0182 bit |
| LeWM paired assignment | 256 | 0.3372 | 0.0198 | 0.3351 | 16/960 | 0.0153 bit |
| LeWM frozen representation control | 256 | 0.3337 | 0.0000 | 0.3333 | 1/960 | 0.0010 bit |
| **LeWM target stop-grad + native SIGReg** | **256** | **0.3337** | **0.2521** | **0.3333** | **1/960** | **0.0010 bit** |
| PLDM | 1,024 | **0.8257** | **0.4875** | **0.8149** | **950/960** | **1.2337 bit** |

直接同预算结论只来自 1,024-step PLDM 与 1,024-step native LeWM；V8 只与 256-step
controls 直接比较。PLDM Stage-1 的通过使用事先冻结、校准后的 min-group 阈值 `0.40`
（旧的 `0.60` 阈值下同一数值会报告失败）。独立的 6×6 Stage-2 只作旁证，其结果为
macro `0.9419`、最差组 `0.8677`、bootstrap lower `0.9336`，不与 Stage-1 数值混排。
这些结果证明 benchmark/data 中存在可学习信号；它们不能单独排除 LeWM 架构或优化的
容量限制。

V8 的最差组从 frozen-representation control 的 `0` 提高到 `0.2521`，但 confusion
rows 和 query response 证明这只是 **全局边缘偏好重新分配**，不是 ICL 进展。只报告最差
组改善会得出错误结论。

## 4. 目前能下的机制结论

### 4.1 已由证据支持

| 结论 | 关键证据 |
|---|---|
| LeWM 在 ActionDelay 表现为系统级条件盲区 | 多类目标、sampler 和 representation 变体均约随机且几乎 history-invariant |
| V8 终点 target 没有按距离判据完全重合 | target pair anti-collapse 通过，但 prediction 仍不响应历史 |
| stop-gradient 关闭了一条真实优化捷径 | MSE 不再能通过更新在线 target 直接缩小条件 target 差异 |
| stop-gradient + 原始 SIGReg **不充分** | 精确 256-step actual-learning gate 三项失败，959/960 query 不响应 |
| benchmark/data 中存在可学习信号 | PLDM 在相同 benchmark 上通过 Stage-1 和 Stage-2；不据此排除 LeWM-specific 容量限制 |
| 全局联合统计不等于条件联合关系 | JTCov/paired global 方法可改善几何或边缘，却未稳定产生正确条件映射 |

此前若干较容易任务中，有无 stop-gradient 的结果差异很小；这只能说明 target 收缩捷径
在那些数据上不是决定瓶颈，不能推出 stop-gradient 在所有 ICL 任务上都不需要。反过来，
ActionDelay V8 又说明关闭这条捷径仍不足以学会条件映射。因此当前证据既不支持
“stop-gradient 普遍不需要”，也不支持“加入 stop-gradient 就是完整修复”；它首先是
一个用于隔离优化路径的简洁因果干预。

### 4.2 仍未被区分

当前证据尚不能把唯一根因指定为以下任意一项：

- Encoder/context aggregation 没有形成可供 Predictor 使用的历史状态；
- target response 虽然距离非零，但尺度、方向或 projection geometry 不适合任务读取；
- 历史表示可读，但 Predictor 或 action conditioning 忽略它；
- prediction response 方向已经出现，但不同 loss/样本的梯度相互抵消；
- 正确方向存在但增益过小、学习动力学过慢。

过去把“模型最终不使用历史”直接写成“Encoder target collapse 是机制级根因”过于强。
现在可以确认的是 **系统级症状和若干被排除的路径**，尚不能确认唯一断点。

### 4.3 为什么很多初筛看起来可行，完整训练却失败

早期筛选大量依赖 step-0 梯度、局部 pair distance、全局方差或某一能力项。这些指标回答
的是“这个目标是否改变某个局部方向”，不是“优化后模型是否会在未见 query 上按历史
作出正确响应”。因此会出现三个常见假阳性：

1. target 被拉开，但 Predictor 不读取历史；
2. 输出三类的边缘比例更均衡，但每个 query 内仍选择同一类；
3. 单任务 ICL 提升来自几何尺度放大，同时原任务 CEM 明显退化。

V8 并不是“初筛通过、完整训练才失败”：其正式 step-0 gate 已失败，256 步训练是在保留
该失败的前提下，为回答 actual learning 问题而额外授权的探索性反证。更广泛的问题仍然
存在——多个更早候选的局部 proxy 改善没有转化为 held-out 条件响应。

这不是需要继续增加候选数量的证据，而是说明此前 preflight 与最终 estimand 不一致。
今后的早停量必须直接测量 held-out matched query 上的条件 response，而不是把梯度代理当作
学习成功。

## 5. 收束后的下一步

### 5.1 先做 LeWM–PLDM 匹配机制审计

在新训练开始前，以匹配 optimizer step、样本曝光、batch identity 和数值精度的 LeWM–
PLDM checkpoints 为对象，并使用各模型内部归一化量，逐层测量：

- `Δh`：只改变历史时，context/history representation 的变化；
- `Δt`：对应 target future 的真实变化；
- `Δp`：prediction 对相同历史变化的响应；
- response gain、与 `Δt` 的 cosine alignment、归一化 response error；
- prediction MSE 与各正则项对这些量的逐模块梯度及 batch 内抵消程度。

审计的决策规则是：

| 观测 | 优先定位 |
|---|---|
| `Δh ≈ 0` | Encoder 或 context aggregation |
| `Δt ≈ 0`、尺度不稳或 projection 不可读 | target representation / projection |
| `Δh > 0`，但 `Δp ≈ 0` | Predictor / action-history conditioning |
| `Δp` 方向存在，但 aggregate gradient 抵消 | objective 或 sampler-level gradient cancellation |
| 方向正确且随步数稳定增长，仅幅度不足 | 预算、尺度或优化速率 |

只有该审计在 LeWM 与成功的 PLDM 之间给出可复现差异，才允许冻结一个新候选。

### 5.2 新候选的最小性约束

新方法仍以简洁为目标：优先替换错误的正则统计量，而不是叠加分类 head、蒸馏、IDM 或
多项 variance/covariance loss。它必须直接作用于 history/action-conditioned response，
而不是再次只匹配 `p(Z_t)` 或未条件化的整轨迹矩。

例如，matched-pair response residual

\[
\left\|(\hat z_i^+-\hat z_j^+)
-\operatorname{sg}(z_i^+-z_j^+)\right\|^2
\]

可以作为机制审计量或待检验假设，但 **现在还不是已批准候选**。是否使用 pair、是否替换
原 SIGReg，以及如何保持原能力，都必须由上述断点审计决定，不能在看到 Development
结果后反向拼装。

下一候选的 preflight 也必须改为直接的条件响应门：使用新冻结、此前未见的 Development
identity/catalog，并在查看结果前预注册阈值；同时检查 response rate、confusion-row
separation、方向正确性和 target geometry，不能再复用已被多轮观察的当前 960 queries。

## 6. 结论边界与可复现性

- V8 的正式 step-0 结论是 `failed_v8_step0_guard_zero_candidate`；其后 256-step 训练是
  明确授权的 exploratory actual-learning falsification，不能追溯性宣称候选通过零步门。
- 256-step 结论严格限于 seed `3072`、exact prefix 和冻结的 ActionDelay Development
  Stage-1 scoring。它是真实的 actual-learning 失败，但不是“任何更长预算都不可能成功”
  的数学定理；当前决策不允许用续训覆盖该失败。
- 因 ICL 门失败，同 checkpoint 的 CEM 没有运行。因此不能声称 V8 保持或损害 CEM。
- 本文不报告新的 Public Test 数值。后续正式 Public 必须在新进程中于文件遍历前排除
  Public 目录，并绑定同一 checkpoint SHA；本阶段不据此作任何 Public 主张。
- ContextWorld release adapter 更新后，Development v2 在 V8 评分前冻结。V6 control
  通过新 adapter 重放时，2,880 条 record、summary、score audit 和 gate 均逐字节一致，
  排除了 release rebind 造成当前结论的解释。
- ActionDelay 内容完整性使用低内存等价审计；原始单进程 `--full` 命令因实现内存问题
  未返回 PASS，本文不将它写成 PASS，也不误写为 benchmark 内容失败。

关键冻结身份：

| 对象 | SHA256 |
|---|---|
| V8 step-0 offset decision | `3a62bc3eca1a3611451d44eddf037eb2ffe92abfa90465bd67eb4452d145c91b` |
| V8 exploratory training addendum v2 | `ea633502bb7e92d4fb3ee3bb83db747db931133733ed68229e8a201a015b58c0` |
| V8 checkpoint | `0401bb123ecc1cb2b63cb30c1e72775016466df10aebb6f058b833be3c4336ff` |
| V8 Development result | `2c3a08b6376e6bb87223673142c0aab0e4088548df585a94740835af1b221a5c` |
| Development v2 config | `a433058704192ec8e688b22cdbf52b64e54c0d37aad2c1664cd6fb0c64602f9f` |
| Development v2 catalog | `d161e713d9ef8834bbd154cf5bc5dcd9083e322beb13ca7d75f7ee12f4cc62c2` |

证据入口：

- [V8 step-0 offset 决策](artifacts/action_delay_h7_target_stopgrad_sigreg_v8_offset_gate_v1/robustness_decision_v1.json)
- [V8 exploratory training addendum](configs/action_delay_h7_target_stopgrad_sigreg_v8_post_gate_training_addendum_v2.yaml)
- [V8 冻结 gate decision](artifacts/paired_terminal_target_stopgrad_sigreg_v8_validation/development/action_delay_stage1_s3072_step256_v1_gate_decision.json)
- [V8 8-rank runtime 审计](artifacts/paired_terminal_target_stopgrad_sigreg_v8_validation/training/target_stopgrad_sigreg_prefix256_v2/full_train_v8_runtime_audit_v1.json)
- [V8 paired sampler 审计](artifacts/paired_terminal_target_stopgrad_sigreg_v8_validation/training/target_stopgrad_sigreg_prefix256_v2/pair_sampler_runtime_audit_v1.json)
- [release rebind 等价回执](artifacts/action_delay_h7_development_v2_release_rebind/release_rebind_equivalence_receipt_v1.json)
- [Encoder-only / history-value 阶段报告](results/history_value_encoder_only_stage_report_v1.md)
- [target-JTCov 任务广度报告](results/joint_temporal_covariance_sigreg_task_breadth_report_v1.md)
- [predictive JTCov 验证摘要](results/causal_predictive_jt_cov_sigreg_validation_summary_v1.json)

机器回执保留完整路径、输入哈希、checkpoint 身份和 gate 字段；本文只呈现支撑当前研究
决策所需的最小结果集，避免把内部执行顺序变成方法叙事。
