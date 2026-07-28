# LeWM 隐藏门规则诊断：默认失效、机制与反证

本文研究的不是“门位置 benchmark 能否刷高分”，而是一个更基础的问题：当当前画面相同、历史决定隐藏动力学规则时，LeWM 是否会在联合训练中删除这部分条件信息。

实验首先确认了默认 LeWM 的真实失效机制；随后，公平的 SIGReg 权重反证实验否定了更强的结构性主张。当前结论是：

> 默认 `0.09 × SIGReg` 在 Door 数据上不足以抵消 prediction MSE 对规则相关表示方向的早期压缩；但在固定 50% 原始 replay、50% Door replay 后，仅把权重提高到 `0.90`，同一 LeWM 已能学会门规则，同时原始域真实环境 CEM 成功率没有下降。因此，现有证据不支持“LeWM 必须引入新的条件正则或多成分防坍缩目标”。

这里需要严格区分三件事：

- 默认超参数下的失败及其优化机制是真实的；
- 只依赖边缘分布的 SIGReg 不提供条件信息保留的数学保证，这一理论边界仍然成立；
- “缺少保证”不等于“当前任务中存在无法靠调参解决的方法缺陷”。公平反证已经否定了后一个经验命题。

结构化数值、哈希和审计结果位于
[结果汇总](results/sigreg_replay50_falsification_summary.json)；本文只保留外部读者理解结论所需的实验设计、证据和边界。

## 1. 如何判断模型是否学会门规则

Door 规则推断和 CEM 规划回答不同问题，不能混成一个指标。

| 问题 | 评测方式 | 真实目标 | 判定 |
|---|---|---|---|
| 模型是否从历史推断出门规则 | 固定相同 history/query/穿门动作，预测一步未来 | 模拟器分别执行 passable 与 blocked 规则得到的两个真实下一状态 | 预测更接近当前真实规则对应的未来 |
| 正确历史是否真正改变预测 | 同一 query 比较正确规则历史与相反规则历史 | 同上 | 正确历史比相反历史更接近真实未来 |
| 模型能否用于原任务规划 | CEM 在模型内搜索动作序列，再把选中序列放回真实环境执行 | 真实环境中的目标到达 | 执行后达到目标即成功 |
| 预测拟合是否变化 | 冻结原始域 query，计算多步 latent rollout MSE | 真实 rollout 的编码 | 仅作诊断，不替代闭环成功率 |

设固定穿门动作为 \(a\)，模型预测为

\[
\hat z^+=G_\phi(H,O_t,a),
\]

模拟器在两条规则下的真实下一状态经当前 checkpoint 编码后分别为
\(z^+_{\mathrm{pass}}\) 和 \(z^+_{\mathrm{block}}\)。若真实规则为 passable，则该 query 的正确判定为

\[
\lVert \hat z^+-z^+_{\mathrm{pass}}\rVert^2
<
\lVert \hat z^+-z^+_{\mathrm{block}}\rVert^2,
\]

blocked 情况反之。固定动作的目的，是隔离“是否使用历史推断规则”，避免把动作搜索失败混入表征诊断。

因此，“让 CEM 选出最小 loss 的动作，再把动作与能否穿门标签比较”并不准确：动作本身不是 passability 标签。端到端 CEM 的正确检查是把选中的动作序列放回真实环境执行，再看是否到达目标；规则能力的正确检查则是固定同一动作，比较预测未来与两个真实执行未来。

正式 Door 评测使用 6 个 eval seed、每个 seed 50 个未见门位置 query。`did_not_attempt_crossing` 历史没有暴露门规则，只报告模型的无证据默认倾向，不参与规则能力否决。能力门槛使用预先存在的 `informative_history_rule_switch_v2`：正确历史必须胜过相反规则历史，并且正确未来选择率在每个 seed×方向×规则单元都超过随机水平。

## 2. 默认 LeWM 为什么失败

### 2.1 可学性与目标对照

相同初始化、15,360 条双规则合成 clip、8 卡、全局 batch 1,024、1,024 个优化步下：

| 训练配方 | 正确未来选择率 | 正确历史胜率 | 最差分层正确率 | 正式通过 |
|---|---:|---:|---:|---:|
| 原始 H3 LeWM | 50.00% | 51.33% | 0.00% | 0/1 |
| LeWM 联合训练，`λ=0.09` | 50.33% | 53.83% | 4.00% | 0/3 |
| LeWM 固定图像表示 | 100.00% | 100.00% | 100.00% | 3/3 |
| PLDM 联合训练 | 99.33% | 99.67% | 92.00% | 3/3 |
| PLDM 固定图像表示 | 100.00% | 100.00% | 100.00% | 3/3 |

这组对照说明，数据、History=3 输入、Predictor 容量和一般性的 Encoder 联合训练都不是瓶颈。默认 LeWM 的失效来自其目标在当前权重下走出的优化路径。

### 2.2 收缩由在线 Encoder 更新携带

LeWM 联合训练后，Encoder/Projector 的“规则配对距离 ÷ 普通不同画面距离”降至 `0.0084 / 0.0051`，正确规则切换率仅 `1.56%`。整体方差仍保留约 81%，有效秩仍保留约 97%，所以这不是整个 Encoder 输出常量，而是规则相关局部方向被选择性压近。

PLDM 联合训练的对应距离比为 `0.6433 / 0.8304`，正确规则切换率为 `98.67%`。模块换件进一步确认，主要收缩由 Encoder 参数更新携带，不是 Projector 参数或 BatchNorm buffer 单独造成。

精确首个训练 batch 上，无穷小梯度步对 Projector 规则配对距离的方向为：

| 梯度来源 | 配对距离变化方向 |
|---|---:|
| prediction target 分支 | -678.83 |
| prediction context / predictor 分支 | -266.37 |
| 完整 prediction MSE | -945.20 |
| `0.09 × SIGReg` | +41.39 |
| LeWM 总目标 | **-903.81** |
| PLDM 已启用正则合计 | +1,691.39 |
| PLDM 总目标 | **+746.19** |

负数表示压近两种规则未来，正数表示分离。SIGReg 不是没有抵抗收缩，而是默认加权梯度远弱于 prediction MSE。

### 2.3 SIGReg loss 降低不等于条件规则被保留

默认 LeWM 的 256-step 轨迹中，SIGReg loss 从 `3.1127` 降到 `1.9539`，但 Projector 规则配对距离比同时从 `1.8862` 降到 `0.0441`。因此，正则标量正常收敛、整体表示统计正常和规则方向坍缩可以同时发生。

LeWM 目标可简写为

\[
\mathcal L =
\mathbb E\lVert G_\phi(H,Q)-E_\theta(O_{t+1})\rVert^2
+\lambda R(P_Z).
\]

对两种隐藏机制 \(m\in\{-1,+1\}\)，令目标表示中的规则幅度为 \(\alpha\)，Predictor 已学会的规则切换比例为 \(\beta\)：

\[
z_m^+=u+\alpha m d,\qquad
\hat z_m^+=u+\alpha\beta m d.
\]

规则相关 prediction loss 为

\[
\mathcal L_{\mathrm{mech}}=C\alpha^2(1-\beta)^2.
\]

当 Predictor 尚未学会规则时，优化器既可以增大 \(\beta\)，也可以缩小在线目标中的 \(\alpha\)。一旦 \(\alpha\) 开始缩小，推动 Predictor 学规则的信号按 \(\alpha^2\) 下降，而继续收缩表示的信号只按 \(\alpha\) 下降，形成自增强的早期失败路径。

SIGReg 匹配每个时间点表示的随机一维投影与标准高斯，约束的是无条件边缘分布 \(P(Z_t)\)。只依赖 \(P_Z\) 的正则无法区分“保留规则耦合”和“由无关视觉因素填充同一边缘统计”的两种表示，因此不能单独保证

\[
I(Z^+;M\mid Q)>0.
\]

这是一个保证范围的边界，不是已经证明的工程不可修复缺陷。当前多卡实现按 rank 的局部 batch 估计 SIGReg；它会影响有限样本噪声和梯度标定，但不改变目标仍是 \(P(Z_t)\) 这一理论结论，也不是本实验采用的根因解释。

## 3. 公平 SIGReg 权重反证

早期纯 Door 续训没有保留原始 replay，所有续训模型都发生明显原任务遗忘，所以不能用那组结果判断“高 SIGReg 是否损害原任务”。最终反证实验重新固定：

- 同一原始 H3 LeWM 初始化和训练 seed `3072`；
- 50% 原始 TwoRoom replay、50% Door mixed-rule replay；
- 相同数据顺序、optimizer、scheduler 和 Encoder 学习率；
- 单机 8 卡、全局 batch 1,024、1,024 个优化步；
- 唯一有意差异是 `λ∈{0.09,0.20,0.30,0.90}`；
- Door 使用 `50×6` 冻结 query；原始域使用配对的 `50×6` CEM query；
- CEM 非劣界为绝对成功率 `−5` 个百分点，按 eval seed 分层做 20,000 次配对 bootstrap。

`0.09、0.20、0.30` 是预先冻结的第一梯级；看到它们的 Door 结果后，`0.90` 作为明确标注的自适应反证点，在训练前单独冻结协议。

### 3.1 训练终点与 Door 规则能力

| SIGReg λ | 训练 pred loss | SIGReg loss | 正确未来选择率 | 正确历史胜率 | 最差分层正确率 | Door 通过 |
|---:|---:|---:|---:|---:|---:|---:|
| `0.09` | 0.02426 | 1.76074 | 51.17% | 67.83% | 20% | 否 |
| `0.20` | 0.03813 | 1.64941 | 56.67% | 92.33% | 32% | 否 |
| `0.30` | 0.04760 | 1.61426 | 67.67% | 98.83% | 44% | 否 |
| `0.90` | 0.06995 | 1.49316 | **99.50%** | **100.00%** | **96%** | **是** |

提高权重时，SIGReg loss 降低、训练 pred loss 上升，规则区分度则持续增强。`0.30` 已能对历史方向作出几乎完全正确的相对响应，但绝对未来校准仍不足，最差分层只有 44%，不能算稳定学会；`0.90` 才通过完整门槛。

Door 的正确真实目标 native MSE 从 `0.03566`（`0.09`）升至 `0.12278`（`0.90`），但正确/错误未来的平均 margin 从 `0.00063` 增至 `0.58632`。不同 checkpoint 的 Encoder 表示尺度不同，所以不能用跨 checkpoint 的绝对 latent MSE 代替同 checkpoint 内的两目标判别。

### 3.2 原始域预测与真实环境 CEM

| SIGReg λ | rollout MSE，1/5 block | CEM 成功 | 相对 `0.09` 差值 | 单侧 95% 下界 | 非劣 |
|---:|---:|---:|---:|---:|---:|
| `0.09` | 0.02256 / 0.09872 | 283/300 = 94.33% | — | — | 对照 |
| `0.20` | 0.02777 / 0.11097 | 278/300 = 92.67% | −1.67 pp | −3.00 pp | 是 |
| `0.30` | 0.03731 / 0.13396 | 279/300 = 93.00% | −1.33 pp | −2.67 pp | 是 |
| `0.90` | 0.05329 / 0.14079 | **285/300 = 95.00%** | **+0.67 pp** | **−1.00 pp** | **是** |

较高权重确实让冻结原始域的 native latent rollout MSE 变大，说明 prediction 拟合数值受到影响；但它没有转化为真实规划能力下降。`0.90` 的 CEM 点估计还比默认权重高 0.67 个百分点，其单侧 95% 下界为 −1.0 个百分点，明显高于预设的 −5 个百分点非劣界。

这正是为什么最终效用判据必须是“CEM 选中计划后在真实环境执行是否成功”，而不是 latent MSE 或候选排序相关性。后两者是诊断量，不能替代任务结果。

### 3.3 反证判决

预先写明的停止条件是：至少一个更高权重点同时通过 Door 规则门槛，并在原始域 CEM 上相对 `0.09` 非劣。`0.90` 同时满足两项，因此：

- “Door 规则只能靠 PLDM、std+cov 或新的条件目标学会”被当前实验反驳；
- “足够强的 SIGReg 必然明显损害原始 TwoRoom 规划”被当前实验反驳；
- “默认 `0.09` 的 Door 失败已经构成开发新目标的充分理由”被当前实验反驳；
- 条件边缘保证不足仍是正确的数学观察，但目前没有显示成必须修复的经验缺陷。

形式化状态记为 `falsified_at_the_replay50_seed3072_gate`。这不是说 `0.90` 已被证明是通用最佳权重：它是一个训练 seed、一个任务上的自适应反证点，足以否定当前绝对主张，但不足以证明跨 seed、Push-T 或其他任务的鲁棒性。

## 4. 评测协议更正

最初生成 replay 验证配置时，materializer 误继承了旧的 `all_histories_strict_v1`，要求正确规则历史还必须在每个分层中胜过 `did_not_attempt_crossing` 历史。后者从未尝试过门，对 passable/blocked 没有辨识信息，不应否决“模型是否利用历史切换规则”。

项目中用于此前正式 LeWM/PLDM 对照表的 `informative_history_rule_switch_v2` 已在 2026-07-24 冻结，早于本次所有 replay checkpoint。发现配置不一致时，`0.90` 已经完成第一次评分，所以更正过程被单独记录：

- 只从既有 v2 源复制 decision contract、metrics 和 gates；
- 不改变 checkpoint、query、原始 loss、阈值或 CEM 结果；
- `did_not_attempt_crossing` 仍完整报告，只从规则能力否决项降为辅助项；
- 四个 checkpoint 全部在同一更正配置下重新评分。

旧协议下 `0.90` 唯一失败项，是 blocked / seed 43 / right-to-left 单元中正确历史相对无证据历史的平均优势为 `−0.00174`；它对相反规则历史的胜率仍为 100%，正确未来选择率为 99.5%。更正后的正式结果不是重新选择有利阈值，而是恢复到本项目在 checkpoint 产生前已经采用的规则能力定义。

## 5. 当前能说什么，不能说什么

### 已建立

- 默认 `0.09` LeWM 在三个训练 seed 上稳定学不到 Door 规则；
- 失败由共享在线 Encoder 的规则方向压缩携带，prediction MSE 的首 batch 收缩梯度远强于默认 SIGReg；
- SIGReg loss、整体方差和有效秩正常，不能证明条件动力学信息仍被保留；
- 固定表示与 PLDM 对照证明数据、History=3 和 Predictor 容量足以完成任务；
- 在公平 replay 协议的 seed 3072 上，`0.90` SIGReg 同时学会规则并保持原始域 CEM 非劣。

### 已被当前反证否定

- 必须加入新的条件正则才能解决 Door；
- LeWM 的单项 SIGReg 设计在该问题上存在不可避免的能力权衡；
- 当前 Door 结果本身足以支撑一个“修复 LeWM 结构缺陷”的顶会方法主张。

### 仍然开放

- `0.90` 在另外两个训练 seed 上是否同样稳定；
- 一个固定权重能否同时覆盖 TwoRoom、Push-T 和其他控制任务；
- LeWorldModel 报告的 Push-T 高权重退化能否在完全相同代码、数据和闭环协议下复现；
- 是否存在真正跨任务的权重冲突，进而需要自动标定或更鲁棒的单项正则。

## 6. 下一步

本阶段不再执行原计划中的隐藏动作重映射 benchmark 和条件感知新目标开发；它们原本用于支持一个已经触发反证停止条件的结构性主张，继续推进会把研究变成结论导向。

如果继续研究，问题应改写为“SIGReg 权重的跨 seed、跨任务鲁棒性与自动标定”，而不是“SIGReg 无法保护条件信息”。合理顺序是：

1. 仅为稳定性确认，在训练 seed `4096、5120` 上复验公平 replay 的有效权重点；
2. 用原论文一致的 Push-T 数据与真实闭环指标复现权重 sweep，和 TwoRoom 使用同一能力非劣原则；
3. 只有发现不存在跨任务可用的固定权重区间时，再研究梯度尺度归一化或自动权重控制；
4. 任何新方法主张都必须同时优于充分调参的 SIGReg，而不是只与默认 `0.09` 比较。

这条路线仍可能产生有价值的问题，但当前结果已经排除了“Door 默认失败即可推出新防坍缩设计”的捷径。

## 7. 可复现资产

- 最终结果：[results/sigreg_replay50_falsification_summary.json](results/sigreg_replay50_falsification_summary.json)
- 最终分析协议：[configs/tworoom_hidden_passage_h3_sigreg_replay50_final_analysis_v1.yaml](configs/tworoom_hidden_passage_h3_sigreg_replay50_final_analysis_v1.yaml)
- 更正后的 Door 协议：[configs/tworoom_hidden_passage_h3_sigreg_replay50_rule_switch_v2_validation_v1.yaml](configs/tworoom_hidden_passage_h3_sigreg_replay50_rule_switch_v2_validation_v1.yaml)
- 第一梯级训练协议：[configs/tworoom_hidden_passage_h3_sigreg_replay50_training_v1.yaml](configs/tworoom_hidden_passage_h3_sigreg_replay50_training_v1.yaml)
- 自适应 `0.90` 协议：[configs/tworoom_hidden_passage_h3_sigreg_replay50_adaptive_extension_training_v1.yaml](configs/tworoom_hidden_passage_h3_sigreg_replay50_adaptive_extension_training_v1.yaml)
- 汇总分析器：[scripts/analyze_sigreg_replay50_falsification.py](scripts/analyze_sigreg_replay50_falsification.py)
- 协议 materializer：[scripts/materialize_sigreg_replay_protocol.py](scripts/materialize_sigreg_replay_protocol.py)、[scripts/materialize_sigreg_replay_extension.py](scripts/materialize_sigreg_replay_extension.py)、[scripts/materialize_sigreg_replay_rule_switch_validation.py](scripts/materialize_sigreg_replay_rule_switch_validation.py)
- 默认机制证据：[results/lewm_gradient_mechanism_exact_tiny_batch_original.json](results/lewm_gradient_mechanism_exact_tiny_batch_original.json)、[results/module_swaps_tiny_trajectory.json](results/module_swaps_tiny_trajectory.json)
- LeWM/PLDM 终点几何：[results/formal_unseen_endpoint/checkpoint_geometry.json](results/formal_unseen_endpoint/checkpoint_geometry.json)、[results/pldm_objective_endpoint/checkpoint_geometry.json](results/pldm_objective_endpoint/checkpoint_geometry.json)

完整原始评测记录位于 ContextWorld artifact root：

```text
evaluation/history3/hidden_passage_sigreg_replay50_falsification_v1/
  aggregate_final_rule_switch_v2.json
  door_rule_switch_v2/
  ability/
```
