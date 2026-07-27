# End-to-End JEPA 世界模型中的条件动力学坍缩

本文讨论一个比“Door ICL 是否成功”更一般的问题：当历史能够识别隐藏动力学机制时，端到端训练的世界模型为什么会主动删除这部分可预测信息。Door benchmark 是受控诊断工具，不是问题定义本身。

逐 checkpoint 数值、哈希和审计结果保存在结构化 artifact 中；本文只保留论证所需的实验设计、关键证据、结论边界和下一步可证伪实验。

## 摘要

在相同初始化、相同双规则数据、相同模型主体和相同训练预算下，默认 LeWM 联合训练的三个种子均未学会隐藏门规则，而 PLDM 联合训练三个种子全部通过。固定原始图像表示后，两种目标都能稳定学会，因此数据、History=3 输入和 Predictor 容量不是瓶颈。

失效发生在共享在线 Encoder：prediction MSE 在 Predictor 学会使用历史之前，先把两种规则对应的真实未来压近。默认 `0.09 × SIGReg` 的梯度方向反对收缩，但强度不足；SIGReg loss、整体方差和有效秩仍可保持正常，因为它约束的是无条件边缘分布 \(P(Z)\)，不知道哪一个局部方向承载了历史可预测的转移机制。

当前证据支持如下机制级结论：

> LeWM 存在一条“表示—预测器竞赛”的退化路径。联合 prediction loss 既可以通过学会历史条件降低，也可以通过让在线目标表示忽略难学的机制降低。一旦机制方向开始缩小，Predictor 学会该机制的梯度会更快衰减，最终形成选择性的条件动力学坍缩。

SIGReg 权重实验同时给出两个重要边界：

- 默认权重是 `0.09`；正式测试的梯级为 `0.30、0.90、2.05`。
- seed 3072 上，`0.09` 失败而 `0.30` 已通过，因此“不加到 2 就学不会”是错误的。当前只知道最低有效点位于 `0.09` 与 `0.30` 之间；`0.20` 尚未测试。
- `0.30、0.90、2.05` 都能在该种子阻断 Door 方向收缩，说明边缘正则在有限任务上可以通过足够强的梯度间接保护机制方向。
- 四个续训 checkpoint 都只看了 Door 合成数据，没有原始 replay。它们的原任务能力均明显低于未续训 checkpoint，因此现有 CEM 退化不能单独归因于高 SIGReg。干净的跨任务权衡实验必须固定原始/合成 replay 比例后重跑。

这使研究问题比“SIGReg 有缺陷”更精确：边缘正则无法单独保证条件转移可辨识，而固定权重又必须在不同任务的梯度尺度之间折中。Door 已经确认这一优化路径；要形成一般性贡献，还需在结构不同的隐藏机制上复现触发边界，并提出不依赖多项 loss 堆叠的条件耦合约束。

## 1. 问题定义

令

- \(H\) 为历史观测与动作；
- \(Q=(O_t,A_t)\) 为当前观测和待执行动作；
- \(M\) 为可由历史推断、但当前画面不可见的动力学机制；
- \(Z^+=E_\theta(O_{t+1})\) 为在线 Encoder 给出的真实未来表示；
- \(\hat Z^+=G_\phi(H,Q)\) 为 Predictor 输出。

当两个样本具有相同 \(Q\)、不同 \(M\)，并且真实未来不同，模型需要保留

\[
I(Z^+;M\mid Q)>0.
\]

本项目把以下失效称为**条件动力学坍缩**：

\[
D\!\left(
P(Z^+\mid Q,M=m_1),
P(Z^+\mid Q,M=m_2)
\right)\rightarrow 0,
\quad m_1\neq m_2,
\]

同时无条件表示仍然非退化：

\[
\operatorname{Var}(Z)>0,\qquad
\operatorname{rank}_{\mathrm{eff}}(Z)\ \text{保持较高}.
\]

它不是整个 Encoder 输出常量，也不等同于通常的维度坍缩。被删除的是一个低占比但对动力学预测必要的条件方向；位置、纹理和其他视觉因素仍可维持整体方差与秩。

## 2. 根因：表示—预测器竞赛

### 2.1 联合目标存在两条降损路径

LeWM 的核心目标可写为

\[
\mathcal L
=
\mathbb E\lVert G_\phi(H,Q)-E_\theta(O_{t+1})\rVert^2
+\lambda R(P_Z).
\]

prediction MSE 有两种降低方式：

1. Predictor 学会从 \(H\) 推断机制 \(M\)；
2. 在线 Encoder 缩小不同机制未来的表示差异，使 Predictor 不再需要区分它们。

第二条路径在优化初期通常更短，因为 Encoder 只需局部修改目标表示，而 Predictor 必须先发现历史中的机制证据，再把它与 query future 对齐。

### 2.2 一个最小动力学模型

把两种机制记作 \(m\in\{-1,+1\}\)，令真实未来在机制方向 \(d\) 上的编码为

\[
z_m^+=u+\alpha m d,
\]

Predictor 已学会的规则切换比例为 \(\beta\)：

\[
\hat z_m^+=u+\alpha\beta m d.
\]

忽略无关常数后，

\[
\mathcal L_{\mathrm{mech}}
=C\alpha^2(1-\beta)^2.
\]

其梯度尺度为

\[
\left|\frac{\partial\mathcal L}{\partial\alpha}\right|
=2C\alpha(1-\beta)^2,
\qquad
\left|\frac{\partial\mathcal L}{\partial\beta}\right|
=2C\alpha^2|1-\beta|.
\]

当 Predictor 尚未学会规则，即 \(\beta\ll1\) 时，梯度下降会减小 \(\alpha\)。更关键的是，\(\alpha\) 缩小后，推动 \(\beta\) 学习的信号按 \(\alpha^2\) 衰减，而继续收缩 \(\alpha\) 的信号只按 \(\alpha\) 衰减。于是早期的小幅收缩会进一步削弱 Predictor 的学习机会，形成自增强的失败路径。

这不是 Door 特有的数学结构。只要在线目标表示可训练、历史机制比表示收缩更慢学会，就存在同样的竞争。

### 2.3 边缘正则为什么没有语义保证

SIGReg 对每个时间点的 batch 表示执行随机一维投影高斯匹配，约束的是

\[
P(Z_t)\approx\mathcal N(0,I).
\]

任意只依赖 \(P_Z\) 的正则 \(R(P_Z)\) 都无法区分两个具有相同边缘分布、但 \(P(Z,M)\) 耦合不同的表示。一个表示可以把机制编码在某个方向，另一个可以用与机制无关的视觉因素占据同一方向；只要二者的 \(P_Z\) 相同，边缘正则得到的值也相同。

因此：

- 边缘高斯性可以抵抗整体收缩；
- 它不能单独保证 \(I(Z^+;M\mid Q)>0\)；
- 提高权重可以在某个有限任务上间接保护该方向，但所需强度依赖任务、初始化、batch 和优化状态；
- 这是一条“无条件几何正常，但条件语义丢失”的允许路径，不是 SIGReg 完全没有梯度。

当前多卡实现按 rank 的局部 batch 计算 SIGReg，再由 DDP 聚合参数梯度。这个实现会影响有限样本估计噪声与梯度标定，但不会改变它所约束的统计对象；即使改成跨 rank 全局 batch，目标仍然是 \(P(Z_t)\)，不是 history–future 条件耦合。因此 per-rank batch 不是本文机制结论的竞争解释。

### 2.4 可迁移的触发条件

Door 实验与最小模型共同给出六个可检验条件：

| 条件 | 对失败路径的作用 |
|---|---|
| 当前 \(Q\) 在不同机制间相同或近似相同 | 强制模型真正使用历史，而非当前画面捷径 |
| 历史可识别机制，机制确实改变未来 | 排除数据不可辨识 |
| 真实未来由共享在线 Encoder 编码 | prediction loss 可以移动监督目标 |
| Predictor 学机制慢于 Encoder 压缩机制方向 | 触发 \(\alpha\)–\(\beta\) 竞赛 |
| 机制方向在整体表示中占比低 | 边缘正则对该方向的恢复力弱 |
| 其他视觉或 nuisance 方向足以维持全局统计 | 允许整体方差、秩和 SIGReg 同时正常 |

由此得到可证伪预测：

- 历史证据越弱、机制频率越低、Predictor/Encoder 有效学习率比越小，联合 LeWM 越容易失败；
- 增加与机制无关的视觉变化会让全局统计更健康，却可能让条件方向更容易被替代；
- 固定或减慢 Encoder、先训练 Predictor、增强历史探针，应把失效边界推向更弱证据；
- 如果方法直接约束 history–future 条件耦合，它应在不全局增大 \(\lambda\) 的情况下移动该边界；
- 若性能下降前没有条件配对距离收缩，或固定表示与联合训练在同一证据强度同时失败，则本机制解释被反驳。

## 3. Door 受控证据

### 3.1 实验协议

正式训练共享：

- 同一个原始 History=3 LeWM checkpoint；
- 7,680 条 passable 和 7,680 条 blocked 合成 clip；
- 训练种子 `3072、4096、5120`；
- 单机 8 卡、全局 batch 1,024、1,024 个优化步；
- 相同 optimizer、scheduler、normalizer 和未见门位置排除表。

每个 checkpoint 使用 `50 query × 6 eval seed = 300` 个未见门位置 query。评分只在同一个 checkpoint、同一个 query 内比较预测与两种真实下一帧表示，不跨 checkpoint 比较原始 latent loss。

### 3.2 训练目标对照

| 训练配方 | 正确目标选择率 | 正确历史胜率 | 最差分层正确率 | 正式通过 |
|---|---:|---:|---:|---:|
| 原始 H3 LeWM | 50.00% | 51.33% | 0.00% | 0/1 |
| LeWM 联合训练，`λ=0.09` | 50.33% | 53.83% | 4.00% | 0/3 |
| LeWM 固定图像表示 | 100.00% | 100.00% | 100.00% | 3/3 |
| PLDM 联合训练 | 99.33% | 99.67% | 92.00% | 3/3 |
| PLDM 固定图像表示 | 100.00% | 100.00% | 100.00% | 3/3 |

这个对照排除了：

- 合成数据或 History=3 结构本身不可学；
- Predictor 容量不足；
- 所有 Encoder 联合训练都会失败；
- PLDM 的成功只是因为另一套初始化或更多训练预算。

### 3.3 表示几何与梯度

LeWM 联合训练后，Encoder/Projector 的“规则配对距离 ÷ 普通不同画面距离”从原始 `1.1720/1.1250` 降至三种子均值 `0.0084/0.0051`，正确规则切换率仅 `1.56%`。与此同时，Encoder 整体方差仍保留约 81%，有效秩仍保留约 97%。PLDM 联合训练则保持 `0.6433/0.8304`，规则切换率为 `98.67%`。

模块交叉换件表明，主要收缩由 Encoder 参数更新携带，而不是 Projector 参数或 BatchNorm buffer 单独造成。

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

负数表示压近两种规则未来，正数表示分离。SIGReg 确实反对收缩，但默认强度只抵消了 prediction 收缩的一小部分。

同一首 batch、三个独立 SIGReg 随机投影流给出的梯度平衡权重约为 `0.80、1.05、1.62`。这些数值是局部诊断，不是任务最小有效权重；正式训练已经证明 `0.30` 可以通过。

### 3.4 SIGReg loss 下降并不代表机制被保留

在原生 LeWM 的 256-step 轨迹中：

| 指标 | step 1 | step 32 | step 128 | step 256 |
|---|---:|---:|---:|---:|
| SIGReg loss | 3.1127 | 2.3088 | 2.0389 | 1.9539 |
| Projector 规则配对/普通距离比 | 1.8862 | 0.9473 | 0.2092 | 0.0441 |

因此，SIGReg 标量持续改善和规则方向持续坍缩可以同时发生。它优化了自己的边缘统计目标，但该标量不报告某个条件机制是否仍可辨识。

### 3.5 机制干预

同一初始化、数据顺序、optimizer 和 256 步预算的 CPU float32 pilot 得到：

| 目标变体 | Encoder 配对比 | Projector 配对比 | 规则切换率 |
|---|---:|---:|---:|
| 原生 LeWM | 0.061 | 0.044 | 0% |
| 只做 target detach | 0.601 | 0.777 | 0% |
| 原生 LeWM + `std(18)` | 0.035 | 0.035 | 0% |
| 原生 LeWM + `std(18)+cov(12)` | 0.465 | 0.877 | 100% |
| PLDM 全部已启用项 | 0.487 | 0.836 | 100% |

这说明 target 分支是最大收缩来源，但 detach 不是充分修复；std 只能保护尺度，不能阻止模型用相关的无关方向满足方差约束。`std+cov` 结果只是机制 pilot，尚未通过正式多卡、多种子和能力回归，不能作为最终方法。

## 4. SIGReg 权重边界与能力归因

### 4.1 正式权重梯级

默认权重为 `0.09`。当前正式跑过的更高梯级是 `0.30、0.90、2.05`，不是从 `2.0` 才开始测试。

下表全部使用 seed 3072、相同纯 Door 合成数据和 1,024-step 训练协议：

| SIGReg 权重 | Door 目标选择率 | Door 严格胜率 | 最差分层目标率 | Door 通过 | 原始域 1/5-block MSE | 原始域 CEM |
|---:|---:|---:|---:|---:|---:|---:|
| `0.09` | 50.50% | 44.50% | 8.00% | 否 | 1.212 / 1.516 | 130/300 = 43.33% |
| `0.30` | 99.00% | 92.17% | 88.00% | 是 | 1.610 / 2.212 | 105/300 = 35.00% |
| `0.90` | 100.00% | 90.83% | 100.00% | 是 | 1.767 / 2.537 | 137/300 = 45.67% |
| `2.05` | 100.00% | 94.17% | 100.00% | 是 | 1.830 / 2.699 | 116/300 = 38.67% |

`0.30` 是**当前最低已测试通过点**，不是已知最小阈值。严格结论只有：

\[
\lambda=0.09:\ \text{fail},
\qquad
\lambda=0.30:\ \text{pass}
\quad\text{（seed 3072、当前协议）}.
\]

只有额外确认该区间内的训练结果随权重单调后，才能把它写成连续阈值区间。

LeWorldModel 的 Push-T sweep 报告 `0.2` 以上开始出现风险，因此 `0.20` 是下一次最有信息量的边界点；在实际训练前不能推断它通过或失败。

### 4.2 原任务退化中的共同混淆

未续训的原始 checkpoint 在同一冻结评测上得到：

| 指标 | 原始 checkpoint |
|---|---:|
| 1-block / 5-block native MSE | 0.0157 / 0.0728 |
| 1-block / 5-block cosine error | 0.0102 / 0.0476 |
| CEM 成功率 | 273/300 = 91.00% |
| CEM 平均最终距离 | 19.91 px |

四个续训模型都远低于这个基线，包括默认 `0.09`。它们共同具有一个重要差异：续训阶段 `original_tworoom_samples=0`。因此当前数据支持：

- 纯 Door 合成续训本身带来严重的原任务遗忘或域偏移；
- 随权重从 `0.09` 增至 `2.05`，native rollout MSE 整体上升，说明较强正则可能进一步影响预测拟合；
- CEM 成功率对权重并不单调，不能把全部规划退化写成“SIGReg 越大越差”；
- native latent MSE 还受 checkpoint 表示尺度影响，所以同时保留了 cosine error 和闭环 CEM，但二者仍不能消除训练数据混淆。

因此，现阶段不能把 `0.30` 或 `2.05` 单独判为“因高权重破坏原任务”，也不能宣称存在干净的 Door—PushT/TwoRoom 不可兼得。正确的排他实验是固定 50% 原始 replay、50% 机制合成数据，只改变 `λ∈{0.09,0.20,0.30}`，再同时运行 Door 与原任务 rollout/CEM。

## 5. 从 Door 推导结构不同的新场景

### 5.1 现有动作延迟结果是重要正对照

默认 LeWM 在另一个 History=3 benchmark 上并非总会失败。使用 50% 原始 replay 和 50% 多延迟合成数据训练时，三个种子的一步动作延迟历史选择率为 `91.61%`，正确目标选择率为 `84.06%`，原始域 CEM 为 `93.56%±1.02%`。

这个结果与表示—预测器竞赛并不矛盾。动作延迟的满幅历史探针产生大范围位移，Predictor 更容易快速发现机制；同时原始 replay 对表示提供了锚点。Door 的规则证据和未来差异更局部。现有动作延迟低幅压力测试只在满幅训练模型上改变 Eval 动作，不能证明联合训练发生条件坍缩。

因此动作延迟应作为“同一 LeWM 可以成功”的正对照，而不是第二个失效复现。

### 5.2 新 benchmark：隐藏动作重映射

为排除门、接触和局部视觉语义，新场景使用两种不可见控制机制：

\[
M_x:\ (+x)\mapsto(+x),
\qquad
M_y:\ (+x)\mapsto(+y).
\]

History=3 配对协议为：

```text
共同中心状态
  ├─ 执行 ρ·(+x)：Mx 沿 x 位移，My 沿 y 位移
  ├─ 执行 ρ·(-x)：两种机制都回到共同中心状态
  └─ 相同满幅 query (+x)：未来分别沿 x / y 位移
```

这样可同时满足：

- query 前的当前位置、当前画面、动作队列和 query 动作完全相同；
- 历史轨迹是唯一机制证据；
- query 始终使用满幅动作，未来目标差异不随 \(\rho\) 改变；
- \(\rho\) 只控制 Predictor 看到的历史证据强度。

第一阶段扫描

\[
\rho\in\{1.0,0.5,0.25,0.125\},
\]

并固定 50% 原始 replay。每个 \(\rho\) 比较：

1. 默认 `0.09` LeWM 联合训练；
2. LeWM 固定 Encoder/Projector；
3. PLDM 联合训练；
4. 仅用于机制验证的 LeWM 低 Encoder 学习率或 Predictor warm-up。

第二阶段再加入 episode-static nuisance 背景：同一配对 query 共享背景，但不同 query 的背景独立变化。它增加边缘表示可用的无关方差，却不提供机制信息，直接检验“其他方向填满边缘统计”的预测。

### 5.3 预注册判据与可证伪结果

每个配置使用三个训练种子和 `50×6` 个未见起点 query，报告：

- 匹配历史的两目标选择率；
- 匹配历史相对错误历史的胜率；
- 最差 seed×机制×方向分层；
- Encoder/Projector 条件配对距离比的训练轨迹；
- SIGReg loss、整体方差和有效秩；
- 原始域 rollout error 与 `50×6` CEM。

关键预测不是“某个模型最终分数低”，而是失效边界的相对位置：

- 若联合 LeWM 随 \(\rho\) 降低先出现条件配对距离收缩，而固定表示和 PLDM 仍能学会，支持表示—预测器竞赛；
- 若降低 Encoder 学习率或 Predictor warm-up 把临界 \(\rho\) 向下移动，提供优化时序的因果证据；
- 若加入 nuisance 后全局统计不变或更好、机制方向却更早丢失，支持边缘覆盖不足；
- 若固定表示与联合训练同时失败，说明是历史证据或 Predictor 容量瓶颈；
- 若性能失败但条件配对距离没有先收缩，则 Door 根因不能推广到该场景。

这个设计比再造一个“门”更有信息量：它改变了动力学机制的语义，同时保留了同 query、历史可辨识、未来分叉和在线目标 Encoder 四个必要结构，并能连续控制触发强度。

## 6. 研究贡献边界

相关工作已经覆盖若干相邻问题：

- [LeJEPA](https://arxiv.org/abs/2511.08544) 与 [LeWorldModel](https://arxiv.org/abs/2603.19312) 展示了 SIGReg 的单项、线性复杂度和无需 teacher 的简洁性；
- [VICReg](https://arxiv.org/abs/2105.04906) 与 PLDM 使用 variance、covariance 等多成分约束；
- [LDReg](https://arxiv.org/abs/2401.10474) 已指出全局高维与局部低维可以共存；
- [VISReg](https://arxiv.org/abs/2606.02572) 处理 SIGReg 在近坍缩区域的梯度和尺度—形状耦合；
- [When Does LeJEPA Learn a World Model?](https://arxiv.org/abs/2605.26379) 的保证依赖平稳加性噪声等条件，历史决定的离散机制切换不在其直接覆盖范围内；
- [On Identifiability of Controlled World Models](https://arxiv.org/abs/2607.22430) 研究受控世界模型的可辨识条件，但与共享在线目标表示中的历史条件收缩问题不同。

因此，潜在贡献不应定义为：

- 再次发现完整或局部维度坍缩；
- 简单提高 SIGReg 权重；
- 把已有 std/covariance 组件重新组合命名；
- 只报告 Door benchmark 上 PLDM 优于 LeWM。

更有价值的研究主线是：

> 证明并系统刻画端到端预测中的条件机制别名化：边缘表示保持健康时，在线目标 Encoder 仍会优先删除慢于 Predictor 学习的历史条件动力学；再用一个条件耦合统计量保护这类信息，而不恢复成多项防坍缩 loss 的手工权重组合。

单个 Door 场景还不足以支撑顶会级一般性结论。至少需要：

1. Door 与隐藏动作重映射两个结构不同的失败复现；
2. 动作延迟成功场景作为触发条件的正对照；
3. \(\alpha\)–\(\beta\) 竞赛的理论命题和可测 proxy；
4. Encoder 学习率或 warm-up 对失效边界的因果移动；
5. 一个条件感知但仍简洁的目标，在 Door、动作重映射和普通控制任务上同时优于默认 SIGReg 与多成分基线；
6. 固定 replay 协议下的原任务非劣验证。

## 7. 下一步实验顺序

### P0：清理当前能力归因

在 seed 3072 上固定 50% 原始 replay、50% Door 合成数据，运行 `λ={0.09,0.20,0.30}`。每个 checkpoint 都执行 Door `50×6`、冻结 rollout error 和原始域 CEM。这个实验回答 `0.20` 是否足够，以及 Door 通过点能否在公平协议下保留原任务能力。

### P1：验证跨场景根因

实现隐藏动作重映射的 \(\rho\) 扫描。先用单种子定位分界，再只对临界相邻点扩展三个种子。冻结表示与 PLDM 是可学性对照，低 Encoder 学习率和 Predictor warm-up 是机制干预，不作为最终方法。

### P2：开发条件感知目标

只有当 P1 复现相同失效轨迹后，才开发新目标。优先考虑同 query 内 history–future 配对与条件打乱之间的统一 sketch statistic，或对匹配历史相对错误历史的预测增益进行尺度归一化。验收标准是：

- 不使用隐藏机制标签；
- 不冻结 Encoder，不引入 EMA teacher；
- 保持一个主要统计原则和至多一个有效外层权重；
- 复杂度对 batch 和表示维度近线性；
- 在普通控制任务上通过固定 replay 的非劣检验。

VISReg、`std+cov`、PLDM 和固定表示继续作为基线与机制对照，不直接当作本项目的新方法。

## 8. 研究资产

- 实验矩阵：[configs/experiment_matrix.yaml](configs/experiment_matrix.yaml)
- 权重与能力汇总：[results/sigreg_weight_sweep_summary.json](results/sigreg_weight_sweep_summary.json)
- 数据、checkpoint 与审计哈希：[results/evidence_manifest.json](results/evidence_manifest.json)
- 正式 LeWM 终点几何：[results/formal_unseen_endpoint/checkpoint_geometry.json](results/formal_unseen_endpoint/checkpoint_geometry.json)
- 正式 PLDM 终点几何：[results/pldm_objective_endpoint/checkpoint_geometry.json](results/pldm_objective_endpoint/checkpoint_geometry.json)
- 模块换件：[results/module_swaps_tiny_trajectory.json](results/module_swaps_tiny_trajectory.json)
- 精确首 batch 梯度：[results/lewm_gradient_mechanism_exact_tiny_batch_original.json](results/lewm_gradient_mechanism_exact_tiny_batch_original.json)
- detach/std 反证：[results/lewm_objective_mechanism_cpu_pilot.json](results/lewm_objective_mechanism_cpu_pilot.json)
- std+cov/PLDM pilot：[results/lewm_std_cov_pldm_active_cpu_pilot.json](results/lewm_std_cov_pldm_active_cpu_pilot.json)
- `2.05` 训练轨迹：[results/lewm_sigreg2p05_formal_trajectory_s3072/checkpoint_geometry.json](results/lewm_sigreg2p05_formal_trajectory_s3072/checkpoint_geometry.json)
- 四档 SIGReg 原始能力明细：`results/lewm_sigreg*_cross_ability_s3072/`
- 几何、换件和梯度分析脚本：[scripts](scripts)
- 诊断测试：[tests](tests)

ContextWorld 中的配套文档：

- [Door Benchmark 设计](../../../ContextWorld/docs/TwoRoom_Door_Benchmark_Design.md)
- [动作延迟正对照](../../../ContextWorld/docs/TwoRoom_Action_Delay_Benchmark_Report.md)

正式 PLDM 汇总位于 ContextWorld artifact root：

```text
evaluation/history3/hidden_passage_pldm_objective_validation_rule_switch_v2/
  aggregate_rule_switch_v2.json
```
