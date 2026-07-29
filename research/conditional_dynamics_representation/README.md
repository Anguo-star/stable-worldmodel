# LeWM 中条件动力学信息的选择性坍缩

本文研究一个比 TwoRoom 门位置本身更一般的问题：当当前观测和动作相同、只有历史能够识别隐藏动力学机制时，世界模型会不会在联合训练中删除这部分条件信息？

结论分为三层：

1. 默认 LeWM 的失败是真实且可定位的。prediction MSE 通过共享在线 Encoder 强烈压缩门规则相关的局部表示方向；默认 `0.09 × SIGReg` 虽然反对收缩，但其加权梯度远弱于 prediction MSE，而且它只约束无条件边缘分布。
2. 这不是一个“原生 SIGReg 永远无法解决”的结构性定理。在固定 50% 原始 replay、50% Door replay 的单训练 seed 反证中，把原生 SIGReg 权重提高到 `0.90` 已能学会门规则，并保持原始域真实 CEM 成功率非劣。
3. 将同一个 SIGReg 统计量改为约束条件匹配的高通差分后，无需提高 `0.09` 权重，也无需加入 variance、covariance 或 VISReg 多成分。在三个训练 seed 上，该方法都完整保留门规则，且原始域 CEM 均相对默认 LeWM 非劣。

因此，当前最准确的判断是：

> 默认边缘 SIGReg 对历史可识别的局部动力学方向缺少定向保护；更高的全局权重可以在当前任务上补偿，但条件化同一个统计量能以默认权重更直接地保护目标方向。三 seed 结果建立了稳定候选机制，还没有建立跨任务方法优势。

机器可读汇总、原始文件哈希和逐 checkpoint 审计见
[三 seed 结果](results/conditional_sigreg_multiseed_v1.json)。

## 1. 问题并不只属于 Door 或 ICL

设当前可见 query 为 \(Q\)，历史为 \(H\)，动作序列为 \(A\)，隐藏动力学机制为 \(M\)。我们关注的场景满足：

\[
P(O^+\mid Q,A,M_1)\neq P(O^+\mid Q,A,M_2),
\qquad
M\ \text{可由}\ H\ \text{识别，但不能由}\ Q\ \text{单独识别}.
\]

Door 只是最小可控实例：相同门位置、相同当前画面和相同穿门动作，在 passable 与 blocked 规则下产生不同下一状态；此前是否成功穿门的历史才暴露规则。

同类问题还可能出现在：

- 隐藏电机极性或动作重映射；
- 只能由历史交互识别的摩擦、质量或执行器延迟；
- 接触模式、锁止状态或工具状态；
- 相同视觉 query 下具有不同转移规律的环境阶段。

这类失败容易出现于以下组合：目标表示由在线 Encoder 共同更新；Predictor 在训练早期尚未识别机制；机制只占表示中的局部方向；其他视觉因素足以维持整体方差与边缘分布。此时优化器可以先删除难预测的机制方向，而不是先让 Predictor 学会使用历史。

这不是“LeWM 对所有隐藏动力学都无能为力”。action-delay H3 负对照中，原生 LeWM 能学到更强的历史信号。当前研究定位的是一种具体失败条件，而不是对所有 ICL 场景作绝对断言。

## 2. 三类评测回答三个不同问题

### 2.1 Door 两真实未来判别：模型是否使用历史规则

对同一个 history/query 和固定穿门动作 \(a\)，模型预测

\[
\hat z^+=G_\phi(H,Q,a).
\]

模拟器分别在 passable 与 blocked 规则下真实执行该动作，再用同一 checkpoint 的 Encoder 得到
\(z^+_{\mathrm{pass}}\) 和 \(z^+_{\mathrm{block}}\)。若当前历史对应 passable，正确判定为

\[
\lVert\hat z^+-z^+_{\mathrm{pass}}\rVert^2
<
\lVert\hat z^+-z^+_{\mathrm{block}}\rVert^2,
\]

blocked 情况反之。

固定动作是有意设计：它把“是否从历史推断规则”与“动作搜索是否成功”分离。正式门槛同时要求：

- 正确真实未来选择率至少 95%；
- 正确规则历史相对相反规则历史的胜率至少 95%；
- 每个 seed × 方向 × 规则分层的最差正确率至少 80%。

Door 使用 eval seed `42–47`，每个 seed 50 个未见门位置 query，共 300 个 query。

### 2.2 原始域真实 CEM：模型是否仍能完成原任务

CEM 在世界模型内搜索动作序列，再把选中的序列放回真实环境执行。只有真实环境到达目标才算成功。协议固定为：

- 6 个 eval seed × 每个 50 个 query；
- horizon 5、receding horizon 5；
- 每轮 300 个候选、30 次迭代、top-k 30；
- 所有 checkpoint 使用同一冻结 catalog 和 normalizer。

候选 checkpoint 与默认 `0.09` LeWM 在同一 300 个 query 上逐条配对。非劣界为绝对成功率 `−5` 个百分点，使用按 eval seed 分层的 20,000 次配对 bootstrap。

### 2.3 冻结 rollout MSE：预测误差诊断

在冻结原始域 query 上报告 1/2/3/5 个 action block 的 native latent rollout MSE。它能说明预测拟合如何变化，但不能替代真实 CEM 成功率。

本文不使用 Spearman、候选排序相关性或其他代理排序指标。

### 2.4 为什么单门 blocked rollout 不是有效闭环证据

当前 Hidden Passage 环境只有一条物理通道。blocked 规则下，跨房间目标本来就不可达，因此“模型没有穿过门”对任何策略都成立，不能证明模型读懂了历史。

有效的第二个闭环任务必须让两种隐藏机制都有可达解，但要求不同最优动作。例如：

- 隐藏电机极性：同一目标在两种极性下需要相反控制；
- 两条通道且仅一条隐藏可通行：两种规则都能到达目标，但必须选择不同路线。

## 3. 默认 LeWM 的机制级根因

### 3.1 数据、History=3 和 Predictor 容量都足够

相同初始化、15,360 条双规则合成数据、8 卡、全局 batch 1,024、1,024 个优化步下：

| 训练配方 | 正确未来选择率 | 正确历史胜率 | 最差分层正确率 | 正式通过 |
|---|---:|---:|---:|---:|
| 原始 H3 LeWM | 50.00% | 51.33% | 0.00% | 0/1 |
| LeWM 联合训练，`λ=0.09` | 50.33% | 53.83% | 4.00% | 0/3 |
| LeWM 固定图像表示 | 100.00% | 100.00% | 100.00% | 3/3 |
| PLDM 联合训练 | 99.33% | 99.67% | 92.00% | 3/3 |
| PLDM 固定图像表示 | 100.00% | 100.00% | 100.00% | 3/3 |

固定表示与 PLDM 对照说明，训练数据、History=3 输入和 Predictor 容量不是瓶颈；一般性的 Encoder 联合训练也不是充分根因。失败来自默认 LeWM 目标在当前尺度下走出的优化路径。

### 3.2 收缩主要由在线 Encoder 参数更新携带

默认 LeWM 联合训练后，Encoder/Projector 的“规则配对距离 ÷ 普通不同画面距离”降至 `0.0084 / 0.0051`，正确规则切换率仅 `1.56%`。整体方差仍保留约 81%，有效秩仍保留约 97%，因此不是整个 Encoder 输出常量，而是规则相关局部方向被选择性压近。

PLDM 联合训练的对应距离比为 `0.6433 / 0.8304`，正确规则切换率为 `98.67%`。模块换件实验进一步把主要收缩定位到 Encoder 参数更新，而不是 Projector 参数或 BatchNorm buffer。

### 3.3 首 batch 的梯度竞争高度不平衡

在精确首个训练 batch 上，沿 Projector 规则配对距离测量无穷小梯度步方向：

| 梯度来源 | 配对距离变化方向 |
|---|---:|
| prediction target 分支 | -678.83 |
| prediction context / predictor 分支 | -266.37 |
| 完整 prediction MSE | **-945.20** |
| 原始 SIGReg | +459.88 |
| `0.09 × SIGReg` | **+41.39** |
| LeWM 总目标 | **-903.81** |
| PLDM 已启用正则合计 | +1,691.39 |
| PLDM 总目标 | **+746.19** |

负数表示压近两种规则未来，正数表示分离。SIGReg 并非没有反抗收缩；默认权重下，它的有效梯度规模远不足以抵消 prediction MSE。

### 3.4 为什么 Predictor 会越来越难学会规则

令目标表示中的机制幅度为 \(\alpha\)，Predictor 已学到的规则切换比例为 \(\beta\)：

\[
z_m^+=u+\alpha m d,\qquad
\hat z_m^+=u+\alpha\beta m d,\qquad m\in\{-1,+1\}.
\]

规则相关 prediction loss 可写为

\[
\mathcal L_{\mathrm{mech}}
=C\alpha^2(1-\beta)^2.
\]

当 Predictor 尚未识别规则时，优化器既可以增大 \(\beta\)，也可以减小在线目标中的 \(\alpha\)。一旦 \(\alpha\) 开始缩小：

- 继续压缩 Encoder 规则方向的梯度按 \(\alpha\) 下降；
- 推动 Predictor 学规则的梯度按 \(\alpha^2\) 更快下降。

这形成自增强的早期失败路径。这里的梯度不是人为定义的“ICL loss 梯度”，而是对规则配对距离和简化机制 loss 的明确方向分析。

### 3.5 SIGReg loss 可以下降，同时条件信息消失

默认 LeWM 的 256-step 轨迹中，SIGReg loss 从 `3.1127` 降到 `1.9539`，但 Projector 规则配对距离比从 `1.8862` 降到 `0.0441`。

原因在于原始 SIGReg 对每个时间点、整个 batch 的随机一维投影匹配标准高斯，约束的是

\[
P(Z_t)\approx\mathcal N(0,I).
\]

Door 真正需要保护的是

\[
I(Z^+;M\mid Q)>0.
\]

只依赖无条件边缘 \(P_Z\) 的正则无法区分：

- 规则方向被保留，其他视觉因素也正常分布；
- 规则方向被删除，但其他视觉因素填满同一个边缘分布。

因此，整体方差、有效秩、随机投影分布和 SIGReg loss 都正常，仍不能证明条件动力学信息存在。多卡实现使用 rank-local batch 估计统计量；这会影响有限样本噪声与梯度标定，但不改变“边缘统计不保证条件信息”这一理论边界，也不是本研究的根因解释。

## 4. 高权重反证：不能把默认失败写成绝对结构缺陷

为避免 Door-only 续训造成原任务遗忘，权重 sweep 固定 50% 原始 TwoRoom replay 和 50% Door replay，只改变原生 SIGReg 权重：

| 原生 SIGReg λ | 训练 pred loss | SIGReg loss | Door 正确未来 | 正确历史 | 最差分层 | Door 通过 |
|---:|---:|---:|---:|---:|---:|---:|
| `0.09` | 0.02426 | 1.76074 | 51.17% | 67.83% | 20% | 否 |
| `0.20` | 0.03813 | 1.64941 | 56.67% | 92.33% | 32% | 否 |
| `0.30` | 0.04760 | 1.61426 | 67.67% | 98.83% | 44% | 否 |
| `0.90` | 0.06995 | 1.49316 | **99.50%** | **100.00%** | **96%** | **是** |

对应原始域结果：

| 原生 SIGReg λ | rollout MSE，1/5 block | 真实 CEM | 相对 `0.09` 差值 | 单侧 95% 下界 | 非劣 |
|---:|---:|---:|---:|---:|---:|
| `0.09` | 0.02256 / 0.09872 | 283/300 = 94.33% | — | — | 对照 |
| `0.20` | 0.02777 / 0.11097 | 278/300 = 92.67% | −1.67 pp | −3.00 pp | 是 |
| `0.30` | 0.03731 / 0.13396 | 279/300 = 93.00% | −1.33 pp | −2.67 pp | 是 |
| `0.90` | 0.05329 / 0.14079 | **285/300 = 95.00%** | **+0.67 pp** | **−1.00 pp** | **是** |

这组反证排除了三种过强说法：

- Door 规则必须靠 PLDM、std+cov 或新条件目标才能学会；
- 足够强的 SIGReg 必然损害当前 TwoRoom 的真实 CEM；
- 默认 `0.09` 的失败本身已足以证明 LeWM 存在不可调参修复的结构缺陷。

同时，它没有证明 `0.90` 是通用最优权重。较高权重显著增大冻结 rollout MSE；跨 Push-T 和其他控制任务能否保持同一可用权重区间仍是开放问题。

## 5. Conditional SIGReg：仍然只有一个正则成分

原始 LeWM 对每个时间点的 batch population \(\{z_{t,b}\}\) 应用 SIGReg。我们只改变统计量看到的 population。

对由可见条件匹配的样本对 \((i,j)\)，若该时间点两条真实未来已经分叉，则使用 Haar 高通差分

\[
h_{t,ij}=\frac{z_{t,i}-z_{t,j}}{\sqrt 2}.
\]

条件正则为

\[
R_{\mathrm{cond}}
=\frac1T\sum_t
\begin{cases}
\operatorname{SIGReg}(\{h_{t,ij}\}), & \text{该时间点存在活跃匹配对},\\
\operatorname{SIGReg}(\{z_{t,b}\}), & \text{否则}.
\end{cases}
\]

若两条表示都是独立标准高斯，高通差分仍是标准高斯；若模型把匹配条件下的两种未来压成同一点，高通 population 变为零，原有 SIGReg 统计量会直接看到这种收缩。

实现保持不变的部分包括：

- 同一个 Epps–Pulley characteristic-function statistic；
- 同一个标准高斯目标；
- 1,024 个随机投影和 17 个 knots；
- 单一标量正则；
- 总权重仍为 `0.09`；
- 不加入 std、covariance、center/scale/shape 或 VISReg loss。

每个差分随机乘 Rademacher 符号，使目标对 pair 内任意排序在期望上不变。正式训练的 pair 只由相同首帧可见像素与相同完整动作序列构造；loss 不读取门规则标签或类别标签。没有 pair 时，代码精确回退到原始 SIGReg。

这不是把 LeWM 改回多成分 VICReg/PLDM。创新点是让原本单一、简洁的 Gaussian sketch 对条件差异方向敏感，而不是重新堆叠 variance、covariance 等多个防坍缩项。

### 5.1 机制消融

在相同 tiny pilot 中：

| 目标 | 规则配对/普通距离比 | 正确规则切换 |
|---|---:|---:|
| paired batch + 原始 SIGReg `0.09` | 0.0289 | 0% |
| Conditional high-pass SIGReg `0.09` | **0.7062** | **100%** |
| 完整可逆 Haar low-pass + high-pass 后再做边缘 SIGReg | 0.0202 | 0% |

完整 Haar 变换仍然保留全部信息，却不能解决问题，说明效果不是来自任意 pair 重排或可逆变换；关键是把高通差异作为独立条件 population 直接约束。只做 target detach、只提高 std 的早期消融也没有恢复规则，进一步排除了单一 detach bug。

## 6. 三训练 seed 正式验证

### 6.1 冻结协议

三组训练 seed 为 `3072、4096、5120`。所有 checkpoint 固定：

- 相同 seed-3072 初始权重；
- 相同数据 split 和 normalizer；
- 50% 原始 replay、50% Door replay；
- 8 卡、每卡 batch 128、全局 batch 1,024；
- 1,024 个 optimizer step；
- 相同 optimizer、scheduler 和 bf16-mixed 精度；
- 唯一方法差异是原始 marginal SIGReg 与 conditional high-pass SIGReg；
- 两者权重都为 `0.09`。

`paired-native` 使用与 conditional 方法完全相同的相邻配对 batch，但仍计算原始无条件 SIGReg，用于隔离 batch 排序或配对呈现本身。所有六个正式训练 run 均保存本地逐步 loss trace；新增的四个多 seed run 均确认 SwanLab 初始化成功。

### 6.2 Door 规则能力

| 训练 seed | 方法 | 正确未来选择率 | 正确历史胜率 | 最差分层正确率 | 通过 |
|---:|---|---:|---:|---:|---:|
| 3072 | paired-native `0.09` | 55.17% | 87.50% | 32% | 否 |
| 3072 | conditional `0.09` | **100.00%** | **100.00%** | **100%** | 是 |
| 4096 | paired-native `0.09` | 52.83% | 83.00% | 24% | 否 |
| 4096 | conditional `0.09` | **100.00%** | **100.00%** | **100%** | 是 |
| 5120 | paired-native `0.09` | 52.33% | 84.00% | 20% | 否 |
| 5120 | conditional `0.09` | **100.00%** | **100.00%** | **100%** | 是 |

Conditional SIGReg 在三个独立优化/data-order seed 上都完整通过；paired-native 在三个 seed 上都失败。因此，效果不能由“模型看到了成对样本”或 batch 顺序解释。

### 6.3 原始域真实 CEM

冻结 native `0.09` 对照为 283/300 = 94.33%。

| 训练 seed | 方法 | CEM 成功 | 相对对照差值 | 单侧 95% 下界 | 非劣 |
|---:|---|---:|---:|---:|---:|
| 3072 | paired-native | 278/300 = 92.67% | −1.67 pp | −3.33 pp | 是 |
| 3072 | conditional | **282/300 = 94.00%** | **−0.33 pp** | **−2.00 pp** | 是 |
| 4096 | paired-native | 282/300 = 94.00% | −0.33 pp | −1.67 pp | 是 |
| 4096 | conditional | **280/300 = 93.33%** | **−1.00 pp** | **−3.00 pp** | 是 |
| 5120 | paired-native | 283/300 = 94.33% | 0.00 pp | −1.00 pp | 是 |
| 5120 | conditional | **281/300 = 93.67%** | **−0.67 pp** | **−2.67 pp** | 是 |

三个 conditional checkpoint 的下界都高于预设 `−5` pp 非劣界。跨训练 seed 的描述性合计中，paired-native 与 conditional 恰好都是 `843/900 = 93.67%`；这个合计不被当作 900 个独立训练重复，只用于说明当前方法没有通过牺牲普通 CEM 效用换取 Door 分数，也没有声称提高普通 CEM。

### 6.4 冻结 rollout 误差

| 训练 seed | 方法 | 训练终点 pred loss | 1-block MSE | 5-block MSE |
|---:|---|---:|---:|---:|
| 3072 | paired-native | 0.02933 | 0.02063 | 0.10899 |
| 3072 | conditional | 0.07490 | **0.01610** | **0.07331** |
| 4096 | paired-native | 0.02783 | 0.02808 | 0.11382 |
| 4096 | conditional | 0.07672 | **0.01611** | **0.07015** |
| 5120 | paired-native | 0.02969 | 0.02546 | 0.10930 |
| 5120 | conditional | 0.07657 | **0.01535** | **0.06886** |

Conditional 训练的终点 batch pred loss 更高，但冻结原始域 1/2/3/5-block rollout MSE 在每个训练 seed、每个 horizon 上都更低。跨 seed 的平均 5-block MSE 为：

- paired-native：`0.11070`；
- conditional：`0.07077`。

这说明“训练日志中的 pred loss 更高”不能直接解释为 eval 预测退化，更不能替代 CEM 成功率。

## 7. 当前结论的边界

### 已建立

- 默认 `0.09` LeWM 在 Door 上稳定删除历史可识别的规则方向；
- 早期压缩由 prediction MSE 与在线 Encoder 更新主导，默认 SIGReg 加权梯度不足；
- SIGReg loss、整体方差和有效秩正常，不保证条件动力学信息仍被保留；
- 高权重原生 SIGReg 的反证说明当前问题不是不可调参解决的绝对结构缺陷；
- Conditional SIGReg 仍是单一正则成分，在默认 `0.09` 权重下跨三个训练 seed 稳定保留规则；
- 三个 conditional checkpoint 均保持原始域真实 CEM 非劣；
- paired-native 三 seed 全失败，排除了配对数据顺序解释。

### 尚未建立

- Conditional SIGReg 跨任务优于充分调参的原生 SIGReg；
- 一个固定权重能同时覆盖 TwoRoom、Push-T 和其他控制任务；
- 条件 pair 在没有合成配对或环境元数据时能够可靠自动发现；
- 当前结果足以支持完整顶会方法主张。

## 8. 下一步研究

下一阶段不应继续修改 Door 接入或增加无信息的 blocked-crossing rollout，而应验证根因能否预测新的失败场景。

1. 构造第二个隐藏动力学任务。两种机制必须在相同可见 query/goal 下都有可达解，但要求不同动作；优先选择隐藏电机极性或双通道单开放规则。
2. 在该任务中预先检查失败条件：历史可识别、当前帧不可识别；原生 LeWM 的整体方差正常但条件差分方向收缩；Conditional SIGReg 在默认权重下保护该方向。
3. 用原论文一致的数据、代码和真实闭环成功率复现 Push-T SIGReg 权重 sweep，验证是否存在 TwoRoom 与 Push-T 之间没有共同可用固定权重的真实冲突。
4. 所有新方法比较都必须包含充分调参的原生 SIGReg、paired-native、Conditional SIGReg 和真实闭环任务结果。
5. 若跨任务权重冲突成立，再研究无需规则标签的 pair/condition 发现、梯度尺度自适应或自动权重控制。

IDM 可以作为增加历史敏感训练信号的独立消融，但它并不自动约束在线目标 Encoder 保留条件差异，不能在验证前视为对当前收缩机制的充分修复。

## 9. 可复现资产

- 三 seed 正式结果：[results/conditional_sigreg_multiseed_v1.json](results/conditional_sigreg_multiseed_v1.json)
- 单 seed candidate screen：[results/conditional_sigreg_screen_v1.json](results/conditional_sigreg_screen_v1.json)
- 机制 pilot：[results/conditional_sigreg_mechanism_pilot_v1.json](results/conditional_sigreg_mechanism_pilot_v1.json)
- 高权重反证：[results/sigreg_replay50_falsification_summary.json](results/sigreg_replay50_falsification_summary.json)
- 多 seed 冻结协议：[configs/tworoom_hidden_passage_h3_conditional_sigreg_multiseed_v1.yaml](configs/tworoom_hidden_passage_h3_conditional_sigreg_multiseed_v1.yaml)
- candidate screen 协议：[configs/tworoom_hidden_passage_h3_conditional_sigreg_screen_v1.yaml](configs/tworoom_hidden_passage_h3_conditional_sigreg_screen_v1.yaml)
- 汇总分析器：[scripts/analyze_conditional_sigreg_multiseed.py](scripts/analyze_conditional_sigreg_multiseed.py)
- Door 评测入口：[scripts/eval_conditional_sigreg_door.py](scripts/eval_conditional_sigreg_door.py)
- 正式训练 overlay：[scripts/run_conditional_sigreg_formal.py](scripts/run_conditional_sigreg_formal.py)
- 正则实现：[stable_worldmodel/wm/loss.py](../../stable_worldmodel/wm/loss.py)
- 公共训练入口：[scripts/train/lewm.py](../../scripts/train/lewm.py)

ContextWorld 原始结果位于：

```text
evaluation/history3/conditional_sigreg_screen_v1/
evaluation/history3/conditional_sigreg_multiseed_v1/
training/reports/h3_passage_replay50_{paired_native,conditional_sigreg0p09}_passage_formal_s{3072,4096,5120}.json
```
