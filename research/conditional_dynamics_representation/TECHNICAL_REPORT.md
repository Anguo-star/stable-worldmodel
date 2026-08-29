# 条件重叠联合对齐与滚动一致性：让世界模型从历史中识别隐藏动力学

## 摘要

世界模型（world model）即使拥有方差充足、无坍缩（collapse）的隐空间表征，也未必能从历史中识别当前回合的隐藏动力学（hidden dynamics）。我们在 ContextWorld 上检验一个更严格的性质：固定当前观测与查询动作、只改变揭示隐藏动力学的历史，模型是否给出不同且方向正确的未来预测。原生 LeWM、stop-gradient、SIGReg、已有的 VISReg 筛查及多种全局协方差目标表明，改善表征的边缘分布不足以建立这种条件关系；其中 VISReg 尚需按完整公开配方补齐正式终点。我们进一步给出一个命题：在缺少条件重叠（conditional overlap）或额外结构假设时，反事实条件响应在非参数意义下不可识别。据此我们提出**条件重叠联合对齐**（Conditional-Overlap Joint Alignment，COJA）：在共享可见条件 $(Q,A)$、历史与真实未来不同的样本组上，直接监督一步条件对应。用于连续动力学的二元目标由去中心响应项和在真实目标处静止的分配势垒（assignment barrier）组成；离散三元组使用对称分配目标。该辅助项只更新已有的预测器路径，不引入新参数或推理开销。COJA 方法族在 ActionDelay、Contact Friction、Portal Exit、Motion Damping 上均给出正的条件响应信号（例如 Motion 的归一化响应误差 NRE 由 $1.130$ 降至 $0.767$）。但一步条件可辨识性不自动传递到自回归滚动：一步 switch 达 $0.938$ 的 checkpoint 在两步隐藏动力学规划中仍近乎失败。我们因此提出 **RC-COJA**（rollout-consistent COJA），在不新增参数的前提下把一部分既有的隐藏样本原生 MSE 重新分配给真实的第二步自回归预测。在从公开初始化的单阶段 4,096 步训练中，RC-COJA 相对同数据一步 COJA 把 Motion 的两步/三步隐藏规划物理误差分别改善 $57.39$ 与 $39.08$ px，Contact 的两步/五步分别改善 $3.16$ 与 $3.09$ px，两项均在第二个训练随机种子上复现；标准 PushT 原任务保持未检测到方法特异的退化。代价是 Motion 上约 $3$ px 的一步误差增加，以及仍然存在的条件重叠数据假设。

---

## 1. 引言

世界模型的训练目标通常由两部分组成：拟合未来表征的预测损失，加上防止表征坍缩的正则项。这一组合在标准基准上有效，但它没有回答一个部署时至关重要的问题：当环境中存在回合内固定、却不可直接观测的动力学因素（摩擦、阻尼、动作延迟、传送出口）时，模型能否从历史中把它识别出来，并据此改变对同一动作的未来预测？

本文以 ContextWorld 为受控平台研究这一性质，并给出三项贡献。

1. **诊断**：我们把“边缘非坍缩但条件失败”的现象拆成可独立测量的环节，并证明只依赖目标表征经验分布的正则项对样本置换不变，因而无法编码“哪段历史对应哪个未来”。我们进一步给出无重叠情形下的不可识别性命题（命题 1）。
2. **方法**：COJA 在共享可见条件的样本组上直接监督一步条件对应；RC-COJA 在同一 Predictor 上展开短自回归，把该对应传递到规划视界（horizon）。两者都不增加参数、模块或推理计算，保存的仍是原族 checkpoint。
3. **证据**：跨四个隐藏动力学任务的条件响应结果，Contact 与 Motion 上的隐藏动力学规划结果与各两个训练随机种子的复现，以及把三类目标量（条件响应、隐藏动力学规划、原任务保持）分开报告的评测协议。

### 1.1 与现有方向的区别

本文与三类相邻工作相关，但解决的条件变量不同。第一类方法改善隐空间的全局几何或抗坍缩性质；例如 [VISReg](https://arxiv.org/abs/2606.02572) 解耦表征的尺度与分布形状，[VIScore](https://arxiv.org/abs/2608.11174) 进一步联合诊断编码器、预测器与规划器。这些工作关注表征或规划质量，却不直接识别固定 $(Q,A)$、只改变历史时的条件对应。第二类方法强化当前状态或动作后果；[Delta-JEPA](https://arxiv.org/abs/2606.31232) 从相邻隐变量位移解码动作，[PhyLatent](https://arxiv.org/abs/2608.05720) 通过物理状态约束和反事实动作分支改善动力学相关表征。本文固定查询动作，研究的是**同一动作在不同回合动力学历史下为何应产生不同未来**。第三类工作直接从交互历史做系统辨识；[ICWM](https://arxiv.org/abs/2606.26025) 将短历史视为无需参数更新的系统上下文。本文不把“使用历史”本身作为贡献，而是研究 reconstruction-free JEPA 中该条件关系为何在常规目标下不可识别，并给出不新增模型模块的训练目标与受控评测。

全文术语在首次出现时给出中英对照，其后以中文为主。

---

## 2. 问题设定与可识别性

### 2.1 记号

- $H$：动作—观测历史；
- $Q$：当前观测；
- $A$：查询动作序列；
- $M$：回合内固定、但不直接提供给模型的隐藏动力学因素；
- $O^+$：执行查询动作后的未来观测；
- $z^+ = \mathrm{Enc}(O^+)$ 与 $\hat z^+$：目标表征与预测表征。

研究对象是条件分布

$$
p(O^+ \mid H, Q, A),
$$

而不是表征的边缘分布 $p(Z)$。在 ContextWorld 的配对历史（matched-history）评测中，$Q$ 与 $A$ 完全相同，只有历史所揭示的 $M$ 不同。一个真正完成回合级系统辨识的模型应满足

$$
p(O^+ \mid H_0, Q, A) \neq p(O^+ \mid H_1, Q, A).
$$

若模型对两段历史输出同一条件均值，它仍可能具有健康的表征方差和较低的平均训练损失。

### 2.2 条件响应的连续度量

令配对条件下的真实响应与预测响应为

$$
r = z_1^+ - z_0^+, \qquad \hat r = \hat z_1^+ - \hat z_0^+ .
$$

我们报告三个连续量：增益（gain）、对齐度（alignment）与归一化响应误差（normalized response error，NRE）

$$
\mathrm{gain} = \frac{\langle \hat r, r\rangle}{\lVert r\rVert^2}, \qquad
\mathrm{alignment} = \frac{\langle \hat r, r\rangle}{\lVert \hat r\rVert\,\lVert r\rVert}, \qquad
\mathrm{NRE} = \frac{\lVert \hat r - r\rVert^2}{\lVert r\rVert^2}.
$$

$\mathrm{NRE} = 1$ 对应完全不随历史变化的预测响应，是一个有物理含义的参照点。ContextWorld 还报告四个离散指标：未来匹配率（future）、历史匹配率（history）、历史切换一致率（switch）和最差条件准确率（worst-condition）。它们度量分配是否正确，不能替代上述连续量：分配准确率可以在 NRE 很差时虚高。

### 2.3 三个互不替代的目标量

- **直接条件响应**：模型是否随历史改变未来预测（gain / alignment / NRE 与离散分配）。
- **隐藏动力学规划**：在含隐藏因素的环境中，正确历史是否让规划器选出更合适的动作；同一 checkpoint 还需比较正确历史与交换历史（correct vs. swapped）。
- **原任务保持（retention）**：在无隐藏因素的标准环境中，训练后的 checkpoint 是否保留原有规划能力。

标准 PushT 的 CEM 成功率只回答第三项。本文对每个结论明确指出它属于哪一个目标量。

### 2.4 无重叠时的不可识别性

**命题 1（缺少跨动力学重叠时的不可识别性）.** 设可见数据为 $(H,Q,A,O^+)$，条件机制为核 $K(\cdot \mid h,q,a)=p(O^+\mid H=h,Q=q,A=a)$。令 $m(H)$ 表示历史所揭示的动力学类别；它只用于表述数据生成过程，不要求作为训练标签。记 $\mathcal{S}=\operatorname{supp}P_{H,Q,A}$。若对 $P_{Q,A}$-几乎所有 $(q,a)$，$\mathcal{S}$ 中与该 $(q,a)$ 共现的历史只来自一个动力学类别 $m(q,a)$，则对任何未与该 $(q,a)$ 共现的类别 $m'\neq m(q,a)$ 及其历史 $h'$，反事实条件响应

$$
\Delta(h',h_0;q,a)=\mathbb{E}[O^+\mid H=h',Q=q,A=a]-\mathbb{E}[O^+\mid H=h_0,Q=q,A=a]
$$

在非参数模型类中不可识别，其中 $h_0$ 是任一与 $(q,a)$ 共现、且满足 $m(h_0)=m(q,a)$ 的历史。具体而言，存在两个核 $K\neq K'$，它们诱导完全相同的可见数据分布，而对应的 $\Delta$ 与 $\Delta'$ 可以任意不同。

**证明要点（proof sketch）.** 可见数据的联合分布由 $P_{H,Q,A}$ 与核在 $\mathcal{S}$ 上的限制唯一确定：

$$
P(h,q,a,\mathrm{d}o) = P_{H,Q,A}(h,q,a)\,K(\mathrm{d}o \mid h,q,a),
$$

而 $P_{H,Q,A}$ 在 $\mathcal{S}$ 之外取零测度。取 $K'=K$ 于 $\mathcal{S}$ 上，并在缺失类别对应的 $(h',q,a)$ 处任意重定义，则 $K$ 与 $K'$ 诱导同一个可见分布，因而任何仅为可见分布泛函的估计量、目标函数或正则项在二者下取值相同。按假设 $(h',q,a)\notin\mathcal{S}$，故 $\Delta$ 依赖于未被数据约束的核取值，可以任意改变。$\square$

**命题 1 不主张什么.** (i) 它不主张精确配对（exact pair）是必要条件：条件重叠只是使 $\Delta$ 成为可见分布泛函的一个充分条件，近邻/条件核平滑、参数化动力学、低秩或其他结构假设同样可以恢复可识别性；(ii) 它不主张 COJA 是唯一或最优的估计器，只说明有效训练信号必须来自上述某一类假设，而不可能由边缘正则凭空产生。

**推论.** 若对一个正测度的 $(q,a)$ 集合，来自至少两个动力学类别的历史 $h_0,h_1$ 与之共现，则相应的在支撑集内条件响应 $\Delta(h_1,h_0;q,a)$ 是可见分布的泛函，可以由观测数据估计。COJA 是利用这一信息的一种具体目标函数。

---

## 3. 失败诊断

### 3.1 边缘正则的置换不变性

考虑任一只依赖目标表征经验集合的正则项 $R(\{z_i^+\}_{i=1}^{n})$。对任意样本置换 $\pi$，

$$
R(\{z_i^+\}) = R(\{z_{\pi(i)}^+\}).
$$

因此，只要表征集合不变，该正则无法区分“哪段历史应对应哪个未来”。它可以防止全局坍缩、调整尺度或改善协方差，却不能单独保证

$$
I(\hat Z^+; M \mid Q, A) > 0 .
$$

这并不意味着条件 MSE 在理论上学不到条件期望；实际困难在于，当 Encoder、Projector 与 Predictor 联合优化时，预测条件均值、移动目标几何或利用与任务无关的表征方差，通常比学习微弱的历史条件响应更容易。

### 3.2 ContextWorld 上的受控反例

| 设置 | 任务与预算 | 结果 |
|---|---|---|
| 原生 LeWM | ActionDelay，1,024 步 | macro 约 $0.333$，接近三分类随机，几乎无历史响应 |
| VISReg | ActionDelay，1,024 步 | macro $0.3316$，worst group $0$，仅 $8/960$ 个查询对历史敏感，同时 $0/2{,}880$ 个目标对坍缩 |
| stop-gradient + SIGReg | ActionDelay | 全部目标对非重合，但 $959/960$ 个查询对不同历史作出相同选择 |
| PLDM | ActionDelay，1,024 步 | macro $0.8257$，说明任务信号与模型容量并非根本不足 |

多个 checkpoint 的历史信息可被 probe 读出，却仍没有方向正确的预测响应。结论是：历史可读、目标可分、边缘非坍缩都是必要信息，但都不等于 Predictor 建立了正确的“历史—动作—未来”对应。该对照不否定 VISReg 或 PLDM 在各自原始目标上的价值。

这里的 VISReg 结果是 1,024 步、匹配预算的机制筛查，不是其完整公开 world-model 配方的最终复现。论文级对照还需在同一冻结数据版本上补齐完整预算的 VIS-WM 原生基线及其 `+COJA` 分支（§8.2）。

### 3.3 六层诊断

| 层面 | 问题 | 主要测量 | 结论 |
|---|---|---|---|
| 数据可辨识性 | 固定 $(Q,A)$ 后是否存在跨动力学的历史变化 | 精确/近似条件重叠 | 缺少条件重叠的回放不能非参数识别该反事实；当前正例利用重叠（命题 1） |
| 表征可用性 | 历史是否保留隐藏动力学信息 | 查询/场景留出 probe | 部分 checkpoint 可读，但可读性不保证被使用 |
| 条件耦合 | Predictor 是否真正使用历史 | switch、Jacobian、模块互换 | 主要断点在 Predictor trunk，而非输出投影 |
| 响应校准 | 变化方向与幅值是否正确 | gain、alignment、NRE | 分配准确率可在 NRE 很差时虚高 |
| 滚动一致性 | 单步历史响应能否在自回归后保持 | 1/2/3 步隐藏规划、正确/交换历史 | 单步 COJA 不充分；真实自回归 MSE 可修复两步并迁移到更长视界 |
| 泛化与保持 | 响应能否跨查询泛化、原任务是否保留 | 训练—开发集差距、同数据标准 CEM | 查询覆盖显著影响 NRE；保持性须与同数据原生对照比较 |

### 3.4 查询模板不足导致响应记忆

早期 Motion Damping 训练在 8,192 步中反复使用 2,048 个查询模板：

| checkpoint | 训练 NRE | 开发集 NRE | 差距 |
|---:|---:|---:|---:|
| 2,048 步 | $0.756$ | $0.901$ | $+0.144$ |
| 8,192 步 | $0.207$ | $1.120$ | $+0.913$ |

训练继续进行时，样本内响应几乎被精确拟合，留出响应反而变差；模块互换显示效果几乎全部随 Predictor trunk 转移，`pred_proj` 影响很小，且变化分布在六个 Transformer block 中。这支持“查询特异的响应记忆”解释，而非“训练不足”。使用完整 8,192 个配对查询训练后，NRE 与分配指标同时显著改善，进一步支持覆盖不足是该失败的重要来源。

### 3.5 单步条件可辨识性不保证自回归保持

规划器消费的不是孤立的一步预测，而是把模型自身输出重新放回历史：

$$
\hat z_{t+1} = F_\theta(H_t, A_t), \qquad \hat z_{t+k} = F_\theta(\hat H_{t+k-1}, A_{t+k-1}).
$$

于是多步响应包含一串在模型自生成状态上取值的 Jacobian。下面只写一阶马尔可夫近似；实际滑动历史还会产生经过更早预测状态的附加路径：

$$
\frac{\partial \hat z_{t+k}}{\partial H_t} \approx \left(\prod_{j=k-1}^{1} \frac{\partial F_\theta}{\partial \hat z_{t+j}}\right) \frac{\partial F_\theta}{\partial H_t},
$$

其中矩阵乘积按 $j$ 递减的顺序左乘。

一步条件目标只约束最右侧一项，后续 Jacobian 可以缩小、放大或旋转历史效应。Motion 的一步 COJA checkpoint 正是这样的反例：单步 switch $= 0.938$，但两步隐藏规划在正确历史下物理误差约 $103$ px，交换历史并不更差。故断点不是“模型完全不读历史”，而是训练分布只校准了真实历史上的一步映射，而部署要求模型在自己生成的历史上保持同一条件动力学解释。

---

## 4. 方法

### 4.1 数据条件

COJA 消费如下形式的样本组：

$$
(H_0, Q, A, O_0^+), \qquad (H_1, Q, A, O_1^+),
$$

两条样本共享可见的 $Q,A$，历史与真实未来不同。COJA 只要求训练管线提供这种分组关系。当前公开云端适配器为保证数据完整性，读取发布资产中的 `public_pair_identity_v1` 并保持对应片段同批；独立零步审计表明，在 Motion 的完整回放资产中，同一分组也可仅由模型可见的查询 RGB 与原始动作字节恢复。由此：

- 隐藏动力学标签不进入模型，也不作为损失目标；
- 分组标识只供采样器和辅助损失索引样本，在模型边界前被剥离；
- 人工的 low/high 命名不是方法所需的模型输入；
- 推理时不需要任何配对；
- 但训练数据仍必须具有条件重叠。

最后一点是当前实现选择的数据假设。命题 1 更一般地说明：条件重叠与额外结构信息至少需要具备其一。对同一 $(Q,A)$ 几乎不重复的普通离线回放，需要主动采集重叠、使用可靠的近邻/条件核，或引入参数化动力学等结构假设。

### 4.2 二元 COJA 目标

设一个宽度为 2 的组。实现先展平特征维，再选取组内各样本的最后一个时间步，并以 FP32 计算辅助项，得到预测 $p_0,p_1\in\mathbb{R}^{d}$ 与目标 $t_0,t_1\in\mathbb{R}^{d}$；目标在该辅助分支内被 detach，记为 $\mathrm{sg}[\cdot]$。

**去中心。** 以组内均值为中心：

$$
\bar p = \tfrac12 (p_0 + p_1), \qquad \bar t = \tfrac12 (t_0 + t_1),
$$

$$
\tilde p_i = p_i - \bar p, \qquad \tilde t_i = t_i - \bar t .
$$

**目标能量与硬性前置条件。** 每组的目标能量按组内成员与特征维一起取均值：

$$
E = \frac{1}{2d} \sum_{i=0}^{1} \lVert \tilde t_i \rVert^2 .
$$

实现要求 $E > \tau$，$\tau = 10^{-8}$（`MINIMUM_TARGET_ENERGY`）；任一组不满足即抛出异常。随后 $E$ **直接作为分母**，不加平滑常数。若记 $u=t_1-t_0$，则二元组恒有 $E=\lVert u\rVert^2/(4d)$，因此该硬门也保证后续响应轴分母严格非零。

**响应项（response）。** 逐组计算并对组取均值：

$$
L_{\mathrm{resp}} = \mathbb{E}_{g}\left[ \frac{\dfrac{1}{2d} \sum_{i=0}^{1} \lVert \tilde p_i - \tilde t_i \rVert^2}{E_g} \right].
$$

对二元组记 $u=t_1-t_0$、$\hat u=p_1-p_0$，则逐组响应项可以化简为

$$
L_{\mathrm{resp},g}=\frac{\lVert \hat u-u\rVert^2}{\lVert u\rVert^2},
$$

即 §2.2 的逐配对 NRE。它直接训练同一 $(Q,A)$ 下由历史引起的相对未来响应；绝对未来主要由基线方法原生的逐样本预测损失负责。需要注意，去中心只适用于 $L_{\mathrm{resp}}$：后续势垒仍通过 $\beta$ 约束组中心误差在真实响应轴上的投影，只对该轴的正交补方向保持中心无关。

**沿真实响应轴的增益与中心偏移。** 令真实响应轴与预测响应轴

$$
u = t_1 - t_0, \qquad \hat u = p_1 - p_0, \qquad \lVert u \rVert^2 = \sum_{\ell=1}^{d} u_\ell^2 ,
$$

其中 $\lVert u \rVert^2$ 按特征维求和（而非取均值）。定义增益 $\alpha$（alpha）与中心偏移 $\beta$（beta）：

$$
\alpha = \frac{\langle \hat u, u \rangle}{\lVert u \rVert^2}, \qquad \beta = \frac{\langle \bar p - \bar t, \; u \rangle}{\lVert u \rVert^2}.
$$

**分配裕度（margin）与势垒。** 两个正确分配对应的裕度为

$$
m_0 = \tfrac{\alpha}{2} - \beta, \qquad m_1 = \tfrac{\alpha}{2} + \beta ,
$$

规范裕度取 $\tfrac12$（`CANONICAL_BINARY_MARGIN`），势垒为逐组、逐裕度的平方缺口均值：

$$
L_{\mathrm{assign}} = \mathbb{E}_{g}\left[ \tfrac12 \sum_{i=0}^{1} \big( \max(0, \tfrac12 - m_i) \big)^2 \right].
$$

**总辅助损失。**

$$
L_{\mathrm{COJA}} = L_{\mathrm{resp}} + L_{\mathrm{assign}} .
$$

**真实目标处静止。** 当 $p_i = t_i$ 时，$\tilde p_i = \tilde t_i$ 故 $L_{\mathrm{resp}} = 0$；同时 $\alpha = 1$、$\beta = 0$，于是 $m_0 = m_1 = \tfrac12$，两个缺口均为 $0$，$L_{\mathrm{assign}} = 0$。两项在该点的梯度也为零：响应项是平方误差取到极小值，单个缺口平方项对 $m_i$ 的导数 $-2\max(0, \tfrac12 - m_i)$ 在 $m_i = \tfrac12$ 处为零；对组和裕度取均值只会再乘常数。与普通排序或对比损失不同，该势垒不会持续把两个预测推得比真实未来更远。

**命题 2（二元目标识别的量）.** 对任一满足 $E>\tau$ 的二元组，$L_{\mathrm{resp}}=0$ 当且仅当 $p_1-p_0=t_1-t_0$。在此条件下，$L_{\mathrm{assign}}=0$ 当且仅当 $\beta=0$。因此，COJA 精确识别历史引起的响应向量，并消除组中心沿真实响应轴的偏移；与响应轴正交的公共中心仍由基线方法的原生预测损失确定。交换组内下标 $0,1$ 只会交换 $m_0,m_1$，不会改变总损失，所以该目标不依赖人为规定的条件方向。

**证明.** 响应项是以正数 $E$ 归一化的平方误差，取零等价于 $\tilde p_i=\tilde t_i$，也就等价于两者的组内差相同。此时 $\alpha=1$，两个势垒同时为零要求 $\tfrac12-\beta\ge\tfrac12$ 且 $\tfrac12+\beta\ge\tfrac12$，即 $\beta=0$。组内翻转使 $u$ 与 $\hat u$ 同时变号、$\alpha$ 不变、$\beta$ 变号，因此两个裕度互换。$\square$

**三元对称分配（triplet）。** 宽度为 3 的组（如 ActionDelay）使用对称分配式。令

$$
D_{ij} = \frac{1}{d}\lVert p_i - t_j \rVert^2, \qquad
s = \frac{1}{6}\sum_{i \neq j} \frac{1}{d}\lVert t_i - t_j \rVert^2, \qquad
N_{ij} = \frac{D_{ij}}{\max(s, \tau)} ,
$$

注意三元分支对目标尺度使用下截断 $\max(s,\tau)$，而二元分支使用前述硬性前置条件。以 $-N$ 为 logits 的双向交叉熵为

$$
\ell_{p \to t} = -\frac{1}{3}\sum_{i=0}^{2} \log \frac{\exp(-N_{ii})}{\sum_{j} \exp(-N_{ij})}, \qquad
\ell_{t \to p} = -\frac{1}{3}\sum_{j=0}^{2} \log \frac{\exp(-N_{jj})}{\sum_{i} \exp(-N_{ij})},
$$

$$
L_{\mathrm{COJA}}^{\mathrm{triplet}} = \mathbb{E}_g\left[\tfrac12\left(\ell_{p \to t} + \ell_{t \to p}\right)\right].
$$

诊断量 `matched_distance` $=\frac13\sum_i D_{ii}$ 与 `counterfactual_distance` $=\frac16\sum_{i\neq j}D_{ij}$ 只用于记录；代码中的分配裕度定义为后者减前者，不参与优化。该三元交叉熵分支继承自离散 ActionDelay 的 PCJA 正例；与二元势垒不同，它在 $p_i=t_i$ 时通常仍有有限梯度。因此，“真实目标处静止”和命题 2 只适用于当前连续动力学使用的二元 COJA 目标。

### 4.3 总目标、跨族路由与部署边界

统一写法为：所选基线方法的原生目标加一个固定权重的条件项，

$$
L = L_{\text{base}} + \lambda_{\mathrm{COJA}} \, L_{\mathrm{COJA}} .
$$

LeWM 是其特例：$L_{\text{base}} = L_{\mathrm{MSE}} + 0.09\, L_{\mathrm{SIGReg}}$，当前配置取 $\lambda_{\mathrm{COJA}} = 0.09$。VIS-WM 则以 VISReg 目标定义自己的 $L_{\text{base}}$。两者共享相同的 LeWM 骨干与通用训练循环，但分别使用独立的训练入口、配置和运行身份；LeWM 配置中不包含 VISReg 选项。

辅助分支是**仅预测器路径路由**的：它关闭 Predictor 路径中的 dropout 等随机层，以 stop-gradient 后的历史嵌入与动作嵌入执行一次额外前向，再与 detach 后的目标计算条件项。因此该项只更新各模型族已有的 Predictor 路径；在带 prediction projection 的模型族中，该投影也随之更新，而 Encoder、目标投影器和动作编码器不接收该项梯度。实现测试分别验证了各族的梯度边界。代价是训练期多一次 Predictor 前向与反向；新增可学习参数、模块与推理计算均为零。训练结束后保存的仍是所在模型族的标准 state dict，规划器与推理调用保持不变。

当前跨方法接口已在 LeWM、VIS-WM、PLDM、DINO-WM（仓库训练入口名为 PreJEPA）四条训练路径上实现并有单元测试覆盖（默认关闭，默认权重 $0.09$，组宽 $2$）。但本文的效果证据主要来自 LeWM；VIS-WM、PLDM 与 DINO-WM 目前只有接口与梯度路由层面的验证，没有对应的完整任务级结果。统一云入口 `coja_v1` 当前只开放 Contact Friction 一个任务。

当前公开 Contact Friction 配置的每个优化步使用 64 条原始环境数据与 64 条 ContextWorld 数据，后者组成 32 个完整配对组。该 $50/50$ 指样本曝光比例，不是 COJA 的理论要求，也不表示各损失项的数值贡献相等。

### 4.4 RC-COJA：从一步条件对应到滚动一致性

COJA 解决“同一查询在不同历史下应对应哪个**一步**未来”，但不约束模型在自生成历史上的后续映射（§3.5）。RC-COJA 不增加网络或推理分支，只在训练时展开现有 Predictor。对隐藏行：

$$
\hat z_{t+1} = F_\theta(H_t, A_t), \qquad \hat z_{t+2} = F_\theta\big(\mathrm{shift}(H_t, \hat z_{t+1}), A_{t+1}\big),
$$

隐藏行的损失为

$$
L_{\mathrm{hidden}}^{\mathrm{RC}} = w_h \Big[ (1-\rho)\, L_{\mathrm{MSE}}^{(1)} + \rho\, L_{\mathrm{MSE}}^{(2,\mathrm{AR})} \Big] + \lambda_{\mathrm{COJA}} \, L_{\mathrm{COJA}}^{(1)} ,
$$

其中 $w_h$ 是隐藏样本在整批原生 MSE 中的总系数。当前 Motion/Contact LeWM 实验采用 64+64 的等量混合，故 $w_h=0.5$；它不是 RC-COJA 额外引入的可调超参数。$\rho$ 只在两个视界之间重新分配这份既有监督，不叠加新的损失。固定配置取 $\rho = 0.25$，即一步/两步系数为 $w_h(1-\rho) = 0.375$ 与 $w_h\rho = 0.125$，$\lambda_{\mathrm{COJA}} = 0.09$ 保持不变。梯度不在 $\hat z_{t+1}$ 处截断，因此第二步误差同时校准第一步输出与 Predictor 在自生成状态上的响应。第二步只使用原生 MSE，不叠加第二步条件项。RC-COJA 当前仍由 LeWM 研究训练器实现，尚未纳入三模型族共享的 `coja_v1` 云端入口。

两者的分工必须区分清楚：**COJA 约束一步条件对应**（同一 $(Q,A)$、不同历史 → 不同一步未来），**RC-COJA 约束滚动一致性**（该对应在模型自身生成的状态上继续成立）。RC-COJA 需要真实的轨迹续段目标，并保留 COJA 的条件重叠假设；其简洁性来自“不改变模型”，而非“无需组织数据”。

---

## 5. 实验协议

### 5.1 对照臂

为把数据配方效应与方法效应分开，规划评测使用三个模型分支：

- **公开参考**：公开原始数据训练的 checkpoint；
- **同数据原生对照**：相同混合数据、预算与训练流程，但不启用新方法；
- **方法分支**：在同数据原生对照基础上启用 COJA 或 RC-COJA。

据此分别报告数据效应（原生 − 公开参考）、方法效应（方法 − 原生）与部署总效应（方法 − 公开参考）。隐藏动力学规划还在同一 checkpoint 上比较正确历史与交换历史，这是判断收益是否真正来自历史使用的直接干预。

### 5.2 评测

- **直接条件响应**：配对查询上的 gain / alignment / NRE 与 future、history、switch、worst 准确率，训练集与开发集分别报告。
- **隐藏动力学规划**：ContextWorld 环境中的 CEM，动作在真实隐藏参数模拟器中执行，报告物理误差与遗憾值（regret），并按视界（h1/h2/h3/h5）分列。Contact 使用一维动作尺度搜索的 $64$ 个样本 $\times\ 6$ 次迭代的筛查式 CEM；物理 oracle 在 $244/256$ 个查询上判定 low/high 的可接受动作区间不重叠，平均最优尺度差为 $0.358$，确认该评测确实需要识别隐藏条件。Motion 的物理 oracle 在 h1/h2/h3 分别保留 $34/156/134$ 个可辨识配对，因此只在同一视界内作配对比较，不跨视界平均原始距离。
- **原任务保持**：标准 PushT 的 CEM 成功率，各分支使用完全相同的查询目录。

### 5.3 统计呈现

1. **任务内效应**：对配对查询报告候选减对照的点估计与 $50\%/80\%/95\%$ 配对自助（paired bootstrap）区间。
2. **训练随机种子**：作为独立层级处理，不把多个种子的查询简单拼成一个更大的样本集。
3. **任务汇总**：每个任务权重相等，先在任务内重采样查询与评测种子，再对任务级效应取均值与中位数，报告分布与最差任务尾部。
4. **能力分轴**：分配、NRE 与 CEM 分别呈现，汇总提升不掩盖单任务 $\mathrm{NRE} > 1$。
5. **实用差异参考**：同时报告 $P(\delta \ge 0)$、$P(\delta \ge -2\text{pp})$、$P(\delta \ge -5\text{pp})$ 的自助重采样频率，并明确它们不是贝叶斯后验概率。
6. **规划三效应**：数据效应、方法效应、总效应分别报告；隐藏规划必须附同 checkpoint 的正确/交换历史对比。

本文不使用单一比例阈值来裁决方法；所有比较均以点估计加区间呈现。

隐藏规划中的“正确历史收益”定义为同一模型在交换历史与正确历史下的误差差；表中的双重差分（difference in differences，DID）进一步减去匹配对照的同一差值：

$$
\mathrm{DID}=\big(e_{\mathrm{swap}}-e_{\mathrm{correct}}\big)_{\mathrm{method}}-\big(e_{\mathrm{swap}}-e_{\mathrm{correct}}\big)_{\mathrm{control}}.
$$

因此，正 DID 表示新方法使正确历史相对交换历史产生了更大的规划收益，而不只是整体移动了预测误差。

---

## 6. 结果

### 6.1 一步条件响应

| 任务与训练 | 离散条件能力（COJA / 对照） | 连续响应（COJA / 对照） | 标准环境保持 |
|---|---|---|---|
| ActionDelay，3 个训练种子 | macro $0.940/0.936/0.942$；worst $0.906/0.901/0.911$ | $960$ 类配对查询上对历史稳定响应 | 一枚训练种子的 900 个配对 CEM 查询：相对同数据对照 $-2.67$pp；另两枚未评测 |
| Contact Friction，4,096 步 | future $0.771/0.521$；switch $1.000/0.668$ | NRE $0.579/0.984$；gain $0.447/0.015$ | CEM100 $72/73$，差值 $-1$pp $[-9,+7]$pp |
| Portal Exit，4,096 步 | future $0.752/0.584$；worst $0.746/0.512$ | NRE $0.284/0.471$；gain $0.604/0.366$ | CEM50 $47/46$，样本量不足以主张提升 |
| Motion Damping，8,192 步完整查询集 | future $0.660/0.494$；history $0.668/0.500$；switch $0.969/0.441$；worst $0.570/0.230$ | NRE $0.767/1.130$；gain $0.370/0.0065$；alignment $0.520/0.017$ | CEM300 $213/232$，差值 $-6.33$pp $[-11.00,-1.67]$pp |

ActionDelay 使用 §4.2 的三元对称分配分支；三个数分别对应三个独立训练种子，而不是同一模型与三个对照的比值。其余连续动力学结果使用二元响应—势垒目标。表中的“真实目标处静止”性质只适用于后二者，不能从 ActionDelay 的交叉熵分支外推。

Motion 的 NRE 配对自助 $95\%$ 区间为 $[0.724, 0.814]$，完整位于零响应参照 $1.0$ 以下，说明连续条件响应可由原 LeWM 学到。Motion 的标准 CEM 下降在多个评测目录上方向一致（种子 $42/43/44$ 各 100 条：$70/79$、$66/75$、$76/82$，等权平均差 $-8.0$pp $[-12.67,-3.67]$pp；种子 42 的 300 条目录：$213/232$，$-6.33$pp，精确 McNemar $p = 0.0145$），它是真实的原任务保持问题，但不能据此推断该 checkpoint 在隐藏动力学环境中没有价值——这两项是不同的目标量。

#### 6.1.1 统一 joint-scratch 完整训练：作用取决于原生条件响应缺口

为排除短预算、初始化和训练数据带来的混淆，我们又通过统一云端入口完成了两个单训练种子的严格对照。每个任务的 native 与 COJA 共享 seed 3072、从头初始化、10 个 epoch、每卡 batch 128、8 卡 BF16、$50/50$ 原始/ContextWorld 数据、优化器和 18,034,478 个可训练参数；COJA 分支只启用关系保持采样与 $\lambda_{\mathrm{COJA}}=0.09$。结果均来自公开 Development 的 256 个配对查询。

| 任务与分支 | future ↑ | worst ↑ | joint pair ↑ | response gain ↑ | NRE ↓ |
|---|---:|---:|---:|---:|---:|
| Robot Arm Mass，native | $0.842$ | $0.793$ | $0.641$ | $0.989$ | $0.141$ |
| Robot Arm Mass，COJA | $0.861$ | $0.848$ | $0.656$ | $0.938$ | $0.233$ |
| Portal Exit，native | $0.508$ | $0.441$ | $0.016$ | $0.043$ | $0.917$ |
| Portal Exit，COJA | $0.717$ | $0.660$ | $0.324$ | $0.605$ | $0.329$ |

Robot Arm Mass 的完整 native 已经形成校准良好的条件响应。配对 query bootstrap 给出的 COJA−native future 差异为 $+1.95$pp（95% 区间 $[-0.39,+4.30]$pp），最弱条件为 $+5.08$pp（$[+0.39,+9.77]$pp）；与此同时，gain 下降 $0.051$（$[-0.085,-0.017]$），NRE 增加 $0.092$（$[+0.064,+0.120]$），switch 下降 $4.30$pp（$[-7.42,-1.17]$pp）。因此早期 1,024-step native 失败不能被用作“COJA 救活该任务”的证据。这个对照表明训练预算本身可以解决部分能力，COJA 不是无条件增益项；在该任务上，它带来最弱条件分配改善与响应校准退化之间的权衡。

Portal Exit 给出相反且更有辨识力的结果。两臂的 switch 都为 $1.0$，但 native 的 gain 只有 $0.043$：它会随历史改变输出，却几乎没有形成真实条件响应的幅值。COJA 将 future、worst 和 joint-pair success 分别提高 $20.90$pp、$21.88$pp 和 $30.86$pp；相同 256 个 query pair 的配对 bootstrap 95% 区间分别为 $[+17.77,+24.02]$pp、$[+13.67,+30.08]$pp 和 $[+25.00,+36.72]$pp。gain 的提升为 $+0.563$（$[+0.503,+0.612]$），NRE 的下降为 $0.589$（$[0.529,0.637]$）。response alignment 单独下降 $0.042$，但 native 的预测响应幅值接近零，因此高余弦值只表示一个几乎静止的向量方向大致正确，不能替代 gain 与 NRE。上述结果支持本文的机制判断：COJA 的主要作用不是让预测“发生任意切换”，而是把同一 $(Q,A)$ 下的历史差异对齐到真实 future response。该 COJA checkpoint 在标准、无隐藏出口变化的 TwoRoom CEM 上为 $300/300$；这是原任务绝对保持证据，不是隐藏 Portal ICL 的代理指标。

以上仍是单训练种子 Development 结果，不承担 Public Test 或跨种子稳定性主张。完整 checkpoint、结果哈希与机器可读指标见[统一完整训练摘要](artifacts/contextworld_joint_scratch_full_single_seed_v1/summary.json)。

### 6.1.2 Contact Friction：seed-3072 full-10-epoch 终点的 Development/Public 报告

最新 Contact Friction endpoint 是从头训练的 LeWM+COJA checkpoint：training seed=`3072`、完整
`10` epoch、checkpoint `weights_epoch_10.pt`。checkpoint SHA256 为
`d05005ba69771a0957e30d03d1ae6f4ddb14af7806afc0c29e1de4f8c43feeda`，StableWM commit 为
`0b6673f9bf0133f713df6303925ea8355b1ded4b`。Development split 用于方法开发与配方选择；
训练终点按既定的完整 10-epoch 配方产生。Public Test split 只用于最终报告，不反馈到调参或选择。

| metric (higher is better unless marked) | Development (tuning/selection) | Public Test (final reporting only) |
|---|---:|---:|
| future | $0.908203$ | $0.910156$ |
| history | $0.888672$ | $0.902344$ |
| switch | $0.996094$ | $1.000000$ |
| low | $0.890625$ | $0.886719$ |
| high | $0.925781$ | $0.933594$ |
| worst | $0.890625$ | $0.886719$ |
| gain | $0.907026$ | $0.904905$ |
| NRE $\downarrow$ | $0.280805$ | $0.297407$ |
| alignment | $0.866844$ | $0.859976$ |
| calibrated-response success | $0.898438$ | $0.937500$ |
| joint-pair success | $0.675781$ | $0.699219$ |

`low`/`high` 是 low/high-friction 的 future rate；calibrated-response success 使用零历史响应
基线，即 `NRE < 1`. Development 与 Public 的离散指标和连续 response geometry 保持同一量级：
future/history/worst 分别为 `0.908203/0.888672/0.890625` 与
`0.910156/0.902344/0.886719`，gain/NRE 分别为 `0.907026/0.280805` 与
`0.904905/0.297407`。这是 split-level stability，不是 training-seed stability。

既有 frozen hard gate 仍要求 future/history/switch/worst 至少为 `0.95/0.95/0.95/0.90`，并
要求 gain 至少为 `0.50`。因此当前 checkpoint 在两侧都通过 switch、gain 与 NRE，但两侧都未
通过 future、history 与 worst；Public Test 的 gate 仍为 failed。Public Test independent rescore
对 future/history/switch/worst/joint-pair success 给出完全相同的聚合值。这个 endpoint 是单 seed
的 descriptive result，不是 formal three-seed method claim。标准 PushT CEM 若在其他实验中出现，
只衡量原任务 retention，不是 ICL test。

身份与上述聚合指标的最小回执见
[contact_friction_lewm_coja_seed3072_full10_summary.json](artifacts/contact_friction_lewm_coja_seed3072_full10_summary.json)。

### 6.2 Contact：一步条件响应转化为隐藏动力学规划收益

| 分支 | 正确历史物理误差 ↓ | 正确历史 scale regret ↓ | 交换历史带来的物理损失 ↑ | 交换历史带来的 regret ↑ |
|---|---:|---:|---:|---:|
| 公开参考 | $87.08$ | $0.3386$ | $-0.37$ | $-0.0023$ |
| 同数据原生对照 | $94.72$ | $0.4102$ | $+0.28$ | $+0.0001$ |
| **COJA** | $89.79$ | $0.3783$ | $+1.70\ [0.80, 2.68]$ | $+0.0121\ [0.0059, 0.0192]$ |

相对同数据原生对照，COJA 的方法效应为物理误差改善 $4.94$ px $[2.31, 7.58]$、scale regret 改善 $0.0319\ [0.0121, 0.0519]$；其正确历史收益相对原生对照又分别增加 $1.41$ px $[0.33, 2.54]$ 与 $0.0120\ [0.0053, 0.0198]$。公开参考的绝对误差仍低于 COJA：三臂分解显示 $50/50$ 混合数据与原生续训相对公开参考带来 $7.64$ px 与 $0.0716$ regret 的不利数据效应，COJA 追回其中一部分但尚未全部追回。该评测使用筛查式一维 CEM（§5.2），支持估计量方向与 checkpoint 排序，不应被读作最终部署成功率。

### 6.3 Motion：单阶段 RC-COJA 与双种子复现

三个分支共享公开 PushT 初始化、$50/50$ 数据、optimizer、4,096 步预算与评测查询；保存模型仍是原 LeWM state dict。

| 视界 | 同数据无辅助项 ↓ | 一步 COJA ↓ | RC-COJA ↓ | RC − COJA 改善 | 正确历史收益 DID |
|---|---:|---:|---:|---:|---:|
| h1 | $23.28$ | $23.23$ | $26.24$ | $-3.01\ [-4.13, -1.86]$ | $-0.24\ [-0.46, -0.03]$ |
| h2（训练） | $100.20$ | $103.17$ | $45.78$ | $+57.39\ [52.19, 62.42]$ | $+4.12\ [2.90, 5.34]$ |
| h3（未训练） | $106.23$ | $108.34$ | $69.26$ | $+39.08\ [33.90, 44.21]$ | $+2.28\ [0.28, 4.01]$ |

RC 分支的直接响应为 future/history/switch/worst $= 0.553/0.617/0.938/0.258$。h2/h3 的绝对误差与历史干预效应同时改善，说明收益不是靠忽略历史取得；h1 保留一个明确的短期代价。

独立训练种子 $14322$ 在完全相同的 31 点动作网格上给出近乎逐值复现：

| 视界 | 种子 14321 RC − COJA | 种子 14322 RC − COJA | 种子 14322 历史收益 DID |
|---|---:|---:|---:|
| h1 | $-3.01\ [-4.13, -1.86]$ | $-3.30\ [-4.53, -2.14]$ | $-0.06\ [-0.28, +0.21]$ |
| h2（训练） | $+57.39\ [52.19, 62.42]$ | $+57.60\ [52.55, 62.55]$ | $+4.24\ [3.16, 5.37]$ |
| h3（未训练） | $+39.08\ [33.90, 44.21]$ | $+41.56\ [36.02, 47.02]$ | $+2.91\ [0.90, 4.95]$ |

标准（无隐藏阻尼）PushT 的 300 查询保持性：同数据无辅助项 $203/300$、一步 COJA $188/300$、RC-COJA $194/300$；RC − COJA $= +2.00$pp $[-2.67, +6.67]$pp，RC − 无辅助项 $= -3.00$pp $[-8.67, +2.67]$pp。种子 $14322$ 的 RC/COJA 为 $192/196$，即 $-1.33$pp $[-6.67, +4.00]$pp。按训练种子分层、再在种子内重采样 300 个配对查询，两种子层级均值为 $+0.33$pp，$50\%/80\%/95\%$ 区间分别为 $[-1.17,+1.83]$、$[-2.50,+3.17]$、$[-4.00,+4.50]$pp。合理结论是：没有检测到 RC 特异的原任务保持损伤，也不能据此宣称提升。

### 6.4 Contact：跨任务单阶段复现

固定 Motion 的 $\rho = 0.25$，不在 Contact 上重新调参；三臂共享初始化、$50/50$ 混合、预算与评测查询。

| 视界 | 公开参考 ↓ | 一步 COJA ↓ | RC-COJA ↓ | RC − COJA 改善 | 正确历史收益 DID |
|---|---:|---:|---:|---:|---:|
| h2（训练） | $25.99$ | $28.45$ | $25.29$ | $+3.16\ [1.58, 4.79]$ | $+1.89\ [1.28, 2.52]$ |
| h5（未训练） | $94.73$ | $89.82$ | $86.72$ | $+3.09\ [0.65, 5.46]$ | $+3.61\ [1.87, 5.40]$ |

一步校准没有可辨认变化：RC 的 future/history/switch/worst $= 0.775/0.844/1.000/0.738$，一步 COJA 为 $0.771/0.850/1.000/0.734$；NRE $= 0.619/0.617$。标准 PushT（同 300 查询）为公开参考 $237$、一步 COJA $206$、RC $207$；RC − COJA $= +0.33$pp $[-4.33, +5.00]$pp，不一致对为 $25/24$。RC − 公开参考 $= -10.00$pp $[-15.00, -5.00]$pp，而一步 COJA − 公开参考已经是 $-10.33$pp $[-15.33, -5.67]$pp：这约 $10$pp 的差距属于共享的混合数据与适配路径，不是 RC 的增量效应。

独立训练种子 $13314$ 只改变随机种子与输出目录：

| 训练种子 | direct future RC−COJA | direct NRE RC−COJA | h2 物理改善 | h5（未训练）物理改善 | 标准 CEM300 RC−COJA |
|---:|---:|---:|---:|---:|---:|
| $13313$ | $+0.20$pp | $+0.00046$ | $+3.16\ [1.58, 4.79]$ px | $+3.09\ [0.65, 5.46]$ px | $+0.33\ [-4.33, +5.00]$pp |
| $13314$ | $-0.78$pp | $-0.00027$ | $+3.67\ [2.06, 5.34]$ px | $+4.37\ [1.76, 6.92]$ px | $+2.00\ [-2.33, +6.33]$pp |

种子 $13314$ 的 h2/h5 正确历史收益 DID 为 $+2.14\ [1.53, 2.78]$ 与 $+3.73\ [1.87, 5.66]$ px，h1 方法效应为 $+0.08\ [-0.27, +0.42]$ px（未检测到短期损伤）。两种子的 direct future/history 平均变化为 $-0.29/-0.59$pp，NRE/gain 平均变化为 $+0.00010/+0.00230$，均远小于长程规划效应；标准 CEM 的训练种子分层自助均值为 $+1.17$pp $[-2.17, +4.50]$pp。

### 6.5 结果小结

- 一步条件响应：ActionDelay（三种子）、Contact、Portal、Motion 四个任务均为正。
- 统一完整训练对照：Portal 的 COJA 相对同预算 native 提高 future $20.90$pp（配对 query 95% 区间 $[+17.77,+24.02]$pp）、worst $21.88$pp（$[+13.67,+30.08]$pp），并把 NRE 从 $0.917$ 降到 $0.329$；Robot Arm Mass 的 native 本身已充分学习，COJA 只呈现较小的分配变化和校准权衡。
- 最新 Contact Friction seed-3072 full-10-epoch checkpoint 在 Development/Public 上保持相近的离散与连续指标，但两侧均错过 frozen gate 的 future/history/worst 主门；该单 seed 结果只作描述性报告。
- 隐藏动力学规划：COJA 在 Contact 上把直接条件能力转化为相对同数据原生对照的规划收益；RC-COJA 在 Motion 与 Contact 的训练视界与未训练更长视界上进一步显著改善，且正确历史收益的 DID 为正。
- 原任务保持：两任务、各两个训练种子均未检测到 RC 特异的退化；Motion 从无辅助项到一步 COJA 的下降与 Contact 相对公开参考的约 $10$pp 差距是共享训练路径的代价，与 RC 增量分离。
- 代价：Motion 上稳定可复现的 $3.01/3.30$ px 一步误差增加。

---

## 7. 消融实验

### 7.1 视界权重与活性成分（Motion，1,024 步续训对照）

所有分支共享同一一步 COJA 起点、$50/50$ 数据、批次采样顺序、优化器与额外步数，只改变视界权重分配：

| 续训目标 | h1 ↓ | h2 ↓ | h3 ↓ |
|---|---:|---:|---:|
| 一步训练对照 | $22.86$ | $102.37$ | $111.45$ |
| 仅第二步条件项（$0.5$） | $23.18$ | $86.17$ | — |
| 第二步原生 MSE（$\rho = 0.25$） | $25.90$ | $58.54$ | $80.53$ |
| 第二步原生 MSE（$\rho = 0.50$） | $29.56$ | $44.40$ | $71.21$ |
| 第二步原生 MSE + 条件项（$0.5/0.5$） | $32.67$ | $43.18$ | $69.69$ |

$\rho = 0.25$ 相对对照的 h1/h2/h3 效应为 $-3.04\ [-4.21,-1.81]$、$+43.83\ [39.57,48.01]$、$+30.92\ [25.32,36.55]$ px。因果拆分表明**第二步原生 MSE 是主要活性成分**：仅条件项把 h2 绝对误差改善约 $16.20$ px，但交换历史效应为负；仅原生 MSE 改善约 $57.97$ px 并给出正的历史收益；在其上再加第二步条件项只带来较小的绝对增益。更大的 $\rho$ 继续提高 h2/h3 收益但加重 h1 代价，说明这是可解释的短期—长程权衡。该阶段的标准 PushT 保持性（同 100 个 episode）为对照 $57/100$、$\rho = 0.50$ $53/100$、$\rho = 0.25$ $60/100$，$\rho=0.25$ 相对对照为 $+3$pp $[-6,+12]$pp。

### 7.2 续段动作支撑（Contact，单因素）

两个 RC 分支唯一的差别是第二个动作块：重复已发布的查询动作，或从普通 PushT 训练总体的每个 episode 确定性抽取一个连续五步块。每个块在 low/high 摩擦下完全相同，$8{,}192$ 个模板全部保留，不按未来、接触、模型输出或隐藏标签筛选。一步直接响应几乎重合（future $0.781/0.779/0.783$，history $0.854/0.846/0.850$，switch 均 $1.000$，NRE $0.617/0.619/0.618$）。

| 视界 | oracle 可辨识配对 | 对照 ↓ | 重复动作 RC ↓ | 经验动作 RC ↓ | 经验 − 对照改善 | 历史收益 DID |
|---|---:|---:|---:|---:|---:|---:|
| h2（训练） | $256/256$ | $29.31$ | $24.84$ | $24.92$ | $4.39\ [2.93, 5.90]$ | $1.26\ [0.67, 1.87]$ |
| h5（未训练） | $244/256$ | $93.41$ | $91.87$ | $89.04$ | $4.37\ [1.87, 6.84]$ | $2.58\ [0.86, 4.34]$ |

标准 PushT（300 配对查询）给出方向明确的单因素结果：一步续训对照 $212/300$；重复动作 RC $197/300$，即 $-5.00$pp $[-9.33, -0.67]$pp；经验动作 RC $216/300$，即 $+1.33$pp $[-3.33, +6.00]$pp；经验 − 重复 $= +6.33$pp $[1.67, 11.00]$pp。由于三臂只改变第二步动作支撑，该结果支持把最初的保持性代价定位到**重复动作导致的滚动监督覆盖过窄**，而不是 RC 原则、$\rho$ 或模型复杂度。经验与重复在 h2 绝对误差上不可区分（$-0.08\ [-1.26, 1.04]$ px），h5 上经验点估计再改善 $2.82$ px（$[-0.79, 6.49]$）。

### 7.3 动作多样性不是单调有益（Motion，负对照）

保持起点、$\rho = 0.25$、损失、种子、批次采样顺序与 1,024 步预算不变，只把与评测动作族一致的零动作保持（zero hold）换成从普通 PushT 回放**无条件**抽取的五步动作块。该分支相对同数据一步对照在 h2/h3 仍改善 $21.17\ [17.80, 24.64]$ 与 $13.28\ [9.08, 17.99]$ px（自回归原生 MSE 仍是活性成分），但相对零动作保持分别差 $23.36\ [20.33, 26.40]$ 与 $16.70\ [12.28, 21.11]$ px，且正确—交换历史收益从 $+1.93/+0.63$ px 翻为 $-0.64/-1.95$ px。因此普适原则不是最大化无条件动作多样性，而是让滚动监督覆盖与查询/部署相关的动作支撑。

### 7.4 条件对应的必要性

| 替代解释 | 关键对照 | 结论 |
|---|---|---|
| 目标坍缩是失败主因 | stop-gradient、VISReg、目标对分离度 | 目标可分仍可能完全忽略历史 |
| Encoder 丢失全部历史信息 | 交叉拟合的历史 probe | 部分信息可读，但 Predictor 不必然使用 |
| 任意配对/对比都有效 | Contact 错配 $(Q,A)$ 对照 | 错误配对退回接近原生水平；需要真正的条件对应 |
| 直接拟合组中心可补全绝对未来 | 精确中心对照 | 中心项数值占优并压低 gain，不能与响应项同权相加 |
| Motion 只需训练更久 | 2,048 → 8,192 步训练—开发差距 | 样本内大幅改善而留出恶化，训练更久加重记忆 |
| 输出投影是主要故障点 | Predictor / `pred_proj` 模块互换 | 效应主要位于 Predictor trunk，并分布于六个 block |
| NRE 改善会自动恢复标准 CEM | Motion 完整查询集训练 | NRE 降到 $0.767$，标准 CEM 仍下降约 $6$–$8$pp |
| 标准 CEM 下降说明条件能力对规划无用 | Contact 三臂隐藏规划 | COJA 同时改善物理误差、regret 与正确历史收益 |
| 单步条件能力可直接用于多步规划 | Motion 1/2/3 步隐藏规划 | 一步 switch 高，两步历史收益可为负 |
| 每个视界都必须增加配对条件项 | 第二步 MSE / 条件项因子拆分（§7.1） | 第二步原生 MSE 是主要活性成分 |
| 第二步训练只是记住第二个终点 | 未训练的 h3/h5 | 两步训练在更长视界仍显著改善 |
| RC 需要先训练 COJA 再续训 | 从公开初始化的单阶段 4,096 步（§6.3、§6.4） | 单阶段复现全部主效应，两阶段安排非必要 |

### 7.5 响应边缘匹配与逐实例条件对齐的区别

一类更早的目标（DynamicsResponseSIGReg）先用配对样本构造响应，随后把不同查询的目标/预测对比放入总体分布统计，并未逐查询强制 $\hat r_i$ 对齐 $r_i$。它在响应方向较一致的 Action Strength 上很强：三种子 future 为 $0.966$、switch 为 $0.996$。在完全相同的 $256$ 对条件、CEM 种子、物理 oracle 与真实执行下，同场复评结果如下。

| 方法或检查点 | 模式分类 | 相对同数据原生对照的主要结果 |
|---|---:|---|
| 公开参考 | $0.488$ | 只作公开起点参照 |
| 同数据原生对照 | $0.822$ | 方法效应的匹配基线 |
| target-JTCov | $0.734$ | 分类低 $8.79$pp $[5.86,11.91]$，regret 高 $0.0725\ [0.0530,0.0921]$，执行距离高 $2.66$ px $[1.91,3.43]$ |
| terminal ConditionalSIGReg | $0.801$ | 来自不匹配的 1,024 步训练，只作检查点筛查，不构成严格方法归因 |
| DynamicsResponseSIGReg | $0.975$ | 分类高 $15.23$pp $[12.30,18.16]$，regret 低 $0.0706\ [0.0544,0.0872]$ |

但 DynamicsResponseSIGReg 在查询依赖更强的 Motion 响应上失败：4,096 步 future/history 仅为 $0.412/0.418$。这组跨任务差异表明，只约束响应集合的总体分布仍可能遗漏每个查询自身的响应方向。

由此可见，COJA 的增量不是“再加一个更强的边缘正则”，而是从响应的边缘匹配转为配对 $(Q,A)$ 下的逐实例条件对应。

## 8. 论文关键待验证项

### 8.1 固定权重敏感性

当前所有 LeWM 结果均使用单一固定权重 $\lambda_{\mathrm{COJA}} = 0.09$，尚未做该权重的敏感性分析。计划中的阶梯为

$$
\lambda_{\mathrm{COJA}} \in \{0,\ 0.03,\ 0.09,\ 0.27\},
$$

在其余配置（初始化、数据混合、关系保持采样器、$\rho = 0.25$、预算、评测查询、随机种子）完全冻结的条件下，报告每一档的直接条件响应、隐藏动力学规划（含正确/交换历史）与标准 PushT 保持性。$\lambda = 0$ 仍使用完全相同的关系保持采样器，只关闭辅助损失，作为阶梯下端的严格对照。**这是尚未执行的验证计划，不是已有结果**；在其完成之前，本文对权重选择的稳健性不作任何主张。

### 8.2 跨训练目标与模型族的可组合性

论文的可组合性主假设不是“COJA 的 LeWM checkpoint 绝对分数高于所有模型”，而是：在保持每种基线方法自身目标不变时，加入同一个 COJA 条件项都能改善该方法原本缺失的条件能力。LeWM、PLDM、VIS-WM 与 DINO-WM 是四个独立的方法基线；其中 LeWM 明确定义为 prediction MSE + SIGReg，VIS-WM 使用 prediction MSE + VISReg。两者仅复用相同的骨干实现与通用训练机械，公开入口分别为 `lewm` 与 `viswm`，目标配置与运行身份不共享。DINO-WM 在本仓库中对应 `prejepa` 入口。待完成矩阵为：

| 方法基线 | 方法原生目标 | 原生终点 | `+COJA` 终点 | 当前状态 |
|---|---|---|---|---|
| LeWM | prediction MSE + SIGReg | 需要与最终数据身份匹配 | 相同 LeWM 配方加 COJA | 已有主要方法证据，最终统一复现待完成 |
| VIS-WM | prediction MSE + VISReg | 需按完整公开配方训练 | 相同 VIS-WM 配方加 COJA | 目前只有 1,024 步原生筛查 |
| PLDM | PLDM 原生目标 | 可复用身份完全匹配的 ContextWorld 基线 | 相同 PLDM 配方加 COJA | 仅接口与梯度边界验证 |
| DINO-WM（`prejepa`） | DINO-WM 原生目标 | 可复用身份完全匹配的 ContextWorld 基线 | 相同 DINO-WM 配方加 COJA | 仅接口与梯度边界验证 |

对基线方法 $f$ 与任务 $t$，主要估计量应是方法内配对增量

$$
\Delta_{f,t}=U_{f+\mathrm{COJA},t}-U_{f,t},
$$

其中 $U$ 按指标方向统一为“越大越好”的任务效用；原始分数、NRE、隐藏动力学规划和标准环境保持仍分别报告。只有当多个基线方法上的 $\Delta_{f,t}$ 在预先冻结的失败能力项上方向一致，才能主张 COJA 是可组合的训练原则。不同方法之间的绝对分数只作次要比较。

为控制成本，先在最新冻结 benchmark 中**所有至少一个原生模型未学会的能力项**上运行一个训练种子的完整终点；不能根据 COJA 的中间结果挑任务。出现跨方法一致正信号后，再补三个训练种子与完整九任务矩阵。已有原生 checkpoint 只有在数据、训练划分、初始化、预算、优化器、精度和代码身份全部一致时才复用，否则必须随 `+COJA` 分支做匹配重跑。第一轮所有方法统一使用 $\lambda_{\mathrm{COJA}}=0.09$，不做逐方法救援式调参；权重敏感性按 §8.1 单独报告。

RC-COJA 依赖真实短续段并且当前只在 LeWM 研究训练器中实现，因此不能与一步 COJA 的跨方法结论混写。论文应先验证上述一步 COJA 的四基线可组合性，再至少选择 PLDM 或 DINO-WM 中的一种验证滚动一致性扩展；在完成以前，不主张 RC-COJA 已跨方法有效。

---

## 9. 局限

1. **条件重叠假设**：COJA 尚未在普通无配对的离线回放上去除条件重叠假设。命题 1 说明，条件重叠、主动干预或额外结构信息至少需要具备其一；它不证明精确配对本身不可替代。如何从普通离线轨迹自动构造可用的重叠仍是开放问题。
2. **轨迹续段数据**：RC-COJA 需要真实的短轨迹续段目标；它不增加参数，但不等于“无需组织数据”。
3. **视界权衡**：Motion 上 RC 相对一步 COJA 的一步误差稳定变差 $3.01/3.30$ px（两种子），这是真实的短期—长程权衡。
4. **动作支撑依赖任务**：续段动作必须落在与查询/部署相关的支撑上（§7.2、§7.3）；目前没有统一的采样公式，各任务需分别确定。
5. **随机种子数量**：滚动一致性在 Motion 与 Contact 上各有两个训练种子，足以确认方向可复现，不足以精确估计训练方差；Portal 的最终方法仍缺多种子结果；RC 尚未扩展到 ActionDelay、Portal 或非 PushT 动力学域。
6. **共享训练路径的原任务代价**：同混合数据的一步 COJA 与 RC 相对公开参考仍有约 $10$pp 的标准 PushT 差距（Contact），Motion 上无辅助项到一步 COJA 也有约 $5$pp 下降。这些不是 RC 的方法效应，但仍是数据混合/适配路径需要单独解决的代价。
7. **评测口径**：隐藏动力学规划使用筛查式 CEM（Contact 为 $64 \times 6$ 一维搜索），支持方向与排序估计，不等价于标准 PushT 的完整多维规划成功率。
8. **跨方法证据不对称**：一步条件项已接入 LeWM、VIS-WM、PLDM、DINO-WM（`prejepa`）并有接口测试，但任务级效果证据主要来自 LeWM；统一云入口 `coja_v1` 当前只开放 Contact Friction。VIS-WM 只有初步筛查，完整四基线矩阵见 §8.2，尚未执行。RC-COJA 尚未进入该共享入口。
9. **权重敏感性未验证**：见 §8.1。
10. **不作过宽主张**：现有结果不支持“所有边缘正则都无用”或“PLDM 整体弱于 LeWM”一类结论。

---

## 10. 结论

ContextWorld 揭示的问题不是普通的表征坍缩，而是**条件联合关系缺失**：模型可以拥有健康的表征边缘分布，却没有学会“这段历史、这个动作，应对应哪个未来”。命题 1 表明，在缺少条件重叠或额外结构假设时，这种反事实条件响应在非参数意义下根本不可识别，因而任何仅依赖边缘统计的正则都不可能补足它。

COJA 用同一可见条件 $(Q,A)$ 下的历史干预直接打破置换对称性，在不改变模型结构与推理路径的前提下建立**一步条件对应**；RC-COJA 用同一 Predictor 上的短自回归原生 MSE 建立**滚动一致性**，把该对应传递到规划视界。Motion 与 Contact 的单阶段实验均在第二个训练随机种子上复现了更长视界的收益，正确历史收益的 DID 为正；与一步 COJA 的匹配比较没有检测到 RC 特异的标准 PushT 保持性退化。完整效应与区间见 §6。

因此，当前最合理的结论是：条件重叠联合对齐是一个有效的一步条件可辨识性方法，与部署相关动作支撑匹配的短自回归一致性是把它转化为规划能力的可迁移机制；两者都不增加模型或部署复杂度。显式条件重叠、短轨迹续段数据、视界权衡、有限的随机种子数与共享混合训练的原任务代价，仍是其明确边界。

---

## 11. 证据索引

带链接的条目已纳入版本控制，可从 GitHub 直接打开；标为“本地归档”的条目在当前研究工作区存在，但尚未进入公开仓库。后续论文发布应从这些本地资产中提取最小、不可变的结果回执，而不是提交完整训练目录。

### 诊断与负对照

- Motion 可见条件分组恢复（零训练步）— 本地归档：`artifacts/pusht_motion_damping_visible_condition_pair_mining_v1/receipt.json`
- Motion 无标签主动重叠采集 MVE — 本地归档：`artifacts/pusht_motion_damping_label_blind_overlap_collection_v1/receipt_templates2048_v1.json`
- Motion 8,192 步响应校准分解 — 本地归档：`artifacts/pusht_motion_damping_full8192_response_calibration_v1/analysis.json`
- Motion 2,048 步响应校准分解 — 本地归档：`artifacts/pusht_motion_damping_full8192_response_calibration_v1/step2048_analysis.json`
- Motion Predictor 模块互换 — 本地归档：`artifacts/pusht_motion_damping_full8192_response_calibration_v1/module_swap_v2.json`
- [VISReg ActionDelay 汇总](artifacts/visreg_action_delay_discovery_v1/summary.json)
- target-JTCov 任务广度与隐藏规划结果 — 本地归档：`results/joint_temporal_covariance_sigreg_task_breadth_summary_v1.json`
- 同场对照有效性复评 — 本地归档：`artifacts/conditional_joint_comparator_validity_v2/summary.json`
- [Action Strength DynamicsResponseSIGReg 三种子结果](https://github.com/Anguo-star/ContextWorld/blob/main/configs/benchmark/pusht_hidden_actuation_replay_matched_results_v2.yaml)
- Action Strength 同场隐藏规划复评 — 本地归档：`artifacts/historical_candidate_reevaluation_v1/summary.json`

### 一步条件响应

- Motion 完整查询集 COJA 开发集响应 — 本地归档：`artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_step8192_v1/s14321_step8192_v1/development_response_analysis_v1.json`
- Motion 完整查询集无辅助项对照 — 本地归档：`artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_native_control_step8192_v1/s14321_step8192_v1/development_response_analysis_v1.json`
- [Contact Friction 迁移汇总](artifacts/pusht_contact_friction_visible_joint_transfer_v1/summary.json)
- [Contact Friction seed-3072 full-10-epoch Development/Public 聚合摘要](artifacts/contact_friction_lewm_coja_seed3072_full10_summary.json)
- Contact 单阶段 RC 响应分解 — 本地归档：`artifacts/pusht_contact_friction_empirical_action_rc_full4096_center_response_vs_coja_v1.json`

### 隐藏动力学规划

- Contact 公开参考/原生/COJA 三臂隐藏规划 — 本地归档：`artifacts/pusht_contact_friction_hidden_cem_h5_three_arm_development256_cpu_v1/summary.json`
- Contact RC h2 隐藏规划 — 本地归档：`artifacts/pusht_contact_friction_rollout_consistent_hidden_cem_h2_dev256_v1/summary.json`
- Contact RC h5 隐藏规划 — 本地归档：`artifacts/pusht_contact_friction_rollout_consistent_hidden_cem_h5_dev256_v1/summary.json`
- Contact 经验动作 h2 隐藏规划 — 本地归档：`artifacts/pusht_contact_friction_empirical_action_hidden_cem_h2_dev256_v1/summary.json`
- Contact 经验动作 h5 隐藏规划 — 本地归档：`artifacts/pusht_contact_friction_empirical_action_hidden_cem_h5_dev256_v1/summary.json`
- Contact 单阶段 h2 隐藏规划 — 本地归档：`artifacts/pusht_contact_friction_empirical_action_rc_full4096_hidden_cem_h2_dev256_v1/summary.json`
- Contact 单阶段 h5 隐藏规划 — 本地归档：`artifacts/pusht_contact_friction_empirical_action_rc_full4096_hidden_cem_h5_dev256_v1/summary.json`
- [Motion 滚动一致性因果拆分汇总](artifacts/pusht_motion_damping_rollout_consistency_mve_v1/summary.json)
- Motion 单阶段 RC h1 隐藏规划 — 本地归档：`artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected34_blocks1_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json`
- Motion 单阶段 RC h2 隐藏规划 — 本地归档：`artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected156_blocks2_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json`
- Motion 单阶段 RC h3 隐藏规划 — 本地归档：`artifacts/pusht_motion_damping_rollout_consistent_zero_hold_full4096_hidden_planning_v1/three_arm_input256_selected134_blocks3_ref0.75_noaux_vs_coja_vs_rc_full4096_v1/summary.json`

### 原任务保持

- ActionDelay 一枚训练种子的 900 查询 CEM 复核 — 本地归档：`artifacts/action_delay_h7_a0_aux_pcja_predictor_only_cem_resolution_v1/consumption_receipt.json`
- Motion 多目录 CEM 连续效应 — 本地归档：`artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_cem_seeds42_43_44_n100_runtimefix_v2/continuous_paired_effect_v1.json`
- Motion seed42×300 CEM 连续效应 — 本地归档：`artifacts/pusht_motion_damping_full_release_visible_joint_absolute_single_stage_cem300_seed42_current_runtime_v1/continuous_paired_effect_v1.json`
- [Motion 单阶段标准 CEM300 配对统计](artifacts/pusht_motion_damping_full4096_standard_cem300_paired_v1/paired_analysis_v1.json)
- Contact 标准 CEM100 — 本地归档：`artifacts/pusht_contact_friction_rollout_consistent_standard_cem100_v1/aggregate.json`
- Contact 经验动作标准 CEM300 — 本地归档：`artifacts/pusht_contact_friction_empirical_action_standard_cem300_v1/aggregate.json`
- Contact 单阶段 RC 标准 CEM300 — 本地归档：`artifacts/pusht_contact_friction_empirical_action_rc_full4096_standard_cem300_v1/aggregate.json`
- Contact 一步 COJA 标准 CEM300 — 本地归档：`artifacts/pusht_contact_friction_coja4096_standard_cem300_v1/aggregate.json`
- Contact 公开参考标准 CEM300 — 本地归档：`artifacts/pusht_contact_friction_native4096_standard_cem300_v1/aggregate.json`

### 复现

- [Motion RC-COJA 双训练种子复现汇总](artifacts/pusht_motion_damping_rc_coja_full4096_replication_v1/replication_summary_v1.json)
- Contact RC-COJA 双训练种子复现汇总 — 本地归档：`artifacts/pusht_contact_friction_rc_coja_full4096_replication_v1/replication_summary_v1.json`

### 实现

- [二元/三元条件联合目标](../../stable_worldmodel/wm/conditional_joint.py)
- [Motion 第二步真实目标构建](scripts/build_pusht_motion_damping_planner_curve_rollout2_targets_v1.py)
- [Motion RC-COJA 续训实现](scripts/run_pusht_motion_damping_planner_curve_rollout_consistent_continuation_v1.py)
- [Motion 无条件经验动作目标构建](scripts/build_pusht_motion_damping_planner_curve_rollout2_empirical_action_targets_v1.py)
- [Motion 经验动作 RC 续训](scripts/run_pusht_motion_damping_planner_curve_rollout_consistent_empirical_action_continuation_v1.py)
- [Motion 单阶段 RC-COJA 实现](scripts/run_pusht_motion_damping_rollout_consistent_zero_hold_full4096_v1.py)
- [Motion 固定配置独立种子训练脚本](scripts/run_pusht_motion_damping_rc_coja_full4096_replication_v1.py)
- Contact 第二步目标构建 — 本地归档：`scripts/build_pusht_contact_friction_rollout2_targets_v1.py`
- Contact RC-COJA 续训 — 本地归档：`scripts/run_pusht_contact_friction_rollout_consistent_continuation_v1.py`
- [Contact 经验动作目标构建](scripts/build_pusht_contact_friction_rollout2_empirical_action_targets_v1.py)
- Contact 经验动作 RC 续训 — 本地归档：`scripts/run_pusht_contact_friction_rollout_consistent_empirical_action_continuation_v1.py`
- Contact 固定配置独立种子训练脚本 — 本地归档：`scripts/run_pusht_contact_friction_rc_coja_full4096_replication_v1.py`
