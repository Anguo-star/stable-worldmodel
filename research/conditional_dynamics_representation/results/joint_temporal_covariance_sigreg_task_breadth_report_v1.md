# target-JTCov 冻结任务广度验证：阶段报告

## 结论

这轮冻结验证不支持把 target-JTCov 作为 LeWM 的通用简洁修复：六项正式
ContextWorld ICL 只通过 `3/6`，九项原始任务 CEM 非劣只通过 `2/9`，对已有 PLDM
参照的覆盖也没有整体竞争力。因此：

- `core_lewm_pareto_improvement = false`；
- `worth_seed_expansion = false`；
- 当前确切方法不应晋级，也不能声称“补上 LeWM 的 ICL 盲区且不牺牲原能力”。

它并非完全无效。Door、Contact Friction、Motion Damping，以及不计入冻结六项主门的
Portal Exit 都学到了明确的历史条件响应。但这种收益不是跨动力学类型稳定成立的，并伴随
大范围原始规划退化。

## 冻结比较契约

方法始终是：

```text
prediction_mse + 0.09 * target_joint_temporal_covariance_sigreg
rho = 1 / T^2
```

相对原生 SIGReg，它新增 `0` 个可学习参数、`1` 个固定超参数规则，没有 pair metadata、
privileged physics、detach、冻结分支、额外 loss 成分或额外 forward。PushT、TwoRoom、
Reacher、Cube 均使用相同权重 `0.09`；公开测试后没有任务特定修补。

本阶段新训练量为 `20 task-epochs`：Reacher `10`、Cube `10`。其他任务使用冻结的固定
step checkpoint。该阶段是 release-declared 单训练 seed 的任务广度筛查，而不是正式多 seed
方法结论。

## ContextWorld ICL

| 组件 | 关键公开指标 | 结果 |
|---|---|---:|
| Speed | 所有 track/horizon 的 loss ratio 约为 `1.0`，query win rate `48.3–51.3%` | 失败 |
| Door | future `90.0%`；history win `99.5%`；strict `60.5%`；worst cell `56%` | 通过 |
| Action Delay | macro `31.34%`；minimum group `28.39%`；bootstrap lower `29.56%` | 失败 |
| Action Strength | future `81.05%`；history `93.95%`；switch `98.05%`；worst `71.09%` | 失败 |
| Contact Friction | future `95.12%`；history `99.80%`；switch `100%`；worst `94.53%` | 通过 |
| Motion Damping | future/history/switch/worst 均为 `100%` | 通过 |
| Portal Exit（addendum） | future/history/switch/worst 均为 `100%` | 通过 |

正式冻结六项为前三个 TwoRoom/后三个 PushT 组件，不含 Portal Exit，因此主计数是 `3/6`。
原生 LeWM 在五个失败组件中只被恢复了三个：Door、Contact Friction、Motion Damping；
Action Delay 和 Action Strength 未恢复。原生本来通过的 Speed 被 target-JTCov 回退，故：

- `all_native_failures_recovered = false`（`3/5`）；
- `all_native_successes_retained = false`（`0/1`）。

## 九项原始任务 CEM

主门使用同 query 的 target/native 配对 episode bootstrap：`100,000` 次重采样，单侧
`95%` 下界必须不低于 `−5 pp`。

| 任务 | target | native | 点差 | 单侧 95% 下界 | 非劣 |
|---|---:|---:|---:|---:|---:|
| PushT Action Strength | 191/300 | 213/300 | −7.33 pp | −12.00 pp | 否 |
| PushT Contact Friction | 212/300 | 235/300 | −7.67 pp | −12.33 pp | 否 |
| PushT Motion Damping | 15/300 | 220/300 | −68.33 pp | −72.67 pp | 否 |
| TwoRoom Speed | 90/300 | 289/300 | −66.33 pp | −70.67 pp | 否 |
| TwoRoom Action Delay | 52/300 | 292/300 | −80.00 pp | −83.67 pp | 否 |
| TwoRoom Portal Exit | 282/300 | 273/300 | +3.00 pp | +0.67 pp | 是 |
| TwoRoom Door | 283/300 | 283/300 | 0.00 pp | −2.00 pp | 是 |
| Reacher | 21/300 | 170/300 | −49.67 pp | −54.67 pp | 否 |
| Cube | 169/300 | 198/300 | −9.67 pp | −14.00 pp | 否 |

总计仅 `2/9` 非劣。Contact Friction 的 target 仍达到预先规定的 `210/300` 绝对 floor，
但这不能替代相对同预算 native 的非劣失败。

Reacher 和 Cube 使用三模型、三 eval seed、每 seed 100 episode 的 fresh 同后端评估。由于
节点没有可用 headless EGL device，冻结 evaluator 在产生 episode 结果前失败；恢复运行只把
MuJoCo 后端从 EGL 改为 OSMesa，模型、数据、query、episode 数和 CEM 参数均未改变。新的
query catalog 与失败前 catalog 的 SHA-256 都是
`f139593c8b75eb5cd142d78223421e3e7b0a76a4ab086908afd9d9171dea9e27`。

渲染后端确实可能改变绝对分数，因此历史 EGL 点值没有进入正式统计；上述 Reacher/Cube
结论只使用同一次 OSMesa 下 fresh target/native/PLDM 的配对结果。

## PLDM 外部参照

PLDM 是当前覆盖最广、仍属于简单联合训练目标的非 LeWM 参照，但不是所有任务都有冻结的
同级结果。

ICL 有四项可比：Door、Action Delay、Contact Friction、Motion Damping。target-JTCov 与
PLDM 的任务门只匹配 `3/4`；Action Delay 上 PLDM 通过而 target-JTCov 失败。共同原始指标
也不满足一致点估计优势。

原始规划有六项可比：

| 任务 | target − PLDM | 单侧 95% 下界 | 非劣 |
|---|---:|---:|---:|
| PushT Contact Friction | +1.00 pp | −3.67 pp | 是 |
| PushT Motion Damping | −61.00 pp | −66.00 pp | 否 |
| TwoRoom Action Delay | −77.67 pp | −81.67 pp | 否 |
| TwoRoom Door | −2.67 pp | −5.00 pp | 是 |
| Reacher | −73.00 pp | −77.33 pp | 否 |
| Cube | +3.33 pp | −0.33 pp | 是 |

即 `3/6` 非劣，因此 `competitive_to_pldm_where_available = false`。Cube 说明了一个有用但
不足以救回方法的局部事实：target-JTCov 虽弱于 native LeWM，却仍非劣于 PLDM。

## 科学解释和下一步边界

JTCov 约束的是未条件化的整段 target trajectory 联合二阶结构。它能在部分任务上保留明显
时间响应，但仍不识别“哪一段历史条件导致哪一个未来响应”。实验结果与这个理论缺口一致：

- Speed 几乎完全无判别，说明联合时间形状仍可在不编码动力学条件的情况下满足；
- Action Delay、Action Strength 没有达到组件门；
- 多个原始任务出现远大于统计波动或 `5 pp` 容差的退化，不是 seed 噪声可以合理解释。

因此不应给这个确切方案扩大 seed，也不应在公开结果之后逐任务调 `rho` 或 loss 权重。若继续
寻找同样简洁的方向，统计量需要更直接地识别 predictor–target 的条件耦合或 conditional
innovation，而不是继续只约束无条件 target trajectory 分布。

Reacher/Cube ICL benchmark 尚未纳入本冻结版本；未来可以补作机制诊断，但它们无法推翻
当前负结论，因为六项已冻结 ICL 和九项原始 CEM 的主门都已经失败。

## 可追溯证据

- 机器可读统一结果：
  [`joint_temporal_covariance_sigreg_task_breadth_summary_v1.json`](joint_temporal_covariance_sigreg_task_breadth_summary_v1.json)，
  SHA-256 `ae5877344dfba8ecb0120cf28271c19e9904903266b12a21c27109b453a19a96`。
- 冻结协议：
  [`joint_temporal_covariance_sigreg_task_breadth_validation_v1.yaml`](../configs/joint_temporal_covariance_sigreg_task_breadth_validation_v1.yaml)。
- 完整执行、失败恢复与偏差记录：
  [`joint_temporal_covariance_sigreg_task_breadth_execution_record_v1.yaml`](../configs/joint_temporal_covariance_sigreg_task_breadth_execution_record_v1.yaml)，
  SHA-256 `19fb89244cd5147efd44c718a7dc0eccfd7b7fa3c54a9a1c4a4a9d0520692a8f`。
- Reacher aggregate SHA-256：
  `6adf7348a241f25d9fe89df17926eec8725992732cdec9a8fd624f3aa4b835e8`。
- Cube aggregate SHA-256：
  `d076d4b9e06c85969c9d5574319d8b4bbfbae6faf6b98bf76da0cdc6e95832f8`。
- 冻结 ContextWorld runtime archive SHA-256：
  `92c0f31b858ff1b313295a55b3dc3bab3c17264d9ccb057d11f32e9e03a7e717`。
- 最终 analyzer SHA-256：
  `23117427940c0024b2a3e51b9d7ceb06563cd68e92069c372dc2184ec2b7391b`。
- 研究测试集：`60 passed in 7.86s`。
