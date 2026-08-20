# 常设审计契约 v1

**这是一份规则文件，不是某次实验的记录。** 它适用于本目录下所有未来的审计器、预注册与回执。
写它的原因：本周同一类缺陷出现了四次，每次都被当作新问题重新发现、重新修复。修复本身没错，
浪费在于**没有把规则写下来**。

冻结日期：2026-08-19。append-only：本文件可增补，不可改写既有条款的含义；作废某条要写明理由并保留原文。

---

## 0. 为什么需要它：实测的审计税

| 指标 | 实测值 |
|---|---|
| 审计/回执类 JSON : 科学结果 JSON | **316 : 40 ≈ 7.9 : 1** |
| 审计脚本 + recovery 脚本 | 193 文件 / 89,782 行（占 `scripts/` 代码 **30%**） |
| `configs/` 中 addendum / recovery / amendment | **275 / 607** |
| 对照：seeds 4096+5120 完整训练 | **73 分钟** |

审计存在的唯一理由是**让结论可信**。当它的产出量接近实验本身，而结论却没有增加时，
它就从保障变成了成本。以下条款的目的是保住保障、砍掉成本。

---

## 1. 内容不变量优先于位置代理（承重条款）

**规则.** 门必须断言合同真正关心的**内容**。当内容不可直接断言时才允许用位置/身份代理，
且代理**必须排在内容检查之后**，并标注为 `recorded_not_gated` 或说明其为何可判定。

**为什么.** 位置代理有两个失效模式，第二个才是严重的：

1. *假警报*：无关变动挪动了位置，门就拒绝。
2. *短路真检查*：代理排在前面，一旦不匹配，审计器停在代理那一行，**真正承载科学含义的内容比较永远不执行**。
   最需要保证的时刻，保证反而消失。反过来，一个被改写的评分 kernel 只要恰好坐在被钉的 commit 上，
   就能通过先跑的那道门。**代理不是更严，是又松又吵。**

**本周的四个实例：**

| 实例 | 断言的代理 | 真正的不变量 | 被什么无关变化打破 |
|---|---|---|---|
| ContextWorld import 闭包 | 模块路径集合精确相等 | 评分时装入的实现不变 | 一次 lazy-import 重构，行为未变 |
| loss trace 终点行 | 最后一行 == `trainer.global_step` | 训练确实跑满授权步数 | 冻结 recipe 自身的有意提前停止——该断言**不可满足** |
| source-rebind HEAD 钉 | `HEAD == 66761639…` | 45 文件评分源闭包不变 | 12 个无关提交；起草替代品期间又来 1 个（宣告许可证） |
| 审计输出目录 | 目录必须为空 | 没有预置回执 | 事先写入的缺陷报告（正当证据） |

**第五个实例（2026-08-20 增补）：第一次出现在冻结的 *builder* 里，不是审计器里。**

| 实例 | 断言的代理 | 真正的不变量 | 被什么无关变化打破 |
|---|---|---|---|
| build 源身份门 | `_git_head(contextworld_root) == config["contextworld_commit"]` | 同一个 45 文件闭包 | 16 个无关提交；且 v2 钉的 commit **根本不是 HEAD 的祖先**（在废弃分支上） |

这一条特别值得记，因为它把第 2 节的失效模式演示得最干净：该比较**排在同一函数内逐文件 sha256 循环之前**，
HEAD 一动，`source_hashes` / `renderer_dependency_hashes` / `scoring_runtime_dependency_hashes` / `overlap_sources`
四个循环全部够不到。**建 release 时最需要保证源码没变，而那一刻保证恰好失效。**

处理：`…_v2_build_source_gate_addendum_v1.yaml` 把 commit 降为记录项，判定权交给内容闭包；
四个已裁定文件按 v3 记录的哈希**继承钉死**，不重新裁定。29 个反例全部拦下（22 配置 + 7 运行时）。

**现场证据（值得单独记一笔）：** 做这件事期间，内容门**真的报警了**——`contextworld/benchmarks/adapters.py`
哈希对不上，几分钟后重跑又过。查清是别人加 PreJEPA adapter 家族的未提交改动，随后回退，**没有任何 commit**。
HEAD 钉对这种改动**完全无话可说**；内容门抓到了并拒绝。这不是论证，是观测。

> **一个自我更正**：第一次比对用正则，报告 `StableWorldModelLeWMAdapter` 变了。那是正则假象——
> 模式依赖尾部 `class` 边界，而改动后的文件没有。用 AST 重做：21 个顶层定义 vs 21 个，共同定义 0 处改动，
> 模块级代码相同，评分路径三个类逐字节相同。**错误的中间结论写进回执了，没有悄悄删掉。**

**处理方式恒定：满足内容不变量来消警，绝不放宽断言。**
- import 闭包 → 显式恢复冻结 import 状态，**不**把集合相等改成子集；
- loss trace → 冻结 recovery runner 依落盘证据补写 phase report，**不**插补缺失行，并记为 `infrastructure_NOGO_not_method_failure`；
- HEAD 钉 → 判定权交给 45 文件内容闭包，4 个差异文件逐一书面裁定并**按裁定时的观测哈希钉死**（再变即作废），HEAD 降为记录项；
- 目录为空 → 允许预存文件**按哈希钉住**；
- build 源身份门 → 同 HEAD 钉的处理，**继承** v3 的四份裁定而不重新裁定；保留"读取期间树必须稳定"（这是真内容属性），去掉"等于某个预定 commit"（这不是）。

**起草新门时的自检（两问）：**
1. 我真正要保证的量是什么？
2. 我写下的断言，在什么**无关**变化下会与它分离？
   若答案是"上游随便提交一次就会"，那写的就是代理，重写。

**第三问（第五实例后增补）：这道门排在哪？**
即使断言本身没错，把它排在真正的内容检查**之前**，就等于给了它一票否决权。
内容检查要么排在最前，要么保证无论前面结果如何都会执行。

---

## 2. 成本必须实测后才能声明（新增，本周踩坑）

**规则.** 任何"这个太贵 / 做不到 / 是重大工程"的判断，**必须附带一次实测**才能写进回执或用作阻断理由。
无实测的成本判断一律标为 `estimate_unverified`，且**不得**单独构成阻断。

**为什么.** `evaluation_blocked_v1.json` 写下：

> "this is a substantial build, not a script invocation"

实测：v1 release staging 306 文件 **23.9 分钟** + 发布 **约 1 分钟**；PLDM v4 发布 306 文件 **2 分钟以内**。
这不是重大工程，就是一次脚本调用。**一个未经测量的成本判断，把 25 分钟的事描述成重大工程，
挡了整整一轮工作。** 这比任何一个技术缺陷都贵。

**怎么做.** 用 mtime 跨度、文件计数、字节数这类**已经落盘的证据**估算，不需要重跑：
`min/max mtime` 之差即建造跨度，`rglob` 计数即规模。代价是几秒钟。

---

## 3. 审计器不得层层 import 上一版（新增）

**规则.** 新审计器可以复用共享工具函数，但**不得**通过 `import v2 → import v1` 的链条继承**判定逻辑**。
需要某个校验时，从**权威来源**取（例如 release builder 自己的常量），不要转述。

**为什么.** v1 审计器的 `POST_FREEZE_DYNAMIC_ROOTS` 从 PLDM stage-2 行拷来，写的是 `.stage2_batch_in_progress`，
而本 release 的 builder 创建的是 `.private_batch_in_progress`。v1 因此把 305 个文件数成 307。
这个 bug 顺着 import 链**原样传到了 v3**。修法不是改 v1（已冻结），而是让 v3 用 **builder 自己的**排除集做普查。

**推论.** 凡是"某个名字/集合/阈值在两处各写一遍"的地方，都是未来的漂移点。取其一为权威，另一处引用它。

---

## 3.5 预注册必须**执行**冻结校验器，不能只对着读（2026-08-20 增补）

**规则.** 新预注册冻结之前，必须**实际调用**它将来要通过的冻结校验函数，把报错贴进回执。
"我读过代码，看起来能过"不算数。

**为什么.** v2 预注册是照着 v1 写的，看起来完备。实际执行冻结校验器：

```
_contract_sections(v2)        -> 缺 training_contract、runtime_rebind
_validate_runtime_rebind(v2)  -> ContractError: previous ContextWorld commit changed
_validate_config_binding(v2)  -> ContractError: prior Private binding chain changed
```

三处缺失，肉眼全没看出来——因为缺的是**继承链上游要求的**东西，不在 v1 预注册的显眼位置。
代价是必须再写一份 append-only addendum 补，而不是冻结时一次写对。**几十秒的执行，省一轮返工。**

**怎么做.** 把配置读成 dict，逐个调用 `_contract_sections` / `_validate_*`，捕获 `ContractError` 打印。
不需要建任何东西，不碰磁盘。

**推论：冻结的校验器不一定适用于它的后代。** v1 的 `_validate_config_binding` 钉死了 v1 **自己的**父辈
（cutoff512）和 v1 自己的 9-release 列表，因此**结构上无法**校验一个以 v1 为父辈的 release。
这不是 bug，是它被正确地钉死了。后代要写自己的等价物，**不要去改冻结件**。

---

## 4. 承重条款，不得以效率为由削弱

以下是数字有意义的**唯一**原因，任何"加快迭代"的理由都不能动它们：

- **一次性 release + checkpoint 盲建**：release 必须在候选 checkpoint 被打开**之前**选定、生成、冻结。
  已消耗的 release **不得**重开或追加候选——重开会**追溯性削弱已发表的旧结果**，不只是新结果。
- **append-only**：`overwrite: forbidden`。错了写更正文件，不改原件。v1 / recovery-v2 / attempt-1/2/3 永不编辑。
- **内容哈希门**：45 文件源闭包、release payload 文件数/字节/树根。
- **per-seed 独立判定**：禁 pooling、禁 averaging、禁 rescue。复现通过 = 每个确认 seed **各自**过全部四门。
- **结果无论好坏都上报**：不得因失败而不发布。
- **诊断量不替代冻结终点**（§1.3）：prefix、训练 loss、梯度方向、target separation、probing 全是诊断量。
  一阶有利是**必要非充分**条件——K4 在训练面板上 9/9 局部纠正，held-out 反而更差，这是本项目自带的反例。

---

## 5. 可以砍掉的开销

- **post-fit recovery 模式**：85 个 recovery 脚本 / 42,911 行，绝大多数在绕同一个"trainer 不干净退出"的问题。
  **修一次根因，整类代码消失。** 在根因修好前，新实验沿用现有 recovery runner，不要再为每个实验新写一个。
- **重复发现同一模式**：本文件存在的意义。新审计器起草时先读本文件第 1、3 节，不要重新踩。
- **凭印象宣布成本**：见第 2 节。

---

## 6. 适用方式

新审计器 / 预注册的文件头引用本文件：

```yaml
audit_contract:
  path: research/conditional_dynamics_representation/AUDIT_CONTRACT.md
  version: 1
```

与本文件冲突的旧条款以本文件为准；但**第 4 节承重条款**若与旧文件冲突，以**更严的一方**为准。

**相关记录**：`configs/…_source_rebind_addendum_v3.yaml`（HEAD 降级为记录项的预注册）、
`artifacts/…_source_rebind_addendum_v3/v1_auditor_exclusion_set_defect_v1.json`（第 3 节的实例）、
`artifacts/…_multi_seed_v1/receipts/evaluation_blocked_v1_correction_v1.json`（第 1、2 节的自我更正）、
`configs/…_v2_build_source_gate_addendum_v1.yaml` + 其回执（第 1 节第五实例，builder 侧）、
`configs/…_v2_contract_sections_addendum_v1.yaml` + 其回执（第 3.5 节的实例）、
README §6.1（同一模式的叙述版）。

---

## 7. 变更记录

| 日期 | 变更 |
|---|---|
| 2026-08-19 | v1 冻结：第 0–6 节，四个位置代理实例 |
| 2026-08-20 | 增补第 1 节第五实例（builder 侧 HEAD 门）与第三问；增补第 3.5 节（预注册必须执行冻结校验器）；第 1 节增补 adapters.py 现场证据与一处自我更正 |
