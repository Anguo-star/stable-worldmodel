# Multitask Scaling Plan

This note records the near-term scaling plan after the MT4 Cube experiment.
The goal is to scale task count while keeping SIGReg, task diversity, and
action-representation effects measurable rather than confounded with each
other.

## Decision

Run two research tracks in parallel, then merge them only after each track has
its own conclusion:

```text
Track A: SIGReg multitask scaling with a fixed action interface
Track B: heterogeneous action unification with SIGReg held fixed
```

The heterogeneous-task pipeline still starts with MT8 and keeps the ladder:

```text
MT4 -> MT8 -> MT16 -> MT32 -> MT64+
```

MT8 is the first operational checkpoint, not the intended upper bound.

## Parallel Research Tracks

### Track A: SIGReg Multitask Scaling

Question:

```text
Does SIGReg remain useful when LeWM is trained on a genuinely multitask
distribution?
```

This track should avoid heterogeneous action dimensions at first. Use datasets
with one shared action interface, such as `lerobot/metaworld_mt50` where the
actions are 4D `[x, y, z, gripper]`. This isolates the regularizer question
from the action-unification question.

Primary comparisons:

| Variant | Purpose |
| --- | --- |
| LeWM + SIGReg | Current LeWM objective. |
| LeWM without SIGReg | Causal test for whether SIGReg matters. |
| LeWM + VCReg/VICReg-style regularizer | Non-SIGReg anti-collapse baseline. |
| PLDM-style multi-loss baseline | Stronger world-model regularization reference if implementation cost is acceptable. |

Primary metrics:

- per-task validation prediction loss;
- latent standard deviation, covariance spectrum, and effective rank;
- task-wise loss imbalance;
- per-task success rate if an eval wrapper exists;
- worst-task degradation as task count rises.

Public materials show LeWorldModel results mainly on individual environments
such as PushT, Reacher, TwoRoom, and OGBench Cube. We should therefore frame
this track as an open system-level validation of SIGReg under multitask mixing,
not as a replication of an established MetaWorld-MT50 LeWM baseline.

### Track B: Heterogeneous Action Unification

Question:

```text
How should tasks with different native action dimensions and meanings share a
single world-model action input?
```

This track should keep the world-model regularizer fixed, initially using
default SIGReg, and vary only the action representation. The first useful
dataset is a heterogeneous MT8-style mix built from the current LeWM tasks plus
new action interfaces such as MetaWorld 4D or additional OGBench/RoboCasa
variants.

Primary comparisons:

| Variant | Role |
| --- | --- |
| A. `pad_to_max` | Compatibility baseline. |
| C. action-space adapter registry | Mature no-global-max controlled baseline. |
| D. descriptor-conditioned universal embedder | Main proposed shared action-embedding method. |
| E. D plus inverse-effect alignment | Later extension after D is stable. |

Track B should not use SIGReg ablations as its first question. Otherwise a
failure cannot be assigned cleanly to action representation or to the
regularizer.

### Merge Rule

Do not combine the two tracks into one large claim until both are locally
understood.

Merge into a combined technical direction only when:

- Track A shows whether SIGReg helps or hurts multitask mixing under a fixed
  action interface;
- Track B has at least one non-padding action representation with complete
  anchor evals;
- both tracks use compatible reporting for per-task loss, latent diagnostics,
  and eval metrics.

The combined target is:

```text
SIGReg-stabilized multitask LeWM + descriptor/effect-aware action embedding
```

## Why Start At MT8

MT8 is large enough to expose the next set of system issues:

- sampling policy beyond simple balanced sampling;
- action representation under more than one action dimension family;
- old-task retention under added task diversity;
- eval and metrics bookkeeping across more tasks;
- per-task normalizer scope and data-manifest hygiene.

It is still small enough that failures can be diagnosed without hiding behind
aggregate metrics.

## Scaling Ladder

| Stage | Purpose | Exit criterion |
| --- | --- | --- |
| MT4 | Current baseline with Cube and action padding. | Existing anchor evals are complete. |
| MT8 | Validate the scaling pipeline and first task expansion. | Anchor tasks do not collapse; new tasks have complete eval metrics. |
| MT16 | Start measuring transfer patterns across task families. | Mean score improves or worst-task degradation is bounded. |
| MT32 | Stress sampling, normalization, and action representation. | No systematic dataset-size or action-space bias dominates. |
| MT64+ | Move toward a generalist world model. | Held-out variants improve over narrower training. |

## MT8 Task Selection

MT8 should add four tasks that are easy to evaluate and close enough to current
tasks to keep diagnosis clean.

Preferred order:

1. OGBench/Cube variants with the same or related action interface.
2. RoboCasa or manipulation tasks already supported by local wrappers.
3. Existing Lance-format datasets with `pixels + action`.
4. More diverse robot datasets only after the action-representation baselines
   are in place.

Avoid mixing pure first-person human video into the main MT8 training run. Use
egocentric video for visual or temporal pretraining experiments, not as direct
action-conditioned LeWM supervision.

## Fixed Eval Panel

Every scaling stage must evaluate a fixed anchor panel:

```text
anchors:
  pusht
  reacher
  tworoom
  cube
```

Each stage also evaluates:

- representative newly added tasks;
- at least one held-out variant from a task family represented in training;
- the same seeds and episode count as prior evals unless explicitly changed.

Primary metrics:

- per-task success rate;
- anchor retention relative to MT4;
- mean score across evaluated tasks;
- worst-task score;
- negative-transfer count;
- per-task exposure and sampling probability.

## Data Manifest Requirements

Before MT8 training, each dataset should have a manifest entry:

```yaml
label: cube_single
path: lance-format/LeWorldModel/data/lewm_cube.lance
task_family: ogbench_cube
observation:
  key: pixels
  view: agentview
  resolution: [224, 224]
action:
  dim: 5
  descriptors:
    - {name: end_effector_delta_x, target: end_effector, control: delta_position, axis: x, frame: workspace, unit: normalized}
    - {name: end_effector_delta_y, target: end_effector, control: delta_position, axis: y, frame: workspace, unit: normalized}
split:
  train: 0.9
eval:
  supported: true
  task_name: cube
```

The manifest is not just documentation. It should become the source of truth
for action descriptors, sampling weights, eval task resolution, and normalizer
scope.

## Sampling Policy

Simple balanced sampling is useful for small task counts, but it can over-repeat
small datasets at MT16/MT32.

Support at least these policies:

```text
balanced
proportional
temperature: p_i proportional to n_i^alpha
capped_balanced
```

Recommended first sweep:

| Sampling | Purpose |
| --- | --- |
| balanced | Compare directly with MT3/MT4 behavior. |
| temperature alpha=0.5 | Reduce large-dataset dominance without fully over-repeating small datasets. |
| capped_balanced | Limit small-dataset repetition in larger task mixtures. |

## Action Representation Ablation

Action representation belongs to Track B. It should be varied independently
from task count and independently from the SIGReg ablation whenever possible.
The first heterogeneous MT8 experiments should compare:

| Variant | Description | Purpose |
| --- | --- | --- |
| A | `pad_to_max` | Current compatibility baseline. |
| B | `pad_to_max + action_mask` | Optional diagnostic if padding artifacts are suspected. |
| C | action-space adapter registry | Mature controlled-dataset baseline without global max action dim. |
| D | descriptor-conditioned universal action embedder | Main proposed route for new action types without adding adapter ids. |
| E | D plus inverse-effect alignment | Test whether embeddings can be calibrated by transition effects. |

Minimum viable comparison:

```text
MT8-A vs MT8-C vs MT8-D
```

Skip B in the main path. B keeps the global max action dimension and mainly
answers whether padding channels are causing artifacts; it is useful only as a
diagnostic. Run E only after D is stable. E adds auxiliary loss complexity and
should not be used to debug the base data pipeline.

C adapters are trained jointly with the world model. They do not require a
separate pretraining stage. C is safe for MT8 eval because each eval task can
provide its `action_interface`, but C should not be presented as an open-world
solution: a genuinely new unknown action stream would need either an existing
interface mapping or a new adapter.

D is the first variant that attacks that limitation. It replaces the
closed-world `action_interface` id with per-dimension descriptors, so a new
action type can be introduced by describing its dimensions rather than adding a
new learned adapter.

## Recommended Execution Order

Run the tracks in parallel but keep their claims separate:

1. Track A: prepare a fixed-action multitask dataset, with
   `lerobot/metaworld_mt50` as the first candidate.
2. Track A: run LeWM + SIGReg, LeWM with `loss.sigreg.weight=0`, and one
   non-SIGReg anti-collapse baseline if feasible.
3. Track B: create an MT8 candidate manifest for heterogeneous action spaces.
4. Track B: build MT8-A using current `pad_to_max`.
5. Track B: add manifest and sampling accounting, then rerun MT8-A if the
   loader changed.
6. Track B: implement MT8-C as the mature no-global-max controlled baseline.
7. Track B: implement MT8-D and compare against C.
8. Track B: add MT8-B only if MT8-A shows suspicious padding-specific behavior
   that C/D cannot explain.
9. Track B: add E only if D is at least competitive with C.
10. Merge the tracks only after Track A and Track B both have complete local
    conclusions.

Move to MT16 after MT8-C/D have complete anchor evals and Track A has at least
one fixed-action multitask result that establishes the SIGReg baseline.

## Success Criteria For Track A

Track A is considered successful if it answers the regularizer question without
action-dimension confounds:

- LeWM + SIGReg, no-SIGReg, and at least one anti-collapse baseline are trained
  under the same fixed-action multitask data regime;
- per-task validation prediction loss is reported, not only aggregate loss;
- latent collapse diagnostics are recorded for each run;
- task-wise loss imbalance is visible;
- eval success is reported when a reliable eval wrapper exists, otherwise the
  missing eval is explicitly marked as a limitation.

The first claim should be comparative and narrow:

```text
Under fixed-action multitask training, SIGReg improves/does not improve
stability relative to no-SIGReg and the chosen regularization baseline.
```

## Success Criteria For Track B / MT8

MT8 is considered successful if:

- all configured tasks train without loader/action-shape exceptions;
- anchor evals are complete for PushT, Reacher, TwoRoom, and Cube;
- no anchor task drops by more than a predeclared tolerance relative to MT4;
- at least one action representation beyond plain `pad_to_max` has complete
  training and eval metrics;
- per-task exposure and sampling policy are recorded in the evidence tables.

Suggested initial tolerance:

```text
anchor degradation <= 5 percentage points relative to MT4 epoch30
```

This is an operational threshold, not a final scientific claim.

## First-Person Video Role

First-person video should be a side track:

```text
egocentric video -> visual/temporal pretraining -> initialize LeWM encoder
```

Do not mix pure actionless video into the main action-conditioned MT8 objective
until there is a separate auxiliary objective with clear accounting.

Useful comparison:

```text
MT8 from scratch
MT8 with generic pretrained visual encoder
MT8 with egocentric-video-pretrained visual encoder
```

The expected benefit is better visual robustness and sample efficiency, not
direct improvement in action controllability without action labels.

## Reporting Rules

Do not report a single aggregate MT8 score without per-task context.

Every Track A report should include:

- dataset and task count;
- action interface and action dimension;
- regularizer variant;
- per-task train exposure;
- per-task validation prediction loss;
- latent diagnostics;
- eval success if available;
- missing evals or failed seeds.

Every Track B / MT8 report should include:

- task list and dataset paths;
- action representation variant;
- sampling policy;
- per-task train exposure;
- per-task eval success;
- anchor retention table;
- negative-transfer notes;
- missing evals or failed seeds.

## Immediate Next Work Items

1. Track A: inspect and convert `lerobot/metaworld_mt50` into the format needed
   for stable LeWM training.
2. Track A: add no-SIGReg and VCReg/VICReg-style config variants.
3. Track A: add latent diagnostics to the summary artifacts.
4. Track B: create an MT8 candidate manifest.
5. Track B: add a manifest-driven config generator for multitask data items.
6. Track B: add sampling accounting for balanced, proportional, and temperature
   sampling.
7. Track B: implement MT8-C action-space adapter registry.
8. Track B: run MT8-A and MT8-C before starting descriptor-conditioned D.
