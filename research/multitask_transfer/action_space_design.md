# Action Space Design Notes

This note records the action-space design options for multitask and mixed-task
LeWM training when tasks have different native action dimensions.

## Current State

The four-task Cube extension currently uses `action_dim_policy: pad_to_max`.
For the active task set, PushT/Reacher/TwoRoom use 2D native actions and Cube
uses 5D native actions. Training pads 2D per-step actions to 5D before the
LeWM dataset reshapes frameskip blocks, so the model sees a 25D action block
when `frameskip=5`.

Evaluation keeps the solver and environment in the native action dimension.
Only the cost-model call pads candidate actions to the model action encoder
dimension. This avoids planning for PushT in a fake 5D action space.

This is acceptable as a narrow compatibility patch, but it should not be the
long-term abstraction.

## Why Plain Padding Is Not Enough

Plain zero padding has useful properties:

- It is simple and keeps the current model architecture unchanged.
- It works for the current small gap between 2D and 5D actions.
- When actions are z-score normalized, zero padded channels often correspond
  to an average normalized value, which is less destabilizing than arbitrary
  constants.
- It preserves native solver dimensions when padding is applied only at the
  model interface.

The long-term issues are more important:

- Missing action dimensions and real zero-valued actions share the same value.
- The model can infer task identity from the padding pattern instead of
  learning transferable dynamics.
- Padding to the global maximum wastes capacity as action spaces diverge.
- Normalization, CEM candidate generation, and action encoders become coupled
  to an artificial global action width.
- Dimension matching is not semantic matching. A 2D planar push action and a
  5D Cube action are not the same control interface with missing coordinates.

The immediate consequence is that `pad_to_max` should remain a baseline and
compatibility mode, not the primary design for broadly mixed action spaces.

## Design Goals

The action representation should satisfy these constraints:

- Native environments and solvers operate in native action spaces.
- Dataset loading records action metadata instead of only a dense tensor.
- The model has an explicit way to distinguish valid action channels from
  missing channels.
- Shared dynamics modules receive a consistent representation.
- Adding a task with a different action dimension should not require changing
  every existing task config.
- The representation should support ablations from low-risk to high-risk
  changes.

## Relationship To The SIGReg Multitask Track

Action unification and SIGReg multitask scaling are separate questions.

The SIGReg track should first use a fixed action interface, such as MetaWorld
MT49/MT50-style 4D actions, so it can test whether SIGReg itself remains useful
under multitask mixing.

This action-space track should do the opposite: keep the regularizer fixed,
initially with default LeWM SIGReg, and vary only how native actions become
the model's shared `act_emb`.

The two tracks answer different failure modes:

| Track | Held fixed | Varied | Failure means |
| --- | --- | --- | --- |
| SIGReg multitask scaling | action interface | regularizer | SIGReg may not stabilize mixed-task representations. |
| Action unification | regularizer | action embedding | the shared action input is not expressive or robust enough. |

Do not treat a heterogeneous-action failure as evidence against SIGReg until a
fixed-action multitask SIGReg baseline has been run. Do not treat a fixed-action
SIGReg result as evidence that the project has solved mixed action spaces.

After both tracks have conclusions, the technical merge target is:

```text
SIGReg-stabilized multitask LeWM
  + descriptor-conditioned or effect-aligned universal action embedding
```

## External Patterns

### Structured Action Specs

Gymnasium treats action and observation spaces as first-class objects. Spaces
can be continuous `Box`, discrete `Discrete`/`MultiDiscrete`, or structured
containers such as `Dict`, `Tuple`, `Sequence`, and `OneOf`; it also provides
flattening and unflattening utilities for learning code.

This is an engineering pattern rather than a model architecture: store action
space metadata, validate samples, and flatten only at the boundary where a
network requires a dense tensor.

### Padding With Masks

The smallest improvement over current padding is:

```text
native_action -> normalize native dims -> pad to max dim
              -> action_mask marks valid channels
```

The model receives both padded action and mask. Any action encoder must either
multiply by the mask before projection or use the mask as an input feature. The
core invariant is:

```text
Changing invalid padded channels must not change model cost.
```

This gives a direct test for whether the model is using padded channels.

### Task-Specific Action Adapters

A stronger and still practical design is to keep native actions until they
enter a task-specific adapter:

```text
action_native[task] -> ActionAdapter(task) -> shared_action_embedding
```

The shared world model consumes `shared_action_embedding`, not padded raw
actions. For the current LeWM cost-model interface, this means replacing the
single global action encoder with a registry keyed by task label or action
space id.

This design has the best fit for the current codebase:

- It preserves native action dimensions for solver and env code.
- It avoids wasting width on global max-dimensional padding.
- It makes task identity explicit instead of leaking it through padding.
- It lets us ablate shared vs task-specific parameters cleanly.
- It can start as a thin wrapper around the existing action encoder.

The cost is that batches need task ids, and the model must support grouped or
per-sample adapter dispatch.

### Shared Semantic Action Schema

For related robot tasks, another common approach is a semantic action schema:

```text
dx, dy, dz, droll, dpitch, dyaw, gripper, terminate, ...
```

Each task maps native controls into this schema and carries a validity mask.
This is useful when tasks share embodiment semantics, such as end-effector
delta pose and gripper state. It is weaker for unrelated tasks where a channel
index has task-specific meaning.

For our current task set, a semantic schema is only partly justified. PushT and
Cube are both control tasks, but a planar 2D push action and a 5D Cube action
do not clearly share enough actuator semantics to make a global schema the
first implementation target.

### Action Tokenization

Generalist agents and VLA systems often turn actions into tokens. Gato uses a
single network across many modalities and embodiments by emitting tokens for
text, button presses, torques, and other action-like outputs. RT-2 expresses
robot actions as text tokens so robotic action prediction and language output
share a sequence modeling interface. FAST targets the weakness of naive
per-dimension binning by compressing action sequences in frequency space and
advertises a tokenizer usable across diverse action spaces and control
frequencies.

This is powerful for autoregressive policy models, but it is a large change
for the current LeWM setup. Our planner proposes continuous action candidates
and asks the world model for a cost. Tokenizing actions would require either a
token-space planner or a decoder from tokens back to continuous candidates.

### Effect-Aware Action Embeddings

A closer precedent for this project is to learn action embeddings with dynamics
supervision. For example, DCT trains action embeddings with both action
reconstruction and expected future-state prediction. The key lesson is that an
action embedding for planning should not be optimized only to encode raw action
values; it should also be predictive of state transitions.

### Continuous Generative Action Heads

Recent robot foundation models also use diffusion or flow-matching action
decoders. pi0 is a representative direction: a VLM backbone plus a flow-based
action expert for continuous robot control across multiple embodiments.

This is attractive for high-frequency dexterous control and policy generation,
but it is not a drop-in replacement for the current CEM-plus-cost-model loop.
It changes the system from "search actions, score with world model" toward
"generate action chunks from a policy/action decoder."

## Recommended Roadmap And Ablations

The central design decision is to keep the world model interface fixed:

```text
obs_embedding + action_embedding -> predictor -> future obs_embedding
```

Current LeWM/PLDM already have this slot: both call
`self.action_encoder(info['action'])` and pass the result as `act_emb` into the
predictor. The question is how to compute `act_emb` from native actions when
action spaces differ.

The current planning set is:

| Variant | Uses global max action dim | Needs new action-space id/table for a new type | Uses multiple encoders | Role |
| --- | --- | --- | --- | --- |
| A. `pad_to_max` | yes | no | no | current baseline |
| B. `pad_to_max + mask` | yes | no | no | optional padding diagnostic |
| C. action-space adapters | no | yes | yes | mature fallback baseline |
| D. descriptor-conditioned universal embedder | no | no, if descriptors are open features | no | main research path |
| E. descriptor embedder + inverse-effect alignment | no | no, if descriptors are open features | no | strongest proposed path |

### Stage 0: Keep Current Baseline As A Control

Keep `pad_to_max` as an explicit baseline. It is useful for the current MT4
experiment and for verifying that mixed action dimensions do not break the
training/eval path.

Do not generalize claims from this mode to highly heterogeneous action spaces.

### Stage 1: Optional Action Masks As Diagnostic B

Add `action_mask` to multitask samples and to the model action encoder path.

Expected behavior:

- `action_mask.shape[-1] == padded_per_step_action_dim`
- valid native channels are 1
- padded channels are 0
- cost does not change when invalid channels are perturbed

Suggested test:

```text
For a PushT batch in an MT4 model:
  cost_a = get_cost(action padded with zeros)
  cost_b = get_cost(same native actions, random values in invalid channels)
  assert close(cost_a, cost_b)
```

This is the lowest-risk fix for the semantic ambiguity in padding, but it still
depends on a global max action dimension. It should be skipped in the main
MT8 path unless MT8-A shows suspicious padding-specific behavior.

### Stage 2: Introduce Action Descriptors

Add task-level action metadata to config or dataset construction. Avoid making
`action_space_id` the core abstraction because it still creates an expanding
enumeration. Prefer per-dimension descriptors:

```yaml
action_descriptors:
  - name: pusher_delta_x
    target: pusher
    control: delta_position
    axis: x
    frame: workspace
    unit: normalized
  - name: pusher_delta_y
    target: pusher
    control: delta_position
    axis: y
    frame: workspace
    unit: normalized
```

For Cube:

```yaml
action_descriptors:
  - name: end_effector_delta_x
    target: end_effector
    control: delta_position
    axis: x
    frame: workspace
    unit: normalized
  - name: end_effector_delta_y
    target: end_effector
    control: delta_position
    axis: y
    frame: workspace
    unit: normalized
  - name: end_effector_delta_z
    target: end_effector
    control: delta_position
    axis: z
    frame: workspace
    unit: normalized
  - name: wrist_or_gripper_command_0
    target: controller
    control: task_specific_continuous
    axis: none
    frame: controller
    unit: normalized
  - name: wrist_or_gripper_command_1
    target: controller
    control: task_specific_continuous
    axis: none
    frame: controller
    unit: normalized
```

The loader should expose at least:

- `task_label`
- `native_action_dim`
- `action_descriptors`
- `padded_action_dim` and `action_mask`, only when padding baselines are used

This makes the contract explicit even before architecture changes.

## Variant C: Action-Space Adapter Baseline

This is the mature baseline. It is not the preferred long-term abstraction, but
it is the clearest comparator for whether descriptor-conditioned embeddings are
actually useful.

C is a controlled-dataset baseline, not an open-world action solution. It
assumes the dataset manifest or eval wrapper knows the action interface. This
is reasonable for curated MT8/MT16 experiments because every dataset already
has an env wrapper, action dimension, normalizer, and eval task. It is not
enough for a new unknown action stream with no metadata.

Replace the single raw action encoder assumption with:

```text
ActionAdapterRegistry:
  pusht_xy -> MLP(2 * frameskip -> action_embed_dim)
  reacher_xy -> MLP(2 * frameskip -> action_embed_dim)
  tworoom_xy -> MLP(2 * frameskip -> action_embed_dim)
  cube_5d -> MLP(5 * frameskip -> action_embed_dim)
```

The shared world model receives `action_embed_dim`. The solver still samples
native actions. Evaluation no longer needs `ActionPaddedCostModel`; it passes
native candidates plus the action interface key used to select the adapter.

Adapters do not need separate pretraining. Train them end-to-end with the
world model:

```text
native action
  -> selected adapter[action_interface]
  -> shared act_emb
  -> shared predictor
  -> prediction loss
```

Backpropagation updates the selected adapter and the shared world-model
parameters in the same optimizer step. This is equivalent to a multitask model
with interface-specific input heads and a shared trunk.

Important design choice:

- If PushT/Reacher/TwoRoom action semantics are close enough, they can share an
  adapter id.
- If we want to avoid hidden task leakage, adapter ids should correspond to
  action semantics, not arbitrary task labels.

Concrete implementation:

1. Add an `action_interface` field to each multitask item. This can be explicit
   in config or inferred from descriptors.
2. Add `stable_worldmodel/wm/common/action_embedding.py` with an
   `ActionAdapterRegistry` module.
3. Replace `action_encoder._target_` in a new config with the registry. Each
   registry entry can initially reuse the existing `Embedder`.
4. Modify LeWM/PLDM calls from:

   ```python
   self.action_encoder(info['action'])
   ```

   to:

   ```python
   self.action_encoder(info['action'], info.get('action_interface'))
   ```

5. Use homogeneous mini-batches by `action_interface` first. This avoids
   custom ragged collation and lets native action tensors keep their real last
   dimension. A later implementation can group mixed batches by interface
   inside the registry.
6. Keep eval native: the solver proposes native candidates, and `get_cost`
   provides the interface key through `info_dict`.

Eval safety:

```text
solver samples native action candidates
model.get_cost(info_dict, native_action_candidates)
action_encoder selects adapter using info_dict["action_interface"]
model returns cost
```

This avoids planning PushT in a fake Cube-sized action space. The only eval
requirement is that the eval task supplies the same action interface metadata
used at training time.

Success criterion:

```text
C should match or beat pad_to_max while removing global max_action_dim.
```

Failure mode:

```text
C can over-specialize because action-interface routing gives the model an
explicit task/action-space split.
```

Open-world limitation:

```text
action_interface = ee_delta_5d
```

is an expanding identifier. A new action type requires either mapping it to an
existing interface or adding a new adapter. If no one can say what the action
numbers mean, C cannot infer that from raw values alone. This is why C should
be treated as a mature no-padding baseline rather than the final generalist
action representation.

## Variant D: Descriptor-Conditioned Universal Embedder

This is the preferred research direction. It keeps one shared action encoder
and avoids both global max action dimensions and expanding action-space id
tables.

D replaces the closed `action_interface` id with open per-dimension
descriptors. The required metadata changes from:

```text
action_interface: ee_delta_5d
```

to:

```text
dimension 0: end-effector delta x in workspace frame
dimension 1: end-effector delta y in workspace frame
...
```

This is still not "metadata-free"; raw action values have no universal meaning
by themselves. The difference is that a new action type can be described
compositionally instead of adding a new learned id table or adapter.

The interface is:

```text
UniversalActionEmbedder(action_values, action_descriptors) -> act_emb
```

The model receives the same `act_emb` shape as today:

```text
act_emb: (B, T, embed_dim)
```

The difference is that `action_values` remain native and variable-dimensional
before embedding.

Concrete representation:

```text
For each sample, time step, frameskip step, and native action dimension:
  scalar item = {
    value,
    time_offset,
    frameskip_offset,
    descriptor_features
  }
```

The shared embedder computes:

```text
descriptor_embedding = descriptor_encoder(descriptor_features)
scalar_token = scalar_mlp([value, time_offset, frameskip_offset, descriptor_embedding])
act_emb[b, t] = pool_or_attention({scalar_token for this b,t action block})
```

Descriptor features should be open rather than an expanding id table. Good
options:

- A frozen text encoder over canonical descriptor strings, followed by a small
  trainable projection.
- Deterministic hashed features over normalized descriptor fields.
- A structured numeric/category vector only if the category vocabulary is
  stable and treated as part of the schema.

Do not use a learned `action_space_id` table as the core D variant. That would
turn D back into a lighter form of C.

Concrete implementation:

1. Add `action_descriptors` to each multitask item in config.
2. Add a descriptor parser that converts each descriptor into a fixed feature
   vector. The first version can be deterministic and cached at dataset
   construction time.
3. Add a custom collate path for native variable-dimensional action blocks.
   Two implementation levels are possible:

   - D1: homogeneous mini-batches by descriptor signature. Simpler, no ragged
     tensor collation.
   - D2: packed scalar collation. Store flat scalar tokens plus
     `(batch_idx, time_idx)` indices and scatter/pool back to `(B, T, D)`.

4. Implement `DescriptorActionEmbedder` with signature:

   ```python
   forward(action, action_descriptors, action_index=None) -> act_emb
   ```

   For D1, `action` can be `(B, T, frameskip, native_dim)`. For D2, `action`
   can be packed scalars and `action_index` maps tokens back to `(B, T)`.
5. Keep LeWM/PLDM predictor unchanged. Only the action encoder call changes.
6. In eval, the solver still proposes native actions. The environment wrapper
   or eval config supplies the same descriptors used during training.

Success criterion:

```text
D should match C without action-space-specific parameters and should beat A
when new action dimensions are introduced.
```

Failure modes:

- Poor descriptors make unrelated dimensions look similar.
- Text descriptors can overfit wording unless canonicalized.
- If descriptors are too generic, the model may learn a weak average action
  representation.

## Variant E: Descriptor Embedder With Inverse-Effect Alignment

Variant E keeps D's interface but adds an auxiliary objective so the action
embedding is shaped by observed transition effects, not just by raw action
values.

E is the open-world-ish extension of D. It still benefits from descriptors, but
it can use a small amount of transition data from a new interface to calibrate
the action embedding toward observed effects instead of relying only on manual
descriptor quality.

Motivation:

```text
The world model cares about what the action does, not just what the action
number looks like.
```

Add an inverse-effect encoder:

```text
EffectEncoder(z_t, z_{t+1}) -> effect_emb
```

where `z_t` and `z_{t+1}` are observation embeddings from the existing image
encoder/projector. Then align:

```text
DescriptorActionEmbedder(a_t, descriptors) ~= EffectEncoder(z_t, z_{t+1})
```

Recommended losses:

```text
L_total = L_world_model
        + lambda_align * L_effect_align
        + lambda_var * L_embedding_regularization
        + optional lambda_recon * L_action_reconstruction
```

Use cosine or normalized MSE for `L_effect_align`:

```text
L_effect_align = 1 - cosine(project_action(act_emb), effect_emb)
```

Add a contrastive version once the simple loss is stable:

```text
positive: action embedding and transition effect from the same (b,t)
negative: transition effects from other samples in the batch
```

Concrete implementation:

1. Implement D first and keep `act_emb` available in `info`.
2. Add `EffectEncoder`, for example an MLP over
   `[z_t, z_{t+1}, z_{t+1} - z_t]`.
3. In `LeWM.encode`, after `info['emb']` and `info['act_emb']` are available,
   compute effect embeddings for adjacent observation pairs:

   ```text
   z_t = emb[:, :-1]
   z_next = emb[:, 1:]
   effect_emb = effect_encoder(z_t, z_next)
   action_emb = act_emb[:, :-1]
   ```

4. Add auxiliary loss outputs to the model or trainer. Keep the first version
   simple: return a dict containing `world_model_loss` and
   `action_effect_align_loss`.
5. Use stop-gradient carefully. A safe first version is:

   ```text
   effect_encoder(stopgrad(z_t), stopgrad(z_next))
   align projected action_emb to effect_emb
   ```

   This prevents the auxiliary loss from distorting the visual representation
   before the basic action embedder is known to work.
6. Keep action reconstruction optional. It is useful only if we later need to
   decode embeddings back into native actions. For CEM cost evaluation, the
   planner already proposes native actions, so reconstruction is not required.

Success criterion:

```text
E should improve D on transfer and should make nearest-neighbor action
embeddings cluster by transition effect rather than by dataset or raw action
dimension.
```

Failure modes:

- Collapse if the alignment loss is too strong.
- Visual transition ambiguity: different actions can produce similar image
  changes under partial observability.
- Effect alignment can learn task-specific shortcuts if descriptors are weak.

## Stage 4: Consider Semantic Schema Or Tokenization

Only move beyond adapters when the target task set justifies it:

- Use a semantic schema when most tasks are robot-control variants with shared
  end-effector/gripper semantics.
- Use action tokenization when the model becomes an autoregressive generalist
  policy or we need to mix discrete, text-like, and continuous actions.
- Use diffusion/flow action heads when the system objective shifts from
  planning with a learned cost to directly generating action chunks.

## Decision For The Current Project

The immediate engineering baseline remains:

```text
MT8-A: pad_to_max
```

The mature fallback baseline is:

```text
C. native action -> action-space adapter -> shared action embedding
```

C should be trained jointly with LeWM/PLDM, not as a separate pretraining job.
It is safe for eval as long as eval metadata provides `action_interface`, but
it is not an open-world solution for unknown action streams.

The main research path is:

```text
D. native action + action descriptors -> universal action embedding
E. D + inverse-effect alignment
```

`pad_to_max + action_mask` remains available only as a diagnostic if plain
padding appears to be the source of a failure. The main path should go directly
from A to C so the project stops relying on global `max_action_dim` sooner.

This path keeps the current LeWM/PLDM predictor contract unchanged: the
predictor still receives a fixed `act_emb`. It also avoids global
`max_action_dim`, avoids per-action-space encoders in the proposed method, and
keeps CEM/eval in native action spaces.

## References

- Gymnasium spaces documentation: https://gymnasium.farama.org/api/spaces/
- DCT action embeddings: https://arxiv.org/abs/2306.15913
- Gato, "A Generalist Agent": https://arxiv.org/abs/2205.06175
- RT-2, "Vision-Language-Action Models Transfer Web Knowledge to Robotic Control": https://arxiv.org/abs/2307.15818
- Open X-Embodiment and RT-X: https://arxiv.org/abs/2310.08864
- Octo, "An Open-Source Generalist Robot Policy": https://arxiv.org/abs/2405.12213
- OpenVLA: https://arxiv.org/abs/2406.09246
- pi0, "A Vision-Language-Action Flow Model for General Robot Control": https://arxiv.org/abs/2410.24164
- FAST action tokenization: https://arxiv.org/abs/2501.09747
