# Multitask Transfer Study

This folder records the working evidence package for the stable-worldmodel
multitask-transfer report.

The process follows the CairnLab rule:

```text
No artifact, no claim.
```

Each claim in `claims.yaml` must point to reproducible evidence:

- raw eval metrics under `/home/ag/dataset/ag_data/outputs/stablewm/checkpoints/*/eval_results/`
  or another root selected by `STABLEWM_CHECKPOINT_ROOT`;
- generated summary tables under `tables/`;
- a parser or verifier script under `scripts/`;
- a CairnLab external-run manifest in `cairn_external_run.yaml`.

`tables/eval_runs.csv` preserves the `weights_path` reported inside each raw
metrics payload for provenance, but model weights are not attached evidence.

Current scope:

- PushT single-task vs three-task LeWM training;
- Reacher single-task vs three-task LeWM training;
- TwoRoom single-task vs three-task LeWM training, including epoch80 as the nearest long-exposure control for MT3 epoch30.
- Four-task LeWM training with Cube added through action padding, focused on
  whether Cube creates sustained negative transfer on PushT.

The main comparison keeps these views separate:

- **Per-task exposure matched**: compare each task to the single-task epoch
  that saw approximately the same number of task samples.
- **Compute matched**: compare three-task epoch 10 to single-task checkpoints
  with approximately similar total optimizer/sample budget.
- **Four-task training progress**: compare MT4 epoch 4/10/30 against MT3 and
  PushT single-task controls; epoch 4 is diagnostic undertraining evidence, not
  a final transfer claim.

Design notes:

- [action_space_design.md](action_space_design.md): action-space options for
  mixed-task training, including why `pad_to_max` is only a baseline and why
  the main path should go from `pad_to_max` to action-space adapters and
  descriptor-conditioned universal action embeddings. This track keeps SIGReg
  fixed while varying the action representation. `pad_to_max + action_mask` is
  kept only as an optional padding diagnostic.
- [scaling_plan.md](scaling_plan.md): MT8-first scaling ladder and action
  representation ablation plan for MT8, MT16, and beyond. It now separates the
  work into two parallel tracks: SIGReg multitask scaling under a fixed action
  interface, and heterogeneous action unification under fixed SIGReg.
