# Multitask Transfer Study

This folder records the working evidence package for the stable-worldmodel
multitask-transfer report.

The process follows the CairnLab rule:

```text
No artifact, no claim.
```

Each claim in `claims.yaml` must point to reproducible evidence:

- raw eval metrics under `/opt/huawei/.../checkpoints/*/eval_results/`;
- generated summary tables under `tables/`;
- a parser or verifier script under `scripts/`;
- a CairnLab external-run manifest in `cairn_external_run.yaml`.

Current scope:

- PushT single-task vs three-task LeWM training;
- Reacher single-task vs three-task LeWM training;
- TwoRoom single-task vs three-task LeWM training, including epoch80 as the nearest long-exposure control for MT3 epoch30.

The main comparison has two explicitly separated fairness views:

- **Per-task exposure matched**: compare each task to the single-task epoch
  that saw approximately the same number of task samples.
- **Compute matched**: compare three-task epoch 10 to single-task checkpoints
  with approximately similar total optimizer/sample budget.
