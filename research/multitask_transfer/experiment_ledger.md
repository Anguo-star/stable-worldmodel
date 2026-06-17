# Experiment Ledger

## Research Question

Does three-task LeWM training produce useful transfer across PushT, Reacher,
and TwoRoom without introducing meaningful negative transfer?

## Current Hypotheses

1. Reacher receives positive transfer from three-task training.
2. PushT does not show meaningful negative transfer under per-task exposure
   matching, but single-task training wins under compute-matched longer
   training.
3. TwoRoom does not show meaningful negative transfer under the available
   epoch10 exposure-matched control or the epoch30 long-exposure control using
   the nearest single-task epoch80 checkpoint.
4. Balanced multitask sampling creates unequal per-task exposure because
   smaller datasets are repeated to match the largest dataset.

## Raw Evidence Locations

Checkpoint root:

```text
/opt/huawei/explorer-env/dataset/ag_data/outputs/stablewm/checkpoints
```

Primary runs:

| Run id | Directory | Notes |
| --- | --- | --- |
| `mt3_epoch10` | `lewm_mt3_lance` | Three-task model trained for 10 epochs. |
| `pusht_single` | `lewm_pusht_lance` | PushT single-task model, eval at epoch 10/30/33. |
| `reacher_single` | `lewm_reacher_lance` | Reacher single-task model, eval at epoch 10/33. |
| `tworoom_single` | `lewm_tworoom_lance` | TwoRoom single-task model, eval at epoch 30/80. |

Evaluation seeds:

```text
42, 43, 44
```

Episodes per seed:

```text
50
```

## Current Results

Regenerate this table with:

```bash
python research/multitask_transfer/scripts/summarize_eval_metrics.py
```

Generated outputs:

- `tables/eval_runs.csv`
- `tables/eval_summary.csv`
- `tables/eval_summary.md`
- `tables/exposure_equivalence.csv`
- `tables/exposure_equivalence.md`

## Interpretation Rules

- A claim about "positive transfer" must compare against a single-task control
  and name the fairness view.
- A claim about "no negative transfer" means the three-task score is within
  the observed seed-level variation of the single-task exposure-matched
  control. It is not the same as proving superiority.
- A compute-matched comparison is not a per-task exposure-matched comparison.
- TwoRoom epoch10 exposure-matched evidence is linked, and MT3 epoch30 is
  compared to the nearest available long-exposure single-task checkpoint
  epoch80.
