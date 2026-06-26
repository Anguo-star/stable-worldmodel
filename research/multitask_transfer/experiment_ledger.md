# Experiment Ledger

## Research Question

Does multitask LeWM training produce useful transfer across PushT, Reacher,
TwoRoom, and the four-task Cube extension without introducing meaningful
negative transfer?

## Current Hypotheses

1. Reacher receives positive transfer from three-task training.
2. PushT does not show meaningful negative transfer under per-task exposure
   matching, but single-task training wins under compute-matched longer
   training.
3. Adding Cube in four-task training does not currently show meaningful
   sustained negative transfer on PushT; epoch 4 is undertrained, but epoch 30
   largely closes the gap to PushT single-task and exceeds MT3 epoch 30.
4. TwoRoom does not show meaningful negative transfer under the available
   epoch10 exposure-matched control or the epoch30 long-exposure control using
   the nearest single-task epoch80 checkpoint.
5. Balanced multitask sampling creates unequal per-task exposure because
   smaller datasets are repeated to match the largest dataset.

## Raw Evidence Locations

Checkpoint root:

```text
/home/ag/dataset/ag_data/outputs/stablewm/checkpoints
```

The summary script also accepts `STABLEWM_CHECKPOINT_ROOT` and has legacy
fallback roots for older machines.

Primary runs:

| Run id | Directory | Notes |
| --- | --- | --- |
| `mt3_epoch10` | `lewm_mt3_lance` | Three-task model trained for 10 epochs. |
| `mt4_epoch10_epoch30` | `lewm_mt4_lance` | Four-task model with Cube, action padding, and eval at epoch 4/10/30. |
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

Key four-task PushT readout:

| Comparison | Success rate | Aggregate |
| --- | ---: | --- |
| MT4 PushT epoch 4 | 11.33% | 17/150 |
| MT4 PushT epoch 10 | 76.67% | 115/150 |
| MT3 PushT epoch 10 | 78.67% | 118/150 |
| PushT single epoch 10 | 80.00% | 120/150 |
| MT4 PushT epoch 30 | 88.67% | 133/150 |
| MT3 PushT epoch 30 | 84.00% | 126/150 |
| PushT single epoch 30 | 90.67% | 136/150 |
| PushT single epoch 33 | 90.67% | 136/150 |

## Interpretation Rules

- A claim about "positive transfer" must compare against a single-task control
  and name the fairness view.
- A claim about "no negative transfer" means the three-task score is within
  the observed seed-level variation of the single-task exposure-matched
  control. It is not the same as proving superiority.
- A compute-matched comparison is not a per-task exposure-matched comparison.
- A four-task early checkpoint is training-progress evidence. The PushT
  transfer claim should use epoch 30 for the sustained effect and can cite
  epoch 4/10 only to show convergence lag.
- TwoRoom epoch10 exposure-matched evidence is linked, and MT3 epoch30 is
  compared to the nearest available long-exposure single-task checkpoint
  epoch80.
