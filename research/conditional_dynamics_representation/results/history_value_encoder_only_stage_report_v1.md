# Encoder-only history-value repair: staged decision

> Historical substage report. The current cross-candidate conclusion, including
> the later ActionDelay target-stop-gradient experiment, is maintained in the
> [canonical Conditional Dynamics ICL report](../README.md). This document keeps
> the narrower Encoder-only decision and its evidence contract.

Status: frozen staged conclusion, Development evidence only for the new
candidates. No new candidate Public Test or CEM was opened.

## Decision

The current evidence does **not** support the claim that a generic
Encoder-only SIGReg modification can make LeWM learn every missing ICL ability
while retaining original-task CEM.

The evidence is narrower but useful:

1. terminal-only ConditionalSIGReg can make the missing conditional response
   learnable on Door and greatly improves Action Strength;
2. it misses the predeclared Action Strength resource gate;
3. explicitly removing the irreducible early MSE contrast fixes Door much
   earlier, but barely changes Action Strength;
4. therefore the remaining cross-task bottleneck is not just early target
   collapse. It includes how the Predictor converts readable history into the
   correct future;
5. CEM safety remains an independent empirical question. It was not tested for
   candidates that failed the ICL resource gate.

No training-seed expansion, formal breadth run, Public Test, or candidate CEM
is authorized from this stage.

## Theory boundary

Let `S_E` be encoded history, `C_E` the current query/action condition, and
`Y_E` the encoded future. The squared-loss value of history is

\[
\Gamma_E = R_C^*(E)-R_{S,C}^*(E)
=\mathbb E\|\mathbb E[Y_E\mid S_E,C_E]
-\mathbb E[Y_E\mid C_E]\|^2.
\]

Target pair variance alone does not imply `Gamma_E > 0`. The cross-fitted
history-value audit found that history is often still linearly readable in
failed checkpoints, so “the framewise Encoder simply erased all history” is
not the general root cause. A successful model needs all three links:

1. readable history representation;
2. a usable conditional-future target scale;
3. learned history-to-future Predictor coupling.

There is also an exact conflict at the first response of an exact-condition
pair. For a deterministic common prediction `p`,

\[
\tfrac12(\|p-h_0\|^2+\|p-h_1\|^2)
=\|p-\bar h\|^2+\tfrac14\|h_0-h_1\|^2.
\]

Thus unchanged all-transition online-target MSE directly penalizes the early
history margin. No Encoder-only regularizer can guarantee that margin without
opposing this MSE term. This is a structural boundary, not a SIGReg weight or
seed issue.

For LeWM's training-time Predictor dropout, subtracting the target variance is
not a sound repair: the resulting objective can be negative and unbounded.
The non-negative Bayes-reduced alternative sends both stochastic predictions
to the differentiable pair midpoint,

\[
L_B=\tfrac12(\|p_0-\bar h\|^2+\|p_1-\bar h\|^2).
\]

It makes the direct target-contrast gradient exactly zero while retaining the
target-center, context, Predictor, later-transition, and SIGReg gradients. It
is not detach, stop-gradient, or freezing, but it does change the prediction
target on a provably non-identifiable paired transition and still requires
visible exact-condition pairs.

## Candidate evidence

| Candidate | Door evidence | Action Strength evidence | CEM | Frozen decision |
|---|---|---|---|---|
| Terminal-only ConditionalSIGReg `0.09` | Development-1024: future/history/switch `100/100/100%`; prediction/target pair ratio `0.959`, cosine `0.991` | Development-1024: future `85.55%`, history `87.89%`, switch `95.31%`, worst `76.17%`; history gate required `90%` | Not opened | Reject exact v1 before 4096/Public/CEM |
| Paired-contrast JTCov `0.09` with unchanged MSE | Coupling direction `+110.10`, but history margin `-827.33` and terminal margin `-902.64` per unit LR | Skipped by sequential gradient gate | Not opened | Gradient-falsified before training |
| Bayes-reduced midpoint MSE + terminal ConditionalSIGReg `0.09` | Development-256 mechanism pass: prediction pair `0.4623`, target pair `1.5314`, ratio `0.3019`, cosine `0.6951`; rule-switch `50%` versus `0%` for the unmodified terminal-256 control | Development-256: future `55.47%`, history `86.91%`, switch `91.80%`, worst `36.72%`; failed history `>=90%` and worst `>=40%` | Not opened | Reject before 1024/Public/CEM |

The Bayes-reduced result is especially diagnostic. Relative to unmodified
terminal-only at the same 256-step contract, it changes Action Strength by
only `+0.78 pp` future and `+0.59 pp` history, with switch and worst mode
unchanged. Removing the early irreducible MSE conflict therefore solves a real
Door bottleneck but is not the missing general Action Strength mechanism.

## Why CEM is still unknown

ICL, representation geometry, and prediction loss cannot substitute for real
planning evaluation. Historical target-JTCov is the counterexample: it learned
some ICL abilities, yet only `2/9` original-task CEM comparisons passed the
per-task non-inferiority gate, with large failures on Damping, Action Delay,
Speed, and Reacher.

The hardened CEM protocol now requires:

- the passed Public ICL result, formal training report, and CEM evaluator to
  bind the exact same candidate checkpoint SHA;
- fixed-step recipe, sample exposure, initialization, data manifest, and
  implementation-source receipts;
- paired same-query analysis, stratified by eval seed, with 100,000 bootstrap
  resamples;
- per-task one-sided 95% lower bound at least `-5 pp`, with no cross-task
  averaging;
- an additional `(eval_seed, source_episode_id)` cluster-bootstrap
  sensitivity result;
- both lower and upper bounds, so a failure is classified as confirmed
  degradation or inconclusive rather than automatically called a drop.

Reacher and Cube CEM can be reused only when the candidate checkpoint is
literally byte-identical to the native checkpoint. Any mixed or separately
trained checkpoint requires fresh CEM.

## Execution order for a future candidate

1. Door and Action Strength Development mechanism/ability gates.
2. Fixed-budget formal Action Strength and Door ICL, without test-driven
   checkpoint selection.
3. Fresh CEM on the same checkpoint SHA immediately after each ICL pass.
4. Action Delay ICL plus fresh CEM as the third hard gate.
5. Only after all three hard gates pass: Contact, Damping, Portal, and Speed,
   each with its own CEM decision.
6. Reacher/Cube paired ICL and fresh CEM when those benchmarks are available.
7. Training-seed expansion last.

## Recommended research fork

Do not continue weight sweeps or seed expansion for the exact candidates above.
The strict proposition “unchanged MSE plus a target-Encoder-only Gaussian
statistic is sufficient” is not supported and has a provable early-transition
boundary.

The next method, if pursued, must explicitly address Predictor
history-to-future coupling or change the probabilistic prediction model. That
is a deliberate relaxation of the current Encoder-only contract and should be
named as such. The cheapest empirical alternative is a separately declared
full-budget exploratory terminal-only run, but it cannot retroactively pass the
failed resource protocol and would still need independent ICL confirmation and
fresh CEM.

## Evidence entry points

- `results/history_value_feasibility_v1.json`
- `results/paired_contrast_jtcov_theory_protocol_v1.md`
- `results/paired_contrast_jtcov_gradient_falsification_v1.json`
- `artifacts/history_value_sigreg_door_decisive_dev1024_s3072_r1/`
- `artifacts/history_value_sigreg_action_strength_resource_dev1024_s13313_r2/`
- `artifacts/identifiability_corrected_mse_development256_v2_s3072_13313_r2/aggregate.json`
- `results/history_value_sigreg_cem_guard_v1/`
