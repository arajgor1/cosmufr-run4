---
license: mit
library_name: pytorch
tags:
  - cosmology
  - astrophysics
  - scientific-ml
  - parameter-inference
  - power-spectrum
pipeline_tag: tabular-regression
---

# CosmUFR Run 4

**A 136,194,617-parameter point predictor of eight cosmological parameters from
the matter power spectrum, released with an audit of its training defects.**

- Code and reproduction: https://github.com/arajgor1/cosmufr-run4
- Checkpoint SHA256: `5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1`

> **Read this first.** The network was designed to refine its answer over sixteen
> steps. In this checkpoint the encoder, the belief proposal and the learned
> step-size and preconditioner networks are bit-identical to their pre-training
> state, and the code that trained it gives them no gradient. The loop still runs,
> following the gradient of trained energy heads, and changes each answer slightly;
> a predictive benefit has not been established. Earlier published figures for
> this model (Ω_m 0.907, σ₈ 0.911, h 0.604) are superseded and should not be
> cited.

Numbers below are labelled **reproduced** (regenerated from this checkpoint and
the bundled benchmark), **saved report** (read from a report saved earlier, not
rerun) or **diagnostic** (measured by an audit run outside this package). Every
table is generated from `reports/results_v1.json` in the code repository.

---

## What this is

A network that maps `log10 P(k)` at two redshifts to eight cosmological
parameters in a single forward pass, trained on emulator-generated spectra. Its
design encodes the spectrum into a 1024-dimensional belief, refines it over
sixteen steps of descent on a learned energy, and reads parameters off the result.
Whether that refinement helps is the research question; this release does not
test it, because the refinement did not train.

## Model details

| | |
|---|---|
| Developed by | Aaditya Rajgor |
| Model type | Feed-forward point predictor followed by a 16-step energy-based refinement loop whose update networks were never trained; no attention |
| Parameters | 136,194,617 |
| Inputs | `log10 P(k)`, 200 log-spaced k bins over k ∈ [0.1, 4.5] h/Mpc, at z = 0 and z = 0.47 |
| Outputs | 8 cosmological parameters. Also 8 σ values and a P(k) reconstruction, neither validated: see limitations |
| Precision | float32 |
| Latency | a few hundred milliseconds per spectrum on CPU |
| Determinism | bit-identical across repeated runs on the same environment (reproduced) |
| Checkpoint | epoch 30, phase 4, run `b200-b4096-run4-20260414_031247` |
| License | MIT |

Parameters, in output order: Ω_m, σ₈, h, n_s, Ω_b, w₀, Σm_ν, w_a.

## What the audit found

<!-- table:modules -->
Diagnostic, run `pre-outreach-m01-gradient-diagnosis-20260913_030043`. Gradient columns count batches, of 20, in which the module received a finite non-zero gradient from the revision's own training step. `attractor_bank` is updated by moving average, not by gradient.

| Module | What it does | Parameters | Identical to Run 2 before any step | Gradient in Run 4 training code | Current code, k_backprop 4 | Current code, k_backprop 16 |
|---|---|---:|---:|---:|---:|---:|
| `gen_head` | redraws P(k) from the belief | 69,316,610 (50.9%) | 0/70 tensors | 20/20 | 20/20 | 20/20 |
| `belief_proposal` | forms the first belief | 11,563,008 (8.5%) | 38/38 tensors | 0/20 | 0/20 | 20/20 |
| `belief_proposal_seq` | sequential proposal, training-only sequential path | 11,563,008 (8.5%) | — | 20/20 | 20/20 | 20/20 |
| `obs_encoder` | reads the two spectra into a 1024-d vector | 9,875,456 (7.2%) | 38/38 tensors | 0/20 | 20/20 | 20/20 |
| `obs_encoder_single` | single-redshift encoder, training-only sequential path | 9,670,656 (7.1%) | 0/38 tensors | 20/20 | 20/20 | 20/20 |
| `settling` | refines the belief over 16 steps | 9,459,728 (7.0%) | 128/128 tensors | 0/20 | 0/20 | 0/20 |
| `attractor_bank` | prototype bank, updated by moving average | 4,194,304 (3.1%) | 0/1 tensors | — | — | — |
| `obs_energy_head` | energy term: fit to the observation | 2,105,345 (1.6%) | 0/22 tensors | 20/20 | 20/20 | 20/20 |
| `dyn_energy_head` | energy term: movement between beliefs | 2,105,345 (1.6%) | 0/22 tensors | 20/20 | 20/20 | 20/20 |
| `halo_head` | side output, not used for the parameters | 1,590,804 (1.2%) | 11/22 tensors | 0/20 | 0/20 | 0/20 |
| `param_head` | computes the eight reported parameters | 1,584,648 (1.2%) | 2/24 tensors | 20/20 | 20/20 | 20/20 |
| `unc_head` | reports a σ per parameter | 1,584,648 (1.2%) | 0/22 tensors | 20/20 | 20/20 | 20/20 |
| `constraint_head` | energy term: learned constraint score | 1,581,057 (1.2%) | 0/22 tensors | 20/20 | 20/20 | 20/20 |
<!-- /table:modules -->

- **Unchanged modules.** The encoder, belief proposal and refinement networks
  (22.7% of parameters) are bit-identical to Run 2's checkpoint saved before any
  optimizer step.
- **Why.** In the code that trained Run 4, the refinement loop detaches the
  belief at every step and computes its step size and preconditioner without
  gradients. Later training code reconnects the encoder, and the proposal only
  when every step is retained. The step-size and preconditioner networks receive
  no gradient in any configuration examined.
- **The loop still runs.** Each step follows the gradient of the trained energy
  heads through the untrained step-size and preconditioner networks. On all 6,000
  benchmark rows it changes every prediction slightly (belief movement median
  0.09%). Whether that helps has not been tested.
- **The energy objective is unbounded below.** It adds the mean energy to a
  difference-only contrastive term, so a constant downward shift lowers the loss
  one-for-one. A diagnostic shift of 1,000 lowered it by exactly 1,000. Run 4's
  logged energy loss fell to −942,080 and stayed there; the log does not show
  which term produced that value.
- **The bias audit is a clue, not proof.** `cosmufr.weight_audit` reports modules
  whose Linear biases are all still zero. Biases are initialised to zero, so this
  is consistent with no update, but `halo_head` has zero biases and uniformly
  rescaled weights.

## Intended use

Research and teaching:

- a worked example of how an iterative-inference design can fail to train while
  the loss curve looks healthy, and how to audit a checkpoint for it;
- a deterministic, checkable point-prediction baseline for `P(k)` → parameters on
  emulator spectra.

**Not** for producing cosmological constraints. The input is not an observable,
the σ output is not validated, and there is no posterior.

## Results

<!-- table:validation_metrics -->
Saved report, not rerun: 162,795 rows of the deterministic validation split (16 source ids), generated 20260904_140841. It needs the master validation rows and `bad_indices.npy` to regenerate.

| Parameter | R², all rows | R², rows where it varies | RMSE, all rows | RMSE, rows where it varies | Rows where it varies |
|---|---:|---:|---:|---:|---:|
| Ω_m | 0.717 | 0.720 | 0.0273 | 0.0271 | 162,733 |
| σ₈ | 0.756 | 0.757 | 0.0285 | 0.0284 | 162,733 |
| h | 0.501 | 0.498 | 0.0402 | 0.0409 | 156,732 |
| n_s | 0.338 | 0.339 | 0.0214 | 0.0215 | 161,733 |
| Ω_b | 0.364 | 0.363 | 0.0045 | 0.0046 | 156,732 |
| w₀ | 0.586 | 0.614 | 0.0254 | 0.0578 | 29,248 |
| Σm_ν (eV) | 0.407 | 0.011 | 0.0993 | 0.1146 | 81,991 |
| w_a | 0.165 | 0.185 | 0.0616 | 0.1576 | 24,247 |

Verdicts (R2 where the parameter varies: above 0.7 recovered, 0.25 to 0.7 partial, below 0.25 not recovered): Ω_m recovered, σ₈ recovered, h partial, n_s partial, Ω_b partial, w₀ partial, Σm_ν not recovered, w_a not recovered.
<!-- /table:validation_metrics -->

<!-- table:benchmark_metrics -->
Reproduced: the bundled 6,000-row benchmark, run `pre-outreach-m00-reproduce-20260913_030009` (torch 2.14.0+cpu, CPU). Agreement with an independent reproduction: largest R² difference 4.8e-06, largest prediction difference 3.1e-05; repeat run bit-identical: yes.

| Parameter | R², all rows | R², rows where it varies | RMSE, all rows | RMSE, rows where it varies | Rows where it varies |
|---|---:|---:|---:|---:|---:|
| Ω_m | 0.687 | 0.689 | 0.0288 | 0.0284 | 5,959 |
| σ₈ | 0.738 | 0.737 | 0.0295 | 0.0293 | 5,959 |
| h | 0.475 | 0.477 | 0.0413 | 0.0418 | 5,791 |
| n_s | 0.331 | 0.336 | 0.0219 | 0.0218 | 5,959 |
| Ω_b | 0.353 | 0.353 | 0.0046 | 0.0046 | 5,791 |
| w₀ | 0.599 | 0.668 | 0.0245 | 0.0510 | 1,078 |
| Σm_ν (eV) | 0.410 | 0.023 | 0.0990 | 0.1144 | 3,028 |
| w_a | 0.148 | 0.174 | 0.0632 | 0.1572 | 910 |
<!-- /table:benchmark_metrics -->

R² is a ratio against the variance of the truth, so where a parameter is held
fixed it measures nothing. For Σm_ν the difference is decisive: about 0.41 on all
rows and 0.01 on rows where it varies. **This model does not recover neutrino
mass.**

These are point-prediction errors on noiseless emulator spectra and are not
comparable to survey posterior widths.

### Per source

<!-- table:validation_per_source -->
Saved report, R² per source. "fixed" means the parameter does not vary in that source, so R² is undefined.

| Source | Rows | Ω_m | σ₈ | h | n_s | Ω_b | w₀ | Σm_ν | w_a |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `bacco` | 23,997 | 0.99 | 0.99 | 0.68 | 0.58 | 0.36 | fixed | fixed | fixed |
| `bcemu` | 23,997 | 0.99 | 0.74 | 0.75 | 0.25 | 0.72 | fixed | fixed | fixed |
| `spk` | 23,997 | 0.98 | 0.98 | 0.66 | 0.66 | 0.35 | fixed | fixed | fixed |
| `bacco_neutrino` | 23,997 | 0.99 | 0.99 | 0.67 | 0.58 | 0.31 | fixed | 0.52 | fixed |
| `bacco_full8` | 23,997 | 0.99 | 0.99 | 0.63 | 0.24 | 0.34 | 0.61 | 0.08 | 0.19 |
| `bacco_multiz` | 23,997 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 | fixed | -0.00 | fixed |
| `bcemu_neutrino` | 10,000 | 0.99 | 0.74 | 0.75 | 0.23 | 0.72 | fixed | -1.34 | fixed |
| `dark_emulator` | 5,001 | 0.87 | 0.93 | fixed | 0.30 | fixed | 0.86 | fixed | fixed |
| `ns_grid` | 2,500 | -0.58 | -0.23 | -7.88 | -0.38 | -0.14 | fixed | fixed | fixed |
| `camb_nl` | 1,000 | 0.98 | 0.99 | fixed | fixed | fixed | fixed | fixed | fixed |
| `camels_astrid_x` | 250 | -0.06 | -0.34 | -5.08 | -0.78 | -0.02 | 0.03 | fixed | -0.03 |
<!-- /table:validation_per_source -->

`bacco_multiz` has byte-identical redshift channels, so it carries no growth
information and scores near zero throughout. On `camels_astrid_x`, the one
hydrodynamic slice, the model scores below the mean predictor on six of the seven
parameters that suite varies; that suite also stores its redshift channels in the
opposite order, so the causes are not separated. Neutrino-mass recovery differs
sharply between `bacco_neutrino` and `bcemu_neutrino`, which are different
emulators; the reason is not established.

## Reproducing these numbers

```bash
git clone https://github.com/arajgor1/cosmufr-run4
cd cosmufr-run4
pip install -e ".[demo]"
python -m cosmufr.reproduce
```

This regenerates the **benchmark** table from the 6,000-row set that ships with
the code and alongside these weights as `cosmufr_benchmark.npz`, within 1e-5 in
R². It does **not** regenerate the validation table, which needs the private
master validation rows and `bad_indices.npy`.

## Limitations and known defects

1. **The refinement's update networks are at initialization, and its benefit is
   untested** (see above).
2. **The stored energy is not informative about the landscape.** On benchmark
   spectra it reads about −926,537, where one float32 step is 0.0625, so small
   changes do not show; in float64 the refinement lowers it slightly on every row
   checked. An added constant would change the value without changing the gradient.
3. **The σ output sits at its clamp floor.** `UncertaintyHead` sits at its clamp floor:
   σ = 0.1 for six of eight parameters on every benchmark input. Do not use these
   as error bars.
4. **Neutrino mass is not recovered** (R² 0.011 where it varies, saved report).
5. **The generative head's output does not follow the input** on the 6,000
   benchmark rows: it stays near 2.6327, varying by less than 1e-6 across k and
   1e-4 across rows.
6. **Two redshifts only.** Multi-redshift generalization is unvalidated.
7. **No ablation and no matched direct network.** Ridge regression on the same 400
   inputs, fitted on 3,000 rows, beats this model on Ω_m (0.738 against 0.716).
8. **Historical cross-run comparisons are unreliable**; epoch-to-epoch R² noise of
   ±0.03 to 0.10 was not controlled.
9. **It fails on hydrodynamic data**, with a channel-order defect confounding the
   cause.
10. **Two bundled suites have reversed redshift channels:** `camb_nl` and
    `camels_astrid_x`, 39 of the 6,000 benchmark rows.
11. **The rebuilt June master arrays have a separately reported ordering problem**
    in `camb_nl` and `camb_wa_grid`, found in a 648-row sample by an independent
    review; not audited source-wide.
12. **The input is not an observable.** Emulator spectra with no survey window,
    shot noise, mask, galaxy bias, noise model or covariance.
13. **Row-level split.** Related spectra of one cosmology may appear in training
    and evaluation.
14. **The earlier "Fisher ceiling" is withdrawn.** It was not derived for a
    specified observation model, and the scores reported at the time exceed it on
    every parameter. Identifiability remains to be investigated.

## What you can conclude

The architecture this model is named for has not been tested. Nothing here is
evidence for or against iterative refinement. On its own terms this is a
deterministic point predictor that recovers matter density and clustering
amplitude on emulator spectra, is beaten by a linear fit on one of them, produces
no validated uncertainty, and fails on the one hydrodynamic slice it is tested
against.

## Training data

Historical documentation for Run 4 gives 84.5 million cosmology → P(k) rows,
mostly emulator output (BACCO, BCemu, SP(k), Dark Emulator, CAMB) with small
hydrodynamic and N-body components. No manifest for that exact corpus has been
located. The master arrays now in storage hold 86,391,712 rows; a June rebuild
that no released model used holds 86,215,712.

A June audit of the corpus found w₀ fixed at −1 in about 86% of rows, w_a at 0 in
about 88% and Σm_ν at 0 in about 74%. On the bundled benchmark the figures are
81.9%, 84.7% and 49.5% (reproduced). How much of the weakness in h, w₀ and w_a is
coverage and how much is a limit of two noiseless spectra over this k range is
not settled here.

## Training procedure

Phase 4 fine-tune on a single B200, batch 4096, BF16, warm-started through Run 3
from Run 2 (the Run 3 checkpoint was not examined in the audit). Two optimizers:
one for the core, one for the energy heads.

## Evaluation protocol

Validation split: for each source, the last rows up to a 0.5% quota, no RNG, rows
in `bad_indices.npy` removed. The split is by row, not grouped by cosmology.
Metrics are computed with the released inference package.

## Citation

```bibtex
@misc{cosmufr_run4_2026,
  title  = {CosmUFR Run 4: a cosmological parameter-prediction prototype from
            the matter power spectrum, with an audit of its training defects},
  author = {Rajgor, Aaditya},
  year   = {2026},
  url    = {https://huggingface.co/arajgor1/cosmufr-run4}
}
```
