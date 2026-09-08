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

**A 136M-parameter belief-settling network that infers eight cosmological parameters from the matter power spectrum, released together with an audit of its own training defects.**

- Code and reproduction: https://github.com/arajgor1/cosmufr-run4
- Checkpoint SHA256: `5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1`

> **Read this first.** Three parts of this network never received a gradient
> during training and still hold their initial random values; five more trained
> and converged on outputs that ignore the input; one part trained and does all
> the useful work. The table under "What I observed" says which is which.
> Earlier published figures for this model (Ω_m 0.907, σ₈ 0.911, h 0.604) are
> superseded and should not be cited.

---

## What this is

A network that maps `log10 P(k)` at two redshifts onto eight cosmological
parameters in a single forward pass, with no simulator and no chain. It was
designed to reach its answer gradually — encode the spectrum into a
1024-dimensional belief, sharpen it over sixteen steps of descent on a learned
score, then read parameters off the settled belief — on the reasoning that a
model which refines can spend more effort on a hard observation than an easy one.
That is the research claim this release does not get to test, for the reason
below.

Trained on 84.5M spectra across fourteen suites, almost all of it emulator
output. Four training runs, then four architectural variants: eight in total.

## Model details

| | |
|---|---|
| Developed by | Aaditya Rajgor |
| Model type | Feed-forward energy-based parameter inference, no attention |
| Parameters | 136,194,617, of which about 1.2% do the work |
| Inputs | `log10 P(k)`, 200 log-spaced k bins over k ∈ [0.1, 4.5] h/Mpc, at z = 0 and z = 0.47 |
| Outputs | 8 cosmological parameters. Also 8 variances and a P(k) reconstruction, both degenerate: see limitations. |
| Precision | float32 |
| Latency | ~300 ms per spectrum on one CPU core, measured |
| Determinism | Bit-identical across repeated calls |
| Checkpoint | epoch 30, phase 4, trained 2026-04-14 |
| License | MIT |

Parameters, in output order: Ω_m, σ₈, h, n_s, Ω_b, w₀, Σm_ν, w_a.

## What I observed

Reading the finished weights part by part gives three outcomes, not two:

| outcome | share | which parts |
|---|---|---|
| **Never trained.** No gradient reached these in any of the eight runs. A `detach()` in the settling loop disconnected them from the loss, so more training could not have moved them. | 24% | `obs_encoder`, `belief_proposal`, `settling`, `halo_head` |
| **Trained, and stopped reading the input.** These received a gradient and converged on outputs that do not depend on the spectrum. | 56% | the three energy heads, `gen_head`, `unc_head` |
| **Trained, and works.** Every reported number comes from here, reading a fixed random projection of the input. | 1.2% | `param_head` |

The rest is an unused single-redshift path plus a prototype bank. Verify in about
seven seconds with `cosmufr.weight_audit(cosmufr.load_model()).table()`.

**Reconnecting the gradient path would not be enough.** The energy heads trained
and converged on a flat landscape: the score varies by about one part in seven
million across completely different spectra, and its gradient has norm 0.11
against a belief of norm 16.5. There is nothing to descend.

## Intended use

Research and teaching. Specifically:

- A worked example of energy-based iterative inference applied to cosmology, and
  of how such a design can fail silently while the loss curve looks healthy.
- A fast, deterministic, reproducible baseline for `P(k)` → parameters.
- A case study in auditing a trained checkpoint rather than trusting it.

**Not** for producing cosmological constraints. The input is not an observable,
there is no uncertainty, and the numbers are four to eight times worse than a
published survey.

## Results

Measured on the deterministic validation split, 162,795 rows across 16 sources, using the released inference package. Full report: `reports/honest_eval.json`.

| Parameter | Full validation R² | R² where it varies | Bundled benchmark (reproducible) | RMSE |
|---|---|---|---|---|
| Ω_m | 0.717 | 0.720 | 0.687 | 0.0273 |
| σ₈ | 0.756 | 0.757 | 0.738 | 0.0285 |
| h | 0.501 | 0.498 | 0.475 | 0.0402 |
| w₀ | 0.586 | 0.614 | 0.599 | 0.0254 |
| Ω_b | 0.364 | 0.363 | 0.353 | 0.0045 |
| n_s | 0.338 | 0.339 | 0.331 | 0.0214 |
| w_a | 0.165 | 0.185 | 0.148 | 0.0616 |
| Σm_ν | 0.407 | **0.011** | 0.410 | 0.0993 |

R² is a ratio against the variance of the truth, so on a slice where a parameter is held at a fiducial constant it measures nothing. The second column restricts each parameter to the sources that actually vary it. For Σm_ν this is decisive: the apparent 0.41 is an artifact of Σm_ν being pinned at zero across most of the training corpus, where predicting near-zero scores well without recovering anything. **This model does not constrain neutrino mass.**

### Per-source breakdown

The aggregate understates performance on sound data and overstates it on defective data. Both are shown.

The eleven sources with more than a handful of rows, including the worst. Five further sources hold 62 rows between them and are too small to score.

| Source | n | Ω_m | σ₈ | h | n_s | Ω_b | w₀ | Σm_ν | w_a |
|---|---|---|---|---|---|---|---|---|---|
| bacco | 23,997 | 0.99 | 0.99 | 0.68 | 0.58 | 0.36 | -- | -- | -- |
| bcemu | 23,997 | 0.99 | 0.74 | 0.75 | 0.25 | 0.72 | -- | -- | -- |
| spk | 23,997 | 0.98 | 0.98 | 0.66 | 0.66 | 0.35 | -- | -- | -- |
| bacco_neutrino | 23,997 | 0.99 | 0.99 | 0.67 | 0.58 | 0.31 | -- | 0.52 | -- |
| bacco_full8 | 23,997 | 0.99 | 0.99 | 0.63 | 0.24 | 0.34 | 0.61 | 0.08 | 0.19 |
| **bacco_multiz** | 23,997 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 | -- | -0.00 | -- |
| bcemu_neutrino | 10,000 | 0.99 | 0.74 | 0.75 | 0.23 | 0.72 | -- | **-1.34** | -- |
| dark_emulator | 5,001 | 0.87 | 0.93 | -- | 0.30 | -- | 0.86 | -- | -- |
| ns_grid | 2,500 | **-0.58** | **-0.23** | **-7.88** | -0.38 | -0.14 | -- | -- | -- |
| camb_nl | 1,000 | 0.98 | 0.99 | -- | -- | -- | -- | -- | -- |
| camels_astrid_x | 250 | -0.06 | -0.34 | **-5.08** | -0.78 | -0.02 | 0.03 | -- | -0.03 |

`bacco_multiz` is 15 percent of the validation set and scores zero on everything, because its z=0.47 spectra are self-paired copies of its z=0 spectra and carry no growth information. That is a data-generation defect. `--` marks parameters pinned in that source, where R² is undefined.

## Reproducing these numbers

```bash
git clone https://github.com/arajgor1/cosmufr-run4
cd cosmufr-run4
pip install -e ".[demo]"
python -m cosmufr.reproduce
```

The 6,000-row benchmark ships in the repository and alongside these weights as `cosmufr_benchmark.npz`.

Be precise about what that reproduces. On a clean machine it regenerates the **benchmark** column below to about 1e-6. It does **not** regenerate the full-validation column: that was measured on 162,795 rows of a private split, and the 6,000-row subsample lands within about 0.03 of it through sampling noise alone. The full-validation numbers remain unverifiable from outside, and the benchmark narrows that gap rather than closing it.

## Limitations and known defects

1. **The belief pipeline never trained.** `obs_encoder`, `belief_proposal` and `settling` are at initialization; 84 Linear biases are still bit-exactly zero after forty epochs. Confirmed independently by comparing the Run 2 and Run 4 checkpoints, where 204 of 204 tensors in those modules are bit-identical while the read-out heads moved 66 to 79 percent. Root cause is an unconditional `detach()` in the settling loop. Verify with `cosmufr.weight_audit(model)`.
2. **Settling does no measurable work.** Mean belief movement 0.09 percent; energy flat to one float32 unit at the magnitude it operates at; 314 of 318 validation batches show exactly zero energy change.
3. **Uncertainties are a constant, not a prediction.** `UncertaintyHead` returns `clamp(softplus(net(b)) + 1e-2, max=4.0)` and sits at the floor, so σ = 0.1 for six of eight parameters on 100 percent of inputs. The reported ECE of 0.39 follows directly. Do not use these as error bars.
4. **Neutrino mass is not recovered** (R² = 0.011 where it varies).
5. **The energy subsystem diverged.** Energy sits near −9.3e5 and its heads drifted ~7e29 in relative norm. The `E_con` anomaly score is around −4.6e5, five orders of magnitude from the −0.999908 quoted in earlier material. It is not a usable out-of-distribution signal.
6. **Two redshifts only** (z = 0, z = 0.47). Multi-redshift generalization is unvalidated, and the multi-redshift corpus has a documented ordering defect.
7. **No ablation, and only a linear baseline.** Ridge regression on the same 400 inputs, fitted on 3,000 rows against this model's 84.5 million, still beats it on Ω_m (0.738 to 0.716). There is no comparison against an amortized posterior estimator, and no network of matched size trained directly on the same inputs. Until that exists, nothing here shows the architecture earns its size.
8. **Historical cross-run comparisons in this project are untrustworthy**, because epoch-to-epoch R² noise of ±0.03 to 0.10 was never controlled for.
9. **The generative head collapsed to a constant.** `GenerativeHead` is documented as reconstructing `log10 P(k)` at arbitrary k. It returns 2.6327 at every k, for every input spectrum, and for a random belief vector, with measured variation of 2e-7 in both directions. Its reported log-space MSE of 0.687 is simply the variance of `log10 P(k)` about a constant, which is what a predictor that ignores its input scores. There is no reconstruction.

10. **It fails on hydrodynamic physics, for reasons not yet separated.** On `camels_astrid_x`, the one evaluation slice from a full hydrodynamic simulation, it scores worse than a constant predictor on six of the seven parameters that suite varies, including −5.08 on h. That slice also has a redshift-ordering defect (item 11), so the physics and the defect are confounded and neither is measured.
11. **Two suites have their redshift channels reversed.** `camb_nl` and `camels_astrid_x` store z = 0.47 where every other suite stores z = 0 — 39 of the 6,000 bundled test spectra. The median log₁₀ gap between channels is about −0.32 in these and +0.32 everywhere else. Found September 2026, after the model shipped. The demo warns when it sees this.
12. **Neutrino mass is confounded with baryonic feedback.** Two suites with the same spread of masses differ only in whether a baryonic correction is applied: R² 0.516 without it, −1.343 with it. Free-streaming suppression and feedback suppression look alike over these scales and nothing in the output distinguishes them.
13. **The input is not an observable.** Emulator matter power spectra, with no survey window, no shot noise, no mask and no galaxy bias. There is no noise model anywhere in the corpus, so there is no covariance and nothing that would make a posterior width a physical quantity.
14. **The "Fisher ceiling" claimed in earlier material is withdrawn.** Substituting the reported scores into the same weights gives 0.559 against a claimed maximum of 0.49, and all eight parameters individually exceed their own claimed maximum. Separately, a Fisher forecast requires a data covariance, and noiseless emulator rows have none, so no such bound can exist.

## What you can conclude

The architecture this model is named for has not been tested, because the
mechanism never ran. Nothing here is evidence for or against iterative belief
refinement.

What it is, on its own terms: a fast, deterministic, fully reproducible baseline
that recovers matter density and clustering amplitude usefully, is beaten by a
linear fit on one of them, produces no uncertainty, and collapses on the one
slice of real gas physics it is tested against. Use it as a baseline, a teaching
example, or a starting point. Do not use it to constrain cosmology.


## Training data

84.5M cosmology → P(k) samples across 14 sources: CAMB (linear and non-linear), CAMELS (IllustrisTNG, SIMBA, Astrid), BACCO, Quijote, BCemu, DarkEmulator, SPk, plus dedicated n_s, w₀ and multi-redshift grids.

The corpus pins hard parameters at fiducial values in a large fraction of samples: w₀ in about 86 percent, w_a in about 88 percent, Σm_ν in about 74 percent. On the bundled benchmark the model is actually scored against, the figures are w₀ 81.9 percent, w_a 84.7 percent and Σm_ν 49.5 percent.

How much of the weakness in h, w₀ and w_a is a coverage limit and how much is a genuine information limit of `log P(k)` at two redshifts is **not settled here**, and I do not currently know how to separate them. Designing that experiment is one of the things I want advice on.

## Training procedure

Phase 4 fine-tune on a single B200, batch 4096, BF16, warm-started from Run 3. Two optimizers, one for the core and one for the energy heads. Completed 2026-04-14.

## Evaluation protocol

Deterministic validation split: for each source, the last `max(1, min(global_quota, n_source × 0.005))` rows. No RNG, so the split is bit-reproducible. Rows listed in `bad_indices.npy` are removed. Metrics are computed with the released inference package rather than the training-time evaluator.

## Citation

```bibtex
@misc{cosmufr_run4_2026,
  title  = {CosmUFR Run 4: a belief-settling network for cosmological parameter
            inference, with an audit of its training defects},
  author = {Rajgor, Aaditya},
  year   = {2026},
  url    = {https://huggingface.co/arajgor1/cosmufr-run4}
}
```
