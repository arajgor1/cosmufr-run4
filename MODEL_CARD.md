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

> **Read this first.** The belief-settling core that gives this architecture its name never received a gradient during training. It sits at initialization. What learned is the set of read-out heads, and of those only the parameter head does useful work: the uncertainty head is stuck at its clamp floor and the generative head returns a constant. Numbers below are measured on that basis and are reproducible from this repository. Earlier published figures for this model (Ω_m 0.907, σ₈ 0.911, h 0.604) are superseded and should not be cited.

---

## Model details

| | |
|---|---|
| Developed by | Aaditya Rajgor |
| Model type | Feed-forward energy-based parameter inference, no attention |
| Parameters | 136,194,617 |
| Inputs | `log10 P(k)`, 200 log-spaced k bins over k ∈ [0.1, 4.5] h/Mpc, at z = 0 and z = 0.47 |
| Outputs | 8 cosmological parameters. Also 8 variances and a P(k) reconstruction, both of which are degenerate: see limitations 3 and 9. |
| Precision | float32 |
| Latency | ~400 ms per spectrum on CPU |
| Determinism | Bit-identical across repeated calls |
| Checkpoint | epoch 30, phase 4, trained 2026-04-14 |
| License | MIT |

Parameters, in output order: Ω_m, σ₈, h, n_s, Ω_b, w₀, Σm_ν, w_a.

## Intended use

Research and teaching. Specifically:

- A worked example of energy-based iterative inference applied to cosmology.
- A case study in how a silent gradient-path defect survives months of training with plausible-looking loss curves, and how to detect one.
- A baseline that a better-trained model can be compared against.

**Not for producing cosmological constraints.** The uncertainties are a constant, so nothing here supports error propagation or likelihood analysis.

## Results

Measured on the deterministic validation split, 162,795 rows across 16 sources, using the released inference package. Full report: `reports/honest_eval.json`.

| Parameter | Full validation R² | R² where it varies | RMSE |
|---|---|---|---|
| Ω_m | 0.717 | 0.720 | 0.0273 |
| σ₈ | 0.756 | 0.757 | 0.0285 |
| h | 0.501 | 0.498 | 0.0402 |
| w₀ | 0.586 | 0.614 | 0.0254 |
| Ω_b | 0.364 | 0.363 | 0.0045 |
| n_s | 0.338 | 0.339 | 0.0214 |
| w_a | 0.165 | 0.185 | 0.0616 |
| Σm_ν | 0.407 | **0.011** | 0.0993 |

R² is a ratio against the variance of the truth, so on a slice where a parameter is held at a fiducial constant it measures nothing. The second column restricts each parameter to the sources that actually vary it. For Σm_ν this is decisive: the apparent 0.41 is an artifact of Σm_ν being pinned at zero across most of the training corpus, where predicting near-zero scores well without recovering anything. **This model does not constrain neutrino mass.**

### Per-source breakdown

The aggregate understates performance on sound data and overstates it on defective data. Both are shown.

| Source | n | Ω_m | σ₈ | h | n_s | Ω_b |
|---|---|---|---|---|---|---|
| bacco | 23,997 | 0.99 | 0.99 | 0.68 | 0.58 | 0.36 |
| bcemu | 23,997 | 0.99 | 0.74 | 0.75 | 0.25 | 0.72 |
| spk | 23,997 | 0.98 | 0.98 | 0.66 | 0.66 | 0.35 |
| bacco_neutrino | 23,997 | 0.99 | 0.99 | 0.67 | 0.58 | 0.31 |
| bacco_full8 | 23,997 | 0.99 | 0.99 | 0.63 | 0.24 | 0.34 |
| **bacco_multiz** | 23,997 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 |
| dark_emulator | 5,001 | 0.87 | 0.93 | -- | 0.30 | -- |
| camb_nl | 1,000 | 0.98 | 0.99 | -- | -- | -- |

`bacco_multiz` is 15 percent of the validation set and scores zero on everything, because its z=0.47 spectra are self-paired copies of its z=0 spectra and carry no growth information. That is a data-generation defect. `--` marks parameters pinned in that source, where R² is undefined.

## Reproducing these numbers

```bash
git clone https://github.com/arajgor1/cosmufr-run4
cd cosmufr-run4
pip install -e ".[demo]"
python -m cosmufr.reproduce
```

The 6,000-row benchmark ships in the repository and alongside these weights as `cosmufr_benchmark.npz`. On a clean machine the reproduction matches the published table to within 1e-6.

## Limitations and known defects

1. **The belief pipeline never trained.** `obs_encoder`, `belief_proposal` and `settling` are at initialization; 84 Linear biases are still bit-exactly zero after forty epochs. Confirmed independently by comparing the Run 2 and Run 4 checkpoints, where 204 of 204 tensors in those modules are bit-identical while the read-out heads moved 66 to 79 percent. Root cause is an unconditional `detach()` in the settling loop. Verify with `cosmufr.weight_audit(model)`.
2. **Settling does no measurable work.** Mean belief movement 0.09 percent; energy flat to one float32 unit at the magnitude it operates at; 314 of 318 validation batches show exactly zero energy change.
3. **Uncertainties are a constant, not a prediction.** `UncertaintyHead` returns `clamp(softplus(net(b)) + 1e-2, max=4.0)` and sits at the floor, so σ = 0.1 for six of eight parameters on 100 percent of inputs. The reported ECE of 0.39 follows directly. Do not use these as error bars.
4. **Neutrino mass is not recovered** (R² = 0.011 where it varies).
5. **The energy subsystem diverged.** Energy sits near −9.3e5 and its heads drifted ~7e29 in relative norm. The `E_con` anomaly score is around −4.6e5, five orders of magnitude from the −0.999908 quoted in earlier material. It is not a usable out-of-distribution signal.
6. **Two redshifts only** (z = 0, z = 0.47). Multi-redshift generalization is unvalidated, and the multi-redshift corpus has a documented ordering defect.
7. **No baseline and no ablation.** There is no comparison against an amortized posterior estimator or a plain MLP, which is the first thing a reviewer should ask for.
8. **Historical cross-run comparisons in this project are untrustworthy**, because epoch-to-epoch R² noise of ±0.03 to 0.10 was never controlled for.
9. **The generative head collapsed to a constant.** `GenerativeHead` is documented as reconstructing `log10 P(k)` at arbitrary k. It returns 2.6327 at every k, for every input spectrum, and for a random belief vector, with measured variation of 2e-7 in both directions. Its reported log-space MSE of 0.687 is simply the variance of `log10 P(k)` about a constant, which is what a predictor that ignores its input scores. There is no reconstruction.

## Training data

84.5M cosmology → P(k) samples across 14 sources: CAMB (linear and non-linear), CAMELS (IllustrisTNG, SIMBA, Astrid), BACCO, Quijote, BCemu, DarkEmulator, SPk, plus dedicated n_s, w₀ and multi-redshift grids.

The corpus pins hard parameters at fiducial values in a large fraction of samples: w₀ in about 86 percent, w_a in about 88 percent, Σm_ν in about 74 percent. This is a substantial part of why those parameters recover poorly, and it is a coverage limit rather than a physics limit.

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
