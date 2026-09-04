# CosmUFR Run 4

**Neural inference of cosmological parameters from the matter power spectrum, released together with the audit that found its central defect.**

CosmUFR takes `log10 P(k)` at two redshifts and infers eight cosmological parameters jointly. It is a 136M-parameter network built around a belief-settling core: encode the spectrum into a 1024-dimensional belief, then refine that belief through 16 steps of gradient descent on a learned energy.

That was the design. This release documents what the trained weights actually do, which is not the same thing.

---

## What this release is

Three artifacts, in order of importance:

1. **A working inference stack.** Loads in under two seconds, runs in ~400 ms on CPU, bit-deterministic across repeated calls.
2. **A benchmark you can check it against.** 6,000 rows of the validation split, committed to this repository. Every number below regenerates on your machine in one command.
3. **An audit of the model's own defects**, reproducible from the released weights with `cosmufr.weight_audit()`.

It is an in-progress PhD project, shared as-is. The defects below are stated because they are real, not as a rhetorical device.

---

## The finding

The belief pipeline this architecture is named for never trained.

```
>>> print(cosmufr.weight_audit(model).table())

module                  on path  n Linear   biases == 0   max |bias|     verdict
--------------------------------------------------------------------------------
belief_proposal             yes        10     10/10        0.000e+00   UNTRAINED
obs_encoder                 yes        10     10/10        0.000e+00   UNTRAINED
settling                    yes        64     64/64        0.000e+00   UNTRAINED
halo_head                   yes         6      6/6         0.000e+00   UNTRAINED
gen_head                    yes        18      0/18        8.653e-01     trained
param_head                  yes         6      0/6         4.632e-01     trained
unc_head                    yes         6      0/6         3.927e-01     trained
```

A `Linear` bias that has taken even one optimizer step essentially never returns to exactly `0.0`. Eighty-four of them are still bit-exactly zero after forty epochs.

The confirming measurement compares Run 2's checkpoint against Run 4's. Run 4 was warm-started from Run 3, itself warm-started from Run 2, then trained forty further epochs:

| Module | Tensors | Bit-identical | Mean relative change |
|---|---|---|---|
| obs_encoder | 38 | 38 / 38 | 0.000e+00 |
| belief_proposal | 38 | 38 / 38 | 0.000e+00 |
| settling | 128 | 128 / 128 | 0.000e+00 |
| param_head | 24 | 2 / 24 | 7.55e-01 |
| gen_head | 70 | 0 / 70 | 6.59e-01 |
| energy heads | 66 | 0 / 66 | 7.4e+29 |

Zero relative change across three training runs is not slow learning. It is no gradient. The read-out heads moved by 66 to 79 percent, so training clearly ran; it simply never reached the encoder. Reproduce with `cosmufr.compare_checkpoints(run2_path, run4_path)`.

**What the model therefore is:** trained read-out heads on a fixed random projection of the input. Random projections preserve a great deal of structure, which is why the results below are respectable. It also means they are a floor for this architecture rather than a ceiling.

**Consequences you will see in the demo.** Settling moves the belief by 0.09 percent of its norm; its energy changes by one float32 unit at the magnitude it operates at, which is the smallest change the number can represent. The uncertainty head is pinned at its clamp floor, so every reported σ is the constant 0.1 and none of them are usable error bars.

---

## Results

Measured with the released package on the deterministic validation split, 162,795 rows across 16 sources. Full report in [`reports/honest_eval.json`](reports/honest_eval.json).

| Parameter | Published 2026-05 | Full validation | Where it varies | Bundled benchmark | RMSE |
|---|---|---|---|---|---|
| Ω_m | 0.907 | **0.717** | 0.720 | 0.687 | 0.0273 |
| σ₈ | 0.911 | **0.756** | 0.757 | 0.738 | 0.0285 |
| h | 0.604 | **0.501** | 0.498 | 0.475 | 0.0402 |
| w₀ | 0.742 | **0.586** | 0.614 | 0.599 | 0.0254 |
| m_ν | 0.410 | 0.407 | **0.011** | 0.410 | 0.0993 |
| Ω_b | 0.406 | **0.364** | 0.363 | 0.353 | 0.0045 |
| n_s | 0.353 | **0.338** | 0.339 | 0.331 | 0.0214 |
| w_a | 0.187 | **0.165** | 0.185 | 0.148 | 0.0616 |

The "published" column is superseded and is shown only so the change is visible. Those numbers came from the training-time evaluator on a validation set that was later corrected, with epoch 30 selected as best from inside ±0.03 to 0.10 evaluation noise. They should not be cited.

**"Where it varies"** computes each parameter only on the sources that actually vary it. R² is a ratio against the variance of the truth, so on a slice where a parameter is held at a fiducial constant it measures nothing. This column is the one to read, and for m_ν it is the whole story: the apparent 0.41 was an artifact of m_ν being pinned at zero throughout most of the corpus, where predicting near-zero scores well without recovering anything. **The model does not constrain neutrino mass.**

### Why the aggregate is lower than it looks

| Source | n | Ω_m | σ₈ | h | n_s | Ω_b | w₀ | m_ν | w_a |
|---|---|---|---|---|---|---|---|---|---|
| bacco | 23,997 | 0.99 | 0.99 | 0.68 | 0.58 | 0.36 | -- | -- | -- |
| bcemu | 23,997 | 0.99 | 0.74 | 0.75 | 0.25 | 0.72 | -- | -- | -- |
| spk | 23,997 | 0.98 | 0.98 | 0.66 | 0.66 | 0.35 | -- | -- | -- |
| bacco_neutrino | 23,997 | 0.99 | 0.99 | 0.67 | 0.58 | 0.31 | -- | 0.52 | -- |
| bacco_full8 | 23,997 | 0.99 | 0.99 | 0.63 | 0.24 | 0.34 | 0.61 | 0.08 | 0.19 |
| **bacco_multiz** | 23,997 | -0.00 | -0.00 | -0.00 | -0.00 | 0.00 | -- | -0.00 | -- |
| bcemu_neutrino | 10,000 | 0.99 | 0.74 | 0.75 | 0.23 | 0.72 | -- | -1.34 | -- |
| dark_emulator | 5,001 | 0.87 | 0.93 | -- | 0.30 | -- | 0.86 | -- | -- |
| ns_grid | 2,500 | -0.58 | -0.23 | -7.88 | -0.38 | -0.14 | -- | -- | -- |
| camb_nl | 1,000 | 0.98 | 0.99 | -- | -- | -- | -- | -- | -- |
| camels_astrid_x | 250 | -0.06 | -0.34 | -5.08 | -0.78 | -0.02 | 0.03 | -- | -0.03 |

`--` marks a parameter pinned in that source, where R² is undefined.

`bacco_multiz` is 15 percent of the validation set and scores zero on everything. Its z=0.47 spectra are self-paired copies of its z=0 spectra, so both input channels are identical and carry no growth information. That is a data-generation defect, documented in the corpus source registry, not a model failure. On sources where the data is sound, Ω_m recovery reaches 0.98 to 0.99.

Both readings are true and both are shown. The aggregate is what the model achieves on the corpus as it stands; the per-source table is what it achieves on data without a known defect.

---

## Reproduce every number above

This is the part that matters. The original Run 4 table could not be checked by anyone outside the training infrastructure. This one can.

```bash
git clone https://github.com/arajgor1/cosmufr-run4
cd cosmufr-run4
pip install -e ".[demo]"
python -m cosmufr.reproduce          # downloads weights, runs the benchmark
```

Or in Python:

```python
import cosmufr

model  = cosmufr.load_model()                  # pulls best.pt from HuggingFace
bench  = cosmufr.load_benchmark()              # ships in this repo, 6,000 rows
result = cosmufr.evaluate(model, bench)

print(result.table())          # the headline table
print(result.source_table())   # the per-source breakdown
```

If your numbers differ from `reports/honest_eval.json`, that is a bug worth reporting.

The bundled benchmark is a proportional subsample of the full split, so its aggregate differs from the full-split value by sampling noise of up to about 0.03. It is proportional rather than evenly stratified for a reason: an earlier version sampled evenly across sources, which changed the source mix and therefore the variance R² divides by, and dropped Ω_m from 0.72 to 0.17 without the model changing at all. `tests/test_reproducibility.py` guards against that.

---

## Quick start

```python
import cosmufr
import numpy as np

model = cosmufr.load_model(ckpt_path="best.pt")

bench  = cosmufr.load_benchmark()
result = cosmufr.infer(bench.pk_z0[0], bench.pk_z047[0], model=model)

print(result.params)   # {'Om': 0.387, 's8': 0.672, 'h': 0.669, ...}
print(result.pk_recon) # reconstructed log10 P(k)
# result.sigmas is the clamp floor on every input. Do not use it as an error bar.
```

Inputs are `P(k)` on a 200-bin log-spaced grid over k ∈ [0.1, 4.5] h/Mpc, at z=0 and z=0.47. Raw or log10 are both accepted and auto-detected.

### Inspect the defects yourself

```python
print(cosmufr.weight_audit(model).table())
print(cosmufr.settling_report(model, bench.pk_z0[0], bench.pk_z047[0]).summary())
print(cosmufr.uncertainty_audit(result.sigmas_array[None, :]).table())
```

---

## Known defects

Stated in full, because a reader will find all of them within ten minutes.

1. **The belief pipeline never trained.** `obs_encoder`, `belief_proposal` and `settling` are at initialization. The architecture's central claim is unsupported by its own weights. Root cause traced to an unconditional `detach()` in the settling loop introduced in commit `a5caac6`.
2. **Settling is a no-op.** 0.09 percent mean belief movement; energy flat to one float32 unit; 314 of 318 validation batches show exactly zero energy change.
3. **Uncertainties are a constant.** `UncertaintyHead` returns `clamp(softplus(net(b)) + 1e-2, max=4.0)` and sits at the floor, so σ = 0.1 for six of eight parameters on 100 percent of inputs. These are not error bars. Reported ECE of 0.39 follows directly from this.
4. **m_ν is not recovered.** R² = 0.011 where m_ν varies.
5. **The energy subsystem is divergent.** Energy sits near −9.3e5 and its heads drifted by ~7e29 in relative norm during training. The `E_con` anomaly score is around −4.6e5, five orders of magnitude from the −0.999908 quoted in earlier material. Do not use it as an out-of-distribution signal.
6. **Two redshifts only** (z=0 and z=0.47). Multi-redshift generalization is not validated, and the multi-redshift corpus has a documented ordering defect.
7. **Evaluation noise was never controlled.** Epoch-to-epoch R² noise of ±0.03 to 0.10 means historical cross-run comparisons in this project's development logs are not trustworthy.
8. **The architecture flags** `use_explicit_z`, `use_source_aware` and `use_multi_z_encoder` are hard-coded off. The checkpoint was trained without them.

---

## What I think should happen next

I am publishing this to get expert judgement on the following, which is also what I would want to work on in a PhD.

**Things I can verify at zero cost and intend to do:**

- Remove the `detach()` that severs the gradient, and gate the fix behind the gradient-flow test in `tests/test_gradient_flow.py`, which would have caught this on the day it was introduced.
- Bound the energy heads so the energy term stops diverging.
- Give the uncertainty head a variance floor it can actually leave.
- Fix the evaluation protocol: a frozen versioned split, metrics averaged over several epochs rather than single-epoch best, and per-parameter R² computed only where the parameter varies.

**Things I do not know how to judge, and want advice on:**

- Whether the belief-settling formulation is worth pursuing once the gradient path is repaired, or whether it is an expensive way to reach what an amortized posterior estimator gets in one forward pass. I have no baseline, and I know that is the first thing a reviewer would ask for.
- How much of the weakness in h, w₀ and w_a is a genuine information limit of `log P(k)` at two redshifts, and how much is the training corpus pinning those parameters at fiducial values in 45 to 88 percent of samples.
- Whether extending to higher k, more redshifts, or explicit BAO features is the right way to make h and m_ν identifiable, and what a defensible experimental design for that looks like.

The broader question I want to work on is using learned inference to shorten the analysis loop between survey observation and parameter constraints. This model is where I started on it, defects included.

---

## Architecture

Input is `log10 P(k)` at two redshifts, concatenated to a 400-d vector. `ObsEncoder` maps it to a 1024-d belief. `BeliefProposal` combines that with a previous belief. `SettlingCore` runs 16 steps of gradient descent on `E = 1.0·E_obs + 0.5·E_con + 0.5·E_dyn`, with a per-step learned preconditioner in [0.01, 1.0] and step size in [0.001, 0.05]. The settled belief feeds `ParameterHead` (8 parameters, sigmoid-clamped to physical priors), `UncertaintyHead` (8 variances) and a k-continuous `GenerativeHead` that reconstructs `log10 P(k)` at arbitrary k. There is no attention anywhere.

Per the audit, only the heads and the energy networks carry trained weights.

## Training summary

- 136,194,617 parameters
- 84.5M cosmology → P(k) samples across 14 source datasets (CAMB, CAMELS, BACCO, Quijote, BCemu, DarkEmulator, SPk, plus n_s, w₀ and multi-redshift grids)
- Phase 4 fine-tune on a single B200, batch 4096, BF16, completed 2026-04-14
- Two optimizers: one for the core, one for the energy heads

## Provenance

| | |
|---|---|
| Checkpoint | `best.pt`, epoch 30, phase 4 |
| SHA256 | `5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1` |
| Size | 544,931,631 bytes |
| Evaluated | 2026-09-04, released package, deterministic split |

## Repository

```
cosmufr/
  inference.py      cosmufr.infer(...)
  benchmark.py      load_benchmark(), evaluate()
  diagnostics.py    weight_audit(), settling_report(), compare_checkpoints()
  figures.py        every figure, from real tensors
  model.py          architecture, frozen for this checkpoint
benchmark/          the 6,000-row evaluation set
reports/            measured results
tests/              determinism, reproducibility, gradient flow
examples/           notebook walkthrough
app.py              Gradio demo
```

## Citation

```
@misc{cosmufr_run4_2026,
  title  = {CosmUFR Run 4: a belief-settling network for cosmological parameter
            inference, with an audit of its training defects},
  author = {Rajgor, Aaditya},
  year   = {2026},
  url    = {https://github.com/arajgor1/cosmufr-run4}
}
```

MIT licensed. See [LICENSE](LICENSE).
