# CosmUFR

**A model that reads a matter power spectrum and returns eight cosmological
parameters — and the audit that found its central mechanism never trained.**

> **Live demo:** https://aadityarajgor27--cosmufr-demo-serve.modal.run
> **Weights:** https://huggingface.co/arajgor1/cosmufr-run4

Everything below is measured with the released package. The test set ships in
this repository, so every number regenerates on your machine without asking
anyone for anything.

---

## Install and run

```bash
git clone https://github.com/arajgor1/cosmufr-run4
cd cosmufr-run4
pip install -e ".[demo]"
```

```python
import cosmufr

model  = cosmufr.load_model()        # pulls best.pt from HuggingFace, 545 MB
bench  = cosmufr.load_benchmark()    # 6,000 held-out spectra, ships here
result = cosmufr.infer(bench.pk_z0[0], bench.pk_z047[0], model=model)

print(result.params)                 # {'Om': 0.387, 's8': 0.672, ...}
```

Inputs are `P(k)` on 200 log-spaced bins over k ∈ [0.1, 4.5] h/Mpc, at z=0 and
z=0.47, in (Mpc/h)³. Raw `P(k)` or `log10 P(k)` are both accepted.

```bash
python -m cosmufr.reproduce          # regenerates every table below
python -c "import cosmufr; print(cosmufr.weight_audit(cosmufr.load_model()).table())"
```

Two things the API will not do quietly: `result.sigmas` is a clamp floor and not
an error bar, and `result.pk_recon` is a constant and not a reconstruction. Both
are defects, listed at the end.

[`examples/01_quickstart.ipynb`](examples/01_quickstart.ipynb) walks the whole
thing end to end.

---

## 1 · What I did

**The problem.** Turning a sky survey into a statement about the universe means
running the physics backwards. The conventional route guesses a cosmology,
simulates it, compares, and repeats a few hundred thousand times. It works, and
it costs CPU-days per analysis.

**The bet.** Train a network on enough simulated universes that it recognises one
on sight, and the inversion becomes a single forward pass. If that holds,
analyses currently priced out by compute become routine.

**The design, and why.** Rather than answering in one shot, the model was built
to arrive gradually: read the spectrum into a 1024-dimensional "belief" about
which universe this is, sharpen that belief over sixteen steps by rolling
downhill on a score the model learns for itself, then read eight parameters off
the settled belief. The reason for the extra machinery is that a single-pass
estimator has to commit immediately, while a model that refines can in principle
spend more effort on a hard observation than an easy one. That is the research
claim, and it is the claim this release does not get to test.

**What it learned from.** 84.5M spectra across fourteen suites — BACCO, SP(k),
BCemu, DarkEmulator, CAMB, and CAMELS IllustrisTNG and SIMBA. Almost all of it is
emulator output: smooth fitted functions, not simulations. About 6,000 rows come
from full hydrodynamic simulations, which is 0.007% of the corpus.

**What I ran.** Four full training runs, then four more architectural variants
during an investigation into why six of the eight parameters would not improve.
Eight runs in total.

---

## 2 · What I observed

### The design did not survive its own weights

I opened the finished checkpoint and checked, part by part, what had actually
changed. There are three outcomes, not two:

| outcome | share | which parts |
|---|---|---|
| **Never trained.** No gradient reached these at any point, in any of the eight runs. They hold their initial random values. A line of code disconnected them from what was being optimised, so more training could not have moved them. | 24% | `obs_encoder` reads the spectrum · `belief_proposal` forms the first guess · `settling` refines it sixteen times · `halo_head` unused side output |
| **Trained, and stopped reading the input.** These received a gradient and changed. What they settled on does not depend on the spectrum. | 56% | the three scoring heads return the same value for wildly different spectra · `gen_head` returns one number at every scale · `unc_head` sits on its clamp floor |
| **Trained, and works.** Every number the model reports comes from here, reading a fixed random projection of the input. | 1.2% | `param_head` |

The remaining 19% is an unused single-redshift copy of the reading and guessing
stages, plus a bank of stored reference states. Both trained. Neither affects an
answer.

**So both of these are true**, and stating only one of them is how this stayed
confusing for months: training ran and moved most of the model, *and* the three
parts the whole design rests on never moved at all.

**How I know.** Training zero-initialises every `Linear` bias, and the first
gradient to reach one moves it off zero; 84 are still bit-exactly `0.0` after
forty epochs. Independently: diffing the finished weights against a checkpoint
thirty-five epochs earlier, all 204 tensors in those three modules are identical
to the last digit, while the read-out layers moved 66 to 79 percent.

**Why.** One line in released source, no checkpoint needed:

```python
for step in range(k):
    b = b.detach()                      # <- severs everything upstream
    with torch.enable_grad():
        b_g = b.requires_grad_(True)
        E = energy_fn(b_g, z.detach(), b_prev.detach())
        grad = torch.autograd.grad(E.sum(), b_g)[0]
    b = b - eta * P * grad.detach()
```

**And reconnecting it would not be enough.** The refinement was meant to work by
rolling downhill on a score. The scoring heads did train, and converged on a
landscape that is flat: the score varies by about one part in seven million
across completely different spectra, its slope has norm 0.11 against a belief of
norm 16.5, and sixteen steps could move the belief half a percent at most,
whatever the input. So there are two faults, not one. The first is a small change
I can verify before spending anything on training. The second is a research
problem.

### How accurate it actually is

Typical error on 162,795 held-out spectra, measured where each parameter varies,
against what a published survey achieves on the same quantity:

| parameter | this model | published | from |
|---|---|---|---|
| Ω_m | 0.027 | 0.007 | Planck 2018 |
| σ₈ | 0.028 | 0.006 | Planck 2018 |
| h | 0.041 | 0.005 | Planck 2018 |
| n_s | 0.021 | 0.004 | Planck 2018 |
| Ω_b | 0.0046 | 0.0006 | Planck 2018 |
| w₀ | 0.058 | 0.055 | DESI DR2 + CMB + SN |
| Σm_ν | 0.115 eV | < 0.07 eV (95%) | DESI DR2 + CMB |
| w_a | 0.158 | 0.2 | DESI DR2 + CMB + SN |

Four to eight times worse than a real survey on everything the CMB constrains
well. The error on neutrino mass is larger than the entire range that parameter
is currently allowed to occupy. The error on h is about 70% of the whole Hubble
tension, so the model cannot speak to that question at all. Every row is generous
to this model: a survey constraint is a marginalised posterior on noisy sky data
with a covariance, and this is point-estimate scatter on noiseless emulator
spectra with no window, no shot noise and no galaxy bias. That matters most for
the two dark-energy rows, which look competitive and are not.

The same result as R², which is what earlier material about this project led
with. R² is a ratio against the spread of the truth in whatever slice you
sampled, so widening a prior raises it without changing the model:

| Parameter | All test data | Where it varies | Bundled benchmark | Verdict |
|---|---|---|---|---|
| Ω_m | 0.717 | **0.720** | 0.687 | recovered |
| σ₈ | 0.756 | **0.757** | 0.738 | recovered |
| w₀ | 0.586 | **0.614** | 0.599 | recovered |
| h | 0.501 | **0.498** | 0.475 | partial |
| Ω_b | 0.364 | **0.363** | 0.353 | partial |
| n_s | 0.338 | **0.339** | 0.331 | partial |
| w_a | 0.165 | **0.185** | 0.148 | weak |
| Σm_ν | 0.407 | **0.011** | 0.410 | not recovered |

**Read the second column.** On data where a parameter is held fixed there is no
spread, so the ratio is meaningless. For neutrino mass that is the whole story:
0.41 on all data, 0.011 once measured honestly.

**What reproduces.** The bundled 6,000-case benchmark ships here and regenerates
its own column to about 1e-6 on any machine. The full-test column came from a
private split and cannot be checked from outside; the benchmark lands within
about 0.03 of it and narrows that gap rather than closing it.

### Is the large model earning its size?

Ridge regression on the same 400 inputs, fit on half the benchmark, both scored
on the same held-out half:

| Parameter | Ridge, 400 features | CosmUFR, 136M | Winner |
|---|---|---|---|
| Ω_m | **0.738** | 0.716 | ridge |
| σ₈ | 0.734 | **0.777** | cosmufr |
| h | 0.417 | **0.496** | cosmufr |
| n_s | 0.205 | **0.343** | cosmufr |
| Ω_b | 0.312 | **0.353** | cosmufr |
| w₀ | −0.014 | **0.612** | cosmufr |
| Σm_ν | 0.242 | **0.423** | cosmufr |
| w_a | 0.018 | **0.170** | cosmufr |

CosmUFR is ahead on seven of eight and loses on matter density, which is written
into the height of the curve and does not need a large network to read. Read the
asymmetry honestly: ridge is fitted on 3,000 rows and CosmUFR trained on 84.5
million, a four-order-of-magnitude advantage, and it still loses that row. The
comparison this does not make, and the one that would settle the architecture, is
a network of matched size trained directly on the same inputs. Rerun with
`python scripts/ridge_baseline.py`.

### Where it breaks

On `camels_astrid_x`, the one evaluation slice from a full hydrodynamic
simulation, the model scores worse than a constant predictor on six of the
seven parameters that suite varies, including −5.08 on h. **That result is confounded, and I say so.**
That suite also stores its two redshift channels in the opposite order to every
other suite, so it differs from the corpus in two ways at once. Swapping the rows
back does not rescue the scores, and the other reversed suite scores normally,
which points at the gas physics rather than the ordering — but pointing is not
measuring. Separating them needs a rerun with the ordering corrected.

**The neutrino number is really a baryon number.** Two suites carry the same
spread of neutrino masses and differ only in whether a baryonic correction is
applied: R² 0.516 without it, −1.343 with it, and the typical error more than
doubles. Massive neutrinos suppress small-scale structure by streaming out of it;
feedback suppresses it by pushing gas out of it. Over these scales the two look
alike, and nothing in the eight output slots distinguishes them.

**One suite is a data defect.** `bacco_multiz` is 15% of the test set and scores
zero on everything, because its two redshift channels are byte-identical copies
and carry no growth information.

### And my explanation for all of it was wrong

Earlier material claimed a Fisher information ceiling of 0.49 and said the model
had reached it, so the limit lay in the observable rather than in the
architecture. **That claim is withdrawn.** Two objections, either fatal on its
own.

*The arithmetic.* Substituting the reported scores into the same weights gives
0.559, above the claimed bound — and all eight parameters individually exceed
their own claimed maximum. A limit every one of your measurements violates is not
a limit.

*The concept.* A Fisher forecast needs a data covariance: a survey volume, a
shot-noise term, a binning. Every row here is a noiseless emulator evaluation, so
the covariance is zero, the information is unbounded, and no such ceiling can
exist. A score short of perfect on noiseless data is a limit of the estimator,
never of the information. What the calculation actually was is a weighted average
of my own scores, with weights I chose, recovering the number the model had
already scored.

### And the run-to-run comparisons were noise

Epoch-to-epoch R² varied by ±0.03 to 0.10, larger than nearly every difference
that drove five months of decisions. There is no per-run score table in this
repository for that reason.

---

## 3 · What I conclude

**The architecture this project set out to test has not been tested.** The
mechanism that makes it interesting never ran, so nothing here is evidence for or
against the idea. What exists is a fast, honest, checkable baseline.

**What it can do today.** Return eight parameters in about a third of a second on
one CPU core, with no simulator and no chain. Recover matter density and
clustering amplitude to roughly four times worse than a survey but the right
shape. Regenerate every published number from a benchmark that ships with the
code. Report its own faults on every run.

**What it cannot do.** Give you an uncertainty. Touch real survey data — the
input is a clean simulated spectrum, and a measured one arrives with a window,
shot noise, a mask and galaxy bias, none of which this has ever seen. Handle gas
physics. Separate neutrino mass from feedback. Support any claim about the
architecture.

**What has to happen next, in order.**

1. **Reconnect the refinement.** A small code change, guarded by a test that
   already ships here. Free, and verifiable before any training spend.
2. **Give the scoring heads something to score.** The harder one. Without it the
   reconnection buys nothing, because there is still no landscape to descend.
3. **Produce a real uncertainty**, and check it the way the field checks one:
   does the stated interval contain the truth as often as it claims?
4. **Rebalance the corpus.** The parameters recovered worst are the ones it
   barely varies — dark energy sits at its fiducial value in 86% of rows.
5. **Then one training run**, with a pass mark written down before it starts
   rather than after.

**What this is aimed at.** A survey measurement interpretable in seconds rather
than weeks, with an uncertainty you can defend to a referee. Get there and the
analyses nobody runs today because they cost too much become ordinary. Nothing
here is at that point, and the honest thing to say about the list above is that
the second item is a research problem rather than a task.

**Open questions I want an outside view on.** Is gradual refinement worth
pursuing at all once the fault is repaired, or does a single-pass estimator reach
the same place? How much of the weakness on h, w₀ and w_a is a real information
limit of `P(k)` rather than training coverage? Would lower k_min, more redshifts,
or explicit acoustic-scale features make h identifiable?

---

## Limitations

Stated in full, because a careful reader finds all of it within ten minutes
anyway and it is better coming from me.

1. **The belief pipeline never trained.** Encoder, belief proposal and settling
   core sit at initialization.
2. **The energy landscape is flat.** The scoring heads collapsed to an
   input-independent constant, so there is nothing to descend, and reconnecting
   the gradient path alone would not change that.
3. **Reported uncertainties are meaningless.** σ = 0.1 for six of eight
   parameters on every input. Do not use them.
4. **The P(k) reconstruction is a constant.** The generative head returns the
   same value at every scale, for every input, and for a random belief vector.
   Its reported MSE of 0.687 is the variance of `log10 P(k)` about a constant.
5. **Neutrino mass is not recovered**, and is confounded with baryonic feedback.
   R² = 0.011 where it varies, and it swings from 0.516 to −1.343 between two
   suites that differ only in whether a baryonic correction is applied.
6. **It fails on hydrodynamic physics, for reasons not yet separated.** On the
   one evaluation slice from a full hydrodynamic simulation it scores worse than
   a constant predictor on six of the seven parameters it varies. That slice also has a
   redshift-ordering defect, so the physics and the defect are confounded and
   neither is measured.
7. **Two suites have their redshift channels reversed.** `camb_nl` and
   `camels_astrid_x` store z=0.47 where every other suite stores z=0, which is 39
   of the 6,000 bundled test spectra. The demo warns when it sees this. Found in
   September 2026, after the model shipped.
8. **The input is not an observable.** Emulator matter power spectra, with no
   survey window, no shot noise, no mask and no galaxy bias. There is no path
   from this to survey data that does not go through all four.
9. **There is no noise model anywhere.** Every row is a deterministic emulator
   evaluation, so there is no covariance, no likelihood, and nothing that would
   make a posterior width a physical quantity.
10. **The anomaly score is not usable.** The energy subsystem diverged; `E_con`
    sits around −4.6e5, five orders of magnitude from the value quoted in earlier
    material.
11. **Two redshifts only.** Multi-redshift generalization is unvalidated and that
    corpus has a documented ordering defect.
12. **The headline table is not externally reproducible.** It was measured on a
    private split; the bundled benchmark narrows that gap to about 0.03.
13. **No ablation.** There is a linear baseline but no ablation of the
    architecture's own components, and no network of matched size trained
    directly on the same inputs. Until that exists, nothing here shows the
    architecture earns its size.
14. **Historical cross-run comparisons in this project are untrustworthy**,
    because epoch-to-epoch R² noise of ±0.03 to 0.10 was never controlled for.

Earlier published figures for this model (Ω_m 0.907, σ₈ 0.911, h 0.604) are
superseded and should not be cited. They came from the training-time evaluator
on a validation set that was later corrected, with a checkpoint selected as best
from inside that noise.

---

## Provenance

| | |
|---|---|
| Checkpoint | `best.pt`, epoch 30, phase 4 |
| SHA256 | `5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1` |
| Parameters | 136,194,617 |
| Trained | 2026-04-14, single B200, batch 4096, BF16 |
| Re-measured | 2026-09-04, released package, deterministic split |

## Repository

```
cosmufr/
  inference.py      cosmufr.infer(...)
  benchmark.py      load_benchmark(), evaluate()
  diagnostics.py    weight_audit(), settling_report(), compare_checkpoints()
  validate.py       input checking with specific errors
  figures.py        every figure, from real tensors
  diagram.py        the architecture diagram, both modes
  reproduce.py      python -m cosmufr.reproduce
benchmark/          the 6,000-case evaluation set
reports/            measured results and the ridge baseline
scripts/            ridge_baseline.py
tests/              determinism, reproducibility, gradient flow
examples/           notebook walkthrough
server.py           the live site and demo
app.py              a Gradio version, for local use
```

## Citation

```bibtex
@misc{cosmufr_run4_2026,
  title  = {CosmUFR: learned inference of cosmological parameters from the
            matter power spectrum, with an audit of its training defects},
  author = {Rajgor, Aaditya},
  year   = {2026},
  url    = {https://github.com/arajgor1/cosmufr-run4}
}
```

MIT licensed. If you work on cosmological inference and any of the open
questions above look answerable, I would like to hear from you.
