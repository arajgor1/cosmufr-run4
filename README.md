# CosmUFR

**A model that reads a matter power spectrum and returns eight cosmological
parameters, released with an audit showing that its refinement core is still at
its initial values.**

> **Live demo:** https://aadityarajgor27--cosmufr-demo-serve.modal.run
> **Weights:** https://huggingface.co/arajgor1/cosmufr-run4

CosmUFR is an independent, synthetic-data prototype for cosmological parameter
prediction. It was designed to refine its answer over several steps. In the
released checkpoint that refinement does not operate, so what this repository
demonstrates is a point predictor and an audit of why the intended mechanism
never trained. Whether refinement would help is untested.

Every number below is labelled by where it comes from. **Reproduced** means
regenerated from the released checkpoint and the bundled benchmark. **Saved
report** means read from a report saved earlier and not rerun. **Diagnostic**
means measured by an audit run that is not part of this package. All numerical
tables are generated from [`reports/results_v1.json`](reports/results_v1.json).

---

## Install and run

```bash
git clone https://github.com/arajgor1/cosmufr-run4
cd cosmufr-run4
pip install -e ".[demo]"
```

```python
import cosmufr

model  = cosmufr.load_model()        # pulls best.pt from Hugging Face, 545 MB
bench  = cosmufr.load_benchmark()    # 6,000 held-out spectra, ships here
result = cosmufr.infer(bench.pk_z0[0], bench.pk_z047[0], model=model)

print(result.params)                 # {'Om': 0.387, 's8': 0.672, ...}
```

Inputs are `P(k)` on 200 log-spaced bins over k ∈ [0.1, 4.5] h/Mpc, at z=0 and
z=0.47, in (Mpc/h)³. Raw `P(k)` or `log10 P(k)` are both accepted.

```bash
python -m cosmufr.reproduce          # regenerates the benchmark table and the diagnostics
```

Two things the API will not do quietly: `result.sigmas` is a clamp floor and not
an error bar, and `result.pk_recon` is a constant and not a reconstruction. Both
are listed under limitations.

[`examples/01_quickstart.ipynb`](examples/01_quickstart.ipynb) walks through it
end to end.

---

## 1 · What I did

**The problem.** Inferring cosmological parameters from a measurement usually
means guessing parameters, generating the observable they imply, comparing, and
repeating many times. That is reliable and slow.

**The approach tested.** Train a network on emulator-generated matter power
spectra so that a new spectrum maps to its parameters in one forward pass.

**The design, and why.** Rather than answering in one shot, the model was built
to arrive gradually: encode the spectrum into a 1024-dimensional belief, refine
that belief over sixteen steps of descent on a score the model learns, then read
eight parameters off the refined belief. The reason for the extra machinery is
that a model which refines could, in principle, spend more effort on a hard
observation than an easy one. That is the research question, and this release
does not test it.

**What it learned from.** Historical documentation for Run 4 gives 84.5 million
training rows, drawn mostly from emulators (BACCO, BCemu, SP(k), Dark Emulator,
CAMB) with small hydrodynamic components. No manifest for that exact corpus has
been located. The master arrays now in storage hold 86,391,712 rows (May 2026)
and, after a June rebuild that no released model used, 86,215,712.

**What I ran.** Runs 1 to 4, of which Run 4 is released, then a series of Run 5
to 8 experiments, several of them short smoke runs.

---

## 2 · What I observed

### The core of the design is still at its initial values

I compared the released weights with Run 2's checkpoint saved before any
optimizer step, and ran the training code itself, one revision at a time, to see
which parts of the model its loss can reach.

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

**What the comparison shows.** The encoder (38 tensors), the belief proposal
(38) and the refinement networks (128) in Run 4 are bit-identical to Run 2's
state before training began. Those three modules are 22.7% of the parameters.
Training changed most of the rest.

**Why they did not move.** In the code that trained Run 4, the refinement loop
detaches the belief at every step, evaluates the energy on a detached encoding,
and computes its step size and preconditioner without gradients. The table's
fifth column shows the result: no gradient reaches the encoder, the proposal or
the refinement networks. Later training code reconnects the encoder, and the
proposal only when every refinement step is retained; the step-size and
preconditioner networks receive no gradient in any configuration examined. A
repair has not been attempted.

**What the answers come from.** The eight parameters are computed by
`param_head` (1.2% of parameters) from a belief that is a fixed random function
of the input.

**The energy score has a second, separate problem.** Its training loss adds the
mean energy to a contrastive term that depends only on energy differences, so
lowering every energy by the same constant lowers the loss one-for-one. A
diagnostic shift of 1,000 lowered it by exactly 1,000. Run 4's logged energy loss
fell from −479,869 at epoch 2 to −942,080 at epochs 36 to 40. That trajectory is
consistent with the unbounded direction; the log does not show what produced the
particular value. On six different benchmark spectra the energy is −926,537.375
and changes by at most one float32 step over the sixteen refinement steps.

**Two notes on the evidence.** Every Linear bias is initialised to zero, so
all-zero biases (which `cosmufr.weight_audit` reports) are a clue, not a proof:
`halo_head` has all-zero biases but its weights were uniformly rescaled by 0.619
between Run 2 and Run 4, for a reason not traced. And the checkpoint comparison
covers Runs 2 to 4. In a later "unfrozen" smoke run the encoder did receive a
small gradient (median norm 1.4e-5 against 5.6 for `param_head`, from the saved
training log), while the refinement networks received none.

The checkpoint comparison and training-path diagnosis need the historical
checkpoints and training source, which are not public. The bias audit and the
settling measurement run from the released weights:

```bash
python -c "import cosmufr; m = cosmufr.load_model(); print(cosmufr.weight_audit(m).table())"
```

### How accurate it is

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

R² divides by the spread of the truth in whatever rows are scored, so on rows
where a parameter is held fixed it measures nothing. The "rows where it varies"
columns restrict each parameter to sources that vary it. For neutrino mass that
is the difference between 0.41 and 0.01 on the validation report.

These are point-prediction errors on noiseless emulator spectra. They are not
comparable to survey constraints, which are posterior widths on real data with a
noise model, and no such comparison is made here.

### Is the large model earning its size?

<!-- table:ridge_baseline -->
Saved report: ridge fitted on 3,000 benchmark rows and both models scored on the other 3,000 (seed 20260904), R² on the held-out half. CosmUFR is higher on 7 of 8.

| Parameter | Ridge, 400 inputs | CosmUFR | Higher |
|---|---:|---:|---|
| Ω_m | 0.738 | 0.716 | ridge |
| σ₈ | 0.734 | 0.777 | cosmufr |
| h | 0.417 | 0.496 | cosmufr |
| n_s | 0.205 | 0.343 | cosmufr |
| Ω_b | 0.312 | 0.353 | cosmufr |
| w₀ | -0.014 | 0.612 | cosmufr |
| Σm_ν | 0.242 | 0.423 | cosmufr |
| w_a | 0.018 | 0.170 | cosmufr |
<!-- /table:ridge_baseline -->

A linear fit on 3,000 rows is competitive on matter density, which is written
into the height of the curve. The comparison is not symmetric in either
direction: the network had far more training data, and the ridge fit is trained
on rows from the same sources it is scored on. The comparison that would say
whether the architecture earns its size, a direct network of matched size trained
on the same inputs, has not been run. Rerun this one with
`python scripts/ridge_baseline.py`.

### Where it breaks

**The hydrodynamic slice.** On `camels_astrid_x`, 250 validation rows from a
hydrodynamic simulation, the model scores worse than predicting the mean on six
of the seven parameters that suite varies, including R² −5.08 on h (saved report).
That suite also stores its two redshift channels in the opposite order: on the
benchmark rows the low-k gap log₁₀ P(z=0) − log₁₀ P(z=0.47) is −0.31 there,
against +0.21 to +0.22 for the suites with more than 100 rows. How much of the
failure is gas physics and how much is channel order is not separated.

**Neutrino mass depends on which emulator produced the spectrum.** R² is 0.516 on
`bacco_neutrino` and −1.343 on `bcemu_neutrino` (saved report). These are
different emulators, so they differ in more than baryonic treatment. Baryonic
feedback and massive neutrinos both suppress small-scale power, which is one
possible reason; without a matched-cosmology comparison this does not separate
them.

**One suite has no second redshift.** In `bacco_multiz` the two channels are
byte-identical in all 926 benchmark rows, so it carries no growth information and
scores near zero on everything.

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

### And my explanation for all of it was wrong

Earlier material claimed a Fisher information ceiling of 0.494 on a weighted
score, said the model had reached it, and concluded that the limit lay in the
observable rather than the model. **That claim is withdrawn.**

*The arithmetic.* Substituting the scores reported at the time into the same
weights gives 0.559, above the claimed bound, and all eight parameters
individually exceed their own claimed maximum.

*The derivation.* A bound on how well parameters can be recovered needs an
observation model: what is measured, with what noise and covariance, over what
distribution of parameters. The earlier bound specified none of these. It was a
weighted average of my own scores with weights I chose. It is withdrawn, and
whether these parameters are identifiable from two noiseless spectra over this k
range, and to what accuracy, remains to be investigated.

### And the run-to-run comparisons were noise

A June review of the training logs found epoch-to-epoch R² varying by ±0.03 to
0.10, larger than most differences that drove decisions between runs. There is
no per-run score table in this repository for that reason.

---

## 3 · What I conclude

**The architecture this project set out to test has not been tested.** The
refinement did not operate in the released model, so nothing here is evidence
for or against it. What exists is a checkable point predictor and a detailed
account of why the intended mechanism did not train.

**What it does today.** Returns eight parameters from a power spectrum in a few
hundred milliseconds on CPU. Recovers matter density and clustering amplitude
with RMSE of about 0.027 and 0.028 on validation rows where they vary.
Reproduces the bundled benchmark within 1e-5 in R².

**What it does not do.** Report an uncertainty: the σ output is constant. Work on
survey data: the input is an emulator spectrum with no window, shot noise, mask
or galaxy bias. Recover neutrino mass. Support any claim about iterative
refinement.

**What would come next, in order.**

1. Build one trustworthy data subset with recorded provenance, returned-redshift
   channel mapping and splits grouped by base cosmology.
2. Make a small model train genuinely: every intended module receiving a gradient
   and an update, checked module by module, and an objective that is bounded
   below.
3. Compare refinement against a tuned direct network of matched size, with
   repeated seeds and a threshold written down before the test set is opened.

**Open questions I want an outside view on.** Is gradual refinement worth
pursuing once it can train, or does a single-pass estimator reach the same place?
How much of the weakness on h, w₀ and w_a is a limit of what two noiseless
spectra over this k range contain, and how much is training coverage? What should
a refinement score be trained to do, given that the current objective is
unbounded below?

**Long-term aim, untested.** A fast inference step that makes expensive analyses
cheaper to run. Nothing here demonstrates that it can be done or that it would
preserve the accuracy of standard methods.

---

## Limitations

1. **The refinement core is at initialization.** The encoder, the belief proposal
   and the refinement networks are bit-identical to Run 2's pre-training state,
   and the training code that produced this checkpoint gives them no gradient.
2. **The energy objective is unbounded below** along a constant shift, and on the
   benchmark the energy is constant to float32 resolution across inputs and
   refinement steps.
3. **Reported uncertainties carry no information.** σ = 0.1 for six of eight
   parameters on every benchmark input. Do not use them.
4. **The P(k) reconstruction is a constant**, 2.6327 at every k and for every
   input.
5. **Neutrino mass is not recovered** (R² 0.011 where it varies, saved report),
   and its recovery differs sharply between two emulators.
6. **It fails on the one hydrodynamic slice**, which also has a channel-order
   defect, so the two causes are not separated.
7. **Redshift channels are reversed in two bundled suites.** `camb_nl` and
   `camels_astrid_x` store the channels in the opposite order, 39 of the 6,000
   benchmark rows. The demo warns when it sees this.
8. **The rebuilt master arrays have their own reported ordering problem.** An
   independent review sampled 648 rows of the June `master_v2` rebuild and found
   `camb_nl` and `camb_wa_grid` with the opposite of the expected growth ordering.
   That is a sample, not a source-wide audit, and it is separate from item 7.
9. **The input is not an observable.** Emulator matter power spectra, with no
   survey window, shot noise, mask, galaxy bias, noise model or covariance.
10. **The split is by row, not by cosmology.** Related spectra of the same base
    cosmology can sit in both training and evaluation; leakage has not been ruled
    out.
11. **Two redshifts only.** Multi-redshift generalization is unvalidated.
12. **The validation table is not externally reproducible.** It needs the master
    validation rows and `bad_indices.npy`. The bundled benchmark reproduces within
    1e-5.
13. **No ablation and no matched direct network.** Until those exist, nothing here
    shows the architecture earns its size.
14. **Historical cross-run comparisons are unreliable**, because epoch-to-epoch
    noise was not controlled.

Earlier published figures for this model (Ω_m 0.907, σ₈ 0.911, h 0.604) are
superseded and should not be cited. They came from a training-time evaluator on a
validation set that was later corrected.

---

## Provenance

| | |
|---|---|
| Checkpoint | `best.pt`, epoch 30, phase 4 |
| SHA256 | `5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1` |
| Parameters | 136,194,617 |
| Training run | `b200-b4096-run4-20260414_031247` (single B200, batch 4096, BF16) |
| Benchmark reproduced | September 2026, torch 2.14.0 CPU, within 1e-5 in R² |
| Results file | [`reports/results_v1.json`](reports/results_v1.json) |

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
reports/            saved reports and results_v1.json, the source of every table
scripts/            ridge_baseline.py, build_results.py, render_tables.py
tests/              determinism, reproducibility, gradient flow, published tables
examples/           notebook walkthrough
server.py           the live site and demo
app.py              a Gradio version, for local use
```

## Citation

```bibtex
@misc{cosmufr_run4_2026,
  title  = {CosmUFR: a cosmological parameter-prediction prototype from the
            matter power spectrum, with an audit of its training defects},
  author = {Rajgor, Aaditya},
  year   = {2026},
  url    = {https://github.com/arajgor1/cosmufr-run4}
}
```

MIT licensed. If you work on cosmological inference and any of the open questions
above look answerable, I would like to hear from you.
