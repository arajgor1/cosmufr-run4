# CosmUFR Run 4

**Neural cosmological inference from matter power spectra.**
136M-parameter belief-settling network. PhD demo release (May 2026).

CosmUFR takes the matter power spectrum P(k) at z=0 and z=0.47 and infers all
8 cosmological parameters jointly with per-parameter uncertainties. The
architecture is *not* a transformer: it iteratively refines a 1024-dim belief
state through 16 steps of energy gradient descent ("settling") guided by three
learned energy heads, then reads cosmology parameters off the settled belief.

The architecture and the energy-based settling procedure are the
research-novelty contribution of an in-progress PhD proposal — this Run 4
release is the best stable checkpoint to date and is shared as a clean,
honest baseline. Subsequent attempts (Runs 5–8, May 1–10 2026) found the
weaker parameter R²s are bounded by a **data-physics ceiling** — the
theoretical maximum from log P(k) at 2 redshifts in k ∈ [0.1, 4.5] h/Mpc
matches the observed composite ceiling around 0.49. Breaking it requires
new data (higher k, more redshifts, BAO features), not new architecture.

## Quick start

```bash
pip install -e .
```

```python
import cosmufr
import numpy as np

# Load the released checkpoint (downloads from HuggingFace on first call)
model = cosmufr.load_model()

# pk_z0, pk_z047 : raw P(k) on a 200-bin log-spaced grid in k = [0.1, 4.5] h/Mpc
pk_z0   = np.load("examples/synthetic_pk.npy")[0]
pk_z047 = np.load("examples/synthetic_pk.npy")[1]

result = cosmufr.infer(pk_z0, pk_z047, model=model)

print(result.params)   # {'Om': 0.32, 's8': 0.82, 'h': 0.74, ...}
print(result.sigmas)   # 1-sigma per-parameter uncertainties
print(result.pk_recon) # reconstructed log10 P(k) at the default 200-bin grid
```

Loading from HuggingFace (private repo — set `HF_TOKEN`):

```bash
export HF_TOKEN=hf_...
```

```python
model = cosmufr.load_model()              # repo_id defaults to arajgor1/cosmufr-run4
# or
model = cosmufr.load_model(ckpt_path="best.pt")   # local file
```

## Performance — honest

Evaluated on Run 4's own validation split (apples-to-apples with the
published numbers in the project's evolution roadmap). Strict GREEN target
follows the roadmap definition (R² >= 0.90 for top-tier params, R² >= 0.80
for the rest).

| Param | R²    | Strict GREEN target | Status                                            |
|-------|-------|---------------------|---------------------------------------------------|
| Om    | 0.907 | >= 0.90             | GREEN                                             |
| s8    | 0.911 | >= 0.90             | GREEN                                             |
| h     | 0.604 | >= 0.90             | YELLOW (best yet, +0.116 over Run 3)              |
| w0    | 0.742 | >= 0.90             | YELLOW                                            |
| ns    | 0.353 | >= 0.80             | YELLOW                                            |
| Ob    | 0.406 | >= 0.80             | YELLOW                                            |
| mv    | 0.410 | >= 0.80             | YELLOW                                            |
| wa    | 0.187 | >= 0.80             | YELLOW                                            |

**Strict GREEN count: 2/8** (Om, s8). A looser ship-readiness definition
(h GREEN at R² >= 0.55) brings this to 3/8. ECE = 0.39 — calibration is
known to be loose; post-hoc temperature scaling is on the roadmap. The
weaker-param R²s are at the **theoretical ceiling for P(k) at 2 redshifts**
(verified empirically across 7 architectural variants, May 2026); breaking
them requires data extensions documented in the development repo.

The exact numbers above come from the checkpoint metadata
(`epoch_30_metrics`, Phase 4) and are reproduced on the Run 4 val set.

## Architecture in one paragraph

Input is the log10 matter power spectrum at two redshifts (concat to a 400-d
vector). An ObsEncoder MLP maps it to a 1024-d "belief" vector. A
BeliefProposal MLP combines that with the previous-step belief; a settling
core then runs 16 steps of energy gradient descent on the belief, where the
energy is the sum of three learned heads (observation-consistency,
prior-consistency, dynamics-consistency) with per-step learned
preconditioning and step-size. The settled belief feeds three task heads:
ParameterHead -> 8 cosmology params (sigmoid-clamped to physical priors),
UncertaintyHead -> 8 variances (softplus, NLL-trained), and a k-continuous
GenerativeHead that reconstructs log10 P(k) at any queried k. There is no
attention.

## Training summary

- 136M parameters (137M reported in the master spec includes a HaloMassHead
  used during pre-training; the inference-time graph is 136M)
- 84.5M cosmology -> P(k) samples across 14 source datasets (CAMB, CAMELS,
  BACCO, Quijote, BCemu, DarkEmulator, SPk, plus ns/w0/multi-z grids)
- 60 epochs of Phase 4 fine-tune on a single B200 (192 GB), warm-started
  from Run 3, B=4096 BF16, completed 2026-04-14
- Two-optimizer setup: opt_core for everything except energy heads,
  opt_energy for energy heads only

## Limitations

- Only two redshifts (z=0, z=0.47) at training time; multi-z generalization
  is not validated.
- The Hubble constant (h), dark-energy CPL parameters (w0, wa), spectral
  index (ns), baryon density (Ob), and neutrino mass sum (mv) are below
  paper-quality target. Treat their point estimates and especially their
  uncertainties accordingly.
- ECE = 0.39 — uncertainty bands are over-confident for some parameters.
- Inference assumes the input k-grid matches the training grid
  (200 log-spaced bins from k=0.1 to k=4.5 h/Mpc).
- Architecture flags (`use_explicit_z`, `use_source_aware`,
  `use_multi_z_encoder`) are hard-coded OFF in this release because the
  Run 4 checkpoint was trained without them. Do not flip them.

## Examples

- [`examples/01_quickstart.ipynb`](examples/01_quickstart.ipynb) — minimal
  five-line inference walkthrough.
- [`examples/02_corner_plot_demo.ipynb`](examples/02_corner_plot_demo.ipynb)
  — full 8x8 corner plot built from analytic Fisher ellipses derived from the
  per-parameter uncertainties returned by the model. No MCMC required.

## Repository

```
cosmufr-run4/
├── README.md              this file
├── MODEL_CARD.md          detailed model card (HF auto-renders this)
├── LICENSE                MIT
├── pyproject.toml         pip-installable
├── cosmufr/
│   ├── __init__.py        public API
│   ├── inference.py       cosmufr.infer(pk_z0, pk_z047) -> CosmUFRResult
│   ├── model.py           CosmUFRLite (Run 4 architecture, inference-only)
│   ├── config.py          CosmUFRConfig (frozen for the released checkpoint)
│   └── load.py            checkpoint loader (local or HF Hub)
├── examples/
│   ├── 01_quickstart.ipynb
│   ├── 02_corner_plot_demo.ipynb
│   └── synthetic_pk.npy   small test sample
└── tests/
    └── test_inference.py  smoke test (load + run + sanity-check shapes)
```

## Citation

```
@misc{cosmufr_run4_2026,
  title  = {CosmUFR Run 4: A 136M-parameter belief-settling network for
            cosmological parameter inference from matter power spectra},
  author = {Rajgor, Aaditya},
  year   = {2026},
  url    = {https://huggingface.co/arajgor1/cosmufr-run4}
}
```

## License

MIT — see [LICENSE](LICENSE).
