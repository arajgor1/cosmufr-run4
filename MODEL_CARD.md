---
license: mit
tags:
  - cosmology
  - astronomy
  - emulator
  - power-spectrum
  - belief-state
  - ufr
  - inverse-problem
pipeline_tag: other
library_name: pytorch
---

# CosmUFR Run 4

CosmUFR is a 136M-parameter neural network that infers all 8 cosmological
parameters from the matter power spectrum P(k) measured at two redshifts
(z=0, z=0.47). Instead of a transformer, the network refines a 1024-dim
**belief** vector through 16 steps of **energy gradient descent**
("settling") guided by three learned energy heads, then reads cosmology
parameters off the settled belief. This is the architectural-novelty
contribution of an in-progress PhD proposal.

This release — **Run 4** — is the best stable checkpoint as of May 2026 and
is shared as a clean baseline. It is honest about which parameters work and
which still need work; see *Performance* below. A follow-on **Run 8** is
planned to push the weaker parameters into the GREEN band.

## TL;DR

| Field | Value |
|---|---|
| Architecture | Belief-settling MLP with 16-step energy GD; no attention |
| Parameters | 136.2M (inference graph; ~137M including pretraining halo head) |
| Inputs | log10 P(k) at z=0 and z=0.47, 200 k-bins each (k in [0.1, 4.5] h/Mpc) |
| Outputs | 8 cosmology params + 8 per-param 1-sigma uncertainties + reconstructed log10 P(k) at any queried k |
| Parameters predicted | Om, s8, h, ns, Ob, w0, mv, wa |
| Training data | 84.5M cosmology -> P(k) samples from 14 source suites (CAMB, CAMELS, BACCO, Quijote, BCemu, DarkEmulator, SPk, plus ns/w0/multi-z grids) |
| Training compute | NVIDIA B200 (192 GB), BF16, batch 4096, 60 epochs of Phase 4 fine-tune warm-started from Run 3 |
| Training completed | 2026-04-14 |
| Strict GREEN params | 2 / 8 (Om R²=0.907, s8 R²=0.911) |
| ECE | 0.39 (loose; under-calibrated, planned post-hoc temperature scaling in Run 8) |

## Performance

Evaluated on Run 4's own validation split — same metric definitions and
data split used during training. Numbers are read directly from the
checkpoint's stored `metrics` dict.

Strict GREEN target follows the project's evolution roadmap: R² >= 0.90 for
top-tier params (Om, s8, h, w0), R² >= 0.80 for the rest.

| Param | RMSE   | R²    | Strict GREEN | Status |
|-------|--------|-------|--------------|--------|
| Om    | 0.0179 | 0.907 | >= 0.90 | GREEN |
| s8    | 0.0228 | 0.911 | >= 0.90 | GREEN |
| h     | 0.0330 | 0.604 | >= 0.90 | YELLOW (best Run 4-series result; +0.116 over Run 3) |
| w0    | 0.0229 | 0.742 | >= 0.90 | YELLOW |
| ns    | 0.0198 | 0.353 | >= 0.80 | YELLOW |
| Ob    | 0.0040 | 0.406 | >= 0.80 | YELLOW |
| mv    | 0.0963 | 0.410 | >= 0.80 | YELLOW |
| wa    | 0.0520 | 0.187 | >= 0.80 | YELLOW |
| ECE   | -      | -     | < 0.10 | RED — calibration is loose (0.39) |
| pk_mse| -      | -     | informational | 0.687 (log-space MSE on val P(k)) |

**Strict GREEN count: 2 / 8** (Om, s8).
A looser ship-readiness definition that counts h GREEN at R² >= 0.55 yields
**3 / 8**. The 5 / 8 GREEN target is a Run 8 goal, NOT achieved by Run 4.

## Intended use

- Sketching cosmological constraints from a measured / simulated matter
  power spectrum at z=0 and z=0.47, on the standard 200-bin k-grid in
  k = [0.1, 4.5] h/Mpc.
- Quick-look inference where speed (one forward pass, ~seconds on CPU)
  matters more than the precision of any individual parameter.
- Demonstrating the belief-settling architecture for research /
  pedagogical purposes.

## Out of scope

- **High-precision constraints on h, ns, Ob, w0, mv, wa.** Their R² and
  uncertainty bands are not yet at paper-quality.
- **Inference at redshifts other than z=0 and z=0.47.** Multi-z
  generalization was not validated for this release.
- **Calibrated uncertainty intervals.** ECE = 0.39 means the 68%/95%
  intervals are over-confident for several parameters.
- **Inputs on a different k-grid.** The model expects 200 log-spaced bins
  from k=0.1 to k=4.5 h/Mpc; resample your data accordingly.

## Architecture

Forward pass:

1. **ObsEncoder** — log10 P(k) [400-d concat of two redshifts] -> z [1024-d]
2. **BeliefProposal** — (z, b_prev) -> b_hat [1024-d] (residual update)
3. **SettlingCore** — 16-step energy gradient descent on b, with a learned
   per-step preconditioner P(b) and step size eta(b). The energy is
   `l_obs * E_obs(b, z) + l_con * E_con(b) + l_dyn * E_dyn(b, b_prev)`,
   where each E_* is its own MLP head.
4. **Task heads** off the settled belief b_star:
   - **ParameterHead** -> 8 cosmology params (sigmoid-clamped to physical priors)
   - **UncertaintyHead** -> 8 variances (softplus + epsilon, trained with NLL)
   - **GenerativeHead** -> implicit field that reconstructs log10 P(k) at any
     queried k value (B, K -> mean, logvar)
5. The **AttractorBank** (4096 prototype belief vectors, EMA-updated during
   training) is included for compatibility with the training graph but is
   inactive at inference.

There is no attention, no MoE, no transformer. All blocks are residual
MLPs with GELU + LayerNorm.

## Training data

84,505,755 (cosmology -> P(k)) pairs from 14 source datasets:

| Source | Samples | Notes |
|---|---|---|
| BACCO | 26,000,000 | linear + non-linear emulator |
| SPk | 18,328,000 | non-linear matter spectrum emulator |
| BCemu | 12,000,000 | with baryonic feedback |
| BACCO_neutrino | 10,000,000 | massive-neutrino sector |
| BACCO_full8 | 10,000,000 | extended 8-param coverage |
| BACCO_multiz | 5,000,000 | multi-redshift coverage |
| BCemu_neutrino | 2,000,000 | with neutrinos + baryons |
| DarkEmulator | 1,000,200 | non-linear from Quijote |
| CAMB_NL | 200,000 | non-linear CAMB anchor |
| CAMB_wa_grid | 50,000 | wa-axis fine grid |
| CAMELS_IllusTNG / SIMBA | 6,000 | hydrodynamic |
| Other (Quijote, multi-z grid, ns/w0 grids) | small | gap-filling |

## Training procedure

- Phases 0-3 (warmup -> recovery -> calibrate -> generative head ramp) on
  T4 / A100; Phase 4 sequential fine-tune on a single B200 (192 GB).
- Run 4 specifically: 60 epochs of Phase 4 fine-tune, B=4096, BF16, AdamW
  with `lr_core = 6e-4` and `lr_energy = 2e-4`, cosine LR decay,
  weight-decay 0.01, gradient clipping 1.0.
- Two-optimizer setup so the energy heads update at a different rate from
  the rest of the model.
- Wall time: ~10 hours on B200, $40 compute.

## Limitations and biases

- Trained mostly on smooth analytic / emulator P(k); real-data noise (shot
  noise, mask effects) is not modeled. Treat as a smooth-P(k) emulator
  inversion, not a fully end-to-end likelihood.
- Performance on h, ns, Ob, w0, mv, wa is below paper-quality target. Do
  not use this release for science-quality constraints on those parameters.
- The model's uncertainty heads are trained with NLL but not separately
  calibrated; ECE is loose (0.39). Post-hoc temperature scaling is on the
  Run 8 roadmap.
- Training data leans heavily on emulators (BACCO, SPk, BCemu, DarkEmulator)
  rather than direct N-body; any systematic in those emulators is inherited.

## How to use

```python
import cosmufr
import numpy as np

model  = cosmufr.load_model()       # private HF download (set HF_TOKEN)
arr    = np.load("examples/synthetic_pk.npy")
result = cosmufr.infer(arr[0], arr[1], model=model)

print(result.params)        # 8-key dict
print(result.sigmas)        # 8-key dict (1-sigma)
print(result.pk_recon)      # reconstructed log10 P(k), shape (200,)
print(result.energy_log)    # 17 floats: settling-energy trajectory
```

For the Fisher-ellipse corner-plot demo, see
`examples/02_corner_plot_demo.ipynb` in the GitHub repo.

## Files

| File | Purpose |
|---|---|
| `best.pt` | model weights (545 MB, PyTorch state_dict) |
| `config.json` | architecture hyperparameters (mirrors `cosmufr.CosmUFRConfig`) |
| `README.md` | this card (HF auto-renders) |

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

MIT.
