"""
cosmufr/inference.py — Public inference API.

Example
-------
>>> import cosmufr
>>> model = cosmufr.load_model(ckpt_path="best.pt")   # or HF download
>>> result = cosmufr.infer(pk_z0, pk_z047, model=model)
>>> result.params      # dict of 8 cosmology params
>>> result.sigmas      # dict of 8 per-param 1-sigma uncertainties
>>> result.pk_recon    # reconstructed log10 P(k) at default 200-bin grid
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Union

import numpy as np
import torch

from cosmufr.config import CosmUFRConfig
from cosmufr.load import PARAM_LABELS, load_model
from cosmufr.model import CosmUFRLite


# Module-level cache so repeated `infer()` calls don't reload the checkpoint.
_DEFAULT_MODEL: Optional[CosmUFRLite] = None


def _to_tensor(x, dtype=torch.float32) -> torch.Tensor:
    if isinstance(x, torch.Tensor):
        return x.to(dtype)
    return torch.as_tensor(np.asarray(x), dtype=dtype)


def _ensure_log10(pk: torch.Tensor) -> torch.Tensor:
    """
    Detect whether input is raw P(k) or log10 P(k).

    Heuristic: P(k) for the trained k-range is positive and typically
    spans 1e-2 to 1e5; log10 P(k) lives roughly in [-2, 5]. If max > 100
    we treat it as raw P(k) and take log10.
    """
    if (pk > 100).any():
        return torch.log10(pk.clamp(min=1e-30))
    return pk


@dataclass
class CosmUFRResult:
    """
    Container for a single inference call.

    Attributes
    ----------
    params : dict[str, float]
        Point estimates for [Om, s8, h, ns, Ob, w0, mv, wa].
    sigmas : dict[str, float]
        1-sigma uncertainties per parameter (sqrt of model-predicted variance).
    params_array : np.ndarray
        Same point estimates as a length-8 array, in PARAM_LABELS order.
    sigmas_array : np.ndarray
        Same sigmas as a length-8 array.
    pk_recon : np.ndarray
        Reconstructed log10 P(k) at the default 200-bin k-grid.
    log_k : np.ndarray
        log(k) values where pk_recon is evaluated (length 200).
    energy_log : list[float]
        Settling energy at each of the 17 steps (initial + 16 GD updates).
        Should be monotonically decreasing for a well-trained model.
    """
    params:        Dict[str, float]
    sigmas:        Dict[str, float]
    params_array:  np.ndarray
    sigmas_array:  np.ndarray
    pk_recon:      np.ndarray
    log_k:         np.ndarray
    energy_log:    list = field(default_factory=list)


def infer(
    pk_z0,
    pk_z047,
    model: Optional[CosmUFRLite] = None,
    ckpt_path: Optional[str] = None,
    device: Union[str, torch.device] = "cpu",
) -> CosmUFRResult:
    """
    Run CosmUFR Run 4 on a single matter power spectrum observation.

    Parameters
    ----------
    pk_z0 : array-like, shape (200,)
        P(k) at z=0 evaluated on a 200-bin log-spaced grid in k=[0.1, 4.5] h/Mpc.
        Either raw P(k) or log10 P(k) — the function auto-detects.
    pk_z047 : array-like, shape (200,)
        Same, at z=0.47.
    model : CosmUFRLite, optional
        Pre-loaded model. If None, loads (and caches) the released checkpoint.
    ckpt_path : str, optional
        Local path used when `model` is None.
    device : str | torch.device
        Target device, default 'cpu'.

    Returns
    -------
    CosmUFRResult
    """
    global _DEFAULT_MODEL

    if model is None:
        if _DEFAULT_MODEL is None:
            _DEFAULT_MODEL = load_model(ckpt_path=ckpt_path, device=device)
        model = _DEFAULT_MODEL

    cfg: CosmUFRConfig = model.cfg
    device = next(model.parameters()).device

    # Validate + log-transform if needed.
    pk_z0_t   = _ensure_log10(_to_tensor(pk_z0)).to(device)
    pk_z047_t = _ensure_log10(_to_tensor(pk_z047)).to(device)

    if pk_z0_t.shape[-1] != cfg.d_obs_single:
        raise ValueError(
            f"pk_z0 must have {cfg.d_obs_single} k-bins, got {pk_z0_t.shape[-1]}"
        )
    if pk_z047_t.shape[-1] != cfg.d_obs_single:
        raise ValueError(
            f"pk_z047 must have {cfg.d_obs_single} k-bins, got {pk_z047_t.shape[-1]}"
        )

    # Add batch dim if missing, concat into the 400-d joint observation.
    if pk_z0_t.dim() == 1:
        pk_z0_t = pk_z0_t.unsqueeze(0)
    if pk_z047_t.dim() == 1:
        pk_z047_t = pk_z047_t.unsqueeze(0)
    obs = torch.cat([pk_z0_t, pk_z047_t], dim=-1)  # [B, 400]

    with torch.inference_mode(False), torch.no_grad():
        # SettlingCore needs autograd locally for energy gradient — overall is no_grad.
        out = model(obs)

    params_arr = out["params"][0].detach().cpu().numpy()
    sigmas_arr = out["variances"][0].detach().cpu().numpy() ** 0.5
    pk_recon   = out["pk_mean"][0].detach().cpu().numpy()
    log_k      = model.log_k_train.detach().cpu().numpy()

    return CosmUFRResult(
        params       = {lbl: float(v) for lbl, v in zip(PARAM_LABELS, params_arr)},
        sigmas       = {lbl: float(v) for lbl, v in zip(PARAM_LABELS, sigmas_arr)},
        params_array = params_arr,
        sigmas_array = sigmas_arr,
        pk_recon     = pk_recon,
        log_k        = log_k,
        energy_log   = list(out["energy_log"]),
    )
