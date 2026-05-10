"""
CosmUFR Run 4 — neural cosmological inference from matter power spectra.

Public API
----------
>>> import cosmufr
>>> model  = cosmufr.load_model(ckpt_path="best.pt")
>>> result = cosmufr.infer(pk_z0, pk_z047, model=model)
>>> result.params      # 8 cosmology parameters
>>> result.sigmas      # per-parameter uncertainties

For the architecture and load paths, see cosmufr.model and cosmufr.load.
"""
from cosmufr.config import CosmUFRConfig
from cosmufr.model import CosmUFRLite
from cosmufr.load import load_model, PARAM_LABELS, HF_REPO_ID
from cosmufr.inference import infer, CosmUFRResult

__version__ = "1.0.0"
__all__ = [
    "CosmUFRConfig",
    "CosmUFRLite",
    "CosmUFRResult",
    "load_model",
    "infer",
    "PARAM_LABELS",
    "HF_REPO_ID",
    "__version__",
]
