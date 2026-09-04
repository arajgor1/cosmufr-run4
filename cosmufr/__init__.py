"""
CosmUFR Run 4 — neural cosmological inference from matter power spectra.

This release ships three things: the model, a benchmark you can check it
against, and the audit that found its central defect.

Inference
---------
>>> import cosmufr
>>> model  = cosmufr.load_model(ckpt_path="best.pt")
>>> result = cosmufr.infer(pk_z0, pk_z047, model=model)
>>> result.params      # 8 cosmology parameters
>>> result.sigmas      # see the note below before using these

Reproducing the released numbers
--------------------------------
>>> bench = cosmufr.load_benchmark()
>>> print(cosmufr.evaluate(model, bench).table())

Checking the model's honesty
----------------------------
>>> print(cosmufr.weight_audit(model).table())
>>> print(cosmufr.settling_report(model, pk_z0, pk_z047).summary())

Known defects, stated up front
------------------------------
1. The observation encoder, the belief proposal network and the settling core
   never received a gradient. They sit at initialization; the read-out heads
   learned to read a fixed random projection. `weight_audit()` shows this.
2. Settling moves the belief by ~0.1% and its energy is flat to float32
   resolution. `settling_report()` shows this.
3. The uncertainty head is pinned at its clamp floor for most parameters, so
   `result.sigmas` is a constant, not a prediction. Do not use it as an error
   bar. `uncertainty_audit()` shows this.
4. m_nu recovery is R² ≈ 0.01 when measured only where m_nu varies. The higher
   number in older material was an artifact of m_nu being pinned at zero
   throughout most of the training corpus.

See README.md and MODEL_CARD.md for the full picture.
"""
from cosmufr.config import CosmUFRConfig
from cosmufr.model import CosmUFRLite
from cosmufr.load import load_model, PARAM_LABELS, HF_REPO_ID
from cosmufr.inference import infer, CosmUFRResult
from cosmufr.benchmark import (
    Benchmark, EvalResult, load_benchmark, evaluate, SOURCE_NAMES,
)
from cosmufr.diagnostics import (
    weight_audit, settling_report, uncertainty_audit, compare_checkpoints,
    WeightAudit, SettlingReport, UncertaintyAudit,
)

__version__ = "2.0.0"
__all__ = [
    "CosmUFRConfig",
    "CosmUFRLite",
    "CosmUFRResult",
    "load_model",
    "infer",
    "PARAM_LABELS",
    "HF_REPO_ID",
    "Benchmark",
    "EvalResult",
    "load_benchmark",
    "evaluate",
    "SOURCE_NAMES",
    "weight_audit",
    "settling_report",
    "uncertainty_audit",
    "compare_checkpoints",
    "WeightAudit",
    "SettlingReport",
    "UncertaintyAudit",
    "__version__",
]
