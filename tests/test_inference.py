"""
Smoke test: load the Run 4 checkpoint, run inference on a synthetic P(k), and
validate output shapes and physical ranges.

Note on the energy: this suite does NOT assert that settling descends. It does
not. See tests/test_gradient_flow.py and cosmufr.settling_report() for the
measurement and the reason.

Run from repo root:
    python -m pytest tests/test_inference.py -v
or
    python tests/test_inference.py
"""
import os
from pathlib import Path

import numpy as np
import pytest

import cosmufr


CKPT_LOCAL = Path(__file__).resolve().parents[1] / "_local_ckpt" / "best.pt"
SYNTH_PK   = Path(__file__).resolve().parents[1] / "examples" / "synthetic_pk.npy"


def _ckpt_path():
    env = os.environ.get("COSMUFR_CKPT")
    if env and Path(env).exists():
        return env
    if CKPT_LOCAL.exists():
        return str(CKPT_LOCAL)
    pytest.skip(
        f"No checkpoint found. Set COSMUFR_CKPT, or place best.pt at "
        f"{CKPT_LOCAL}. Download it from "
        f"https://huggingface.co/arajgor1/cosmufr-run4"
    )


def test_load_model():
    model = cosmufr.load_model(ckpt_path=_ckpt_path())
    n_params = sum(p.numel() for p in model.parameters())
    # Run 4 = 137M params (architecture spec).
    assert 130_000_000 < n_params < 145_000_000, f"Param count off: {n_params:,}"


def test_infer_shapes_and_ranges():
    model = cosmufr.load_model(ckpt_path=_ckpt_path())

    # Synthetic LCDM-shaped P(k): rough power-law decline.
    k = np.logspace(np.log10(0.1), np.log10(4.5), 200)
    pk_z0   = 1e4 * k ** (-1.5)
    pk_z047 = 0.6 * pk_z0          # crude growth-factor scaling

    result = cosmufr.infer(pk_z0, pk_z047, model=model)

    # Shapes
    assert result.params_array.shape == (8,)
    assert result.sigmas_array.shape == (8,)
    assert result.pk_recon.shape == (200,)
    assert result.log_k.shape == (200,)
    assert len(result.energy_log) == 17  # 1 init + 16 settling steps

    # Param ranges (sigmoid-clamped to physical priors)
    p = result.params
    assert 0.10 <= p["Om"] <= 0.50
    assert 0.52 <= p["s8"] <= 1.00
    assert 0.60 <= p["h"]  <= 0.80
    assert 0.90 <= p["ns"] <= 1.02
    assert 0.03 <= p["Ob"] <= 0.07
    assert -1.20 <= p["w0"] <= -0.80
    assert  0.00 <= p["mv"] <=  0.40
    assert -0.50 <= p["wa"] <=  0.50

    # All sigmas are strictly positive
    assert (result.sigmas_array > 0).all()

    # Guard against energy blow-up only. The released checkpoint's energy sits
    # near -9.3e5 and is flat across all 16 steps; asserting a descent here
    # would be asserting something this model does not do.
    assert abs(result.energy_log[-1]) < 1e7


def test_dict_keys_match_param_labels():
    model = cosmufr.load_model(ckpt_path=_ckpt_path())
    pk = np.ones(200, dtype=np.float32)
    result = cosmufr.infer(pk, pk, model=model)
    assert list(result.params.keys()) == cosmufr.PARAM_LABELS
    assert list(result.sigmas.keys()) == cosmufr.PARAM_LABELS


if __name__ == "__main__":
    # Allow `python tests/test_inference.py` direct execution
    test_load_model()
    print("[OK] load_model — checkpoint loads with correct param count.")
    test_infer_shapes_and_ranges()
    print("[OK] infer — shapes and physical ranges valid.")
    test_dict_keys_match_param_labels()
    print("[OK] dict keys — params/sigmas keyed by PARAM_LABELS.")
    print("\nAll smoke tests passed.")
