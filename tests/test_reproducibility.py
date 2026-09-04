"""
Reproducibility: the released numbers must regenerate on any machine.

The original Run 4 table could not be checked by anyone outside the author's
training infrastructure, because the validation split lived only there. These
tests assert that the bundled benchmark closes that gap: running it here
reproduces reports/honest_eval.json.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest

import cosmufr

CKPT = os.environ.get("COSMUFR_CKPT", "_local_ckpt/best.pt")
REPORT = Path(__file__).resolve().parent.parent / "reports" / "honest_eval.json"

needs_ckpt = pytest.mark.skipif(
    not os.path.exists(CKPT), reason=f"checkpoint not found at {CKPT}"
)

# The bundled benchmark is a 6,000-row proportional subsample of a 162,733-row
# split, so its R2 differs from the full-split value by sampling noise. This is
# the tolerance that difference must stay inside.
BENCH_VS_FULL_TOL = 0.06


def test_benchmark_is_present_and_well_formed():
    b = cosmufr.load_benchmark()
    assert len(b) == 6000
    assert b.pk_z0.shape == (6000, 200)
    assert b.pk_z047.shape == (6000, 200)
    assert b.params.shape == (6000, 8)
    assert b.k_bins.shape == (200,)
    assert np.isfinite(b.pk_z0).all() and np.isfinite(b.pk_z047).all()
    assert np.isclose(b.k_bins.min(), 0.1, rtol=1e-3)
    assert np.isclose(b.k_bins.max(), 4.5, rtol=1e-3)
    assert (b.pk_z0 > 0).all(), "raw P(k) must be positive"


@needs_ckpt
def test_inference_is_deterministic():
    """Three identical calls must give bit-identical parameters."""
    model = cosmufr.load_model(ckpt_path=CKPT)
    b = cosmufr.load_benchmark()
    runs = [cosmufr.infer(b.pk_z0[0], b.pk_z047[0], model=model).params_array
            for _ in range(3)]
    assert np.array_equal(runs[0], runs[1])
    assert np.array_equal(runs[0], runs[2])


@needs_ckpt
@pytest.mark.slow
def test_benchmark_reproduces_the_published_table():
    """
    The headline claim of this release: our numbers regenerate on your machine.

    If this fails, either the checkpoint changed or something in preprocessing
    drifted. Both are worth knowing about.
    """
    published = json.loads(REPORT.read_text())["benchmark"]["metrics"]

    model = cosmufr.load_model(ckpt_path=CKPT)
    result = cosmufr.evaluate(model, cosmufr.load_benchmark())

    for lbl, blk in published.items():
        got = result.overall[lbl]["r2"]
        want = blk["r2"]
        if want is None:
            assert got is None, f"{lbl}: expected undefined R2, got {got}"
        else:
            assert got is not None, f"{lbl}: expected R2 {want}, got undefined"
            assert abs(got - want) < 1e-3, (
                f"{lbl}: R2 {got:.4f} does not match published {want:.4f}"
            )


@needs_ckpt
@pytest.mark.slow
def test_bundled_benchmark_tracks_the_full_validation_split():
    """
    The bundled subset must be representative, not flattering.

    An earlier version of the benchmark sampled evenly across sources instead
    of proportionally. That changed the source mix, and therefore the variance
    R2 divides by, dropping Om from 0.72 to 0.17 without the model changing at
    all. This test guards against that class of mistake.
    """
    report = json.loads(REPORT.read_text())
    full = report["full_val_metrics"]
    bench = report["benchmark"]["metrics"]

    for lbl, blk in full.items():
        a, b = blk["r2"], bench[lbl]["r2"]
        if a is None or b is None:
            continue
        assert abs(a - b) < BENCH_VS_FULL_TOL, (
            f"{lbl}: bundled benchmark R2 {b:.3f} drifts from the full split "
            f"{a:.3f} by more than {BENCH_VS_FULL_TOL}. The subset is not "
            f"representative."
        )
