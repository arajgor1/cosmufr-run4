"""
cosmufr/validate.py — tell the user exactly what is wrong with their input.

This existed only as a helper cell inside the quickstart notebook, which meant
the library and the public demo both accepted whatever they were handed. An
all-zero spectrum returned a confident-looking cosmology; a spectrum full of
1e300 produced NaN parameters and no error at all. Neither is acceptable on a
public endpoint, and a demo that argues carefully about what its outputs do not
mean should not silently invent one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# The grid the model was trained on: 200 log-spaced bins over k in [0.1, 4.5].
K_MIN, K_MAX, N_K = 0.1, 4.5, 200
K_GRID = np.logspace(np.log10(K_MIN), np.log10(K_MAX), N_K)

# log10 P(k) for any plausible cosmology over this k range sits inside roughly
# [-2, 6]. Outside that, the input is not a matter power spectrum in the units
# the model expects, whatever else it may be.
LOG10_PK_MIN, LOG10_PK_MAX = -3.0, 7.0


@dataclass
class Validation:
    ok: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    interpreted_as: str = ""

    def __bool__(self) -> bool:
        return self.ok


def _check_one(name: str, a, errors: List[str]) -> Optional[np.ndarray]:
    arr = np.asarray(a)
    if arr.ndim != 1:
        errors.append(f"{name}: expected a 1-D array of {N_K} values, got shape {arr.shape}.")
        return None
    if arr.size != N_K:
        errors.append(
            f"{name}: expected {N_K} k-bins, got {arr.size}. Interpolate your "
            f"spectrum onto the training grid first: "
            f"np.exp(np.interp(np.log(K_GRID), np.log(your_k), np.log(your_pk)))."
        )
        return None
    if not np.isfinite(arr).all():
        n = int((~np.isfinite(arr)).sum())
        errors.append(f"{name}: contains {n} non-finite value(s) (NaN or inf).")
        return None
    return arr.astype(np.float64)


def validate_spectra(pk_z0, pk_z047, k=None) -> Validation:
    """
    Check an input pair before it reaches the model.

    Returns a Validation. Errors mean the input cannot be run. Warnings mean it
    will run but the result should not be trusted.
    """
    errors: List[str] = []
    warnings: List[str] = []

    a = _check_one("pk_z0", pk_z0, errors)
    b = _check_one("pk_z047", pk_z047, errors)
    if a is None or b is None:
        return Validation(ok=False, errors=errors)

    if k is not None:
        kk = np.asarray(k, dtype=np.float64)
        if kk.size == N_K and not np.allclose(kk, K_GRID, rtol=0.02):
            warnings.append(
                "The supplied k grid differs from the training grid by more than "
                "2%. Interpolate log10 P(k) onto the training grid, or the model "
                "is reading your values at the wrong scales."
            )

    # Raw P(k) or already log10? The library decides by whether any value
    # exceeds 100, so make that decision explicit and flag it when ambiguous.
    looks_raw = bool((a > 100).any() or (b > 100).any())
    interpreted = "raw P(k), log10 will be applied" if looks_raw else "already log10 P(k)"

    if looks_raw:
        for name, arr in (("pk_z0", a), ("pk_z047", b)):
            if (arr <= 0).any():
                errors.append(
                    f"{name}: looks like raw P(k) (values above 100) but contains "
                    f"non-positive entries. P(k) is a variance and cannot be "
                    f"negative or zero."
                )
        log_a, log_b = np.log10(np.clip(a, 1e-30, None)), np.log10(np.clip(b, 1e-30, None))
    else:
        log_a, log_b = a, b
        if not (0.05 < np.ptp(log_a) < 12):
            warnings.append(
                f"pk_z0 was read as log10 P(k) because no value exceeds 100, but "
                f"it spans only {np.ptp(log_a):.3g} in that reading. If you meant "
                f"raw P(k), your amplitudes are far below anything physical."
            )

    for name, lg in (("pk_z0", log_a), ("pk_z047", log_b)):
        if np.ptp(lg) < 1e-9:
            errors.append(
                f"{name}: every value is identical ({lg.flat[0]:.4g}). A flat "
                f"spectrum carries no cosmological information, and the model "
                f"will still return eight confident-looking numbers. Refusing."
            )
            continue
        if lg.min() < LOG10_PK_MIN or lg.max() > LOG10_PK_MAX:
            errors.append(
                f"{name}: log10 P(k) spans [{lg.min():.2f}, {lg.max():.2f}], "
                f"outside the plausible range "
                f"[{LOG10_PK_MIN:.0f}, {LOG10_PK_MAX:.0f}]. Check your units: the "
                f"model expects P(k) in (Mpc/h)^3 on a k grid in h/Mpc."
            )
        if np.all(np.diff(lg) > 0):
            warnings.append(
                f"{name}: increases monotonically with k. A matter power spectrum "
                f"over k = 0.1 to 4.5 h/Mpc falls with k. Your array may be "
                f"reversed."
            )

    if not errors:
        # Growth check: P(k) at z=0.47 should sit below z=0 at every scale.
        if np.median(log_b - log_a) > 0.02:
            warnings.append(
                "The z=0.47 spectrum has more power than the z=0 spectrum. "
                "Structure grows with time, so this is backwards; the two rows "
                "may be swapped."
            )

    return Validation(ok=not errors, errors=errors, warnings=warnings,
                      interpreted_as=interpreted)


def to_training_grid(k_src, pk_src) -> np.ndarray:
    """Log-log interpolate an arbitrary spectrum onto the 200-bin training grid."""
    k_src = np.asarray(k_src, dtype=np.float64)
    pk_src = np.asarray(pk_src, dtype=np.float64)
    return np.exp(np.interp(np.log(K_GRID), np.log(k_src), np.log(pk_src)))
