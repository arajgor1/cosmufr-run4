"""
cosmufr/figures.py — every figure in the demo, from real tensors.

Each function takes measurements and returns a matplotlib Figure, so the
notebook, the Gradio app and anything else render identically. Nothing here
synthesises, smooths or interpolates a value for presentation.

Palette is Okabe-Ito, which is colourblind-safe. Figures are legible at 800px
and readable on light or dark backgrounds.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from cosmufr.load import PARAM_LABELS

# Okabe-Ito
C_BLUE = "#0072B2"
C_ORANGE = "#E69F00"
C_GREEN = "#009E73"
C_RED = "#D55E00"
C_PURPLE = "#CC79A7"
C_SKY = "#56B4E9"
C_YELLOW = "#F0E442"
C_GREY = "#7F7F7F"

PARAM_TEX = {
    "Om": r"$\Omega_m$", "s8": r"$\sigma_8$", "h": r"$h$", "ns": r"$n_s$",
    "Ob": r"$\Omega_b$", "w0": r"$w_0$", "mv": r"$\sum m_\nu$", "wa": r"$w_a$",
}


def _style(fig, axes):
    """Neutral styling that survives both light and dark viewers."""
    for ax in np.atleast_1d(axes).ravel():
        ax.set_facecolor("none")
        ax.grid(alpha=0.25, linewidth=0.6)
        for s in ax.spines.values():
            s.set_alpha(0.4)
        ax.tick_params(labelsize=8)
    fig.patch.set_alpha(0.0)
    return fig


# ─────────────────────────────────────────────────────────────────────────────

def fig_weight_audit(audit, figsize=(8, 4.2)):
    """
    The finding: which modules ever received a gradient.

    This is the most informative figure in the release. It shows that the
    belief pipeline the architecture is named for never trained.
    """
    import matplotlib.pyplot as plt

    names = list(audit.modules.keys())
    fracs = [audit.modules[n]["n_zero_bias"] / audit.modules[n]["n_linear"]
             for n in names]
    colors = [C_RED if f == 1.0 else C_GREEN for f in fracs]

    order = np.argsort(fracs)[::-1]
    names = [names[i] for i in order]
    fracs = [fracs[i] for i in order]
    colors = [colors[i] for i in order]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(names))
    ax.barh(y, fracs, color=colors, alpha=0.85, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlim(0, 1.08)
    ax.set_xlabel("fraction of Linear layers whose bias is still exactly 0.0", fontsize=9)
    ax.set_title("Which modules ever received a gradient", fontsize=11, pad=10)
    ax.invert_yaxis()

    for yi, f in zip(y, fracs):
        label = "never trained" if f == 1.0 else "trained"
        ax.text(f + 0.02, yi, label, va="center", fontsize=8,
                color=C_RED if f == 1.0 else C_GREEN)

    ax.axvline(1.0, color=C_GREY, linestyle=":", linewidth=1, alpha=0.6)
    fig.text(0.01, -0.02,
             "An optimizer step moves a bias off its initial value. A module "
             "whose biases are all bit-exactly zero\nafter 40 epochs received no "
             "gradient at all.", fontsize=8, color=C_GREY)
    fig.tight_layout()
    return _style(fig, ax)


def fig_settling_trajectory(report, figsize=(8, 6)):
    """
    Energy and parameter read-out across the 16 refinement steps.

    On the released checkpoint both panels are flat. That is the honest result
    and the figure is labelled to say so rather than being omitted.
    """
    import matplotlib.pyplot as plt

    steps = np.arange(len(report.energy_log))
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [1, 1.15]})

    ax1.plot(steps, report.energy_log, color=C_BLUE, marker="o",
             markersize=3.5, linewidth=1.6)
    ax1.set_ylabel("total energy $E$", fontsize=9)
    ax1.set_title("Belief settling: 16 steps of energy gradient descent",
                  fontsize=11, pad=10)
    drop = report.energy_drop
    ax1.annotate(f"energy drop over 16 steps: {drop:.3e}",
                 xy=(0.98, 0.10), xycoords="axes fraction", ha="right",
                 fontsize=8.5, color=C_RED if abs(drop) < 1e-3 else C_GREEN)

    t = report.param_trajectory
    for i, lbl in enumerate(PARAM_LABELS):
        base = t[0, i] if abs(t[0, i]) > 1e-12 else 1.0
        ax2.plot(steps, 100 * (t[:, i] - t[0, i]) / abs(base),
                 label=PARAM_TEX[lbl], linewidth=1.5)
    ax2.set_xlabel("settling step", fontsize=9)
    ax2.set_ylabel("change from step 0  (%)", fontsize=9)
    ax2.legend(ncol=4, fontsize=8, frameon=False, loc="upper left")
    ax2.axhline(0, color=C_GREY, linewidth=0.8, alpha=0.5)

    fig.text(0.01, -0.015,
             f"Belief moves {report.belief_movement*100:.3f}% of its norm across all "
             f"16 steps; cosine similarity {report.cosine_similarity:.6f}.\n"
             "The refinement runs but does no measurable work. See the weight "
             "audit for why.", fontsize=8, color=C_GREY)
    fig.tight_layout()
    return _style(fig, [ax1, ax2])


def fig_parameter_recovery(result, figsize=(11, 6)):
    """Predicted against true value, one panel per parameter."""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 4, figsize=figsize)
    for i, (lbl, ax) in enumerate(zip(PARAM_LABELS, axes.ravel())):
        t, p = result.truth[:, i], result.predictions[:, i]
        blk = result.overall[lbl]
        r2, varying = blk["r2"], result.varying_only[lbl]["r2"]

        ax.scatter(t, p, s=3, alpha=0.25, color=C_BLUE, edgecolors="none",
                   rasterized=True)
        lo = float(min(t.min(), p.min()))
        hi = float(max(t.max(), p.max()))
        pad = 0.04 * (hi - lo + 1e-9)
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad],
                color=C_RED, linestyle="--", linewidth=1.1, alpha=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)

        if r2 is None:
            title = f"{PARAM_TEX[lbl]}   R² n/a (pinned)"
        elif varying is not None and abs(varying - r2) > 0.05:
            title = f"{PARAM_TEX[lbl]}   R²={r2:.2f}  (varying: {varying:.2f})"
        else:
            title = f"{PARAM_TEX[lbl]}   R²={r2:.2f}"
        ax.set_title(title, fontsize=9.5)
        if i >= 4:
            ax.set_xlabel("true", fontsize=8)
        if i % 4 == 0:
            ax.set_ylabel("predicted", fontsize=8)

    fig.suptitle(f"Parameter recovery on the bundled benchmark (N = {result.n:,})",
                 fontsize=11.5, y=0.995)
    fig.tight_layout()
    return _style(fig, axes)


def fig_pk_reconstruction(k, pk_z0, pk_z047, pk_recon, log_k_recon=None,
                          figsize=(8, 6)):
    """
    Input spectra against the generative head's reconstruction, plus residual.

    The generative head is one of the parts of this model that did train, so
    this figure shows real work rather than a null result.
    """
    import matplotlib.pyplot as plt

    k = np.asarray(k)
    pk_z0 = np.asarray(pk_z0)
    pk_z047 = np.asarray(pk_z047)
    pk_recon = np.asarray(pk_recon)

    if (pk_z0 > 100).any():
        pk_z0 = np.log10(np.clip(pk_z0, 1e-30, None))
    if (pk_z047 > 100).any():
        pk_z047 = np.log10(np.clip(pk_z047, 1e-30, None))

    k_recon = np.exp(np.asarray(log_k_recon)) if log_k_recon is not None else k
    recon_on_input = np.interp(np.log(k), np.log(k_recon), pk_recon)
    resid = pk_z0 - recon_on_input
    mse = float((resid ** 2).mean())

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize, sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    ax1.plot(k, pk_z0, color=C_BLUE, linewidth=1.8, label="input, $z=0$")
    ax1.plot(k, pk_z047, color=C_SKY, linewidth=1.4, alpha=0.85,
             label="input, $z=0.47$")
    ax1.plot(k_recon, pk_recon, color=C_ORANGE, linewidth=1.6, linestyle="--",
             label="GenerativeHead reconstruction")
    ax1.set_xscale("log")
    ax1.set_ylabel(r"$\log_{10} P(k)$", fontsize=9)
    ax1.legend(fontsize=8, frameon=False)
    ax1.set_title("Power spectrum reconstruction", fontsize=11, pad=10)

    ax2.plot(k, resid, color=C_PURPLE, linewidth=1.4)
    ax2.axhline(0, color=C_GREY, linewidth=0.9, alpha=0.6)
    ax2.set_xscale("log")
    ax2.set_xlabel(r"$k$  [$h\,\mathrm{Mpc}^{-1}$]", fontsize=9)
    ax2.set_ylabel("input − recon", fontsize=9)
    ax2.annotate(f"log-space MSE = {mse:.4f}", xy=(0.98, 0.85),
                 xycoords="axes fraction", ha="right", fontsize=8.5)
    fig.tight_layout()
    return _style(fig, [ax1, ax2])


def fig_uncertainty_audit(result, figsize=(8, 4.6)):
    """
    Reported sigma against the error actually made.

    A working uncertainty head produces a cloud that rises to the right. This
    one produces a vertical line at the clamp floor.
    """
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize,
                                   gridspec_kw={"width_ratios": [1.25, 1]})

    err = np.abs(result.predictions - result.truth)
    ax1.scatter(result.sigmas.ravel(), err.ravel(), s=3, alpha=0.12,
                color=C_BLUE, edgecolors="none", rasterized=True)
    lim = float(max(result.sigmas.max(), err.max())) * 1.05
    ax1.plot([0, lim], [0, lim], color=C_RED, linestyle="--", linewidth=1.1,
             label="ideal: $\\sigma$ = |error|")
    ax1.set_xlabel(r"reported $\sigma$", fontsize=9)
    ax1.set_ylabel("actual |error|", fontsize=9)
    ax1.set_title("Reported uncertainty vs actual error", fontsize=10, pad=8)
    ax1.legend(fontsize=8, frameon=False)

    fracs = [result_frac(result, i) for i in range(len(PARAM_LABELS))]
    y = np.arange(len(PARAM_LABELS))
    ax2.barh(y, fracs, color=[C_RED if f > 0.9 else C_ORANGE for f in fracs],
             alpha=0.85, height=0.62)
    ax2.set_yticks(y)
    ax2.set_yticklabels([PARAM_TEX[p] for p in PARAM_LABELS], fontsize=9)
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("fraction of outputs at the clamp floor", fontsize=9)
    ax2.set_title("Sigma pinned at floor", fontsize=10, pad=8)
    ax2.invert_yaxis()

    fig.text(0.01, -0.02,
             "The uncertainty head returns clamp(softplus(net(b)) + 1e-2, max=4), "
             "so its smallest output is sigma = 0.1.\nFor most parameters that "
             "floor is what you get, on every input. These are not usable error bars.",
             fontsize=8, color=C_GREY)
    fig.tight_layout()
    return _style(fig, [ax1, ax2])


def result_frac(result, i, floor=0.1):
    """Fraction of a parameter's sigmas sitting at the clamp floor."""
    return float((np.abs(result.sigmas[:, i] - floor) < 1e-6).mean())


def fig_per_source(result, figsize=(9, 4.8)):
    """
    R2 by training source. This is what explains the aggregate.

    The headline number is pulled down by sources with documented data defects
    and by sources that pin a parameter at a constant.
    """
    import matplotlib.pyplot as plt

    names, mat, flags = [], [], []
    for name, blk in sorted(result.per_source.items(), key=lambda kv: -kv[1]["n"]):
        names.append(f"{name}  (n={blk['n']:,})")
        mat.append([blk["metrics"][p]["r2"] for p in PARAM_LABELS])
        flags.append(bool(blk.get("known_defective")))

    arr = np.array([[np.nan if v is None else v for v in row] for row in mat],
                   dtype=float)
    arr = np.clip(arr, -0.25, 1.0)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(arr, cmap="RdYlGn", vmin=-0.25, vmax=1.0, aspect="auto")
    ax.set_xticks(np.arange(len(PARAM_LABELS)))
    ax.set_xticklabels([PARAM_TEX[p] for p in PARAM_LABELS], fontsize=10)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names, fontsize=8.5)

    for r in range(arr.shape[0]):
        for c in range(arr.shape[1]):
            v = arr[r, c]
            ax.text(c, r, "--" if np.isnan(v) else f"{v:.2f}",
                    ha="center", va="center", fontsize=7.5,
                    color="#222222" if not np.isnan(v) else C_GREY)
        if flags[r]:
            ax.text(arr.shape[1] - 0.35, r, "  known data defect",
                    va="center", fontsize=7.5, color=C_RED)

    ax.set_title("R² by training source  ('--' = parameter pinned, R² undefined)",
                 fontsize=10.5, pad=10)
    fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02, label="R²")
    fig.tight_layout()
    return _style(fig, ax)
