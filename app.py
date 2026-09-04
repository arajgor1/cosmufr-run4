"""
app.py — the hosted CosmUFR demo.

Drops straight into a HuggingFace Space (SDK: gradio, hardware: CPU basic).

Design note: the defects are not hidden behind a disclosure triangle. The
"Audit" tab is a first-class part of the demo, because the honest finding about
this model is more interesting than its parameter table.
"""
from __future__ import annotations

import io
import json
import os
import time
from pathlib import Path

import gradio as gr
import matplotlib
import numpy as np

matplotlib.use("Agg")

import cosmufr
from cosmufr import figures as F
from cosmufr.load import PARAM_LABELS

PARAM_TEX = {
    "Om": "Ω_m", "s8": "σ₈", "h": "h", "ns": "n_s",
    "Ob": "Ω_b", "w0": "w₀", "mv": "Σm_ν", "wa": "w_a",
}

# ── Load once at startup ─────────────────────────────────────────────────────
CKPT = os.environ.get("COSMUFR_CKPT")
MODEL = cosmufr.load_model(ckpt_path=CKPT, device="cpu")
BENCH = cosmufr.load_benchmark()
N_PARAMS = sum(p.numel() for p in MODEL.parameters())
AUDIT = cosmufr.weight_audit(MODEL)

_REPORT_PATH = Path(__file__).parent / "reports" / "honest_eval.json"
REPORT = json.loads(_REPORT_PATH.read_text()) if _REPORT_PATH.exists() else {}

EXAMPLE_IDS = list(range(0, min(len(BENCH), 4000), 137))[:24]


def _example_label(i: int) -> str:
    src = cosmufr.SOURCE_NAMES.get(int(BENCH.source_lid[i]), f"lid_{BENCH.source_lid[i]}")
    p = BENCH.params[i]
    return f"#{i}  {src}   Ω_m={p[0]:.3f}  σ₈={p[1]:.3f}  h={p[2]:.3f}"


EXAMPLE_CHOICES = [(_example_label(i), i) for i in EXAMPLE_IDS]

HEADER = f"""
# CosmUFR Run 4

Infers eight cosmological parameters from the matter power spectrum at two
redshifts. **136,194,617 parameters · CPU inference · bit-deterministic.**

This demo runs the real released checkpoint
(`sha256 5db09d4f…`), not a mock. Every number below comes from a tensor the
model computed.

**Before you read the outputs:** an audit of these weights found that the
belief-settling core this architecture is named for never received a gradient
during training. The encoder, the belief proposal network and the settling core
sit at their initialization; what learned is the read-out heads, reading a fixed
random projection. The **Audit** tab shows the evidence. The reported
uncertainties are a clamp constant, not a prediction, and should not be used as
error bars.
"""

FOOTER = """
---

### Known defects, in full

1. `obs_encoder`, `belief_proposal` and `settling` never trained: 84 Linear
   biases are still bit-exactly zero after 40 epochs.
2. Settling moves the belief by ~0.09% and its energy is flat to one float32
   unit. It does no measurable work.
3. Every reported σ is the clamp floor (0.1) for six of eight parameters, on
   100% of inputs. Not error bars.
4. Σm_ν is not recovered: R² = 0.011 where it varies. The higher figure in
   older material was an artifact of Σm_ν being pinned at zero in most of the
   training corpus.
5. The energy subsystem diverged; `E_con` is ~−4.6e5 and is not a usable
   out-of-distribution signal.
6. Two redshifts only. No baseline, no ablation.

This is a research prototype from an in-progress PhD project. Code, benchmark
and full report: [github.com/arajgor1/cosmufr-run4](https://github.com/arajgor1/cosmufr-run4)
"""


def _fig_to_image(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="white")
    buf.seek(0)
    matplotlib.pyplot.close(fig)
    from PIL import Image
    return Image.open(buf)


def _param_rows(result, truth=None):
    rows = []
    for i, lbl in enumerate(PARAM_LABELS):
        v = result.params_array[i]
        s = result.sigmas_array[i]
        row = [PARAM_TEX[lbl], f"{v:.5f}",
               f"{s:.4f}  (clamp floor)" if abs(s - 0.1) < 1e-6 else f"{s:.4f}"]
        if truth is not None:
            row += [f"{truth[i]:.5f}", f"{v - truth[i]:+.5f}"]
        rows.append(row)
    return rows


def run_example(idx, progress=gr.Progress()):
    idx = int(idx)
    pk0, pk047 = BENCH.pk_z0[idx], BENCH.pk_z047[idx]
    truth = BENCH.params[idx]
    return _run(pk0, pk047, truth, progress)


def run_upload(file, progress=gr.Progress()):
    if file is None:
        raise gr.Error("Upload a .npy or .csv with shape (2, 200): "
                       "row 0 = P(k) at z=0, row 1 = P(k) at z=0.47.")
    path = file.name if hasattr(file, "name") else file
    try:
        arr = np.load(path) if str(path).endswith(".npy") else \
            np.loadtxt(path, delimiter=",")
    except Exception as e:
        raise gr.Error(f"Could not read the file: {e}")

    arr = np.asarray(arr, dtype=np.float64)
    if arr.shape != (2, 200):
        raise gr.Error(
            f"Expected shape (2, 200), got {arr.shape}. Row 0 is P(k) at z=0 "
            f"and row 1 is P(k) at z=0.47, both on 200 log-spaced k bins "
            f"covering k = 0.1 to 4.5 h/Mpc."
        )
    if not np.isfinite(arr).all():
        raise gr.Error("Input contains NaN or inf.")
    return _run(arr[0], arr[1], None, progress)


def _run(pk0, pk047, truth, progress):
    progress(0.1, desc="running inference")
    t0 = time.time()
    result = cosmufr.infer(pk0, pk047, model=MODEL)
    dt = (time.time() - t0) * 1000

    progress(0.4, desc="tracing the settling loop")
    report = cosmufr.settling_report(MODEL, pk0, pk047)

    progress(0.7, desc="rendering")
    headers = ["parameter", "predicted", "reported σ"]
    if truth is not None:
        headers += ["true", "residual"]
    rows = _param_rows(result, truth)

    fig_pk = _fig_to_image(F.fig_pk_reconstruction(
        BENCH.k_bins, pk0, pk047, result.pk_recon, result.log_k))
    fig_settle = _fig_to_image(F.fig_settling_trajectory(report))

    meta = (
        f"**Inference time** {dt:.0f} ms on CPU  ·  "
        f"**belief movement during settling** {report.belief_movement*100:.3f}%  ·  "
        f"**energy drop** {report.energy_drop:.2e} "
        f"({report.energy_drop_in_ulps:.1f} float32 ULP)"
    )
    return (
        gr.Dataframe(value=rows, headers=headers, wrap=True),
        fig_pk, fig_settle, meta,
        json.dumps(
            {"params": result.params,
             "sigmas_are_clamp_floor_not_predictions": result.sigmas,
             "settling": {
                 "belief_movement_frac": report.belief_movement,
                 "energy_log": report.energy_log,
                 "energy_drop": report.energy_drop,
             },
             "inference_ms": dt,
             "model_version": "run4-2026-04-14",
             "checkpoint_sha256":
                 "5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1"},
            indent=2),
    )


def audit_view():
    """The finding, rendered."""
    fig = _fig_to_image(F.fig_weight_audit(AUDIT))
    text = AUDIT.table()
    text += "\n\nUntrained modules on the default inference path: "
    text += ", ".join(AUDIT.untrained_on_default_path)
    text += (
        "\n\nConfirming measurement, Run 2 checkpoint vs Run 4 (40 epochs apart):"
        "\n"
        "\n  module            tensors   bit-identical   mean rel. change"
        "\n  obs_encoder            38        38 / 38         0.000e+00"
        "\n  belief_proposal        38        38 / 38         0.000e+00"
        "\n  settling              128      128 / 128         0.000e+00"
        "\n  param_head             24         2 / 24         7.55e-01"
        "\n  gen_head               70         0 / 70         6.59e-01"
        "\n"
        "\nZero relative change across three training runs is not slow learning."
        "\nIt is no gradient. Reproduce with cosmufr.compare_checkpoints()."
    )
    return fig, text


def results_view():
    if not REPORT:
        return "Report not bundled with this deployment.", None
    full = REPORT["full_val_metrics"]
    vary = REPORT["full_val_metrics_varying_only"]
    rows = []
    for lbl in PARAM_LABELS:
        a, v = full[lbl]["r2"], vary[lbl]["r2"]
        rows.append([
            PARAM_TEX[lbl],
            "n/a" if a is None else f"{a:.3f}",
            "n/a" if v is None else f"{v:.3f}",
            f"{full[lbl]['rmse']:.4f}",
        ])
    return gr.Dataframe(
        value=rows,
        headers=["parameter", "R² full validation", "R² where it varies", "RMSE"],
    )


with gr.Blocks(title="CosmUFR Run 4", theme=gr.themes.Soft()) as demo:
    gr.Markdown(HEADER)

    with gr.Tab("Infer"):
        with gr.Row():
            with gr.Column(scale=1):
                ex = gr.Dropdown(choices=EXAMPLE_CHOICES,
                                 value=EXAMPLE_CHOICES[0][1],
                                 label="Example spectrum (from the bundled benchmark, truth known)")
                btn = gr.Button("Run inference", variant="primary")
                gr.Markdown("**or** upload your own")
                up = gr.File(label=".npy or .csv, shape (2, 200)",
                             file_types=[".npy", ".csv"])
                upbtn = gr.Button("Run on upload")
                gr.Markdown(
                    "Rows are P(k) at z=0 and z=0.47, on 200 log-spaced bins "
                    "over k = 0.1 to 4.5 h/Mpc. Raw P(k) or log10 P(k) are both "
                    "accepted."
                )
            with gr.Column(scale=2):
                meta = gr.Markdown()
                table = gr.Dataframe(label="Parameters")
        with gr.Row():
            pk_img = gr.Image(label="P(k) reconstruction", type="pil")
            settle_img = gr.Image(label="Settling trajectory", type="pil")
        js = gr.Code(label="Result JSON", language="json")

        btn.click(run_example, [ex], [table, pk_img, settle_img, meta, js])
        upbtn.click(run_upload, [up], [table, pk_img, settle_img, meta, js])

    with gr.Tab("Audit — start here"):
        gr.Markdown(
            "### The belief-settling core never trained\n"
            "A `Linear` bias that has taken even one optimizer step essentially "
            "never returns to exactly `0.0`. Eighty-four of them are still "
            "bit-exactly zero after forty epochs of training."
        )
        audit_img = gr.Image(label="Which modules ever received a gradient",
                             type="pil")
        audit_txt = gr.Code(label="Measurements")
        demo.load(audit_view, None, [audit_img, audit_txt])

    with gr.Tab("Results"):
        gr.Markdown(
            "### Measured on 162,795 validation rows with the released package\n"
            "'Where it varies' restricts each parameter to the sources that "
            "actually vary it. R² divides by the variance of the truth, so on a "
            "slice where a parameter is pinned at a fiducial constant it "
            "measures nothing. For Σm_ν this is the whole story."
        )
        res_table = gr.Dataframe()
        demo.load(results_view, None, [res_table])
        gr.Markdown(
            "Reproduce with `python -m cosmufr.reproduce` after cloning the "
            "repository. On a clean machine this matches to within 1e-6."
        )

    gr.Markdown(FOOTER)


if __name__ == "__main__":
    demo.launch()
