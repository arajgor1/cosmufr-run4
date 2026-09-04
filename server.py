"""
server.py — the hosted demo, server-rendered.

Why not Gradio for the hosted build
-----------------------------------
`app.py` is a Gradio version of this demo and works fine locally. Behind
Modal's proxy its Server-Sent Events stream (`/gradio_api/queue/data`) is
aborted, which puts the Gradio client into a failed state: after that no button
submits anything and the page looks alive but is inert. That is the worst
possible failure for a demo whose entire job is to work on the first click.

This version renders every page server-side. No SSE, no websockets, no
client-side framework, no JavaScript required. A form POST returns a complete
HTML document with the figures embedded as data URIs. It works behind any
proxy and degrades to plain HTML.

Run locally:  uvicorn server:app --port 8000
"""
from __future__ import annotations

import base64
import io
import json
import os
import time
from pathlib import Path
from typing import Optional

import matplotlib

matplotlib.use("Agg")

import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse

import cosmufr
from cosmufr import figures as F
from cosmufr.load import PARAM_LABELS

SHA256 = "5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1"
REPO_URL = "https://github.com/arajgor1/cosmufr-run4"
HF_URL = "https://huggingface.co/arajgor1/cosmufr-run4"

PARAM_TEX = {
    "Om": "Ω<sub>m</sub>", "s8": "σ<sub>8</sub>", "h": "h", "ns": "n<sub>s</sub>",
    "Ob": "Ω<sub>b</sub>", "w0": "w<sub>0</sub>", "mv": "Σm<sub>ν</sub>",
    "wa": "w<sub>a</sub>",
}

MODEL = cosmufr.load_model(ckpt_path=os.environ.get("COSMUFR_CKPT"), device="cpu")
BENCH = cosmufr.load_benchmark()
AUDIT = cosmufr.weight_audit(MODEL)
N_PARAMS = sum(p.numel() for p in MODEL.parameters())

_report_path = Path(__file__).parent / "reports" / "honest_eval.json"
REPORT = json.loads(_report_path.read_text()) if _report_path.exists() else {}

EXAMPLE_IDS = list(range(0, min(len(BENCH), 5400), 211))[:24]

app = FastAPI(title="CosmUFR Run 4")


# ── rendering helpers ────────────────────────────────────────────────────────

def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                facecolor="white")
    matplotlib.pyplot.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


CSS = """
:root { color-scheme: light dark; --fg:#16181d; --bg:#fbfbfd; --mut:#5b6270;
        --line:#d9dde5; --accent:#0072B2; --warn:#D55E00; --card:#ffffff; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8eaee; --bg:#14161a; --mut:#9aa2b1; --line:#2b3038;
          --accent:#56B4E9; --warn:#E69F00; --card:#1b1e24; }
}
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.6 -apple-system,
       BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif; }
.wrap { max-width:1080px; margin:0 auto; padding:32px 20px 72px; }
h1 { font-size:30px; margin:0 0 4px; letter-spacing:-0.4px; }
h2 { font-size:19px; margin:36px 0 10px; }
.sub { color:var(--mut); margin:0 0 22px; }
nav { display:flex; gap:6px; margin:20px 0 26px; flex-wrap:wrap; }
nav a { padding:7px 14px; border:1px solid var(--line); border-radius:7px;
        text-decoration:none; color:var(--fg); background:var(--card); font-size:14px; }
nav a.on { background:var(--accent); color:#fff; border-color:var(--accent); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:18px 20px; margin:16px 0; }
.warn { border-left:4px solid var(--warn); }
table { border-collapse:collapse; width:100%; font-size:14px; }
th,td { text-align:left; padding:7px 10px; border-bottom:1px solid var(--line); }
th { color:var(--mut); font-weight:600; font-size:12.5px; text-transform:uppercase;
     letter-spacing:.04em; }
td.num, th.num { text-align:right; font-variant-numeric:tabular-nums; }
code,pre { font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12.5px; }
pre { background:var(--bg); border:1px solid var(--line); border-radius:8px;
      padding:14px; overflow-x:auto; }
img { max-width:100%; height:auto; border-radius:8px; margin:10px 0; }
select,input[type=file],button { font:inherit; padding:9px 12px; border-radius:7px;
      border:1px solid var(--line); background:var(--card); color:var(--fg); }
button { background:var(--accent); color:#fff; border-color:var(--accent);
         cursor:pointer; font-weight:600; }
form { display:flex; gap:10px; flex-wrap:wrap; align-items:center; }
select { min-width:min(560px,100%); }
.muted { color:var(--mut); font-size:13px; }
.flag { color:var(--warn); font-weight:600; }
footer { margin-top:44px; padding-top:20px; border-top:1px solid var(--line);
         color:var(--mut); font-size:13.5px; }
"""


def _page(title: str, active: str, body: str) -> HTMLResponse:
    nav = "".join(
        f'<a href="{href}" class="{"on" if key == active else ""}">{label}</a>'
        for key, href, label in [
            ("infer", "/", "Run the model"),
            ("audit", "/audit", "The audit — start here"),
            ("results", "/results", "Results"),
        ]
    )
    html = f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title}</title><style>{CSS}</style></head><body><div class="wrap">
<h1>CosmUFR Run 4</h1>
<p class="sub">Eight cosmological parameters from the matter power spectrum at two
redshifts. {N_PARAMS:,} parameters, CPU inference, bit-deterministic.<br>
Runs the real released checkpoint <code>sha256 {SHA256[:8]}…</code>, not a mock.</p>
<div class="card warn">
<strong>Before you read any output.</strong> An audit of these weights found that the
belief-settling core this architecture is named for never received a gradient during
training. It sits at its initialization; what learned is the read-out heads, reading a
fixed random projection. The reported uncertainties are a clamp constant, and the P(k)
"reconstruction" is a single constant. Neither is a result.
</div>
{nav}
{body}
<footer>
<strong>Known defects, in full.</strong>
<ol>
<li><code>obs_encoder</code>, <code>belief_proposal</code> and <code>settling</code>
never trained: 84 Linear biases are still bit-exactly zero after 40 epochs.</li>
<li>Settling moves the belief ~0.09% and its energy is flat to one float32 unit.</li>
<li>Every reported σ is the clamp floor (0.1) for six of eight parameters, on 100% of
inputs. Not error bars.</li>
<li>The generative head returns one constant at every k, for every input, and even for
a random belief vector. Its MSE of 0.687 is the variance of log10 P(k) about a constant.</li>
<li>Σm<sub>ν</sub> is not recovered: R² = 0.011 where it varies.</li>
<li>The energy subsystem diverged; E_con ≈ −4.6e5 and is not a usable
out-of-distribution signal.</li>
<li>Two redshifts only. No baseline, no ablation.</li>
</ol>
Research prototype from an in-progress PhD project.
Code and benchmark: <a href="{REPO_URL}">{REPO_URL}</a> ·
Weights: <a href="{HF_URL}">{HF_URL}</a>
</footer></div></body></html>"""
    return HTMLResponse(html)


def _example_label(i: int) -> str:
    src = cosmufr.SOURCE_NAMES.get(int(BENCH.source_lid[i]), f"lid_{BENCH.source_lid[i]}")
    p = BENCH.params[i]
    return f"#{i} · {src} · Om={p[0]:.3f} s8={p[1]:.3f} h={p[2]:.3f}"


def _form(selected: Optional[int] = None) -> str:
    opts = "".join(
        f'<option value="{i}"{" selected" if i == selected else ""}>{_example_label(i)}</option>'
        for i in EXAMPLE_IDS
    )
    return f"""
<div class="card">
<form method="post" action="/infer">
  <select name="example_id">{opts}</select>
  <button type="submit">Run inference</button>
</form>
<p class="muted" style="margin:14px 0 0">These are held-out spectra from the bundled
benchmark, so the true parameters are known and shown alongside the prediction.</p>
</form>
</div>
<div class="card">
<form method="post" action="/infer" enctype="multipart/form-data">
  <input type="file" name="upload" accept=".npy,.csv">
  <button type="submit">Run on your own spectrum</button>
</form>
<p class="muted" style="margin:14px 0 0">A <code>.npy</code> or <code>.csv</code> of shape
(2, 200): row 0 is P(k) at z=0, row 1 at z=0.47, on 200 log-spaced bins over
k = 0.1 to 4.5 h/Mpc. Raw P(k) or log10 P(k) are both accepted.</p>
</div>"""


# ── routes ───────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
def index():
    return _page("CosmUFR Run 4", "infer", _form(EXAMPLE_IDS[0]))


@app.post("/infer", response_class=HTMLResponse)
async def infer(example_id: Optional[str] = Form(None),
                upload: Optional[UploadFile] = File(None)):
    truth = None
    selected = None

    if upload is not None and upload.filename:
        raw = await upload.read()
        if len(raw) > 4_000_000:
            return _page("Error", "infer",
                         '<div class="card warn">File too large (limit 4 MB).</div>'
                         + _form())
        try:
            if upload.filename.endswith(".npy"):
                arr = np.load(io.BytesIO(raw))
            else:
                arr = np.loadtxt(io.StringIO(raw.decode("utf-8")), delimiter=",")
        except Exception as e:
            return _page("Error", "infer",
                         f'<div class="card warn">Could not read that file: {e}</div>'
                         + _form())
        arr = np.asarray(arr, dtype=np.float64)
        if arr.shape != (2, 200):
            return _page("Error", "infer",
                         f'<div class="card warn">Expected shape (2, 200), got '
                         f'{arr.shape}. Row 0 is P(k) at z=0 and row 1 at z=0.47, '
                         f'both on 200 log-spaced bins over k = 0.1 to 4.5 h/Mpc.'
                         f'</div>' + _form())
        if not np.isfinite(arr).all():
            return _page("Error", "infer",
                         '<div class="card warn">Input contains NaN or inf.</div>'
                         + _form())
        pk0, pk047 = arr[0], arr[1]
        source = "your uploaded spectrum"
    else:
        selected = int(example_id) if example_id else EXAMPLE_IDS[0]
        selected = max(0, min(selected, len(BENCH) - 1))
        pk0, pk047 = BENCH.pk_z0[selected], BENCH.pk_z047[selected]
        truth = BENCH.params[selected]
        source = _example_label(selected)

    t0 = time.time()
    result = cosmufr.infer(pk0, pk047, model=MODEL)
    dt = (time.time() - t0) * 1000
    report = cosmufr.settling_report(MODEL, pk0, pk047)

    head = ("<tr><th>parameter</th><th class='num'>predicted</th>"
            "<th class='num'>reported σ</th>")
    if truth is not None:
        head += "<th class='num'>true</th><th class='num'>residual</th>"
    head += "</tr>"

    rows = ""
    for i, lbl in enumerate(PARAM_LABELS):
        v = float(result.params_array[i])
        sg = float(result.sigmas_array[i])
        sg_txt = (f"{sg:.4f} <span class='flag'>clamp floor</span>"
                  if abs(sg - 0.1) < 1e-6 else f"{sg:.4f}")
        rows += f"<tr><td>{PARAM_TEX[lbl]}</td><td class='num'>{v:.5f}</td>" \
                f"<td class='num'>{sg_txt}</td>"
        if truth is not None:
            rows += (f"<td class='num'>{truth[i]:.5f}</td>"
                     f"<td class='num'>{v - float(truth[i]):+.5f}</td>")
        rows += "</tr>"

    payload = {
        "params": result.params,
        "sigmas_are_the_clamp_floor_not_predictions": result.sigmas,
        "settling": {
            "belief_movement_fraction": report.belief_movement,
            "energy_drop": report.energy_drop,
            "energy_drop_in_float32_ulp": report.energy_drop_in_ulps,
            "energy_log": report.energy_log,
        },
        "inference_ms": round(dt, 1),
        "model_version": "run4-2026-04-14",
        "checkpoint_sha256": SHA256,
    }

    body = f"""
{_form(selected)}
<h2>Result</h2>
<p class="muted">Input: {source} · inference {dt:.0f} ms on CPU ·
belief movement during settling {report.belief_movement*100:.3f}% ·
energy change {report.energy_drop:.2e} ({report.energy_drop_in_ulps:.1f} float32 ULP)</p>
<div class="card"><table>{head}{rows}</table></div>

<h2>Settling trajectory</h2>
<img src="{_png(F.fig_settling_trajectory(report))}" alt="settling trajectory">

<h2>Power spectrum</h2>
<img src="{_png(F.fig_pk_reconstruction(BENCH.k_bins, pk0, pk047, result.pk_recon, result.log_k))}"
     alt="power spectrum">

<h2>Result JSON</h2>
<pre>{json.dumps(payload, indent=2)}</pre>
"""
    return _page("CosmUFR Run 4 — result", "infer", body)


@app.get("/audit", response_class=HTMLResponse)
def audit_page():
    body = f"""
<h2>The belief-settling core never trained</h2>
<p>A <code>Linear</code> bias that has taken even one optimizer step essentially never
returns to exactly <code>0.0</code>. Eighty-four of them are still bit-exactly zero
after forty epochs.</p>
<img src="{_png(F.fig_weight_audit(AUDIT))}" alt="weight audit">
<pre>{AUDIT.table()}</pre>
<p>Untrained modules on the default inference path:
<strong>{", ".join(AUDIT.untrained_on_default_path)}</strong></p>

<h2>The confirming measurement</h2>
<p>Run 4 was warm-started from Run 3, itself warm-started from Run 2, then trained
forty further epochs. Comparing Run 2 against Run 4:</p>
<div class="card"><table>
<tr><th>module</th><th class="num">tensors</th><th class="num">bit-identical</th>
<th class="num">mean relative change</th></tr>
<tr><td>obs_encoder</td><td class="num">38</td><td class="num">38 / 38</td><td class="num">0.000e+00</td></tr>
<tr><td>belief_proposal</td><td class="num">38</td><td class="num">38 / 38</td><td class="num">0.000e+00</td></tr>
<tr><td>settling</td><td class="num">128</td><td class="num">128 / 128</td><td class="num">0.000e+00</td></tr>
<tr><td>param_head</td><td class="num">24</td><td class="num">2 / 24</td><td class="num">7.55e-01</td></tr>
<tr><td>gen_head</td><td class="num">70</td><td class="num">0 / 70</td><td class="num">6.59e-01</td></tr>
<tr><td>energy heads</td><td class="num">66</td><td class="num">0 / 66</td><td class="num">7.4e+29</td></tr>
</table></div>
<p>Zero relative change across three training runs is not slow learning. It is no
gradient. The read-out heads moved by 66 to 79 percent, so training clearly ran; it
simply never reached the encoder. Reproduce with
<code>cosmufr.compare_checkpoints(run2, run4)</code>.</p>

<h2>You can see it without the checkpoint</h2>
<p>One synthetic training step on a freshly initialized model shows that
<code>param_head</code> is the only module that receives any gradient, because
<code>SettlingCore.forward</code> detaches the belief on entry to every step:</p>
<pre>fresh = CosmUFRLite(CosmUFRConfig())
out = fresh(obs, return_full=False)
mse_loss(out["params"], target).backward()

modules that received any gradient: ['param_head']</pre>
<p><code>tests/test_gradient_flow.py</code> pins this, and is the test that would have
caught the defect the day it was introduced.</p>
"""
    return _page("CosmUFR Run 4 — audit", "audit", body)


@app.get("/results", response_class=HTMLResponse)
def results_page():
    if not REPORT:
        return _page("Results", "results",
                     '<div class="card">Report not bundled.</div>')
    full = REPORT["full_val_metrics"]
    vary = REPORT["full_val_metrics_varying_only"]
    OLD = {"Om": 0.907, "s8": 0.911, "h": 0.604, "ns": 0.353,
           "Ob": 0.406, "w0": 0.742, "mv": 0.410, "wa": 0.187}

    rows = ""
    for lbl in PARAM_LABELS:
        a, v = full[lbl]["r2"], vary[lbl]["r2"]
        hl = " class='flag'" if lbl == "mv" else ""
        rows += (f"<tr><td>{PARAM_TEX[lbl]}</td>"
                 f"<td class='num'>{OLD[lbl]:.3f}</td>"
                 f"<td class='num'>{'n/a' if a is None else f'{a:.3f}'}</td>"
                 f"<td class='num'{hl}>{'n/a' if v is None else f'{v:.3f}'}</td>"
                 f"<td class='num'>{full[lbl]['rmse']:.4f}</td></tr>")

    src = ""
    for name, blk in sorted(REPORT["per_source_metrics"].items(),
                            key=lambda kv: -kv[1]["n"]):
        cells = ""
        for lbl in PARAM_LABELS:
            r2 = blk["metrics"][lbl]["r2"]
            cells += (f"<td class='num'>{'--' if r2 is None else f'{r2:.2f}'}</td>")
        flag = " <span class='flag'>known data defect</span>" if name == "bacco_multiz" else ""
        src += f"<tr><td>{name}{flag}</td><td class='num'>{blk['n']:,}</td>{cells}</tr>"

    hdr = "".join(f"<th class='num'>{PARAM_TEX[l]}</th>" for l in PARAM_LABELS)

    body = f"""
<h2>Measured on {REPORT['val_split']['n']:,} validation rows</h2>
<p class="muted">Computed with the released package on the deterministic validation
split, {REPORT['val_split']['n_sources']} sources, no RNG anywhere.</p>
<div class="card"><table>
<tr><th>parameter</th><th class="num">published 2026-05</th>
<th class="num">full validation</th><th class="num">where it varies</th>
<th class="num">RMSE</th></tr>{rows}</table></div>
<p>The published column is superseded and shown only so the change is visible. Those
numbers came from the training-time evaluator on a validation set that was later
corrected, with epoch 30 selected as best from inside ±0.03 to 0.10 evaluation noise.
They should not be cited.</p>
<p><strong>“Where it varies”</strong> restricts each parameter to the sources that
actually vary it. R² is a ratio against the variance of the truth, so on a slice where
a parameter is held at a fiducial constant it measures nothing. For Σm<sub>ν</sub> that
is the whole story: the apparent 0.41 is an artifact of it being pinned at zero across
most of the corpus.</p>

<h2>Why the aggregate is lower than it looks</h2>
<div class="card"><table>
<tr><th>source</th><th class="num">n</th>{hdr}</tr>{src}</table></div>
<p><code>--</code> marks a parameter pinned in that source, where R² is undefined.
<code>bacco_multiz</code> is 15 percent of the validation set and scores zero on
everything: its z=0.47 spectra are self-paired copies of its z=0 spectra and carry no
growth information. On sources with sound data, Ω<sub>m</sub> recovery reaches 0.98
to 0.99.</p>

<h2>Reproduce this yourself</h2>
<pre>git clone {REPO_URL}
cd cosmufr-run4
pip install -e ".[demo]"
python -m cosmufr.reproduce</pre>
<p class="muted">The 6,000-row benchmark ships inside the repository. On a clean
machine the reproduction matches this table to within 1e-6.</p>
"""
    return _page("CosmUFR Run 4 — results", "results", body)


@app.get("/health")
def health():
    return JSONResponse({
        "status": "ok",
        "model_version": "run4-2026-04-14",
        "checkpoint_sha256": SHA256,
        "parameters": N_PARAMS,
        "benchmark_n": len(BENCH),
        "untrained_on_default_path": AUDIT.untrained_on_default_path,
        "code": REPO_URL,
        "weights": HF_URL,
    })
