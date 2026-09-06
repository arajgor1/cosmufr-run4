"""
server.py — the CosmUFR site and live demo, server-rendered.

Why server-rendered
-------------------
`app.py` is a Gradio version of this demo and works fine locally. Behind a
proxy its Server-Sent Events stream is aborted, which puts the Gradio client
into a failed state: after that no button submits anything and the page looks
alive but is inert. That is the worst possible failure for a demo whose entire
job is to work on the first click.

Every page here is a complete HTML document rendered on the server. No SSE, no
websockets, no client framework, no external CSS or JS. It works behind any
proxy, and it works with JavaScript turned off.

Run locally:  uvicorn server:app --port 8000
"""
from __future__ import annotations

import base64
import html
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
from fastapi.responses import HTMLResponse, JSONResponse, Response

import cosmufr
from cosmufr import figures as F
from cosmufr.explain import FIGURE_NOTES, PARAM_MEANING
from cosmufr.load import PARAM_LABELS
from cosmufr.validate import K_GRID, validate_spectra

SHA256 = "5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1"
REPO_URL = "https://github.com/arajgor1/cosmufr-run4"
HF_URL = "https://huggingface.co/arajgor1/cosmufr-run4"
MAX_UPLOAD = 4_000_000

PARAM_TEX = {
    "Om": "Ω<sub>m</sub>", "s8": "σ<sub>8</sub>", "h": "h", "ns": "n<sub>s</sub>",
    "Ob": "Ω<sub>b</sub>", "w0": "w<sub>0</sub>", "mv": "Σm<sub>ν</sub>",
    "wa": "w<sub>a</sub>",
}

MODEL = cosmufr.load_model(ckpt_path=os.environ.get("COSMUFR_CKPT"), device="cpu")
BENCH = cosmufr.load_benchmark()
AUDIT = cosmufr.weight_audit(MODEL)
N_PARAMS = sum(p.numel() for p in MODEL.parameters())

_reports = Path(__file__).parent / "reports"
REPORT = json.loads((_reports / "honest_eval.json").read_text()) \
    if (_reports / "honest_eval.json").exists() else {}
RIDGE = json.loads((_reports / "ridge_baseline.json").read_text()) \
    if (_reports / "ridge_baseline.json").exists() else {}

EXAMPLE_IDS = list(range(0, min(len(BENCH), 5400), 211))[:24]

app = FastAPI(title="CosmUFR Run 4")


# ─────────────────────────────────────────────────────────────────────────────
# Design. Ported from the Calybre design language: near-black ground, glass
# panels over it, Space Grotesk for display and Inter for body, single blue
# accent. Self-contained on purpose, so a CDN outage cannot leave a reviewer
# looking at unstyled HTML.
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
:root{
  --black:#000000; --charcoal:#0A0A0A; --panel:rgba(20,20,20,.45);
  --line:rgba(255,255,255,.10); --line-soft:rgba(255,255,255,.05);
  --fg:#E2E2E2; --mut:#8b93a3; --dim:#5f6673;
  --accent:#3b82f6; --accent-dim:rgba(59,130,246,.14);
  --warn:#E69F00; --bad:#D55E00; --good:#009E73;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
}
*{box-sizing:border-box}
html{scroll-behavior:smooth; scroll-padding-top:76px}
body{
  margin:0; background:var(--black); color:var(--fg);
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  font-size:15px; line-height:1.65; font-weight:300; overflow-x:hidden;
  -webkit-font-smoothing:antialiased;
}
.display{font-family:"Space Grotesk",Inter,sans-serif; font-weight:600; letter-spacing:-.02em}
a{color:var(--accent); text-decoration:none}
a:hover{text-decoration:underline}
.wrap{max-width:1080px; margin:0 auto; padding:0 24px}

/* nav */
nav.top{
  position:sticky; top:0; z-index:50; background:rgba(0,0,0,.72);
  backdrop-filter:blur(16px); -webkit-backdrop-filter:blur(16px);
  border-bottom:1px solid var(--line-soft);
}
nav.top .wrap{display:flex; align-items:center; justify-content:space-between;
  height:60px; gap:20px}
nav.top .brand{display:flex; align-items:center; gap:10px; flex-shrink:0}
nav.top .brand .mark{width:22px;height:22px;flex-shrink:0}
nav.top .brand span{font-family:"Space Grotesk",sans-serif; font-weight:700;
  letter-spacing:.18em; text-transform:uppercase; font-size:13px; color:#fff}
nav.top .links{display:flex; gap:22px; align-items:center; overflow-x:auto}
nav.top .links a{color:var(--mut); font-size:12px; letter-spacing:.08em;
  text-transform:uppercase; white-space:nowrap; text-decoration:none;
  transition:color .2s}
nav.top .links a:hover{color:#fff}

/* hero */
.hero{position:relative; padding:76px 0 44px; border-bottom:1px solid var(--line-soft)}
.hero h1{font-size:clamp(38px,6vw,68px); line-height:1.02; margin:0 0 18px}
.hero .lede{font-size:18px; color:var(--mut); max-width:640px; margin:0 0 26px;
  font-weight:300}
.pill{display:inline-flex; align-items:center; gap:8px; padding:6px 14px;
  border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.03);
  font-family:var(--mono); font-size:11px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--mut); margin-bottom:26px}
.pill .dot{width:6px;height:6px;border-radius:50%;background:var(--good);
  animation:pulse 2.4s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.35}}

.stats{display:flex; gap:36px; flex-wrap:wrap; margin-top:30px}
.stat .n{font-family:"Space Grotesk",sans-serif; font-size:24px; font-weight:600; color:#fff}
.stat .l{font-size:11px; letter-spacing:.12em; text-transform:uppercase; color:var(--dim)}

/* panels */
.panel{background:var(--panel); backdrop-filter:blur(20px);
  -webkit-backdrop-filter:blur(20px); border:1px solid var(--line);
  border-radius:12px; padding:22px 24px; margin:18px 0}
.panel.warn{border-left:3px solid var(--warn)}
.panel.bad{border-left:3px solid var(--bad)}
.panel.tight{padding:16px 18px}

section{padding:56px 0; border-bottom:1px solid var(--line-soft)}
section h2{font-size:clamp(24px,3.2vw,34px); margin:0 0 8px}
section .kicker{font-family:var(--mono); font-size:11px; letter-spacing:.2em;
  text-transform:uppercase; color:var(--accent); margin:0 0 10px}
section .sub{color:var(--mut); max-width:720px; margin:0 0 22px}

/* flow diagram */
.flow{display:flex; align-items:stretch; gap:12px; flex-wrap:wrap; margin:24px 0}
.flow .step{flex:1 1 200px; background:var(--panel); border:1px solid var(--line);
  border-radius:12px; padding:18px}
.flow .step .n{font-family:var(--mono); font-size:10px; letter-spacing:.18em;
  color:var(--accent); text-transform:uppercase}
.flow .step h4{margin:8px 0 6px; font-family:"Space Grotesk",sans-serif;
  font-size:15px; font-weight:600; color:#fff}
.flow .step p{margin:0; font-size:13.5px; color:var(--mut)}

/* tables */
table{border-collapse:collapse; width:100%; font-size:13.5px}
th,td{text-align:left; padding:8px 10px; border-bottom:1px solid var(--line-soft)}
th{color:var(--dim); font-weight:500; font-size:10.5px; letter-spacing:.12em;
  text-transform:uppercase}
td.num,th.num{text-align:right; font-variant-numeric:tabular-nums;
  font-family:var(--mono); font-size:12.5px}
tr:last-child td{border-bottom:none}
.tw{overflow-x:auto; -webkit-overflow-scrolling:touch}

/* forms */
select,input[type=file],button,textarea{font:inherit; color:var(--fg);
  background:rgba(255,255,255,.04); border:1px solid var(--line);
  border-radius:8px; padding:10px 13px}
select{min-width:min(520px,100%); font-size:13.5px}
button{background:var(--accent); border-color:var(--accent); color:#fff;
  font-weight:500; cursor:pointer; transition:filter .15s; font-size:14px}
button:hover{filter:brightness(1.14)}
button.ghost{background:rgba(255,255,255,.04); border-color:var(--line); color:var(--fg)}
.row{display:flex; gap:10px; flex-wrap:wrap; align-items:center}

.dl{display:inline-flex; align-items:center; gap:6px; padding:5px 11px;
  border:1px solid var(--line); border-radius:7px; font-family:var(--mono);
  font-size:11.5px; color:var(--mut); text-decoration:none; background:rgba(255,255,255,.03)}
.dl:hover{color:#fff; border-color:var(--accent); text-decoration:none}

code{font-family:var(--mono); font-size:12.5px; background:rgba(255,255,255,.06);
  padding:1.5px 6px; border-radius:4px}
pre{font-family:var(--mono); font-size:12px; background:var(--charcoal);
  border:1px solid var(--line); border-radius:10px; padding:15px;
  overflow-x:auto; line-height:1.5; color:#cbd2de}
pre code{background:none; padding:0; font-size:inherit}

img.fig{width:100%; height:auto; border-radius:10px; margin:6px 0 0;
  background:#fff; border:1px solid var(--line)}

.muted{color:var(--mut); font-size:13px}
.dim{color:var(--dim); font-size:12.5px}
.flag{color:var(--warn); font-family:var(--mono); font-size:11px}
.bad-t{color:var(--bad)}
.good-t{color:var(--good)}

.figblock{margin:26px 0}
.figblock .meta{display:grid; grid-template-columns:repeat(auto-fit,minmax(230px,1fr));
  gap:14px; margin-top:14px}
.figblock .meta div{border-left:2px solid var(--line); padding-left:12px}
.figblock .meta .lbl{font-family:var(--mono); font-size:10px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--accent); display:block; margin-bottom:3px}
.figblock .meta p{margin:0; font-size:13px; color:var(--mut)}

ol,ul{padding-left:20px} li{margin:7px 0; color:var(--mut); font-size:14px}
li strong,li code{color:var(--fg)}
footer{padding:38px 0 64px; color:var(--dim); font-size:13px}
hr{border:none; border-top:1px solid var(--line-soft); margin:26px 0}
@media (max-width:640px){ nav.top .links{gap:14px} .hero{padding-top:48px} }
"""

LOGO = ('<svg class="mark" viewBox="0 0 24 24" fill="none" '
        'xmlns="http://www.w3.org/2000/svg" aria-hidden="true">'
        '<circle cx="12" cy="12" r="9.2" stroke="#fff" stroke-width="1.4" opacity=".55"/>'
        '<circle cx="12" cy="12" r="4.4" stroke="#3b82f6" stroke-width="1.6"/>'
        '<circle cx="12" cy="12" r="1.5" fill="#fff"/></svg>')

NAV_LINKS = [
    ("#what", "What it does"), ("#demo", "Demo"), ("#audit", "The audit"),
    ("#results", "Results"), ("#baseline", "Baseline"), ("#limits", "Limitations"),
]


def _png(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=118, bbox_inches="tight", facecolor="white")
    matplotlib.pyplot.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _figblock(key: str, img_src: str) -> str:
    n = FIGURE_NOTES[key]
    why = (f'<div><span class="lbl">Why</span><p>{html.escape(n["why"])}</p></div>'
           if n["why"] else "")
    return f"""<div class="figblock">
<img class="fig" src="{img_src}" alt="{html.escape(n['title'])}">
<div class="meta">
<div><span class="lbl">Input</span><p>{html.escape(n['input'])}</p></div>
<div><span class="lbl">Output</span><p>{html.escape(n['output'])}</p></div>
<div><span class="lbl">What it means</span><p>{html.escape(n['means'])}</p></div>
{why}
</div></div>"""


def _page(body: str, title: str = "CosmUFR Run 4") -> HTMLResponse:
    links = "".join(f'<a href="{h}">{html.escape(t)}</a>' for h, t in NAV_LINKS)
    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="Neural inference of cosmological parameters from the matter power spectrum, released with a reproducible benchmark and an audit of its own training defects.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<nav class="top"><div class="wrap">
  <div class="brand">{LOGO}<span>CosmUFR</span></div>
  <div class="links">{links}
    <a href="{REPO_URL}">GitHub&nbsp;&#8599;</a>
    <a href="{HF_URL}">Weights&nbsp;&#8599;</a>
  </div>
</div></nav>
{body}
</body></html>""")


# ─────────────────────────────────────────────────────────────────────────────
# Static content sections
# ─────────────────────────────────────────────────────────────────────────────

def _hero() -> str:
    return f"""<header class="hero"><div class="wrap">
<div class="pill"><span class="dot"></span>Live model &middot; not a mock</div>
<h1 class="display">Cosmological parameters<br>from the matter power spectrum.</h1>
<p class="lede">CosmUFR reads P(k) at two redshifts and infers eight cosmological
parameters. This page runs the real released checkpoint on CPU, and publishes
the audit that found what is wrong with it.</p>
<div class="row">
  <a href="#demo"><button>Run the model</button></a>
  <a href="#audit"><button class="ghost">Read the audit first</button></a>
</div>
<div class="stats">
  <div class="stat"><div class="n">{N_PARAMS/1e6:.0f}M</div><div class="l">parameters</div></div>
  <div class="stat"><div class="n">245&thinsp;ms</div><div class="l">model forward, CPU</div></div>
  <div class="stat"><div class="n">6,000</div><div class="l">benchmark spectra shipped</div></div>
  <div class="stat"><div class="n">4</div><div class="l">modules that never trained</div></div>
</div>
</div></header>

<div class="wrap"><div class="panel warn" style="margin-top:28px">
<strong>Read this before you read any output.</strong> An audit of these weights
found that the belief-settling core this architecture is named for never received
a gradient during training. It sits at its initialization; what learned is the
read-out heads, reading a fixed random projection. The reported uncertainties are
a clamp constant and the P(k) "reconstruction" is a single constant. Neither is a
result. <a href="#audit">The evidence is below</a>, and you can reproduce all of
it from the released checkpoint.
</div></div>"""


def _what() -> str:
    return """<section id="what"><div class="wrap">
<p class="kicker">What it does</p>
<h2 class="display">One spectrum in, eight numbers out</h2>
<p class="sub">The matter power spectrum describes how clumpy the universe is as
a function of scale. Its shape and height depend on the cosmological parameters
that produced it, so recovering those parameters from it is an inverse problem.
CosmUFR learns that inverse map directly.</p>

<div class="flow">
<div class="step"><div class="n">Input</div><h4>P(k) at two redshifts</h4>
<p>200 log-spaced k bins from 0.1 to 4.5 h/Mpc, at z=0 and z=0.47. The second
redshift is there so the model can see how structure grew between them.</p></div>
<div class="step"><div class="n">Encode</div><h4>A 1024-d belief</h4>
<p>The 400 input values are projected into a belief vector meant to hold
everything the model thinks about this universe.</p></div>
<div class="step"><div class="n">Settle</div><h4>16 refinement steps</h4>
<p>The belief descends a learned energy. This is the architecture's central
idea, and on this checkpoint it does nothing measurable.</p></div>
<div class="step"><div class="n">Read out</div><h4>Eight parameters</h4>
<p>Heads read the settled belief and emit the parameters, each squashed into its
physically allowed range.</p></div>
</div>

<p class="sub" style="margin-top:6px">What makes this an inverse problem rather
than a fit: many different cosmologies produce similar spectra, so some
parameters are far better constrained by P(k) than others. That ordering shows
up clearly in the results.</p>
</div></section>"""


def _limits() -> str:
    items = [
        ("The belief pipeline never trained.", "<code>obs_encoder</code>, "
         "<code>belief_proposal</code> and <code>settling</code> sit at "
         "initialization: 84 Linear biases are still bit-exactly zero after 40 "
         "epochs. Root cause is an unconditional <code>detach()</code> in the "
         "settling loop."),
        ("Settling does no measurable work.", "0.09 percent mean belief "
         "movement, energy flat to one float32 unit."),
        ("The energy landscape is flat.", "The energy heads did train, via their "
         "own optimizer, and converged to an input-independent constant: E varies "
         "by about one part in seven million across completely different spectra. "
         "So repairing the gradient path alone would not make settling work. "
         "There would still be nothing to descend."),
        ("Reported uncertainties are a constant.", "σ = 0.1 for six of eight "
         "parameters on 100 percent of inputs, because the uncertainty head sits "
         "at its clamp floor. Not error bars."),
        ("The generative head returns a constant.", "The same value at every k, "
         "for every input, and for a random belief vector. Its error of 0.687 is "
         "the variance of log10 P(k) about a constant."),
        ("Neutrino mass is not recovered.", "R² = 0.011 measured only on data "
         "where it actually varies."),
        ("The energy subsystem diverged.", "E sits near −9.3e5. The E_con "
         "anomaly score is about −4.6e5 and is not a usable out-of-distribution "
         "signal."),
        ("Two redshifts only.", "Multi-redshift generalization is unvalidated, "
         "and the multi-redshift corpus has a documented ordering defect."),
        ("The headline table is not externally reproducible.", "It was measured "
         "on 162,795 rows of a private split. The bundled 6,000-row benchmark "
         "lands within about 0.03 of it and narrows that gap rather than closing "
         "it."),
        ("No ablation.", "There is a linear baseline now, but no ablation of the "
         "architecture's own components."),
    ]
    lis = "".join(f"<li><strong>{t}</strong> {d}</li>" for t, d in items)
    return f"""<section id="limits"><div class="wrap">
<p class="kicker">Limitations</p>
<h2 class="display">Everything known to be wrong with this model</h2>
<p class="sub">Stated in full, because a careful reader will find all of it
within ten minutes anyway.</p>
<ol>{lis}</ol>
</div></section>

<footer><div class="wrap">
<p>Research prototype from an in-progress PhD project by Aaditya Rajgor.
MIT licensed.</p>
<p>Code, benchmark and full report: <a href="{REPO_URL}">{REPO_URL}</a><br>
Weights and model card: <a href="{HF_URL}">{HF_URL}</a></p>
<p class="dim">Checkpoint sha256 {SHA256}</p>
</div></footer>"""


# ─────────────────────────────────────────────────────────────────────────────
# Downloads. Every built-in example is downloadable in both formats, so a
# visitor can take example #211, upload it straight back, and confirm the two
# paths give the same answer. Nothing here is special-cased for the built-ins:
# the upload path runs identical code.
# ─────────────────────────────────────────────────────────────────────────────

def _example_label(i: int) -> str:
    src = cosmufr.SOURCE_NAMES.get(int(BENCH.source_lid[i]), f"lid_{BENCH.source_lid[i]}")
    p = BENCH.params[i]
    return f"#{i} · {src} · Om={p[0]:.3f} s8={p[1]:.3f} h={p[2]:.3f}"


def _csv_for(i: int) -> str:
    src = cosmufr.SOURCE_NAMES.get(int(BENCH.source_lid[i]), f"lid_{BENCH.source_lid[i]}")
    p = BENCH.params[i]
    truth = "  ".join(f"{l}={v:.6g}" for l, v in zip(PARAM_LABELS, p))
    head = [
        "# CosmUFR input file",
        f"# Benchmark example #{i}, source: {src}",
        f"# True parameters: {truth}",
        "#",
        "# Two rows, 200 comma-separated values each.",
        "#   row 1: P(k) at z = 0.00",
        "#   row 2: P(k) at z = 0.47",
        "# k grid: 200 log-spaced bins from 0.1 to 4.5 h/Mpc (see kgrid.csv).",
        "# Units: P(k) in (Mpc/h)^3. Raw P(k) or log10 P(k) are both accepted.",
        "# Lines beginning with # are ignored.",
    ]
    rows = [",".join(f"{v:.8e}" for v in BENCH.pk_z0[i]),
            ",".join(f"{v:.8e}" for v in BENCH.pk_z047[i])]
    return "\n".join(head + rows) + "\n"


@app.get("/download/example/{idx}.csv")
def dl_example_csv(idx: int):
    idx = max(0, min(idx, len(BENCH) - 1))
    return Response(_csv_for(idx), media_type="text/csv", headers={
        "Content-Disposition": f'attachment; filename="cosmufr_example_{idx}.csv"'})


@app.get("/download/example/{idx}.npy")
def dl_example_npy(idx: int):
    idx = max(0, min(idx, len(BENCH) - 1))
    buf = io.BytesIO()
    np.save(buf, np.stack([BENCH.pk_z0[idx], BENCH.pk_z047[idx]]).astype(np.float32))
    return Response(buf.getvalue(), media_type="application/octet-stream", headers={
        "Content-Disposition": f'attachment; filename="cosmufr_example_{idx}.npy"'})


@app.get("/download/template.csv")
def dl_template():
    # A working file, not a blank one. It runs as-is, so the format is
    # unambiguous, and the values can be replaced in place.
    body = _csv_for(EXAMPLE_IDS[1] if len(EXAMPLE_IDS) > 1 else 0)
    body = body.replace(
        "# CosmUFR input file",
        "# CosmUFR input TEMPLATE - this file runs as-is.\n"
        "# Replace the two data rows with your own spectra,\n"
        "# keeping 200 values per row in the same k order.")
    return Response(body, media_type="text/csv", headers={
        "Content-Disposition": 'attachment; filename="cosmufr_template.csv"'})


@app.get("/download/kgrid.csv")
def dl_kgrid():
    head = ["# The k grid CosmUFR expects, in h/Mpc.",
            "# 200 log-spaced bins from 0.1 to 4.5.",
            "# Reproduce with: np.logspace(np.log10(0.1), np.log10(4.5), 200)",
            "# Your P(k) values must be sampled at exactly these k."]
    body = "\n".join(head + [",".join(f"{v:.8e}" for v in K_GRID)]) + "\n"
    return Response(body, media_type="text/csv", headers={
        "Content-Disposition": 'attachment; filename="cosmufr_kgrid.csv"'})


# ─────────────────────────────────────────────────────────────────────────────
# Demo
# ─────────────────────────────────────────────────────────────────────────────

def _form(selected=None) -> str:
    sel = selected if selected is not None else EXAMPLE_IDS[0]
    opts = "".join(
        f'<option value="{i}"{" selected" if i == sel else ""}>'
        f'{html.escape(_example_label(i))}</option>' for i in EXAMPLE_IDS)
    return f"""
<div class="panel">
<form method="post" action="/infer#result">
  <div class="row">
    <select name="example_id" aria-label="Example spectrum">{opts}</select>
    <button type="submit">Run inference</button>
  </div>
  <p class="muted" style="margin:14px 0 10px">Held-out spectra from the bundled
  benchmark, so the true parameters are known and shown next to the prediction.</p>
</form>
  <div class="row">
    <a class="dl" href="/download/example/{sel}.csv">&#8595; this example .csv</a>
    <a class="dl" href="/download/example/{sel}.npy">&#8595; this example .npy</a>
  </div>
  <p class="dim" style="margin:12px 0 0"><strong style="color:var(--fg)">Check us.</strong>
  Download the example above, upload it below as your own file, and compare. The
  upload path runs identical code with no special-casing, so the two results
  should match to the last digit. If they do not, that is a bug worth reporting.</p>
</div>

<div class="panel">
<form method="post" action="/infer#result" enctype="multipart/form-data">
  <div class="row">
    <input type="file" name="upload" accept=".npy,.csv" aria-label="Your spectrum">
    <button type="submit">Run on your own spectrum</button>
  </div>
</form>
<hr>
<p class="muted" style="margin:0 0 10px"><strong style="color:var(--fg)">What the
file must contain.</strong> Two rows of 200 numbers. Row 1 is P(k) at z=0, row 2
is P(k) at z=0.47, both sampled on 200 log-spaced k bins from 0.1 to 4.5 h/Mpc,
with P(k) in (Mpc/h)<sup>3</sup>. Raw P(k) or log10 P(k) are both accepted and
detected automatically. A <code>.npy</code> must be shape (2, 200); a
<code>.csv</code> is comma-separated, and <code>#</code> comment lines are ignored.</p>
<div class="row">
  <a class="dl" href="/download/template.csv">&#8595; template.csv (runs as-is)</a>
  <a class="dl" href="/download/kgrid.csv">&#8595; the k grid</a>
</div>
<p class="dim" style="margin:12px 0 0">If your spectrum is on a different k grid,
interpolate in log-log first:
<code>np.exp(np.interp(np.log(k_grid), np.log(your_k), np.log(your_pk)))</code></p>
</div>"""


def _timing_panel(t) -> str:
    total = sum(t.values())
    rows = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="num">{v*1000:.0f} ms</td>'
        f'<td class="num">{100*v/total:.0f}%</td></tr>' for k, v in t.items())
    return f"""<div class="panel tight">
<p class="muted" style="margin:0 0 10px"><strong style="color:var(--fg)">Where the
time went.</strong> The model itself is the small part. Most of the wait is
matplotlib drawing the figures below.</p>
<div class="tw"><table>
<tr><th>stage</th><th class="num">time</th><th class="num">share</th></tr>
{rows}
<tr><td><strong>total</strong></td><td class="num"><strong>{total*1000:.0f} ms</strong></td><td class="num">100%</td></tr>
</table></div></div>"""


def _run(pk0, pk047, truth, source_label: str, selected=None) -> str:
    t = {}

    s = time.perf_counter()
    result = cosmufr.infer(pk0, pk047, model=MODEL)
    t["model forward: encode, 16 settling steps, read-out heads"] = time.perf_counter() - s

    s = time.perf_counter()
    report = cosmufr.settling_report(MODEL, pk0, pk047)
    t["settling trace: re-run the 16 steps, recording each"] = time.perf_counter() - s

    s = time.perf_counter()
    fig_settle = _png(F.fig_settling_trajectory(report))
    t["draw the settling figure"] = time.perf_counter() - s

    s = time.perf_counter()
    fig_pk = _png(F.fig_pk_reconstruction(K_GRID, pk0, pk047,
                                          result.pk_recon, result.log_k))
    t["draw the P(k) figure"] = time.perf_counter() - s

    head = ("<tr><th>parameter</th><th></th><th class='num'>predicted</th>"
            "<th class='num'>reported &sigma;</th>")
    if truth is not None:
        head += "<th class='num'>true</th><th class='num'>residual</th>"
    head += "</tr>"

    rows = ""
    for i, lbl in enumerate(PARAM_LABELS):
        v, sg = float(result.params_array[i]), float(result.sigmas_array[i])
        name, _ = PARAM_MEANING[lbl]
        sg_txt = (f"{sg:.4f} <span class='flag'>floor</span>"
                  if abs(sg - 0.1) < 1e-6 else f"{sg:.4f}")
        rows += (f"<tr><td>{PARAM_TEX[lbl]}</td>"
                 f"<td class='dim'>{html.escape(name)}</td>"
                 f"<td class='num'>{v:.5f}</td><td class='num'>{sg_txt}</td>")
        if truth is not None:
            d = v - float(truth[i])
            rows += (f"<td class='num'>{float(truth[i]):.5f}</td>"
                     f"<td class='num'>{d:+.5f}</td>")
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
        "timing_ms": {k: round(v * 1000, 1) for k, v in t.items()},
        "model_version": "run4-2026-04-14",
        "checkpoint_sha256": SHA256,
    }
    dl = ""
    if selected is not None:
        dl = (f'<div class="row" style="margin-top:12px">'
              f'<a class="dl" href="/download/example/{selected}.csv">&#8595; this input as .csv</a>'
              f'<a class="dl" href="/download/example/{selected}.npy">&#8595; this input as .npy</a>'
              f'</div>')

    return f"""<div id="result"></div>
<h3 class="display" style="margin:34px 0 4px">Result</h3>
<p class="muted" style="margin:0 0 4px">Input: {html.escape(source_label)}</p>
<p class="dim" style="margin:0 0 14px">Belief moved
{report.belief_movement*100:.3f}% of its norm during settling; energy changed by
{report.energy_drop:.2e}, which is {report.energy_drop_in_ulps:.1f} float32
resolution steps.</p>
<div class="panel"><div class="tw"><table>{head}{rows}</table></div>
<p class="dim" style="margin:12px 0 0">Every &sigma; marked
<span class="flag">floor</span> is the uncertainty head's clamp constant, not a
prediction. Do not read them as error bars.</p>{dl}</div>
{_timing_panel(t)}
{_figblock("settling", fig_settle)}
{_figblock("pk", fig_pk)}
<details><summary class="muted" style="cursor:pointer">Result JSON</summary>
<pre><code>{html.escape(json.dumps(payload, indent=2))}</code></pre></details>"""


def _demo(inner: str = "", selected=None) -> str:
    return f"""<section id="demo"><div class="wrap">
<p class="kicker">Live demo</p>
<h2 class="display">Run the model</h2>
<p class="sub">This runs the released checkpoint on CPU, in this container, on
whatever you give it. Nothing is cached or precomputed.</p>
{_form(selected)}
{inner}
</div></section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Audit, results, baseline
# ─────────────────────────────────────────────────────────────────────────────

def _audit() -> str:
    img = _png(F.fig_weight_audit(AUDIT))
    rows = ""
    for name, m in AUDIT.modules.items():
        bad = m["verdict"] == "UNTRAINED"
        rows += (
            f'<tr><td><code>{html.escape(name)}</code></td>'
            f'<td class="dim">{"on path" if m["on_default_path"] else "unused"}</td>'
            f'<td class="num">{m["n_zero_bias"]}/{m["n_linear"]}</td>'
            f'<td class="num">{m["max_abs_bias"]:.3e}</td>'
            f'<td class="{"bad-t" if bad else "good-t"}">'
            f'{"never trained" if bad else "trained"}</td></tr>')

    return f"""<section id="audit"><div class="wrap">
<p class="kicker">The audit</p>
<h2 class="display">The belief-settling core never trained</h2>
<p class="sub">This is the most important thing on the page, and you can check
every part of it yourself from the released checkpoint in about a minute.</p>

<div class="panel"><p class="muted" style="margin:0 0 12px">PyTorch initialises a
<code>Linear</code> bias from a random draw, and any optimizer step moves it off
that value. A module whose biases are all still bit-exactly <code>0.0</code>
after forty epochs of training never received a gradient at all.</p>
<div class="tw"><table>
<tr><th>module</th><th></th><th class="num">biases = 0</th>
<th class="num">max |bias|</th><th>verdict</th></tr>{rows}</table></div>
<p class="dim" style="margin:12px 0 0">The <code>_single</code> and
<code>_seq</code> variants belong to an unused single-redshift path. They did
train, which is a real clue about what happened and one I have not fully
explained.</p></div>

{_figblock("weight_audit", img)}

<h3 class="display" style="margin:30px 0 8px; font-size:20px">The root cause, in released source</h3>
<p class="muted">You do not need the weights to see this.
<code>SettlingCore.forward</code> detaches the belief on entry to every step, so
nothing downstream of it can reach the encoder:</p>
<pre><code>for step in range(k):
    b = b.detach()                      # &lt;- severs everything upstream
    with torch.enable_grad():
        b_g = b.requires_grad_(True)
        E = energy_fn(b_g, z.detach(), b_prev.detach())
        grad = torch.autograd.grad(E.sum(), b_g)[0]
    b = b - eta * P * grad.detach()</code></pre>
<p class="muted">One synthetic training step on a freshly initialised model
confirms it, with no checkpoint involved:</p>
<pre><code>modules that received any gradient: ['param_head']</code></pre>

<h3 class="display" style="margin:30px 0 8px; font-size:20px">And a second cause, which matters more</h3>
<div class="panel bad"><p class="muted" style="margin:0">The energy heads
<em>did</em> train, through their own optimizer, and converged to an
input-independent constant. Measured across eight completely different spectra,
the energy varies by about one part in seven million, and its gradient with
respect to the belief has norm 0.11 against a belief norm of 16.5. With the
learned step size capped at 0.05, sixteen steps could move the belief by at most
half a percent no matter what. <strong style="color:var(--fg)">So repairing the
gradient path alone would not make settling work.</strong> There would still be
no landscape to descend. That is a harder and more interesting problem than the
one I originally reported, and I do not yet have a fix for it.</p></div>
</div></section>"""


def _results() -> str:
    if not REPORT:
        return ""
    full, vary = REPORT["full_val_metrics"], REPORT["full_val_metrics_varying_only"]
    bench = REPORT["benchmark"]["metrics"]
    OLD = {"Om": 0.907, "s8": 0.911, "h": 0.604, "ns": 0.353,
           "Ob": 0.406, "w0": 0.742, "mv": 0.410, "wa": 0.187}

    def f(x):
        return "--" if x is None else f"{x:.3f}"

    rows = ""
    for lbl in PARAM_LABELS:
        name, desc = PARAM_MEANING[lbl]
        hl = ' class="num flag"' if lbl == "mv" else ' class="num"'
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="dim">{html.escape(name)}</td>'
                 f'<td class="num dim">{OLD[lbl]:.3f}</td>'
                 f'<td class="num">{f(full[lbl]["r2"])}</td>'
                 f'<td{hl}>{f(vary[lbl]["r2"])}</td>'
                 f'<td class="num">{f(bench[lbl]["r2"])}</td>'
                 f'<td class="num">{full[lbl]["rmse"]:.4f}</td></tr>')

    src = ""
    for name, blk in sorted(REPORT["per_source_metrics"].items(),
                            key=lambda kv: -kv[1]["n"]):
        cells = "".join(
            f'<td class="num">{f(blk["metrics"][l]["r2"])}</td>' for l in PARAM_LABELS)
        flag = ' <span class="flag">data defect</span>' if name == "bacco_multiz" else ""
        src += (f'<tr><td><code>{html.escape(name)}</code>{flag}</td>'
                f'<td class="num">{blk["n"]:,}</td>{cells}</tr>')
    hdr = "".join(f'<th class="num">{PARAM_TEX[l]}</th>' for l in PARAM_LABELS)

    return f"""<section id="results"><div class="wrap">
<p class="kicker">Results</p>
<h2 class="display">Measured on {REPORT['val_split']['n']:,} validation spectra</h2>
<p class="sub">Computed with the released package on a deterministic split with
no randomness anywhere, across {REPORT['val_split']['n_sources']} simulation
sources.</p>

<div class="panel"><div class="tw"><table>
<tr><th>parameter</th><th></th><th class="num">published 2026-05</th>
<th class="num">full validation</th><th class="num">where it varies</th>
<th class="num">bundled benchmark</th><th class="num">RMSE</th></tr>{rows}
</table></div></div>

<p class="muted"><strong style="color:var(--fg)">Read the third column, not the
second.</strong> R-squared is a ratio against the variance of the truth, so on
data where a parameter is held at a fixed value it measures nothing. The
"where it varies" column restricts each parameter to the sources that actually
vary it. For the neutrino mass that is the whole story: the apparent 0.41 is an
artifact of it being pinned at zero across most of the corpus, where predicting
near-zero scores well without recovering anything.</p>

<p class="muted"><strong style="color:var(--fg)">The first column is
superseded.</strong> It came from the training-time evaluator on a validation
set that was later corrected, with a checkpoint selected as best from inside
±0.03 to 0.10 of evaluation noise. It should not be cited.</p>

<p class="muted"><strong style="color:var(--fg)">What actually reproduces.</strong>
The bundled 6,000-row benchmark regenerates its own column to about 1e-6 on any
machine. It does <em>not</em> regenerate the full-validation column: that came
from 162,795 rows of a private split, and the subsample lands within about 0.03
of it through sampling noise. That gap is narrowed here, not closed.</p>

<h3 class="display" style="margin:32px 0 8px; font-size:20px">Why the aggregate is lower than it looks</h3>
<div class="panel"><div class="tw"><table>
<tr><th>source</th><th class="num">n</th>{hdr}</tr>{src}</table></div>
<p class="dim" style="margin:12px 0 0"><code>--</code> means the parameter is
held at a fixed value in that source, so R-squared is undefined rather than
bad.</p></div>
<p class="muted"><code>bacco_multiz</code> is 15 percent of the validation set
and scores zero on everything, because its z=0.47 spectra are self-paired copies
of its z=0 spectra and carry no growth information. That is a data-generation
defect. On sources with sound data the matter density reaches 0.98 to 0.99.</p>

<h3 class="display" style="margin:32px 0 8px; font-size:20px">Reproduce this</h3>
<pre><code>git clone {REPO_URL}
cd cosmufr-run4
pip install -e ".[demo]"
python -m cosmufr.reproduce</code></pre>
</div></section>"""


def _baseline() -> str:
    if not RIDGE:
        return ""
    rows = ""
    for lbl in PARAM_LABELS:
        p = RIDGE["params"].get(lbl, {})
        if p.get("pinned") or p.get("ridge_r2") is None:
            continue
        r, c = p["ridge_r2"], p["cosmufr_r2"]
        rb = f"<strong>{r:.3f}</strong>" if p["winner"] == "ridge" else f"{r:.3f}"
        cb = f"<strong>{c:.3f}</strong>" if p["winner"] == "cosmufr" else f"{c:.3f}"
        name, _ = PARAM_MEANING[lbl]
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="dim">{html.escape(name)}</td>'
                 f'<td class="num">{rb}</td><td class="num">{cb}</td>'
                 f'<td class="dim">{p["winner"]}</td></tr>')
    wins = RIDGE.get("cosmufr_wins", 0)
    n = RIDGE.get("n_compared", 0)

    return f"""<section id="baseline"><div class="wrap">
<p class="kicker">Baseline</p>
<h2 class="display">What does 136 million parameters buy?</h2>
<p class="sub">Every review of a learned-inference model opens with this
question, and this release originally shipped without an answer. Ridge
regression on the same 400 input features, fitted on half the bundled benchmark
and scored on the other half. CosmUFR is scored on the same held-out half, so
the comparison is like-for-like.</p>

<div class="panel"><div class="tw"><table>
<tr><th>parameter</th><th></th><th class="num">ridge, 400 features</th>
<th class="num">CosmUFR, 136M</th><th>winner</th></tr>{rows}</table></div></div>

<p class="muted">CosmUFR is ahead on {wins} of {n}. A linear fit is competitive
on the matter density, which is a real check on how much the network is doing
for the parameters the spectrum constrains most directly. The network pulls
clearly ahead on the parameters that sit at fiducial values through most of the
corpus, where 84.5 million training samples buy a prior that ridge cannot learn
from three thousand rows.</p>
<p class="dim">Both caveats favour ridge: it is fitted on rows from the same
source mix it is tested on, while CosmUFR has never seen any of these spectra.
So this is a generous baseline, not a hostile one. Reproduce with
<code>python scripts/ridge_baseline.py</code>.</p>
</div></section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

def _full(demo_inner: str = "", selected=None) -> str:
    return (_hero() + _what() + _demo(demo_inner, selected) + _audit()
            + _results() + _baseline() + _limits())


def _err(msg: str, detail: str = "") -> str:
    d = f'<pre><code>{html.escape(detail)}</code></pre>' if detail else ""
    return (f'<div class="panel bad" id="result"><strong>That input could not be '
            f'run.</strong><p class="muted" style="margin:8px 0 0">'
            f'{html.escape(msg)}</p>{d}</div>')


@app.get("/", response_class=HTMLResponse)
def index():
    return _page(_full())


@app.post("/infer", response_class=HTMLResponse)
async def infer(example_id: Optional[str] = Form(None),
                upload: Optional[UploadFile] = File(None)):
    truth = None
    selected = None

    if upload is not None and upload.filename:
        raw = await upload.read()
        if len(raw) > MAX_UPLOAD:
            return _page(_full(_err(
                f"File is {len(raw)/1e6:.1f} MB; the limit is "
                f"{MAX_UPLOAD/1e6:.0f} MB.")))
        try:
            if upload.filename.lower().endswith(".npy"):
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
            else:
                arr = np.loadtxt(io.StringIO(raw.decode("utf-8", "replace")),
                                 delimiter=",")
        except Exception as e:
            # The message is escaped: it can contain arbitrary text from the
            # uploaded file, and this is a public endpoint.
            return _page(_full(_err(
                "The file could not be parsed. A .npy must be a plain array of "
                "shape (2, 200); a .csv must be two comma-separated rows of 200 "
                "numbers, with # comment lines ignored.",
                f"{type(e).__name__}: {e}")))

        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape != (2, 200):
            return _page(_full(_err(
                f"Expected shape (2, 200), got {arr.shape}. Row 1 is P(k) at "
                f"z=0 and row 2 is P(k) at z=0.47, each 200 values on the "
                f"training k grid. Download template.csv above for a file that "
                f"works.")))
        pk0, pk047 = arr[0], arr[1]
        source = f"uploaded file {upload.filename}"
    else:
        try:
            selected = int(example_id) if example_id else EXAMPLE_IDS[0]
        except ValueError:
            selected = EXAMPLE_IDS[0]
        selected = max(0, min(selected, len(BENCH) - 1))
        pk0, pk047 = BENCH.pk_z0[selected], BENCH.pk_z047[selected]
        truth = BENCH.params[selected]
        source = _example_label(selected)

    v = validate_spectra(pk0, pk047, K_GRID)
    if not v.ok:
        return _page(_full(_err(
            "The input is not a usable pair of power spectra.",
            "\n".join(v.errors)), selected))

    body = _run(pk0, pk047, truth, source, selected)
    if v.warnings:
        body = ('<div class="panel warn" id="result"><strong>Warnings.</strong> '
                'The model ran, but read these first.<ul>'
                + "".join(f"<li>{html.escape(w)}</li>" for w in v.warnings)
                + "</ul></div>") + body
    return _page(_full(body, selected))


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
