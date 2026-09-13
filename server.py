"""
server.py — the CosmUFR site and live demo, server-rendered.

Structure is a demo, not a report. A visitor should learn what problem this
attacks, see it work on real data, understand what came out, and only then meet
the caveats and the roadmap. The audit that found this model's defects is real
and stays on the page, but it sits where evidence of rigour belongs rather than
in front of the thing being demonstrated.

Rendered entirely on the server: no SSE, no client framework, no CDN. The only
JavaScript is a decorative background and a scroll-into-view helper, both purely
additive. It works behind any proxy and with JavaScript disabled.

Run locally:  uvicorn server:app --port 8000
"""
from __future__ import annotations

import base64
import html
import io
import json
import os
import re
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
from cosmufr.diagram import architecture_svg
from cosmufr.explain import FIGURE_NOTES, PARAM_MEANING
from cosmufr.interpret import (PRIOR_RANGE, read_parameters, read_pk,
                               read_run, read_settling)
from cosmufr.narrate import narrate, run_facts
from cosmufr.load import PARAM_LABELS
from cosmufr.validate import K_GRID, validate_spectra

SHA256 = "5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1"
REPO_URL = "https://github.com/arajgor1/cosmufr-run4"
HF_URL = "https://huggingface.co/arajgor1/cosmufr-run4"
MAX_UPLOAD = 4_000_000

PARAM_TEX = {
    "Om": "&Omega;<sub>m</sub>", "s8": "&sigma;<sub>8</sub>", "h": "h",
    "ns": "n<sub>s</sub>", "Ob": "&Omega;<sub>b</sub>", "w0": "w<sub>0</sub>",
    "mv": "&Sigma;m<sub>&nu;</sub>", "wa": "w<sub>a</sub>",
}

# Plain-language description of each simulation suite the examples come from.
# Without this a visitor sees "camb_nl" in a dropdown and learns nothing.
SOURCE_BLURB = {
    "camb_nl": "CAMB, the standard Boltzmann solver, with a non-linear correction",
    "bacco": "BACCO, an emulator trained on high-resolution N-body simulations",
    "bcemu": "BCemu, which adds the effect of gas and feedback on small scales",
    "spk": "SP(k), a model for how baryons suppress small-scale structure",
    "dark_emulator": "Dark Emulator, from the Dark Quest project (Nishimichi et al. 2019)",
    "bacco_neutrino": "BACCO, with massive neutrinos included",
    "bacco_full8": "BACCO, varying all eight parameters at once",
    "bcemu_neutrino": "BCemu, with massive neutrinos included",
    "bacco_multiz": "BACCO multi-redshift, a set with a known data defect",
    "ns_grid": "a dedicated grid varying the spectral index",
    "camels_astrid_x": "CAMELS Astrid, a hydrodynamic simulation suite",
}

MODEL = cosmufr.load_model(ckpt_path=os.environ.get("COSMUFR_CKPT"), device="cpu")
BENCH = cosmufr.load_benchmark()
AUDIT = cosmufr.weight_audit(MODEL)
CONTACT_EMAIL = "aadityarajgor27@gmail.com"
N_PARAMS = sum(p.numel() for p in MODEL.parameters())
# What fraction of the network each top-level module holds. The headline fact of
# the audit is not which parts failed but how much of the model they are.
MODULE_SHARE = {n: sum(p.numel() for p in mod.parameters()) / N_PARAMS * 100
                for n, mod in MODEL.named_children()}

_reports = Path(__file__).parent / "reports"
REPORT = json.loads((_reports / "honest_eval.json").read_text()) \
    if (_reports / "honest_eval.json").exists() else {}
RIDGE = json.loads((_reports / "ridge_baseline.json").read_text()) \
    if (_reports / "ridge_baseline.json").exists() else {}
# The single source for every number the page prints: reproduced, saved-report and
# diagnostic results, each labelled. Built by scripts/build_results.py.
RESULTS = json.loads((_reports / "results_v1.json").read_text(encoding="utf-8")) \
    if (_reports / "results_v1.json").exists() else {}

EXAMPLE_IDS = list(range(0, min(len(BENCH), 5400), 211))[:24]

# log10 P(k) at z=0 across the benchmark. Used only to tell a visitor whether
# the spectrum they uploaded resembles what the model was trained on, which is
# the most useful thing available when there is no truth to score against.
def _src_name(i: int) -> str:
    return cosmufr.SOURCE_NAMES.get(int(BENCH.source_lid[i]), f"lid_{BENCH.source_lid[i]}")



BENCH_LOGPK = np.log10(np.clip(BENCH.pk_z0, 1e-30, None))

# The spectrum shown in the hero, and what the model actually returns for it.
# Prefer a suite that varies all eight parameters: a fiducial universe reads as
# placeholder text because w0 sits at -1.000 and mv at 0.000.
HERO_ID = next((j for j in EXAMPLE_IDS if _src_name(j) == "bacco_full8"),
               EXAMPLE_IDS[1] if len(EXAMPLE_IDS) > 1 else 0)
_t0 = time.perf_counter()
HERO_RESULT = cosmufr.infer(BENCH.pk_z0[HERO_ID], BENCH.pk_z047[HERO_ID],
                            model=MODEL)
HERO_MS = (time.perf_counter() - _t0) * 1000

app = FastAPI(title="CosmUFR")


# ─────────────────────────────────────────────────────────────────────────────
# Design system, ported from calybre.ai: black ground, a plexus canvas behind
# the hero, technical corner brackets, one accent, Space Grotesk over Inter,
# numbered sections with a category label and a declarative headline.
# Self-contained so no CDN can leave a visitor looking at unstyled HTML.
# ─────────────────────────────────────────────────────────────────────────────

CSS = """
:root{
  --bg:#000; --bg2:#070709; --panel:rgba(18,18,21,.62); --panel2:rgba(255,255,255,.024);
  --line:rgba(255,255,255,.10); --hair:rgba(255,255,255,.055);
  --fg:#EDEEF0; --mut:#9aa1ad; --dim:#666d79;
  --accent:#3b82f6; --accent2:#60a5fa; --accent-bg:rgba(59,130,246,.10);
  --warn:#E6A23C; --bad:#E2643B; --good:#3FBF8F;
  --mono:ui-monospace,"SF Mono",SFMono-Regular,Menlo,Consolas,monospace;
  --maxw:1260px;
}
*{box-sizing:border-box}
/* Never put overflow-x on body: in Chromium it moves the scrolling element off
   <html> and silently breaks every in-page anchor, including the jump to the
   result after a run. Wide content scrolls inside .tw wrappers instead. */
html{scroll-behavior:smooth; scroll-padding-top:80px; background:var(--bg)}
#result{scroll-margin-top:88px}
body{
  margin:0; background:var(--bg); color:var(--fg);
  font-family:Inter,-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
  font-size:15.5px; line-height:1.7; font-weight:300;
  -webkit-font-smoothing:antialiased; text-rendering:optimizeLegibility;
}
.display{font-family:"Space Grotesk",Inter,sans-serif; font-weight:600;
  letter-spacing:-.028em; line-height:1.04}
a{color:var(--accent2); text-decoration:none}
a:hover{color:#fff}
.wrap{max-width:var(--maxw); margin:0 auto; padding:0 32px}

/* ── nav ─────────────────────────────────────────────────────────────── */
nav.top{position:sticky; top:0; z-index:60; background:rgba(0,0,0,.68);
  backdrop-filter:blur(18px) saturate(140%); -webkit-backdrop-filter:blur(18px);
  border-bottom:1px solid var(--hair)}
nav.top .wrap{display:flex; align-items:center; gap:24px; height:66px}
.brand{display:flex; align-items:center; gap:11px; flex-shrink:0}
.brand svg{width:19px; height:19px; color:#fff}
.brand b{font-family:"Space Grotesk",sans-serif; font-weight:700; font-size:14px;
  letter-spacing:.22em; text-transform:uppercase; color:#fff}
.navlinks{display:flex; gap:26px; margin-left:auto; align-items:center;
  overflow-x:auto; min-width:0; scrollbar-width:none}
.navlinks::-webkit-scrollbar{display:none}
/* On a phone the chapter names cannot fit and scrolling them is not a real
   navigation. The page is short enough to scroll, and the CTA still works. */
@media (max-width:640px){.navlinks{display:none}}
.navlinks a{color:var(--mut); font-size:11.5px; letter-spacing:.13em;
  text-transform:uppercase; white-space:nowrap; font-weight:400; transition:color .18s}
.navlinks a:hover{color:#fff}
.navcta{flex-shrink:0; display:inline-flex; align-items:center; gap:7px;
  padding:8px 17px; border:1px solid rgba(255,255,255,.28); border-radius:999px;
  color:#fff !important; font-size:11.5px; letter-spacing:.13em;
  text-transform:uppercase; transition:background .2s,border-color .2s}
.navcta:hover{background:#fff; color:#000 !important; border-color:#fff}

/* ── hero ────────────────────────────────────────────────────────────── */
.hero{position:relative; overflow:hidden; border-bottom:1px solid var(--hair);
  background:radial-gradient(1100px 620px at 74% 34%, rgba(59,130,246,.10), transparent 62%)}
.hero canvas{position:absolute; inset:0; z-index:0; pointer-events:none}
.hero .wrap{position:relative; z-index:2; padding-top:96px; padding-bottom:84px;
  display:grid; grid-template-columns:minmax(0,1.18fr) minmax(0,.82fr);
  gap:52px; align-items:center; min-height:calc(100vh - 66px)}
@media (max-width:980px){
  .hero .wrap{grid-template-columns:1fr; gap:36px; min-height:0; padding-top:72px}
  .heroviz{order:2}
}
.brackets{position:absolute; inset:26px 24px; z-index:1; pointer-events:none}
.brackets i{position:absolute; width:34px; height:34px; border:1px solid rgba(255,255,255,.20)}
.brackets i:nth-child(1){top:0;left:0;border-right:0;border-bottom:0}
.brackets i:nth-child(2){top:0;right:0;border-left:0;border-bottom:0}
.brackets i:nth-child(3){bottom:0;left:0;border-right:0;border-top:0}
.brackets i:nth-child(4){bottom:0;right:0;border-left:0;border-top:0}
.eyebrow{display:inline-flex; align-items:center; gap:9px; padding:6px 14px;
  border:1px solid var(--line); border-radius:999px; background:rgba(255,255,255,.03);
  font-family:var(--mono); font-size:10.5px; letter-spacing:.19em;
  text-transform:uppercase; color:var(--mut); margin-bottom:30px}
.eyebrow .dot{width:6px;height:6px;border-radius:50%;background:var(--good);
  box-shadow:0 0 9px var(--good); animation:pulse 2.6s ease-in-out infinite}
@keyframes pulse{0%,100%{opacity:1}50%{opacity:.32}}
.hero h1{font-size:clamp(34px,3.5vw,50px); margin:0 0 22px; max-width:17ch}
.hero h1 em{font-style:normal; color:var(--dim)}
.hero .lede{font-size:16.5px; line-height:1.62; color:var(--mut); max-width:54ch; margin:0 0 32px}
.ctas{display:flex; gap:12px; flex-wrap:wrap}
.btn{display:inline-flex; align-items:center; gap:9px; padding:13px 24px;
  border-radius:9px; font-size:14px; font-weight:500; border:1px solid transparent;
  cursor:pointer; transition:filter .18s,background .18s,border-color .18s;
  font-family:Inter,sans-serif}
.btn-p{background:var(--accent); color:#fff !important; border-color:var(--accent)}
.btn-p:hover{filter:brightness(1.16); color:#fff !important}
.btn-g{background:rgba(255,255,255,.05); color:var(--fg) !important; border-color:var(--line)}
.btn-g:hover{background:rgba(255,255,255,.10); color:#fff !important}

.metrics{display:grid; grid-template-columns:repeat(auto-fit,minmax(112px,1fr));
  gap:1px; margin-top:38px; background:var(--hair); border:1px solid var(--hair)}
.metrics div{background:rgba(0,0,0,.55); padding:15px 16px}
.metrics .v{font-family:"Space Grotesk",sans-serif; font-size:21px; font-weight:600;
  color:#fff; letter-spacing:-.02em; line-height:1.15}
.metrics .k{font-family:var(--mono); font-size:9.5px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--mut); margin-top:6px; line-height:1.35}

/* hero visual: the real thing the model does, drawn from real benchmark data */
.heroviz{position:relative}
.heroviz .frame{border:1px solid var(--line); border-radius:14px;
  background:rgba(8,8,10,.62); backdrop-filter:blur(10px); padding:18px 20px 16px}
.heroviz .cap{display:flex; justify-content:space-between; align-items:baseline;
  font-family:var(--mono); font-size:9.5px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--dim); margin-bottom:10px}
.heroviz svg{display:block; width:100%; height:auto}
.hgrid{display:grid; grid-template-columns:repeat(4,1fr); gap:1px;
  background:var(--hair); border:1px solid var(--hair); border-radius:9px;
  overflow:hidden; margin-top:14px}
.hgrid div{background:rgba(0,0,0,.7); padding:9px 10px}
.hgrid .p{font-family:var(--mono); font-size:9.5px; color:var(--dim);
  letter-spacing:.08em}
.hgrid .q{font-family:"Space Grotesk",sans-serif; font-size:14.5px; color:#fff;
  font-weight:500; margin-top:2px}
.hgrid .t{font-family:var(--mono); font-size:9px; color:var(--dim); margin-top:3px}

/* ── sections ────────────────────────────────────────────────────────── */
section{padding:104px 0; border-bottom:1px solid var(--hair); position:relative}
.display.fold{font-size:23px; font-weight:600; color:#fff; margin:56px 0 18px;
  padding-top:30px; border-top:1px solid var(--hair); letter-spacing:-.015em}
.lede.caveat{font-size:14.5px; color:var(--mut); border-left:2px solid var(--warn);
  padding-left:16px; margin-top:20px}
.shead{display:grid; grid-template-columns:88px minmax(0,1.12fr) minmax(0,1fr);
  gap:26px 46px; margin-bottom:48px; align-items:start}
.shead .num{font-family:var(--mono); font-size:12px; color:var(--accent);
  letter-spacing:.1em; padding-top:9px}
.shead .cat{font-family:var(--mono); font-size:10.5px; letter-spacing:.21em;
  text-transform:uppercase; color:var(--dim); margin:0 0 13px}
.shead h2{font-size:clamp(27px,3.9vw,44px); margin:0 0 16px; max-width:20ch}
.shead p{color:var(--mut); max-width:54ch; margin:0; font-size:16px}
.shead .sh-p{padding-top:9px}
/* A heading and the sentence under it, side by side, so a wide screen
   carries two columns of content instead of one and a margin. */
.subhead{display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr);
  gap:10px 46px; align-items:start; margin:56px 0 22px; padding-top:30px;
  border-top:1px solid var(--hair)}
.subhead.plain{margin:44px 0 16px; padding-top:0; border-top:0}
.subhead.plain h3{margin:0; font-size:21px}
.subhead .fold{margin:0; padding-top:0; border-top:0}
.subhead .sub-lede{margin:0; max-width:54ch; padding-top:5px}
@media (max-width:980px){
  .shead{grid-template-columns:88px minmax(0,1fr)}
  .shead .sh-p{grid-column:2; padding-top:14px}
  .subhead{grid-template-columns:1fr}
  .subhead .sub-lede{padding-top:0; margin-top:-8px}
}
.sbody{margin-left:114px}
@media (max-width:820px){
  .shead{grid-template-columns:1fr; gap:0} .shead .num{padding:0 0 10px}
  .sbody{margin-left:0} section{padding:72px 0} .wrap{padding:0 22px}
}

/* ── panels, cards, chips ────────────────────────────────────────────── */
.panel{background:var(--panel); backdrop-filter:blur(14px);
  -webkit-backdrop-filter:blur(14px); border:1px solid var(--line);
  border-radius:14px; padding:26px 28px; margin:20px 0}
.panel.tight{padding:18px 20px}
.panel.warn{border-left:3px solid var(--warn)}
.panel.bad{border-left:3px solid var(--bad)}
.panel.accent{border-left:3px solid var(--accent)}

.cards{display:grid; grid-template-columns:repeat(auto-fit,minmax(228px,1fr)); gap:14px}
.card{background:var(--panel2); border:1px solid var(--line); border-radius:13px;
  padding:22px; transition:border-color .2s,background .2s}
.card:hover{border-color:rgba(255,255,255,.20); background:rgba(255,255,255,.04)}
.card .n{font-family:var(--mono); font-size:10px; letter-spacing:.18em;
  color:var(--accent); text-transform:uppercase}
.card h4{margin:11px 0 8px; font-family:"Space Grotesk",sans-serif; font-size:16.5px;
  font-weight:600; color:#fff; letter-spacing:-.01em}
.card p{margin:0; font-size:14px; color:var(--mut); line-height:1.62}

.diagram{border:1px solid var(--line); border-radius:14px; padding:22px 20px 14px;
  background:linear-gradient(180deg,rgba(255,255,255,.03),rgba(255,255,255,.008));
  margin:22px 0; overflow-x:auto}
.diagram svg{display:block; width:100%; min-width:720px; height:auto}
.diagram .cap{font-family:var(--mono); font-size:9.5px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--dim); margin:0 0 12px}
.chips{display:flex; gap:8px; flex-wrap:wrap; margin-top:20px}
.chip{font-family:var(--mono); font-size:10.5px; letter-spacing:.09em;
  color:var(--dim); border:1px solid var(--hair); border-radius:6px;
  padding:6px 11px; background:rgba(255,255,255,.02)}
.chip b{color:var(--fg); font-weight:500}

/* ── tables ──────────────────────────────────────────────────────────── */
.tw{overflow-x:auto; -webkit-overflow-scrolling:touch; border-radius:11px}
table{border-collapse:collapse; width:100%; font-size:13.5px}
th,td{text-align:left; padding:11px 13px; border-bottom:1px solid var(--hair)}
th{color:var(--dim); font-weight:500; font-size:10px; letter-spacing:.15em;
  text-transform:uppercase; white-space:nowrap}
td.num,th.num{text-align:right; font-variant-numeric:tabular-nums;
  font-family:var(--mono); font-size:12.5px}
tbody tr:hover{background:rgba(255,255,255,.022)}
tr:last-child td{border-bottom:none}

/* ── forms ───────────────────────────────────────────────────────────── */
select,input[type=file],button,textarea{font:inherit; color:var(--fg);
  background:rgba(255,255,255,.045); border:1px solid var(--line);
  border-radius:9px; padding:12px 14px; font-size:14px}
select{min-width:min(560px,100%); cursor:pointer}
select:focus,input:focus{outline:2px solid var(--accent); outline-offset:1px}
input[type=file]{cursor:pointer; color:var(--mut)}
.row{display:flex; gap:11px; flex-wrap:wrap; align-items:center}
/* Example picker. A native <select> renders as an opaque white OS menu that
   ignores the page's styling entirely, and collapses 24 entries into one line
   of unreadable text. These are plain submit buttons carrying their own value,
   so the picker is still one form post and still needs no JavaScript. */
.picker{display:grid; grid-template-columns:repeat(auto-fill,minmax(196px,1fr));
  gap:8px; margin:4px 0 18px}
button.exam{display:block; width:100%; text-align:left; padding:11px 13px;
  border-radius:10px; border:1px solid var(--line); background:rgba(255,255,255,.028);
  color:var(--fg); cursor:pointer; font-weight:400; transition:border-color .16s,background .16s}
button.exam:hover{border-color:rgba(96,165,250,.6); background:rgba(96,165,250,.07);
  filter:none}
button.exam.on{border-color:var(--accent); background:var(--accent-bg)}
/* These are spans inside a button: without display:block the id, the suite
   name and the values all run together on one line. */
button.exam .eid{display:block; font-family:var(--mono); font-size:9.5px;
  letter-spacing:.1em; color:var(--dim)}
button.exam.on .eid{color:var(--accent2)}
button.exam .esrc{display:block; font-size:13px; color:#fff; font-weight:500;
  margin:3px 0 6px; font-family:"Space Grotesk",Inter,sans-serif;
  overflow:hidden; text-overflow:ellipsis; white-space:nowrap}
button.exam .etv{display:block; font-family:var(--mono); font-size:10.5px;
  color:var(--mut); line-height:1.55}
.dl{display:inline-flex; align-items:center; gap:7px; padding:7px 13px;
  border:1px solid var(--line); border-radius:8px; font-family:var(--mono);
  font-size:11.5px; color:var(--mut); background:rgba(255,255,255,.028);
  transition:border-color .18s,color .18s}
.dl:hover{color:#fff; border-color:var(--accent)}

code{font-family:var(--mono); font-size:12.5px; background:rgba(255,255,255,.07);
  padding:2px 6px; border-radius:5px; color:#dfe4ec}
pre{font-family:var(--mono); font-size:12px; background:var(--bg2);
  border:1px solid var(--line); border-radius:11px; padding:17px;
  overflow-x:auto; line-height:1.58; color:#c8cfda}
pre code{background:none; padding:0}
img.fig{width:100%; height:auto; border-radius:11px; margin:4px 0 0;
  background:#0B0B0E; border:1px solid var(--line); display:block}

.muted{color:var(--mut); font-size:14.5px}
.dim{color:var(--dim); font-size:13px}
.flag{color:var(--warn); font-family:var(--mono); font-size:10.5px}
.bad-t{color:var(--bad)} .good-t{color:var(--good)}
.lead-in{color:var(--fg); font-weight:400}

/* ── figure blocks ───────────────────────────────────────────────────── */
.figblock{margin:30px 0}
.figcap{border:1px solid var(--line); border-top:0; border-radius:0 0 11px 11px;
  padding:18px 20px; background:var(--panel2)}
.figblock img.fig{border-radius:11px 11px 0 0}
.narration{border:1px solid rgba(59,130,246,.32); border-radius:12px;
  background:rgba(59,130,246,.055); padding:20px 24px; margin:0 0 20px}
.narration .nlbl{display:block; font-family:var(--mono); font-size:9.5px;
  letter-spacing:.17em; text-transform:uppercase; color:var(--accent);
  margin-bottom:12px}
.narration p{margin:0 0 12px; font-size:15px; line-height:1.7; color:var(--fg);
  max-width:74ch}
.narration p:first-of-type{font-size:16.5px; color:#fff;
  font-family:"Space Grotesk",Inter,sans-serif; font-weight:500;
  letter-spacing:-.008em; line-height:1.5}
.narration p.nfoot{font-size:12px; color:var(--dim); line-height:1.6;
  margin:16px 0 0; padding-top:14px; border-top:1px solid var(--hair)}
.reading{border-left:2px solid var(--accent); padding:2px 0 2px 16px;
  margin:0 0 20px}
.reading .rlbl{display:block; font-family:var(--mono); font-size:9.5px;
  letter-spacing:.17em; text-transform:uppercase; color:var(--accent);
  margin-bottom:8px}
.reading p{margin:0 0 10px; font-size:14.5px; color:var(--mut); line-height:1.68}
.reading p.rbottom{color:#fff; font-size:16.5px; font-weight:500;
  font-family:"Space Grotesk",Inter,sans-serif; letter-spacing:-.008em;
  line-height:1.45; margin-bottom:12px}
.reading p:last-child{margin-bottom:0}
.panel .reading{margin:16px 0 0}
#result + h3 + p + .reading{border-left-width:3px; padding:4px 0 4px 20px;
  margin:0 0 22px}
#result + h3 + p + .reading .rbottom{font-size:19px; line-height:1.4}
.figcap .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:16px}
.figcap .lbl{font-family:var(--mono); font-size:9.5px; letter-spacing:.17em;
  text-transform:uppercase; color:var(--accent); display:block; margin-bottom:5px}
.figcap p{margin:0; font-size:13.5px; color:var(--mut); line-height:1.6}

/* ── take it away ─────────────────────────────────────────────────────── */
.two-up{display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr)); gap:14px}
.two-up .panel{margin:0}
h4.ch{margin:0 0 12px; font-family:var(--mono); font-size:10px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--accent)}

/* ── open questions ──────────────────────────────────── */
.opens{display:grid; grid-template-columns:repeat(auto-fit,minmax(300px,1fr));
  gap:14px; margin:22px 0 26px}
.op{border:1px solid var(--hair); border-left:3px solid var(--accent);
  border-radius:11px; padding:20px 22px; background:rgba(20,20,23,.42)}
.op h4{margin:0 0 10px; font-family:"Space Grotesk",Inter,sans-serif;
  font-size:16.5px; font-weight:600; color:#fff; letter-spacing:-.012em;
  line-height:1.35}
.op p{margin:0; font-size:13.5px; line-height:1.68; color:var(--mut)}

/* ── the standing of the work ─────────────────────────────────────────── */
.verdicts{display:grid; grid-template-columns:repeat(auto-fit,minmax(268px,1fr));
  gap:14px; margin:24px 0}
.vd{border:1px solid var(--hair); border-top-width:3px; border-radius:11px;
  padding:20px 22px; background:rgba(20,20,23,.42)}
.vd.good{border-top-color:var(--good)}
.vd.bad{border-top-color:var(--bad)}
.vd.next{border-top-color:var(--accent)}
.vd h4{margin:0 0 12px; font-family:"Space Grotesk",Inter,sans-serif;
  font-size:16px; font-weight:600; color:#fff; letter-spacing:-.01em}
.vd ul{margin:0; padding-left:17px}
.vd li{margin-bottom:9px; font-size:13.5px; line-height:1.6; color:var(--mut)}
.vd li:last-child{margin-bottom:0}
.sub-lede{max-width:72ch; margin:-6px 0 20px; color:var(--mut)}
.hashline{overflow-wrap:anywhere; word-break:break-all}

/* ── the three outcomes ───────────────────────────────────────────────── */
.outcomes{display:grid; gap:14px; margin:22px 0 26px}
.outcome{border:1px solid var(--hair); border-left-width:3px; border-radius:11px;
  padding:20px 22px; background:rgba(20,20,23,.42)}
.outcome.bad{border-left-color:var(--bad)}
.outcome.warn{border-left-color:var(--warn)}
.outcome.good{border-left-color:var(--good)}
.oc-head{display:flex; align-items:baseline; justify-content:space-between;
  gap:14px; flex-wrap:wrap; margin-bottom:8px}
.oc-head h4{margin:0; font-family:"Space Grotesk",Inter,sans-serif; font-size:17px;
  font-weight:600; color:#fff; letter-spacing:-.01em}
.oc-share{font-family:var(--mono); font-size:10px; letter-spacing:.13em;
  text-transform:uppercase; color:var(--dim); white-space:nowrap}
.outcome p{margin:0 0 14px; font-size:14px; line-height:1.65; max-width:74ch}
.oc-tbl{font-size:13px}
.oc-tbl td{padding:5px 14px 5px 0; border:0}
.oc-tbl td:first-child{color:var(--fg)}
.panel.warn{border-color:rgba(230,162,60,.34)}

/* ── evolution ladder ─────────────────────────────────────────────────── */
.steps{border-left:1px solid var(--line); margin-left:6px}
.step-row{display:grid; grid-template-columns:60px 1fr; gap:18px;
  padding:22px 0 22px 26px; position:relative}
.step-row::before{content:""; position:absolute; left:-5px; top:31px; width:9px;
  height:9px; border-radius:50%; background:var(--dim); border:2px solid var(--bg)}
.step-row.ok::before{background:var(--good)}
.step-row.next::before{background:var(--accent)}
.step-row.bad::before{background:var(--bad)}
.step-row.blocked::before{background:var(--warn)}
.step-row .sn{font-family:var(--mono); font-size:19px; color:var(--dim)}
.step-row.ok .sn{color:var(--good)}
.step-row.next .sn{color:var(--accent)}
.step-row.bad .sn{color:var(--bad)}
.step-row.blocked .sn{color:var(--warn)}
.st-head{display:flex; align-items:baseline; gap:12px; flex-wrap:wrap; margin-bottom:6px}
.st-head h4{margin:0; font-family:"Space Grotesk",Inter,sans-serif; font-size:18px;
  font-weight:600; color:#fff; letter-spacing:-.01em}
.pill-ok,.pill-next,.pill-bad,.pill-blocked,.pill-later{font-family:var(--mono);
  font-size:9.5px; letter-spacing:.14em; text-transform:uppercase; padding:3px 9px;
  border-radius:999px; border:1px solid var(--line); color:var(--dim)}
.pill-ok{color:var(--good); border-color:rgba(63,191,143,.45); background:rgba(63,191,143,.09)}
.pill-next{color:var(--accent); border-color:rgba(59,130,246,.45); background:rgba(59,130,246,.09)}
.pill-bad{color:var(--bad); border-color:rgba(226,100,59,.45); background:rgba(226,100,59,.09)}
.pill-blocked{color:var(--warn); border-color:rgba(230,162,60,.45); background:rgba(230,162,60,.09)}
@media (max-width:700px){.step-row{grid-template-columns:1fr; gap:6px; padding-left:20px}}

/* ── roadmap ─────────────────────────────────────────────────────────── */
.road{display:grid; grid-template-columns:repeat(auto-fit,minmax(260px,1fr)); gap:16px}
.road .col{border:1px solid var(--line); border-radius:13px; padding:22px;
  background:var(--panel2)}
.road .col.now{border-color:rgba(63,191,143,.34); background:rgba(63,191,143,.045)}
.road .col.next{border-color:rgba(59,130,246,.34); background:rgba(59,130,246,.045)}
.road .st{font-family:var(--mono); font-size:10px; letter-spacing:.18em;
  text-transform:uppercase; margin-bottom:12px}
.road .col.now .st{color:var(--good)} .road .col.next .st{color:var(--accent)}
.road .col.later .st{color:var(--dim)}
.road ul{margin:0; padding-left:17px} .road li{font-size:13.8px; margin:9px 0}
.road h4{margin:0 0 4px; font-family:"Space Grotesk",sans-serif; font-size:16px;
  color:#fff; font-weight:600}

ol,ul{padding-left:19px} li{margin:9px 0; color:var(--mut); font-size:14.5px}
li strong{color:var(--fg); font-weight:500}
details summary{cursor:pointer; color:var(--mut); font-size:13.5px; user-select:none}
details[open] summary{margin-bottom:11px}
footer{padding:56px 0 78px; color:var(--dim); font-size:13.5px}
footer a{color:var(--mut)}
hr{border:none; border-top:1px solid var(--hair); margin:28px 0}
"""

# Plexus background, ported from the Calybre site's ParticleCanvas. Drifting
# points joined by faint lines: for a cosmology page it reads as large-scale
# structure, which is what the model is actually looking at.
PARTICLES_JS = """
(function(){
  var c = document.getElementById('plexus');
  if (!c || !c.getContext) return;
  if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) return;
  var ctx = c.getContext('2d'), W = 0, H = 0, ps = [], raf, mx = -9999, my = -9999;
  var LINK = 165;
  function size(){
    var p = c.parentElement, dpr = window.devicePixelRatio || 1;
    W = p.clientWidth; H = p.clientHeight;
    c.width = W * dpr; c.height = H * dpr;
    c.style.width = W + 'px'; c.style.height = H + 'px';
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    var n = W < 760 ? 34 : 74; ps = [];
    for (var i = 0; i < n; i++) ps.push({
      x: Math.random()*W, y: Math.random()*H,
      vx:(Math.random()-0.5)*0.24, vy:(Math.random()-0.5)*0.24,
      r: Math.random()*1.4 + 0.4
    });
  }
  function frame(){
    ctx.clearRect(0,0,W,H);
    for (var i=0;i<ps.length;i++){
      var a = ps[i];
      a.x += a.vx; a.y += a.vy;
      if (a.x<0||a.x>W) a.vx*=-1;
      if (a.y<0||a.y>H) a.vy*=-1;
      ctx.beginPath(); ctx.arc(a.x,a.y,a.r,0,6.2832);
      ctx.fillStyle='rgba(255,255,255,0.50)'; ctx.fill();
      for (var j=i+1;j<ps.length;j++){
        var b=ps[j], dx=a.x-b.x, dy=a.y-b.y, d=Math.sqrt(dx*dx+dy*dy);
        if (d<LINK){
          ctx.beginPath();
          ctx.strokeStyle='rgba(255,255,255,'+((1-d/LINK)*0.13)+')';
          ctx.lineWidth=1; ctx.moveTo(a.x,a.y); ctx.lineTo(b.x,b.y); ctx.stroke();
        }
      }
      var mdx=a.x-mx, mdy=a.y-my, md=Math.sqrt(mdx*mdx+mdy*mdy);
      if (md<210){
        ctx.beginPath();
        ctx.strokeStyle='rgba(96,165,250,'+((1-md/210)*0.20)+')';
        ctx.lineWidth=1; ctx.moveTo(a.x,a.y); ctx.lineTo(mx,my); ctx.stroke();
      }
    }
    raf = requestAnimationFrame(frame);
  }
  size(); frame();
  window.addEventListener('resize', size);
  c.parentElement.addEventListener('mousemove', function(e){
    var r=c.getBoundingClientRect(); mx=e.clientX-r.left; my=e.clientY-r.top;
  });
  c.parentElement.addEventListener('mouseleave', function(){ mx=-9999; my=-9999; });
})();
"""

ANCHOR_JS = """
/* In-page links jump rather than glide. Smooth scrolling over a document this
   tall is cancelled part-way by the inline figures decoding and reflowing
   underneath the animation, so a visitor clicks a nav link, watches it
   highlight, and does not move. Instant landing cannot be interrupted.
   Progressive enhancement: without JS the browser's own jump still works. */
(function(){
  var land = function(el){
    if (!el) return;
    el.scrollIntoView({block:'start', behavior:'instant'});
  };
  document.addEventListener('click', function(ev){
    var a = ev.target.closest ? ev.target.closest('a[href^="#"]') : null;
    if (!a) return;
    var id = a.getAttribute('href').slice(1);
    if (!id) return;
    var el = document.getElementById(id);
    if (!el) return;
    ev.preventDefault();
    history.replaceState(null, '', '#' + id);
    land(el);
    /* Figures below the fold can still be decoding; land once more when they
       have finished, so the section header does not drift off the top. */
    window.addEventListener('load', function(){ land(el); }, {once: true});
  });
  /* Arriving with a fragment already in the URL hits the same problem. */
  if (location.hash.length > 1) {
    var t = document.getElementById(location.hash.slice(1));
    if (t) { land(t); window.addEventListener('load', function(){ land(t); }); }
  }
})();
"""


SCROLL_JS = """
/* Land on the result after a POST. The URL fragment alone is unreliable: the
   page paints before the inline figures decode and the browser settles back at
   the top, so a visitor clicks Run and sees nothing change. Purely additive. */
(function(){
  var e = document.getElementById('result');
  if (!e) return;
  var go = function(){ e.scrollIntoView({block:'start', behavior:'instant'}); };
  go(); window.addEventListener('load', go);
})();
"""

MARK = ('<svg viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">'
        '<rect x="50" y="7.5" width="60" height="60" transform="rotate(45 50 7.5)" '
        'fill="none" stroke="currentColor" stroke-width="9"/>'
        '<circle cx="50" cy="50" r="9" fill="currentColor"/></svg>')

NAV_LINKS = [("#what-i-did", "What I did"),
             ("#what-i-observed", "What I observed"),
             ("#what-i-conclude", "What I conclude"),
             ("#demo", "Try it"),
             ("#the-code", "The code")]


# The figures are generated by cosmufr.figures, which draws for a paper: black
# text on white. Dropped into a dark page as-is they read as screenshots from a
# PDF rather than part of the product. This re-skins them at render time to the
# site's palette, without touching the plotting code, so the notebook and the
# README still get the light version they want.
FIG_BG = "#0B0B0E"
FIG_FG = "#C9CFD9"
FIG_GRID = "#2A2E36"

_DARK_RC = {
    "text.color": FIG_FG, "axes.labelcolor": FIG_FG,
    "axes.edgecolor": FIG_GRID, "axes.titlecolor": "#F2F4F7",
    "xtick.color": FIG_FG, "ytick.color": FIG_FG,
    "grid.color": FIG_GRID, "figure.facecolor": FIG_BG,
    "axes.facecolor": FIG_BG, "savefig.facecolor": FIG_BG,
    "legend.facecolor": FIG_BG, "legend.edgecolor": FIG_GRID,
}


def _png(fig, dark: bool = True) -> str:
    if dark:
        # The figure already exists, so restyle its artists rather than relying
        # on rcParams, which only apply at creation time.
        fig.patch.set_facecolor(FIG_BG)
        for ax in fig.get_axes():
            ax.set_facecolor(FIG_BG)
            ax.title.set_color("#F2F4F7")
            ax.xaxis.label.set_color(FIG_FG)
            ax.yaxis.label.set_color(FIG_FG)
            ax.tick_params(colors=FIG_FG, which="both")
            for sp in ax.spines.values():
                sp.set_color(FIG_GRID)
            for gl in ax.get_xgridlines() + ax.get_ygridlines():
                gl.set_color(FIG_GRID)
            # Annotations added with ax.annotate/ax.text default to black and
            # disappear on a dark plate. Only recolour the ones that are
            # actually dark, so deliberate colour coding survives.
            for t in ax.texts:
                c = t.get_color()
                if c in ("black", "k", "#000000", (0.0, 0.0, 0.0, 1.0)):
                    t.set_color(FIG_FG)
            leg = ax.get_legend()
            if leg is not None:
                leg.get_frame().set_facecolor(FIG_BG)
                leg.get_frame().set_edgecolor(FIG_GRID)
                for t in leg.get_texts():
                    t.set_color(FIG_FG)
        # Captions written with fig.text default to black and vanish on dark.
        for t in fig.texts:
            if t.get_color() in ("black", "#000000", (0.0, 0.0, 0.0, 1.0)):
                t.set_color(FIG_FG)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=118, bbox_inches="tight",
                facecolor=FIG_BG if dark else "white")
    matplotlib.pyplot.close(fig)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _reading(lines) -> str:
    """
    What this particular run shows, computed from its own numbers.

    Separate from FIGURE_NOTES, which describe the figure in general. These
    sentences are derived from the result on screen, so they stay true for an
    uploaded spectrum the model has never seen, and for a future checkpoint
    where the current defects are fixed.

    Items are (kind, text). The "bottom" line leads and is styled as the
    headline, because a reader wants the conclusion before the evidence.
    """
    if not lines:
        return ""
    body = ""
    for kind, text in lines:
        cls = "rbottom" if kind == "bottom" else ""
        body += f'<p class="{cls}">{html.escape(text)}</p>'
    return (f'<div class="reading"><span class="rlbl">What this is telling '
            f'you</span>{body}</div>')


def _figblock(key: str, img_src: str, reading=None) -> str:
    n = FIGURE_NOTES[key]
    cells = [("Input", n["input"]), ("Output", n["output"]),
             ("What it means", n["means"])]
    if n["why"]:
        cells.append(("Why it looks like this", n["why"]))
    inner = "".join(
        f'<div><span class="lbl">{lbl}</span><p>{html.escape(txt)}</p></div>'
        for lbl, txt in cells)
    return (f'<div class="figblock"><img class="fig" src="{img_src}" '
            f'alt="{html.escape(n["title"])}">'
            f'<div class="figcap">{_reading(reading)}'
            f'<div class="grid">{inner}</div></div></div>')


def _sub(section: str, heading: str, standfirst: str = "") -> str:
    """One rendered section's body, relabelled for its place in the argument."""
    OPEN = '<div class="sbody">'
    start = section.index(OPEN) + len(OPEN)
    # Sections close their body then their wrapper, sometimes on one line and
    # sometimes on two. Cut at the wrapper either way.
    end = section.rindex("</div></section>")
    body = section[start:end].rstrip()
    if body.endswith("</div>"):
        body = body[:-6]
    if standfirst:
        head = (f'<div class="subhead"><h3 class="display fold">{heading}</h3>'
                f'<p class="muted sub-lede">{standfirst}</p></div>')
    else:
        head = f'<h3 class="display fold">{heading}</h3>'
    return f'{head}{body}'


def _chapter(num: str, cat: str, title: str, sub: str, *blocks: str) -> str:
    """A top-level question, and the evidence that answers it."""
    anchor = cat.lower().replace(" ", "-")
    return (f'<section id="{anchor}"><div class="wrap">'
            f'{_shead(num, cat, title, sub)}'
            f'<div class="sbody">{"".join(blocks)}</div></div></section>')


def _extend(primary: str, *fragments: str) -> str:
    """Append already-prepared fragments into a rendered section's body."""
    CLOSE = "</div></div></section>"
    return primary[:primary.rindex(CLOSE)] + "".join(fragments) + CLOSE


def _fold(primary: str, *rest: str) -> str:
    """
    Render several sections as one, the follow-on ones demoted to sub-headings.

    Each section function owns a large f-string, so combining them by editing
    those strings would be fragile. This works on the rendered HTML instead: keep
    the first section's frame, and graft the others' bodies into it under their
    own titles.
    """
    OPEN, CLOSE = '<div class="sbody">', "</div></div></section>"
    out = primary[:primary.rindex(CLOSE)]
    for r in rest:
        title = re.search(r'<h2 class="display">(.*?)</h2>', r, re.S)
        body = r[r.index(OPEN) + len(OPEN):r.rindex(CLOSE)]
        head = (f'<h3 class="display fold">{title.group(1)}</h3>'
                if title else "")
        out += head + body
    return out + CLOSE


def _shead(num: str, cat: str, title: str, sub: str = "") -> str:
    """The section title, with its standfirst in a column beside it."""
    p = f'<div class="sh-p"><p>{sub}</p></div>' if sub else ""
    return (f'<div class="shead"><div class="num">{num}</div>'
            f'<div><p class="cat">{cat}</p><h2 class="display">{title}</h2></div>'
            f'{p}</div>')


def _page(body: str, title: str = "CosmUFR — cosmology from a power spectrum") -> HTMLResponse:
    links = "".join(f'<a href="{h}">{html.escape(t)}</a>' for h, t in NAV_LINKS)
    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="A cosmological parameter-prediction prototype that reads the matter power spectrum, published with its benchmark and an audit of its training defects.">
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap" rel="stylesheet">
<style>{CSS}</style></head><body>
<nav class="top"><div class="wrap">
  <div class="brand">{MARK}<b>CosmUFR</b></div>
  <div class="navlinks">{links}</div>
  <a class="navcta" href="#demo">Run it &rarr;</a>
</div></nav>
{body}
<script>{PARTICLES_JS}{ANCHOR_JS}{SCROLL_JS}</script>
</body></html>""")


# ─────────────────────────────────────────────────────────────────────────────
# Narrative sections
# ─────────────────────────────────────────────────────────────────────────────


def _hero_visual() -> str:
    """
    The two input curves and the eight numbers the model returns for them, run
    on a real benchmark spectrum at import.

    The true values sit underneath the predictions on purpose. An earlier version
    of this panel printed the truth alone under a caption reading "Output", which
    is the answer key dressed as a result, on a page whose whole claim is that it
    does not do that.
    """
    i = HERO_ID
    lk = np.log10(K_GRID)
    a = np.log10(np.clip(BENCH.pk_z0[i], 1e-30, None))
    b = np.log10(np.clip(BENCH.pk_z047[i], 1e-30, None))

    W, H, PL, PR, PT, PB = 520.0, 250.0, 38.0, 12.0, 14.0, 26.0
    x0, x1 = lk.min(), lk.max()
    y0 = float(min(a.min(), b.min())); y1 = float(max(a.max(), b.max()))
    pad = 0.06 * (y1 - y0); y0 -= pad; y1 += pad

    def px(v): return PL + (v - x0) / (x1 - x0) * (W - PL - PR)
    def py(v): return PT + (1 - (v - y0) / (y1 - y0)) * (H - PT - PB)

    def path(arr):
        return "M " + " L ".join(f"{px(lk[j]):.1f},{py(arr[j]):.1f}"
                                 for j in range(0, len(arr), 2))

    grid = ""
    for kv in (0.1, 0.3, 1.0, 3.0):
        gx = px(np.log10(kv))
        grid += (f'<line x1="{gx:.1f}" y1="{PT}" x2="{gx:.1f}" y2="{H-PB}" '
                 f'stroke="rgba(255,255,255,.07)"/>'
                 f'<text x="{gx:.1f}" y="{H-PB+15}" fill="#666d79" font-size="9" '
                 f'font-family="ui-monospace,monospace" text-anchor="middle">{kv:g}</text>')

    pred = HERO_RESULT.params_array
    true = BENCH.params[i]
    fmt = lambda l, v: f"{v:+.3f}" if l in ("w0", "wa") else f"{v:.3f}"
    cells = "".join(
        f'<div><div class="p">{PARAM_TEX[l]}</div>'
        f'<div class="q">{fmt(l, pred[j])}</div>'
        f'<div class="t">true {fmt(l, true[j])}</div></div>'
        for j, l in enumerate(PARAM_LABELS))

    return f"""<div class="heroviz"><div class="frame">
<div class="cap"><span>Input &middot; log&#8321;&#8320; P(k)</span><span>{html.escape(_src_name(i))}</span></div>
<svg viewBox="0 0 {W:.0f} {H:.0f}" role="img"
     aria-label="Matter power spectrum at two redshifts">
  {grid}
  <line x1="{PL}" y1="{PT}" x2="{PL}" y2="{H-PB}" stroke="rgba(255,255,255,.14)"/>
  <line x1="{PL}" y1="{H-PB}" x2="{W-PR}" y2="{H-PB}" stroke="rgba(255,255,255,.14)"/>
  <path d="{path(a)}" fill="none" stroke="#60a5fa" stroke-width="2"/>
  <path d="{path(b)}" fill="none" stroke="#3b82f6" stroke-width="1.6"
        stroke-dasharray="4 3" opacity=".85"/>
  <text x="{W-PR}" y="{PT+12}" fill="#60a5fa" font-size="9.5" text-anchor="end"
        font-family="ui-monospace,monospace">z = 0.00</text>
  <text x="{W-PR}" y="{PT+26}" fill="#3b82f6" font-size="9.5" text-anchor="end"
        font-family="ui-monospace,monospace">z = 0.47</text>
  <text x="{(PL+W-PR)/2:.0f}" y="{H-4}" fill="#666d79" font-size="9"
        font-family="ui-monospace,monospace" text-anchor="middle">k  [h/Mpc]</text>
</svg>
<div class="cap" style="margin:16px 0 0"><span>Output &middot; what the model infers, against the truth</span><span>{HERO_MS:.0f} ms</span></div>
<div class="hgrid">{cells}</div>
</div></div>"""


def _hero() -> str:
    m = RESULTS["modules"]
    unchanged = m["unchanged_share_percent"]
    head = m["by_module"]["param_head"]["share_percent"]
    om = RESULTS["validation_report"]["varying_only"]["Om"]["rmse"]
    return f"""<header class="hero">
<canvas id="plexus"></canvas>
<div class="brackets"><i></i><i></i><i></i><i></i></div>
<div class="wrap">
<div>
<div class="eyebrow"><span class="dot"></span>Run 4 &middot; research preview &middot; live model</div>
<h1 class="display">A 136M-parameter model for cosmological inference &mdash;<br>
<em>and the audit that found its core still at initialization.</em></h1>
<p class="lede">CosmUFR predicts eight cosmological parameters from the matter
power spectrum in one forward pass. It was designed to reach them by refining a
belief over sixteen steps. In the released checkpoint the encoder, the belief
proposal and the refinement networks are bit-identical to their state before
training, and the code that trained it passes them no gradient, so the answers
come from a read-out head on a fixed random projection of the input. This page
covers what I did, what I observed, what I conclude, the model running live, and
the code. Each number is labelled as reproduced, taken from a saved report, or
from a diagnostic run.</p>
<p class="lede caveat"><span class="lead-in">Before anything else.</span> The
input is an emulator matter power spectrum, not a measured observable: no survey
window, shot noise, mask, galaxy bias or noise model. Nothing here has been
pointed at survey data.</p>
<div class="ctas">
  <a class="btn btn-p" href="#what-i-observed">What I found &rarr;</a>
  <a class="btn btn-g" href="#demo">Run it on a spectrum</a>
</div>
<div class="metrics">
  <div><div class="v">{unchanged:.1f}%</div><div class="k">of parameters still at initialization</div></div>
  <div><div class="v">{head:.1f}%</div><div class="k">compute the reported parameters</div></div>
  <div><div class="v">{om:.3f}</div><div class="k">RMSE on matter density, validation report</div></div>
  <div><div class="v">6,000</div><div class="k">benchmark rows, reproduced within 1e-5</div></div>
</div>
</div>
{_hero_visual()}
</div></header>"""


def _idea() -> str:
    return f"""<section id="idea"><div class="wrap">
{_shead("02", "What this is", "Surveys are fast now. Interpreting them is not.",
        "A telescope survey measures where hundreds of millions of galaxies are. "
        "Turning that into a statement about the universe means running the "
        "physics backwards, and that inversion is the slow part.")}
<div class="sbody">

<div class="panel accent">
<p class="muted" style="margin:0 0 16px"><span class="lead-in">The conventional
route.</span> You guess a set of cosmological parameters, simulate the universe
they would produce, compare it to what was measured, and repeat. Hundreds of
thousands of times, until the guesses converge. It is reliable and it is slow:
CPU-days to CPU-weeks for a single analysis.</p>
<p class="muted" style="margin:0"><span class="lead-in">The approach this
tests.</span> Train a network on tens of millions of emulator-generated spectra
until it maps a spectrum to its parameters directly, so a new spectrum costs a
single forward pass. If that worked well enough, some analyses could become much
cheaper to run. That is a long-term aim, and nothing here tests it.</p>
</div>

<div class="cards" style="margin-top:26px">
<div class="card"><div class="n">One measurement in</div>
<h4>The power spectrum</h4>
<p>A single curve summarising how clumpy the universe is at every scale, measured
at two moments in cosmic history.</p></div>
<div class="card"><div class="n">One forward pass</div>
<h4>No sampling, no likelihood</h4>
<p>No simulator in the loop and no chain to converge. The network was trained
once; using it costs one pass.</p></div>
<div class="card"><div class="n">Eight numbers out</div>
<h4>A candidate cosmology</h4>
<p>How much matter, how clumpy, how fast the expansion, what the dark energy is
doing. Plus which of those it does not recover.</p></div>
</div>

<div class="subhead plain"><h3 class="display">The path a spectrum takes</h3>
<p class="muted sub-lede">Four hundred numbers go in on the left. The encoder
turns them into a 1024-dimensional "belief" about which universe this is, sixteen
refinement steps are meant to sharpen that belief, and three read-out heads turn
it into answers.</p></div>
<div class="diagram"><p class="cap">Inference path &middot; one forward pass</p>
{architecture_svg()}</div>

<p class="muted" style="margin-top:26px">This release is an early prototype of
that idea. The benchmark ships with the code, each number says where it comes
from, and the parts that do not work are named.</p>
</div></div></section>"""


def _input_section() -> str:
    src_rows = ""
    seen = set()
    for i in EXAMPLE_IDS:
        name = cosmufr.SOURCE_NAMES.get(int(BENCH.source_lid[i]), "")
        if name in seen or name not in SOURCE_BLURB:
            continue
        seen.add(name)
        src_rows += (f'<tr><td><code>{html.escape(name)}</code></td>'
                     f'<td class="muted">{html.escape(SOURCE_BLURB[name])}</td></tr>')

    return f"""<section id="input"><div class="wrap">
{_shead("--", "The input", "What the model actually reads.",
        "Two curves. Everything the model knows about a universe comes from "
        "these 400 numbers.")}
<div class="sbody">

<div class="cards">
<div class="card"><div class="n">What P(k) is</div>
<h4>Clumpiness, scale by scale</h4>
<p>Matter is not spread evenly. The power spectrum says how much structure exists
at each size: large scales on the left, small ones on the right. It falls to the
right because there is less power in each mode at small scales.</p></div>
<div class="card"><div class="n">Why two of them</div>
<h4>Now, and about five billion years ago</h4>
<p>One curve at z=0 and one at z=0.47. Comparing them shows how fast structure
grew, which is what separates parameters that would otherwise look identical
from a single snapshot.</p></div>
<div class="card"><div class="n">The fixed grid</div>
<h4>200 points, k = 0.1 to 4.5</h4>
<p>Both curves are sampled at the same 200 scales, in units of h/Mpc. That fixed
grid is the contract: anything you feed the model has to be on it.</p></div>
</div>

<div class="chips">
<span class="chip"><b>400</b> input numbers</span>
<span class="chip"><b>2</b> redshifts &middot; z=0.00, z=0.47</span>
<span class="chip"><b>200</b> log-spaced k bins each</span>
<span class="chip"><b>0.1 &ndash; 4.5</b> h/Mpc</span>
<span class="chip">units <b>(Mpc/h)&sup3;</b></span>
</div>

<div class="subhead plain"><h3 class="display">The examples in the demo</h3>
<p class="muted sub-lede">The dropdown is not a set of toy inputs. Each entry is
the spectrum a published cosmology code produces for one specific set of
parameters, held out of training, with those parameters recorded. The label shows
the suite it came from and three of its true values, so you can see before you
run it what the right answer is.</p></div>
<p class="dim" style="max-width:66ch">They are simulation output, not
measurements. Nobody observes a matter power spectrum: a survey records where
galaxies are, and the spectrum is inferred from that through a survey window, a
selection function, shot noise and a galaxy bias model with its own free
parameters. None of that is present here, in the training data or in these
examples, and none of it is solved. It is the gap between this and anything that
could be pointed at a real survey.</p>
<div class="panel tight"><div class="tw"><table>
<tr><th>label in the dropdown</th><th>what it is</th></tr>{src_rows}
</table></div></div>
<p class="dim">Every one of these is downloadable, so you can take a spectrum
away, feed it back through the upload box, and confirm you get the same answer.</p>
</div></div></section>"""


def _output_section() -> str:
    rows = ""
    for lbl in PARAM_LABELS:
        name, desc = PARAM_MEANING[lbl]
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td style="white-space:nowrap"><strong>{html.escape(name)}</strong></td>'
                 f'<td class="muted">{html.escape(desc)}</td></tr>')
    return f"""<section id="output"><div class="wrap">
{_shead("--", "The output", "Reading the eight numbers.",
        "Every run returns the same eight parameters. Here is what each one is, "
        "and how to tell a good answer from a bad one.")}
<div class="sbody">
<div class="panel tight"><div class="tw"><table>
<tr><th>symbol</th><th>name</th><th>what it controls</th></tr>{rows}
</table></div></div>

<div class="cards" style="margin-top:24px">
<div class="card"><div class="n">How to read it</div><h4>Predicted vs true</h4>
<p>Because the examples are held-out simulations, the demo shows the true value
next to the prediction and the difference between them. A small residual means
the model recovered that parameter for that universe.</p></div>
<div class="card"><div class="n">Read with care</div><h4>The &sigma; column</h4>
<p>The model reports an uncertainty, and it is not trustworthy. It emits the same
constant on every input, so it is flagged in the results table and should be
ignored. What I observed explains why.</p></div>
<div class="card"><div class="n">Expect a spread</div><h4>Not all eight are equal</h4>
<p>On this data the model recovers matter density and clumpiness best, expansion
rate and dark energy partly, and neutrino mass not at all. Whether that ordering
reflects what the spectrum contains or what the training data covered is an open
question.</p></div>
</div>
</div></div></section>"""



# The author's run notes. The September 2026 audit examined Run 2's and Run 4's
# checkpoints and the saved logs of later smoke runs directly; the rest of this
# table is the historical account, and the page says so under the table.
RUNS = [
    ("Run 1",
     "First end-to-end training.",
     "The pipeline ran end to end; parameter recovery did not."),
    ("Run 2",
     "Longer curriculum from a fresh initialization.",
     "Matter density and clustering amplitude came in. Its checkpoint saved before "
     "any optimizer step is the reference the audit compares against."),
    ("Run 3",
     "Reweighted objective, fixed sequential-consistency loss, torch.compile.",
     "Faster; the expansion rate got worse."),
    ("Run 4",
     "Separate joint and sequential belief proposals; warm-started through Run 3.",
     "The released checkpoint."),
    ("Runs 5&ndash;8",
     "Further architectural attempts, several of them short smoke runs.",
     "Inconclusive: epoch-to-epoch noise in the saved evaluations was as large as "
     "the differences being compared."),
    ("The audit",
     "Stopped training and examined the weights and the training code.",
     "The encoder, proposal and refinement networks never left their initial "
     "values in Runs 2 to 4, and the energy objective is unbounded below."),
]


def _evolution() -> str:
    rows = "".join(
        f'<tr><td><strong>{name}</strong></td>'
        f'<td class="muted">{changed}</td>'
        f'<td class="muted">{learned}</td></tr>'
        for name, changed, learned in RUNS)
    return f"""<section id="evolution"><div class="wrap">
{_shead("07", "How it got here", "Training runs, then a decision to stop training.",
        "The useful history is not a score table. It is what each run changed "
        "and what that taught, because the scores turned out to be less "
        "trustworthy than they looked.")}
<div class="sbody">
<div class="panel"><div class="tw"><table>
<tr><th>run</th><th>what changed</th><th>what it taught</th></tr>
{rows}</table></div>
<p class="dim" style="margin:12px 0 0">From the author's run notes, except where
the audit examined checkpoints or saved logs directly.</p></div>

<div class="panel warn">
<p class="muted" style="margin:0"><span class="lead-in">Why there are no
per-run scores here.</span> There were, for months. A June review of the saved
training logs found that epoch-to-epoch evaluation noise was &plusmn;0.03 to 0.10, larger than nearly
every difference being compared between runs, and that the validation set had
changed partway through without the earlier numbers being re-measured against
it. So the run-to-run deltas that drove five months of decisions were mostly
noise. Publishing them as a progression would repeat the mistake. What survives
is the architectural lesson from each run, which is what the table records.</p>
</div>

<p class="muted">The most expensive lesson in the run notes was Run 5: several
changes shipped at once, a regression, and no way to tell which change caused it.
Any future run should change one thing at a time, with a threshold written down
before it starts.</p>
</div></div></section>"""


def _conclusion() -> str:
    """The standing of the work, stated plainly enough to disagree with."""
    vary = RESULTS["validation_report"]["varying_only"]
    om, s8, mv = vary["Om"]["rmse"], vary["s8"]["rmse"], vary["mv"]["r2"]

    return f"""
<div class="panel accent">
<p class="muted" style="margin:0 0 14px"><span class="lead-in">In one
paragraph.</span> The architecture this project set out to test has not been
tested. Its refinement did not operate in the released model, so nothing here is
evidence for or against the idea. What exists is a checkable point predictor: it
recovers two of eight parameters on emulator spectra, is beaten by a linear fit on
one of them, and fails on the one hydrodynamic slice it is tested against. That is
a starting point, not a result.</p>
<p class="muted" style="margin:0">The audit and the retraction under it are about
the same thing: how easy it was to run training, watch the loss fall, and be wrong
about what the model was doing.</p>
</div>

<div class="verdicts">
  <div class="vd good">
    <h4>What it does today</h4>
    <ul>
      <li>Returns eight parameters from a power spectrum in a few hundred
          milliseconds on CPU, with no simulator in the loop. The demo prints the
          time of each call.</li>
      <li>Recovers matter density and clustering amplitude with RMSE of about
          {om:.3f} and {s8:.3f} on validation rows where they vary.</li>
      <li>Reproduces the bundled 6,000-row benchmark within 1e-5 in R&sup2;.</li>
      <li>Reports its own degenerate outputs on every run.</li>
    </ul>
  </div>
  <div class="vd bad">
    <h4>What it does not do</h4>
    <ul>
      <li>Report an uncertainty. The &sigma; output is a constant.</li>
      <li>Work on survey data. The input is an emulator spectrum with no window,
          shot noise, mask, galaxy bias or noise model.</li>
      <li>Recover neutrino mass: R&sup2; {mv:.3f} where it varies.</li>
      <li>Beat predicting the mean on the hydrodynamic slice, where a channel-order
          defect also confounds the cause.</li>
      <li>Support any claim about iterative refinement.</li>
    </ul>
  </div>
  <div class="vd next">
    <h4>What would come next</h4>
    <ul>
      <li><strong>A trustworthy data subset.</strong> Recorded provenance, redshift
          channels mapped from what the generator returns, and splits grouped by
          base cosmology.</li>
      <li><strong>A model that genuinely trains.</strong> Every intended module
          receiving a gradient and an update, checked module by module, and an
          objective bounded below. The diagnosis of what the current code reaches
          is done; the repair is not.</li>
      <li><strong>A fair comparison.</strong> Refinement against a tuned direct
          network of matched size, repeated seeds, and a threshold written down
          before the test set is opened.</li>
    </ul>
  </div>
</div>

<div class="subhead plain"><h3 class="display">What I do not know how to fix</h3>
<p class="muted sub-lede">These are the questions this project cannot settle from
the inside. Each one changes what the next training run should be, and I cannot
tell which answer is right.</p></div>

<div class="opens">
  <div class="op">
    <h4>Is refining an answer worth doing at all?</h4>
    <p>The design assumes a model that sharpens its answer over several steps can
    spend more effort on a hard observation than an easy one. That has not been
    tested, because the mechanism never operated. A single pass may reach the same
    place for a fraction of the machinery, and nothing here argues otherwise.</p>
  </div>
  <div class="op">
    <h4>What should the internal score be trained to do?</h4>
    <p>The refinement descends a score the model learns for itself. One thing is
    known: the current objective adds the mean energy to a term that depends only
    on differences, so it is unbounded below. What it should be instead, tied to
    something observable and never shown the answer at inference, I do not have a
    formulation I believe in.</p>
  </div>
  <div class="op">
    <h4>How much of the weakness is the measurement, and how much is mine?</h4>
    <p>Two of the eight parameters recover well and six do not. I once claimed this
    was a limit of the observable; that bound was never derived and is withdrawn.
    I cannot separate what two noiseless spectra over this range of scales contain
    from what the training data covered, and the two imply different next
    moves.</p>
  </div>
  <div class="op">
    <h4>Would this ever reach real survey data?</h4>
    <p>Nobody measures a matter power spectrum directly. A survey records where
    galaxies are, and everything between that and this model's input is unbuilt
    here. I do not know whether the right route is to model that gap or to rebuild
    the input around what a survey produces.</p>
  </div>
</div>

<div class="panel accent">
<p class="muted" style="margin:0"><span class="lead-in">Long-term aim,
untested.</span> A fast inference step that makes expensive analyses cheaper to
run, with uncertainties that could be defended. Nothing here shows that it can be
done, that it would preserve the accuracy of standard methods, or what it would
save.</p>
</div>
"""


def _roadmap() -> str:
    """
    The model-evolution ladder from the design documents, restated as intentions.

    Each rung names a limitation to remove. Nothing above rung 01 exists, and
    every rung above 00 depends on modules that did not train, so each is stated
    as a design intention rather than a property of this model.
    """
    cards = """<div class="step-row bad"><div class="sn">00</div><div class="sc"><div class="st-head"><h4>Make the architecture actually train</h4><span class="pill-bad">blocked, and first</span></div><p class="dim" style="margin:0 0 8px">the defect</p><p class="muted" style="margin:0">Not in the original plan; the audit put it here. In the released model the encoder, proposal and refinement networks never trained, and the energy objective is unbounded below. Every rung above rests on this one. The diagnosis of what the training code reaches is done; the repair is not.</p></div></div><div class="step-row ok"><div class="sn">01</div><div class="sc"><div class="st-head"><h4>A working inference path</h4><span class="pill-ok">done</span></div><p class="dim" style="margin:0 0 8px">shipped</p><p class="muted" style="margin:0">One observable, eight parameters, a single forward pass, and a benchmark that reproduces. It recovers matter density and clustering amplitude on emulator spectra and does not recover the rest.</p></div></div><div class="step-row next"><div class="sn">02</div><div class="sc"><div class="st-head"><h4>Training data with hydrodynamic physics</h4><span class="pill-next">next</span></div><p class="dim" style="margin:0 0 8px">the constraint: mostly gravity-only emulator data</p><p class="muted" style="margin:0">Most of the training set has no gas, feedback or AGN physics, and the model fails on the one hydrodynamic slice it is tested against. CAMELS is one candidate source. More realistic training data would not by itself make any score interpretable as new physics.</p></div></div><div class="step-row later"><div class="sn">03</div><div class="sc"><div class="st-head"><h4>Parameters chosen at runtime</h4><span class="pill-later">design intent</span></div><p class="dim" style="margin:0 0 8px">the constraint: a fixed output list</p><p class="muted" style="margin:0">The output is eight fixed parameters with compiled-in ranges. The intention is a read-out that takes a parameter specification at runtime. Whether the belief state would carry that information is untested, since the modules that build it did not train.</p></div></div><div class="step-row blocked"><div class="sn">3B</div><div class="sc"><div class="st-head"><h4>Distributions, not point estimates</h4><span class="pill-blocked">blocked</span></div><p class="dim" style="margin:0 0 8px">the constraint: no usable uncertainty</p><p class="muted" style="margin:0">A posterior needs a stated observation model, noise and prior, and calibration checks. A density model trained against a network whose &sigma; is a constant would learn nothing useful, so this waits on the rungs below.</p></div></div><div class="step-row later"><div class="sn">04</div><div class="sc"><div class="st-head"><h4>More than one instrument</h4><span class="pill-later">design intent</span></div><p class="dim" style="margin:0 0 8px">the constraint: one observable</p><p class="muted" style="margin:0">The intention is a separate encoder per observable, each updating a shared belief. Nothing here shows a belief state can be shared that way; the encoder that would feed it never trained.</p></div></div><div class="step-row later"><div class="sn">05</div><div class="sc"><div class="st-head"><h4>Anomaly as a research direction</h4><span class="pill-later">not built</span></div><p class="dim" style="margin:0 0 8px">the constraint: you must know what to look for</p><p class="muted" style="margin:0">The idea is that an energy left high after refinement could point at scales the model cannot explain. The energy here is constant and its objective unbounded below, so no such signal exists. It would need calibrated false-alarm rates on injected signals long before anyone trusted it on sky data.</p></div></div>"""
    return f"""<section id="roadmap"><div class="wrap">
{_shead("08", "Where it goes", "Each step removes one constraint.",
        "The ladder below is the design plan for the model, stated as intentions. "
        "At every step it names one limitation someone built in and proposes "
        "taking it out.")}
<div class="sbody">

<div class="panel accent">
<p class="muted" style="margin:0 0 12px"><span class="lead-in">Planning by
subtraction.</span> Most of what this model cannot do is a design choice rather
than physics: eight outputs fixed at build time, one instrument, no distribution
over answers. A roadmap that names the limitation each rung removes can be held to
account: a rung is either removed or it visibly is not.</p>
<p class="muted" style="margin:0"><span class="lead-in">What the audit
changed.</span> This ladder now starts at 00, because a plan whose first rungs sit
on an architecture that never trained is not a plan. Making it train comes before
extending it.</p>
</div>

<div class="steps">{cards}</div>

<h3 class="display" style="font-size:21px; margin:46px 0 10px">What I want advice on</h3>
<div class="panel"><ul style="margin:0">
<li>Is iterative belief refinement worth pursuing at all once its gradient path
works, or does a direct estimator reach the same place in one forward pass? There
is no ablation, and that is the first thing a reviewer should ask for.</li>
<li>How much of the weakness in expansion rate and dark energy is a limit of what
the power spectrum at two redshifts contains, and how much is training coverage?
An earlier version of this project answered "physics" with some confidence and
was wrong. I cannot separate them cleanly.</li>
<li>Would higher k, more redshifts, or explicit acoustic-scale features make the
expansion rate identifiable, and what would a defensible experiment to settle
that look like?</li>
<li>Step 05 is the one I am least able to judge. An anomaly signal pointing at
unexplained physics is either genuinely useful or an excellent way to fool
yourself, and I would rather know which before spending on it.</li>
</ul></div>

<div class="panel accent" style="margin-top:22px">
<p class="muted" style="margin:0">Nothing above rung 01 is done. What exists is an
inference path, a benchmark that reproduces, a linear baseline, and an account of
which parts do not work.</p>
</div>
</div></div></section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Downloads. Every example is downloadable in both formats so a visitor can take
# one, upload it straight back, and confirm the answer matches. Nothing here is
# special-cased for the built-ins: the upload path runs identical code.
# ─────────────────────────────────────────────────────────────────────────────

def _example_label(i: int) -> str:
    p = BENCH.params[i]
    return (f"#{i} · {_src_name(i)} · true Om={p[0]:.3f} s8={p[1]:.3f} h={p[2]:.3f}")


def _csv_for(i: int) -> str:
    p = BENCH.params[i]
    truth = "  ".join(f"{l}={v:.6g}" for l, v in zip(PARAM_LABELS, p))
    head = [
        "# CosmUFR input file",
        f"# Benchmark example #{i}, source: {_src_name(i)}",
        f"# True parameters: {truth}",
        "#",
        "# Two rows, 200 comma-separated values each.",
        "#   row 1: P(k) at z = 0.00",
        "#   row 2: P(k) at z = 0.47",
        "# k grid: 200 log-spaced bins from 0.1 to 4.5 h/Mpc (see kgrid.csv).",
        "# Units: P(k) in (Mpc/h)^3. Raw P(k) or log10 P(k) both accepted.",
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

def _picker(sel: int) -> str:
    """One submit button per example, each carrying its own example_id."""
    out = []
    for i in EXAMPLE_IDS:
        p = BENCH.params[i]
        on = " on" if i == sel else ""
        out.append(
            f'<button class="exam{on}" type="submit" name="example_id" value="{i}" '
            f'title="{html.escape(SOURCE_BLURB.get(_src_name(i), ""))}">'
            f'<span class="eid">#{i}</span>'
            f'<span class="esrc">{html.escape(_src_name(i))}</span>'
            f'<span class="etv">&Omega;m {p[0]:.3f} &nbsp; &sigma;8 {p[1]:.3f}<br>'
            f'h {p[2]:.3f} &nbsp; w&#8320; {p[5]:+.2f}</span></button>')
    return '<div class="picker">' + "".join(out) + "</div>"


def _form(selected=None) -> str:
    sel = selected if selected is not None else EXAMPLE_IDS[0]
    blurb = SOURCE_BLURB.get(_src_name(sel), "a held-out simulated universe")
    return f"""
<div class="panel">
<p class="muted" style="margin:0 0 6px"><span class="lead-in">Pick a universe and
run it.</span> Each card is a simulated cosmology held out of training. The label
is the code that produced it, and the numbers underneath are its <em>true</em>
parameters, so you know the right answer before the model gives you one.</p>
<p class="dim" style="margin:0 0 14px">Currently selected: #{sel}, from
{html.escape(blurb)}. Suites that vary all eight parameters, like
<code>bacco_full8</code>, make the most interesting runs.</p>
<form method="post" action="/infer#result">
{_picker(sel)}
</form>
<div class="row" style="margin-top:16px">
  <a class="dl" href="/download/example/{sel}.csv">&#8595; this spectrum .csv</a>
  <a class="dl" href="/download/example/{sel}.npy">&#8595; this spectrum .npy</a>
</div>
<p class="dim" style="margin:14px 0 0"><span class="lead-in">Check it.</span>
Download the spectrum above, upload it below as your own file, and compare. The
upload path runs identical code with no special-casing, so the two answers should
match to the last digit. If they do not, that is a bug worth reporting.</p>
</div>

<div class="panel">
<p class="muted" style="margin:0 0 14px"><span class="lead-in">Or bring your own.</span>
Two rows of 200 numbers: row 1 is P(k) at z=0, row 2 at z=0.47, both on the fixed
k grid from 0.1 to 4.5 h/Mpc, with P(k) in (Mpc/h)&sup3;. Raw P(k) or log10 P(k)
are both accepted and detected automatically. A <code>.npy</code> must be shape
(2,&nbsp;200); a <code>.csv</code> is comma-separated with <code>#</code> comment
lines ignored.</p>
<form method="post" action="/infer#result" enctype="multipart/form-data">
  <div class="row">
    <input type="file" name="upload" accept=".npy,.csv" aria-label="Your spectrum">
    <button class="btn btn-g" type="submit">Run on your file &rarr;</button>
  </div>
</form>
<div class="row" style="margin-top:16px">
  <a class="dl" href="/download/template.csv">&#8595; template.csv (runs as-is)</a>
  <a class="dl" href="/download/kgrid.csv">&#8595; the k grid</a>
</div>
<p class="dim" style="margin:14px 0 0">On a different k grid? Interpolate in
log-log first:
<code>np.exp(np.interp(np.log(k_grid), np.log(your_k), np.log(your_pk)))</code></p>
</div>"""


def _timing_panel(t) -> str:
    total = sum(t.values())
    rows = "".join(
        f'<tr><td>{html.escape(k)}</td><td class="num">{v*1000:.0f} ms</td>'
        f'<td class="num">{100*v/total:.0f}%</td></tr>' for k, v in t.items())
    return f"""<div class="panel tight">
<p class="muted" style="margin:0 0 12px"><span class="lead-in">Where the time went.</span>
The inference really is that fast. Most of what you waited for was matplotlib
drawing the pictures below, not the model.</p>
<div class="tw"><table>
<tr><th>stage</th><th class="num">time</th><th class="num">share</th></tr>{rows}
<tr><td><strong>total</strong></td><td class="num"><strong>{total*1000:.0f} ms</strong></td>
<td class="num">100%</td></tr></table></div></div>"""


# How each parameter behaves on the validation report, in words, using the same
# verdict rule as the results table. The narrator is given this so it can tell a
# reader which of eight numbers to weigh on a spectrum with no known answer.
_VERDICTS = RESULTS["validation_report"]["verdicts"]
_VARYING = RESULTS["validation_report"]["varying_only"]
TRACK_RECORD = {
    l: f"{_VERDICTS[l]}, R-squared {_VARYING[l]['r2']:.2f} on validation rows where it varies"
    for l in PARAM_LABELS
}
TYPICAL_ERROR = {l: _VARYING[l]["rmse"] for l in PARAM_LABELS}


def _narration(result, truth, pk0, source: str, report) -> str:
    """
    An explanation of this run written for this run, or nothing.

    Returns an empty string whenever the narrator is unavailable or its draft
    failed the grounding check, and the caller then shows the computed reading
    alone. The page must read correctly in both cases, so nothing here is load
    bearing.
    """
    a = np.log10(np.clip(np.asarray(pk0, dtype=float), 1e-30, None))
    lo = np.percentile(BENCH_LOGPK, 1, axis=0)
    hi = np.percentile(BENCH_LOGPK, 99, axis=0)
    outside = float(np.mean((a < lo) | (a > hi))) * 100

    # The same thresholds the computed summary uses, so the two paragraphs
    # cannot disagree about how many parameters missed.
    tally = None
    if truth is not None:
        frac = {l: abs(float(result.params_array[i]) - float(truth[i]))
                   / PRIOR_RANGE[l] for i, l in enumerate(PARAM_LABELS)}
        close = sum(1 for v in frac.values() if v < 0.05)
        missed = sum(1 for v in frac.values() if v > 0.10)
        tally = (f"{close} of the eight landed close, {missed} clearly missed, "
                 f"and the rest sit in between")

    facts = run_facts(
        result.params_array, result.sigmas_array, PARAM_LABELS, truth=truth,
        source=source, outside_pct=outside, tally=tally,
        belief_movement=report.belief_movement,
        energy_ulps=report.energy_drop_in_ulps,
        track_record=TRACK_RECORD, typical_error=TYPICAL_ERROR)

    text = narrate(facts)
    if not text:
        return ""
    paras = "".join(f"<p>{html.escape(p.strip())}</p>"
                    for p in text.splitlines() if p.strip())
    return (f'<div class="narration"><span class="nlbl">Reading this result</span>'
            f'{paras}<p class="nfoot">Written for this run by a language model '
            f'that was given the measurements above and nothing else, and is not '
            f'permitted to state a number they do not contain. It has no access '
            f'to the model being described. The computed summary below is '
            f'produced by the code itself and does not depend on it.</p></div>')


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
        head += "<th class='num'>true</th><th class='num'>off by</th>"
    head += "</tr>"

    rows = ""
    for i, lbl in enumerate(PARAM_LABELS):
        v, sg = float(result.params_array[i]), float(result.sigmas_array[i])
        name, _ = PARAM_MEANING[lbl]
        sg_txt = (f"{sg:.4f} <span class='flag'>ignore</span>"
                  if abs(sg - 0.1) < 1e-6 else f"{sg:.4f}")
        rows += (f"<tr><td>{PARAM_TEX[lbl]}</td>"
                 f"<td class='dim' style='white-space:nowrap'>{html.escape(name)}</td>"
                 f"<td class='num'>{v:.5f}</td><td class='num'>{sg_txt}</td>")
        if truth is not None:
            rows += (f"<td class='num'>{float(truth[i]):.5f}</td>"
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
        "timing_ms": {k: round(v * 1000, 1) for k, v in t.items()},
        "model_version": "run4-2026-04-14",
        "checkpoint_sha256": SHA256,
    }
    dl = ""
    if selected is not None:
        dl = (f'<div class="row" style="margin-top:14px">'
              f'<a class="dl" href="/download/example/{selected}.csv">&#8595; this exact input as .csv</a>'
              f'<a class="dl" href="/download/example/{selected}.npy">&#8595; as .npy</a></div>')
    truth_note = ("" if truth is None else
                  "<p class='dim' style='margin:12px 0 0'>The true values come "
                  "from the simulation that produced this spectrum. It is from "
                  "the held-out split, which is made by row, so related spectra "
                  "of the same cosmology in training are not ruled out.</p>")

    s = time.perf_counter()
    narration = _narration(result, truth, pk0, source_label, report)
    if narration:
        t["explain the result: one call to a small language model"] =             time.perf_counter() - s

    return f"""<div id="result"></div>
<h3 class="display" style="font-size:24px; margin:38px 0 6px">Result</h3>
<p class="muted" style="margin:0 0 4px">Input: {html.escape(source_label)}</p>
{narration}
{_reading(read_run(result.params_array, result.sigmas_array, truth,
                  reference=BENCH_LOGPK, pk_z0=pk0, settling=report))}
<div class="panel"><div class="tw"><table>{head}{rows}</table></div>
{_reading(read_parameters(result.params_array, result.sigmas_array, truth))}
{truth_note}{dl}</div>
<p class="dim" style="margin:14px 0 0">Diagnostics for this run: the belief moved
{report.belief_movement*100:.3f}% of its norm across the sixteen refinement
steps, and the score it was meant to be descending changed by
{report.energy_drop:.2e}, which is {report.energy_drop_in_ulps:.1f} float32
resolution steps &mdash; the smallest amounts the arithmetic can represent.</p>
{_timing_panel(t)}
{_figblock("settling", fig_settle, read_settling(report))}
{_figblock("pk", fig_pk, read_pk(K_GRID, pk0, pk047, result.pk_recon,
                                 result.log_k, reference=BENCH_LOGPK,
                                 has_truth=truth is not None))}
<details><summary>Full result as JSON</summary>
<pre><code>{html.escape(json.dumps(payload, indent=2))}</code></pre></details>"""


def _demo(inner: str = "", selected=None) -> str:
    return f"""<section id="demo"><div class="wrap">
{_shead("04", "Try it yourself", "Run the model on a spectrum.",
        "The released checkpoint, loaded in this container, running on whatever "
        "you give it. Nothing is cached and nothing is precomputed.")}
<div class="sbody">{_form(selected)}{inner}</div>
</div></section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy, baseline, audit, limitations
# ─────────────────────────────────────────────────────────────────────────────




def _error_table() -> str:
    """Point-prediction error per parameter, on rows where the parameter varies."""
    val = RESULTS["validation_report"]["varying_only"]
    ben = RESULTS["benchmark"]["varying_only"]
    rows = ""
    for lbl in PARAM_LABELS:
        name, _ = PARAM_MEANING[lbl]
        unit = " eV" if lbl == "mv" else ""
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="dim" style="white-space:nowrap">{html.escape(name)}</td>'
                 f'<td class="num">{val[lbl]["rmse"]:.4f}{unit}</td>'
                 f'<td class="num dim">{val[lbl]["n"]:,}</td>'
                 f'<td class="num">{ben[lbl]["rmse"]:.4f}{unit}</td>'
                 f'<td class="num dim">{ben[lbl]["n"]:,}</td></tr>')
    return (f'<div class="panel"><div class="tw"><table>'
            f'<tr><th>parameter</th><th></th>'
            f'<th class="num">RMSE, validation report</th><th class="num">rows</th>'
            f'<th class="num">RMSE, benchmark (reproduced)</th><th class="num">rows</th></tr>'
            f'{rows}</table></div></div>')


def _results() -> str:
    if not RESULTS:
        return ""
    val = RESULTS["validation_report"]
    full, vary = val["overall"], val["varying_only"]
    bench = RESULTS["benchmark"]["overall"]
    agree = RESULTS["benchmark"]["agreement"]

    def f(x):
        return "&ndash;" if x is None else f"{x:.3f}"

    label = {"recovered": "<span class='good-t'>recovered</span>",
             "partial": "<span class='flag'>partial</span>",
             "not recovered": "<span class='bad-t'>not recovered</span>",
             "undefined": "<span class='dim'>undefined</span>"}
    rows = ""
    for lbl in PARAM_LABELS:
        name, _ = PARAM_MEANING[lbl]
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="dim" style="white-space:nowrap">{html.escape(name)}</td>'
                 f'<td class="num">{f(full[lbl]["r2"])}</td>'
                 f'<td class="num">{f(vary[lbl]["r2"])}</td>'
                 f'<td class="num">{f(bench[lbl]["r2"])}</td>'
                 f'<td>{label[val["verdicts"][lbl]]}</td></tr>')

    src = ""
    for name, blk in sorted(val["per_source"].items(), key=lambda kv: -kv[1]["n"]):
        cells = "".join(f'<td class="num">{f(blk["metrics"][l]["r2"])}</td>'
                        for l in PARAM_LABELS)
        flag = (' <span class="flag">channels identical</span>'
                if name == "bacco_multiz" else "")
        src += (f'<tr><td><code>{html.escape(name)}</code>{flag}</td>'
                f'<td class="num">{blk["n"]:,}</td>{cells}</tr>')
    hdr = "".join(f'<th class="num">{PARAM_TEX[l]}</th>' for l in PARAM_LABELS)

    multiz = RESULTS["benchmark"]["redshift_channel_checks"]["per_source"]["bacco_multiz"]
    om = {n: b["metrics"]["Om"]["r2"] for n, b in val["per_source"].items()
          if b["metrics"]["Om"]["r2"] is not None}
    high = sorted(n for n, r in om.items() if r >= 0.97)
    low = sorted((n for n, r in om.items() if r < 0.97), key=lambda n: om[n])
    low_txt = ", ".join(f"<code>{n}</code> {om[n]:.2f}" for n in low)

    return f"""<section id="results"><div class="wrap">
{_shead("04", "Accuracy", "How well it actually works.",
        "Point-prediction accuracy, from a saved validation report and from the "
        "bundled benchmark reproduced in the audit.")}
<div class="sbody">

<p class="muted"><span class="lead-in">Two sources for every number.</span> The
validation report covers {val["rows"]:,} rows of a deterministic held-out split
across {val["source_ids_in_split"]} source ids. It is a saved report, and
regenerating it needs the private master validation rows. The bundled benchmark is
a 6,000-row subset that ships with the code; I reproduced it in the audit, and it
agrees with an independent reproduction to {agree["max_abs_r2_diff_vs_independent_review"]:.1e}
in R&sup2;.</p>

{_error_table()}

<p class="muted">These are point-prediction errors on noiseless emulator spectra.
They are not comparable to survey constraints, which are posterior widths on real
data with a noise model, and no such comparison is made here.</p>

<h3 class="display" style="font-size:21px; margin:44px 0 10px">The same result as R&sup2;</h3>
<p class="muted">R&sup2; is a ratio against the spread of the truth in whatever
rows are scored, so widening a parameter's range raises it without changing the
model. The verdict uses R&sup2; on validation rows where the parameter varies:
above 0.7 recovered, 0.25 to 0.7 partial, below 0.25 not recovered.</p>
<div class="panel"><div class="tw"><table>
<tr><th>parameter</th><th></th><th class="num">all validation rows</th>
<th class="num">where it varies</th><th class="num">benchmark, all rows</th>
<th>verdict</th></tr>{rows}</table></div></div>

<p class="muted"><span class="lead-in">The column to use is "where it
varies".</span> On rows where a parameter is held fixed there is no spread and
R&sup2; means nothing. Neutrino mass is the clearest case:
{full["mv"]["r2"]:.2f} across all validation rows, {vary["mv"]["r2"]:.3f} on rows
where it varies.</p>

<h3 class="display" style="font-size:21px; margin:40px 0 10px">One broken suite drags every average down</h3>
<div class="panel tight"><div class="tw"><table>
<tr><th>source</th><th class="num">n</th>{hdr}</tr>{src}</table></div>
<p class="dim" style="margin:12px 0 0"><code>&ndash;</code> means the parameter is
held fixed in that source, so R&sup2; is undefined rather than bad. Saved
report.</p></div>
<p class="muted">In <code>bacco_multiz</code> the two redshift channels are
byte-identical in all {multiz["rows_z0_identical_to_z047"]:,} benchmark rows, so it
carries no growth information and scores near zero on everything. Matter density
comes back at 0.98 to 0.99 on {len(high)} of the {len(om)} sources, and lower on
the rest: {low_txt}.</p>

{_hydro_finding()}
{_neutrino_finding()}

</div></div></section>"""


def _hydro_finding() -> str:
    """The one hydrodynamic slice, and the channel-order defect that confounds it."""
    ps = RESULTS["validation_report"]["per_source"]
    a = ps.get("camels_astrid_x")
    if not a:
        return ""
    m = a["metrics"]
    gaps = RESULTS["benchmark"]["redshift_channel_checks"]["per_source"]
    rows = ""
    for lbl in PARAM_LABELS:
        r2, rm, sd = m[lbl]["r2"], m[lbl]["rmse"], m[lbl]["truth_std"]
        if r2 is None:
            continue
        cls = "bad-t" if r2 < 0 else ""
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="num {cls}">{r2:.3f}</td>'
                 f'<td class="num">{rm:.4f}</td>'
                 f'<td class="num dim">{sd:.4f}</td></tr>')
    varied = [l for l in PARAM_LABELS if m[l]["r2"] is not None]
    below = [l for l in varied if m[l]["r2"] < 0]
    worst = min(varied, key=lambda l: m[l]["r2"])
    big = [g["median_lowk_log10_z0_over_z047"] for g in gaps.values()
           if g["n"] > 100 and g["rows_z0_identical_to_z047"] == 0]
    astrid = gaps["camels_astrid_x"]["median_lowk_log10_z0_over_z047"]
    camb = gaps["camb_nl"]["median_lowk_log10_z0_over_z047"]
    return f"""
<h3 class="display" style="font-size:21px; margin:44px 0 10px">Where it breaks: the one hydrodynamic slice</h3>
<p class="muted">Most of the training data is emulator output. One slice of the
evaluation, <code>camels_astrid_x</code>, is different: {a["n"]} spectra from a
hydrodynamic simulation, where gas cools, stars form and feedback pushes matter
back out. <strong>The model scores worse than predicting the mean on
{len(below)} of the {len(varied)} parameters this source varies</strong>,
including the {PARAM_MEANING[worst][0]} at R&sup2; {m[worst]["r2"]:.2f}. Saved
report.</p>
<div class="panel tight"><div class="tw"><table>
<tr><th>parameter</th><th class="num">R&sup2;</th><th class="num">RMSE</th>
<th class="num">spread in the truth</th></tr>{rows}</table></div>
<p class="dim" style="margin:12px 0 0">A negative R&sup2; means predicting the mean
of the truth would have done better.</p></div>

<div class="panel warn">
<p class="muted" style="margin:0 0 12px"><span class="lead-in">A confound.</span>
This source also stores its two redshift channels in the opposite order. On the
benchmark rows, the median over k &lt; 0.2 h/Mpc of
log&#8321;&#8320; P(z=0) &minus; log&#8321;&#8320; P(z=0.47) is {astrid:+.2f}
here and {camb:+.2f} for <code>camb_nl</code>, the other reversed source, against
{min(big):+.2f} to {max(big):+.2f} for the other sources with more than 100
benchmark rows and two distinct channels.</p>
<p class="muted" style="margin:0">So the failure has two candidate causes, gas
physics the model was not trained on and a channel order it was not trained on,
and they are not separated. Separating them needs the source corrected and
evaluated again.</p>
</div>

<p class="muted">What holds either way: every other number on this page describes
the model on emulator spectra that resemble its training data.</p>"""


def _neutrino_finding() -> str:
    """Neutrino-mass recovery on the two sources that vary it, without a causal story."""
    ps = RESULTS["validation_report"]["per_source"]
    a, b = ps.get("bacco_neutrino"), ps.get("bcemu_neutrino")
    if not (a and b):
        return ""
    am, bm = a["metrics"]["mv"], b["metrics"]["mv"]
    return f"""
<h3 class="display" style="font-size:21px; margin:44px 0 10px">Neutrino mass depends on which emulator produced the spectrum</h3>
<div class="panel tight"><div class="tw"><table>
<tr><th>source</th><th></th><th class="num">n</th>
<th class="num">R&sup2; on neutrino mass</th><th class="num">RMSE</th></tr>
<tr><td><code>bacco_neutrino</code></td><td class="dim">BACCO emulator</td>
<td class="num">{a["n"]:,}</td><td class="num">{am["r2"]:.3f}</td>
<td class="num">{am["rmse"]:.3f} eV</td></tr>
<tr><td><code>bcemu_neutrino</code></td><td class="dim">BCemu emulator, with a baryonic-feedback model</td>
<td class="num">{b["n"]:,}</td><td class="num bad-t">{bm["r2"]:.3f}</td>
<td class="num">{bm["rmse"]:.3f} eV</td></tr>
</table></div></div>
<p class="muted">Saved report. The same parameter comes back with R&sup2;
{am["r2"]:.2f} from one emulator and {bm["r2"]:.2f} from the other. They are
different emulators, so they differ in more than baryonic treatment. Baryonic
feedback and massive neutrinos both suppress small-scale power, which is one
possible reason; without spectra of matched cosmologies from both, this does not
separate them.</p>"""


def _baseline() -> str:
    g = RESULTS.get("ridge_baseline")
    if not g:
        return ""
    rows = ""
    for lbl in PARAM_LABELS:
        p = g["params"].get(lbl, {})
        if p.get("pinned") or p.get("ridge_r2") is None:
            continue
        r, c = p["ridge_r2"], p["cosmufr_r2"]
        rb = f"<strong>{r:.3f}</strong>" if p["winner"] == "ridge" else f"{r:.3f}"
        cb = f"<strong>{c:.3f}</strong>" if p["winner"] == "cosmufr" else f"{c:.3f}"
        name, _ = PARAM_MEANING[lbl]
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="dim" style="white-space:nowrap">{html.escape(name)}</td>'
                 f'<td class="num">{rb}</td><td class="num">{cb}</td>'
                 f'<td class="dim">{p["winner"]}</td></tr>')
    return f"""<section id="baseline"><div class="wrap">
{_shead("05", "Baseline", "Is the big model earning its keep?",
        "The first question anyone should ask about a 136-million-parameter "
        "network is whether a simple method does just as well.")}
<div class="sbody">
<p class="muted" style="max-width:66ch">Ridge regression on the same 400 input
numbers, fitted on {g["n_fit"]:,} benchmark rows, with both models scored on the
other {g["n_test"]:,}. Saved report, independently re-derived with the same
seed.</p>
<div class="panel"><div class="tw"><table>
<tr><th>parameter</th><th></th><th class="num">ridge, 400 features</th>
<th class="num">CosmUFR, 136M</th><th>higher</th></tr>{rows}</table></div></div>
<p class="muted">CosmUFR is higher on {g["cosmufr_wins"]} of {g["n_compared"]}. A
plain linear fit is competitive on matter density, which is written into the
height of the curve and does not need a large network to read.</p>
<p class="dim">The comparison is not symmetric in either direction. The network was
trained on far more data: historical documentation gives 84.5 million rows. Ridge
is fitted on rows from the same sources it is scored on, and the benchmark split
is by row, so related spectra in the network's training data are not ruled out
either. The comparison that would say whether the architecture earns its size, a
direct network of matched size trained on the same inputs, has not been run.
Rerun this one with <code>python scripts/ridge_baseline.py</code>.</p>
</div></div></section>"""


# What each module is for, and which of the audit's findings applies to it. The
# grouping follows measurements in reports/results_v1.json: identity against Run 2's
# checkpoint saved before any optimizer step, and whether the training step reaches
# the module. Shares are parameter counts, not a measure of usefulness.
MODULE_ROLE = {
    "obs_encoder":         "Reads the spectrum",
    "belief_proposal":     "Forms the first guess",
    "settling":            "Refines the guess, 16 times",
    "param_head":          "Turns the guess into eight numbers",
    "unc_head":            "Says how sure the model is",
    "gen_head":            "Redraws the spectrum from the guess",
    "obs_energy_head":     "Scores how well a guess fits the data",
    "dyn_energy_head":     "Scores how far a guess has moved",
    "constraint_head":     "Scores a learned constraint on the guess",
    "halo_head":           "A side output, not used for the parameters",
    "obs_encoder_single":  "Sequential path, used only in training",
    "belief_proposal_seq": "Sequential path, used only in training",
    "attractor_bank":      "Stored reference beliefs, moving-average update",
}

GROUP_OF = {
    "obs_encoder": "unchanged", "belief_proposal": "unchanged", "settling": "unchanged",
    "obs_energy_head": "degenerate", "dyn_energy_head": "degenerate",
    "constraint_head": "degenerate", "gen_head": "degenerate", "unc_head": "degenerate",
    "param_head": "works",
}

GROUP_COPY = {
    "unchanged": (
        "Still at initialization",
        "bad",
        "Bit-identical to Run 2's checkpoint saved before any optimizer step. In "
        "the code that trained Run 4 the refinement loop detaches the belief at "
        "every step and computes its step size and preconditioner without "
        "gradients, so the training loss never reaches these modules."),
    "degenerate": (
        "Changed, and their outputs carry no information",
        "warn",
        "These received gradients and changed. What they produce does not vary "
        "usefully with the input: the energy is the same to float32 resolution "
        "across spectra, the redrawn spectrum is one constant at every scale, and "
        "the reported &sigma; sits at its clamp floor."),
    "works": (
        "Changed, and produces the answers",
        "good",
        "The eight reported parameters are computed here, from a belief that is a "
        "fixed random function of the input because everything upstream of it is "
        "unchanged. It still recovers matter density and clustering amplitude."),
}


def _module_groups():
    """Group modules by the audit's measurements; the rest is listed under the table."""
    out = {"unchanged": [], "degenerate": [], "works": [], "other": []}
    for name, e in RESULTS["modules"]["by_module"].items():
        out[GROUP_OF.get(name, "other")].append(
            (name, MODULE_ROLE.get(name, ""), e["share_percent"]))
    return out


def _share(group: str) -> float:
    """Share of all parameters held by one group."""
    return sum(share for _, _, share in _module_groups()[group])


def _outcome_table() -> str:
    """The three outcomes, with what each part is for and how big it is."""
    groups = _module_groups()
    body = ""
    for key in ("unchanged", "degenerate", "works"):
        rows = groups[key]
        if not rows:
            continue
        title, tone, gloss = GROUP_COPY[key]
        share = sum(sh for _, _, sh in rows)
        cells = "".join(
            f'<tr><td class="dim">{html.escape(job)}</td>'
            f'<td><code>{html.escape(n)}</code></td>'
            f'<td class="num dim">{sh:.1f}%</td></tr>'
            for n, job, sh in sorted(rows, key=lambda r: -r[2]))
        body += (f'<div class="outcome {tone}">'
                 f'<div class="oc-head"><h4>{title}</h4>'
                 f'<span class="oc-share">{share:.1f}% of parameters</span></div>'
                 f'<p class="muted">{gloss}</p>'
                 f'<div class="tw"><table class="oc-tbl">{cells}</table></div>'
                 f'</div>')
    other = groups["other"]
    names = ", ".join(f"<code>{html.escape(n)}</code>"
                      for n, _, _ in sorted(other, key=lambda r: -r[2]))
    rest = sum(sh for _, _, sh in other)
    halo = RESULTS["modules"]["halo_head_weight_rescale_run2_to_run4"]
    foot = (f'<p class="dim" style="margin:6px 0 0">The remaining {rest:.0f}% '
            f'({names}) does not produce the parameters. The sequential path trains '
            f'only through a training-time consistency loss; the reference bank is '
            f'updated by moving average; <code>halo_head</code> receives no gradient, '
            f'but its weights were uniformly rescaled by {min(halo):.3f} between Run 2 '
            f'and Run 4, for a reason not traced.</p>')
    return f'<div class="outcomes">{body}</div>{foot}'


def _audit() -> str:
    img = _png(F.fig_weight_audit(AUDIT))
    by = RESULTS["modules"]["by_module"]
    later = RESULTS["later_training_code_saved_log"]["median_grad_norm"]
    shift = RESULTS["energy"]["loss_change_for_shift_minus_1000"]["run4_launch_384ad38__run4_weights"]
    traj = RESULTS["energy"]["run4_training_log"]["trajectory"]
    e_first, e_last = traj[0], traj[-1]
    bench_e = list(RESULTS["energy"]["benchmark_energy_values"]["rows"].values())

    def ident(n):
        x = by[n]["run2_initial_vs_run4"]
        return f'{x["identical"]}/{x["shared"]}'

    def reach(n, key):
        x = by[n]["training_step"][key]
        return f'{x["grad_batches"]}/{x["of"]}'

    return f"""<section id="what-i-observed"><div class="wrap">
{_shead("02", "What I observed", "The core of the design is still at its initial values.",
        "I compared the released weights with earlier checkpoints, including one "
        "saved before any optimizer step, and ran the training code itself to see "
        "which modules its loss can reach. What follows is the evidence: the "
        "weights, the runs, accuracy, where it breaks, a linear baseline, and the "
        "argument I got wrong.")}
<div class="sbody">

<div class="panel bad">
<p class="muted" style="margin:0 0 12px"><span class="lead-in">The finding in one
paragraph.</span> The idea this model is built on is that it reaches its answer
gradually: read the spectrum, form a rough guess about the universe, then sharpen
that guess sixteen times. In the released checkpoint the modules meant to do that
are bit-identical to their state before training, and the code that trained it
never passes them a gradient. The eight parameters are computed by a small
read-out head from a fixed random function of the input.</p>
<p class="muted" style="margin:0">Most of the other parameters did change during
training, which is why this was not visible in the loss curves.</p>
</div>

<h3 class="display" style="font-size:21px; margin:44px 0 14px">What happened to each part</h3>
{_outcome_table()}

<h3 class="display" style="font-size:21px; margin:48px 0 10px">How I know</h3>
<p class="muted"><span class="lead-in">Against the starting point.</span> Run 2
saved a checkpoint before its first optimizer step. Against it, the released
checkpoint's encoder matches in {ident("obs_encoder")} tensors, the belief proposal
in {ident("belief_proposal")} and the refinement networks in {ident("settling")},
bit for bit, while the read-out heads changed. Run 4 was warm-started through
Run 3 from Run 2, so this covers Runs 2 to 4.</p>
<p class="muted"><span class="lead-in">In the training code.</span> I ran the code
that trained Run 4 for twenty diagnostic steps from the released weights, and
twenty from a fresh initialization. The encoder received a gradient in
{reach("obs_encoder", "run4_code_run4_weights")} steps, the proposal in
{reach("belief_proposal", "run4_code_run4_weights")} and the refinement networks in
{reach("settling", "run4_code_run4_weights")}; the read-out head in
{reach("param_head", "run4_code_run4_weights")}. The loop detaches the belief at
every step and computes its step size and preconditioner without gradients. Later
training code reconnects the encoder ({reach("obs_encoder", "current_code_kbp4")}),
and the proposal only when every refinement step is retained
({reach("belief_proposal", "current_code_kbp16")}); the refinement networks stay at
{reach("settling", "current_code_kbp16")}. In a saved log from a later smoke run
with that code, the encoder's median gradient was {later["encoder"]:.1e} against
{later["param_head"]:.1f} for the read-out head.</p>
<p class="muted"><span class="lead-in">A clue, not a proof.</span> Every Linear
bias is initialised to zero, and the figure below counts modules whose biases are
all still zero. That is consistent with no update, but <code>halo_head</code> also
has all-zero biases and its weights changed. The comparison and the training-step
diagnosis above are the evidence.</p>
{_figblock("weight_audit", img)}

<p class="muted"><span class="lead-in">A second, separate problem.</span> The
energy the refinement is meant to descend is trained with a loss that adds the
mean energy to a contrastive term depending only on differences. Lowering every
energy by the same constant therefore lowers the loss one-for-one: a diagnostic
shift of 1,000 changed it by {shift:,.1f}. Run 4's logged energy loss went from
{e_first["train_L_energy"]:,.0f} at epoch {e_first["epoch"]} to
{e_last["train_L_energy"]:,.0f} by epoch {e_last["epoch"]}, and on six different
benchmark spectra the energy is {bench_e[0]:,.3f} to within one float32 step. That
is consistent with the unbounded direction; the log does not show which term
produced the value. Reconnecting the gradient path alone would leave this in
place.</p>

<div class="panel accent">
<p class="muted" style="margin:0 0 12px"><span class="lead-in">What I do not
know, and would like to.</span> What the energy should be trained to do instead:
bounded below, tied to something observable, and never shown the answer at
inference. I do not have a formulation I believe in.</p>
<p class="muted" style="margin:0 0 12px">Underneath that sits the question the
project was built to ask and has not asked: <strong style="color:var(--fg)">does
refining an answer over several steps buy anything a single pass does not?</strong>
Nothing in this release is evidence either way.</p>
<p class="muted" style="margin:0">Both are open, and the second is the one worth
someone else's opinion before another training run.</p>
</div>

<p class="dim">The bias audit and the settling measurement run from the released
weights with <code>cosmufr.weight_audit(model)</code> and
<code>cosmufr.settling_report(...)</code>. The checkpoint comparison and the
training-step diagnosis need the historical checkpoints and training source,
which are not public.</p>
</div></div></section>"""


# The ceiling as it was asserted: a weight, a claimed per-parameter maximum, and
# the value the model was reported to reach at the time. Kept as data so the
# table and the totals below it are computed from one source and cannot drift.
CEILING_CLAIM = [
    ("h",  "expansion rate",  0.25, 0.55, 0.604),
    ("wa", "dark energy drift", 0.20, 0.18, 0.187),
    ("w0", "dark energy",     0.15, 0.50, 0.742),
    ("Om", "matter density",  0.12, 0.88, 0.907),
    ("s8", "lumpiness",       0.08, 0.85, 0.911),
    ("ns", "spectral tilt",   0.08, 0.34, 0.353),
    ("Ob", "ordinary matter", 0.08, 0.36, 0.406),
    ("mv", "neutrino mass",   0.04, 0.40, 0.410),
]


def _ceiling() -> str:
    """
    The retraction: the arithmetic that undoes the claimed bound, and the reason it
    was never a bound in the first place.
    """
    claimed = sum(w * cap for _, _, w, cap, _ in CEILING_CLAIM)
    scored = sum(w * got for _, _, w, _, got in CEILING_CLAIM)
    over = [l for l, _, _, cap, got in CEILING_CLAIM if got > cap]
    weak_weight = sum(w for l, _, w, _, _ in CEILING_CLAIM if l in ("h", "wa", "w0"))

    rows = ""
    for lbl, name, w, cap, got in CEILING_CLAIM:
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="dim" style="white-space:nowrap">{name}</td>'
                 f'<td class="num dim">{w:.2f}</td>'
                 f'<td class="num">{cap:.2f}</td>'
                 f'<td class="num bad-t">{got:.3f}</td></tr>')

    return f"""<section id="ceiling"><div class="wrap">
{_shead("06", "What I got wrong", "I claimed a physics ceiling. It was never derived.",
        "The audit above is a defect in the code. This one is a mistake in "
        "reasoning, asserted in writing and defended for nine days.")}
<div class="sbody">

<div class="panel accent">
<p class="muted" style="margin:0"><span class="lead-in">The claim.</span> Six of
the eight parameters would not improve across several architectural variants. I
then wrote that a Fisher analysis put the maximum recoverable score at
{claimed:.2f}, that my model had plateaued there, and that the limit therefore lay
in the observable rather than in anything I had built. I called it the result I
was most confident in.</p>
</div>

<h3 class="display" style="font-size:21px; margin:40px 0 10px">The arithmetic does not hold</h3>
<div class="panel"><div class="tw"><table>
<tr><th>parameter</th><th></th><th class="num">my weight</th>
<th class="num">claimed maximum</th><th class="num">what I reported scoring</th></tr>
{rows}</table></div>
<p class="dim" style="margin:12px 0 0">Weighted, the claimed maximum is
{claimed:.3f} and the scores reported at the time come to {scored:.3f}.</p></div>

<p class="muted">A model cannot exceed a bound it is sitting under. The scores I
reported did, by {(scored - claimed) / claimed * 100:.0f} percent, and not through
one outlier: <strong>all {len(over)} of the 8 parameters individually score above
their own claimed maximum.</strong></p>

<h3 class="display" style="font-size:21px; margin:40px 0 10px">And it was never derived for a stated observation model</h3>
<p class="muted">A bound on how well parameters can be recovered needs an
observation model: what is measured, with what noise and covariance, over what
distribution of parameters. The earlier bound specified none of these. What I had
actually computed was a weighted average of my own scores, with weights I chose,
and the three parameters I could not recover carried {weak_weight * 100:.0f}
percent of that weight. The bound is withdrawn. Whether these parameters are
identifiable from two noiseless spectra over this range of scales, and to what
accuracy, remains to be investigated.</p>

<div class="panel accent">
<p class="muted" style="margin:0 0 12px"><span class="lead-in">What it cost, and
why it is here.</span> Nine days of runs spent defending a number that had never
been derived, when a calculation that could have tested local identifiability
&mdash; finite-difference an emulator over the training range and look at the
singular values of the resulting Jacobian &mdash; needs no GPU at all. What
actually limited those runs is not established: the saved evaluations were too
noisy to tell them apart.</p>
<p class="muted" style="margin:0">I found this because the audit forced me back
through the reasoning, not because I checked whether my own calculation was the
thing I had called it. The defect was found by a test; the bad reasoning was found
only by accident.</p>
</div>
</div></div></section>"""


def _the_code() -> str:
    """Install it, run it, check it, get in touch."""
    return f"""<section id="the-code"><div class="wrap">
{_shead("05", "The code", "Everything needed to check this.",
        "The code, the checkpoint and the benchmark are public, and the benchmark "
        "numbers regenerate from them. The validation report and the audit "
        "diagnostics need data and checkpoints that are not public, and say so.")}
<div class="sbody">

<div class="two-up">
<div class="panel">
<h4 class="ch">Install and run</h4>
<pre><code>git clone {REPO_URL}
cd cosmufr-run4
pip install -e ".[demo]"</code></pre>
<p class="dim" style="margin:12px 0 0">Weights download automatically on first
use, about 545 MB, checked against a recorded hash.</p>
</div>
<div class="panel">
<h4 class="ch">Infer on one spectrum</h4>
<pre><code>import cosmufr

model  = cosmufr.load_model()
bench  = cosmufr.load_benchmark()
result = cosmufr.infer(bench.pk_z0[0],
                       bench.pk_z047[0],
                       model=model)
print(result.params)</code></pre>
<p class="dim" style="margin:12px 0 0">Raw P(k) or log&#8321;&#8320; P(k) are both
accepted, on 200 bins over k in [0.1, 4.5] h/Mpc at z = 0 and z = 0.47.</p>
</div>
</div>

<div class="two-up" style="margin-top:14px">
<div class="panel">
<h4 class="ch">Regenerate the benchmark table</h4>
<pre><code>python -m cosmufr.reproduce</code></pre>
<p class="dim" style="margin:12px 0 0">Rebuilds the benchmark table and the defect
diagnostics from the bundled 6,000 rows. In the audit it agreed with an
independent reproduction within 1e-5 in R&sup2;.</p>
</div>
<div class="panel">
<h4 class="ch">Check the audit for yourself</h4>
<pre><code>import cosmufr
m = cosmufr.load_model()
print(cosmufr.weight_audit(m).table())
print(cosmufr.settling_report(m, pk0, pk047))</code></pre>
<p class="dim" style="margin:12px 0 0">About seven seconds. The first prints
which modules still have all-zero biases, which is a clue rather than proof; the
second measures how far the refinement moves the belief.</p>
</div>
</div>

<h3 class="display" style="font-size:21px; margin:46px 0 12px">Where everything lives</h3>
<div class="panel tight"><div class="tw"><table>
<tr><th>what</th><th>where</th></tr>
<tr><td>Code, benchmark, tests, saved reports and the results file</td>
    <td><a href="{REPO_URL}">{REPO_URL.replace("https://", "")}</a></td></tr>
<tr><td>Weights and model card</td>
    <td><a href="{HF_URL}">{HF_URL.replace("https://", "")}</a></td></tr>
<tr><td>This page, running the real model</td>
    <td><a href="/">the demo above</a></td></tr>
</table></div>
<p class="dim hashline" style="margin:12px 0 0">Checkpoint sha256 {SHA256}</p></div>

<div class="panel accent" style="margin-top:24px">
<p class="muted" style="margin:0 0 10px"><span class="lead-in">Get in
touch.</span> CosmUFR is built and released by Aaditya Rajgor, open under MIT. The two questions I would most value an outside view on:
whether gradual refinement is worth pursuing at all once it can train, or whether
a single-pass estimator reaches the same place; and how much of the weakness on
the expansion rate is a limit of what this measurement contains rather than a
limit of the training data.</p>
<p class="muted" style="margin:0"><a href="mailto:{CONTACT_EMAIL}">{CONTACT_EMAIL}</a></p>
</div>
</div></div></section>"""


def _limits() -> str:
    items = [
        ("The refinement core is at initialization.",
         "The encoder, the belief proposal and the refinement networks are "
         "bit-identical to Run 2's pre-training state, and the code that trained "
         "this checkpoint gives them no gradient. See what I observed, above."),
        ("The energy objective is unbounded below.",
         "It lowers one-for-one under a constant downward shift of every energy, and "
         "on the benchmark the energy is constant to float32 resolution across "
         "inputs and refinement steps."),
        ("Reported uncertainties carry no information.",
         "&sigma; = 0.1 for six of eight parameters on every benchmark input, "
         "because the uncertainty head sits at its clamp floor. Do not use them."),
        ("The P(k) reconstruction is a constant.",
         "The generative head returns the same value at every scale and for every "
         "input."),
        ("Neutrino mass is not recovered.",
         "R&sup2; = 0.011 on validation rows where it varies, and recovery differs "
         "sharply between two emulators."),
        ("It fails on the one hydrodynamic slice.",
         "That slice also has a redshift-channel-order defect, so the two causes "
         "are not separated."),
        ("Redshift channels are reversed in two bundled sources.",
         "<code>camb_nl</code> and <code>camels_astrid_x</code> store the channels "
         "in the opposite order, 39 of the 6,000 benchmark rows. The demo warns "
         "when it sees this."),
        ("The rebuilt master arrays have a separately reported ordering problem.",
         "An independent review sampled 648 rows of the June rebuild and found "
         "<code>camb_nl</code> and <code>camb_wa_grid</code> with the opposite of "
         "the expected growth ordering. That is a sample, not a source-wide audit, "
         "and no released model used the rebuild."),
        ("The input is not an observable.",
         "Emulator matter power spectra, with no survey window, shot noise, mask, "
         "galaxy bias, noise model or covariance."),
        ("The split is by row, not by cosmology.",
         "Related spectra of the same base cosmology can sit in both training and "
         "evaluation; leakage has not been ruled out."),
        ("Two redshifts only.",
         "Multi-redshift generalization is unvalidated."),
        ("The validation table is not externally reproducible.",
         "It needs the private master validation rows. The bundled benchmark "
         "reproduces within 1e-5."),
        ("No ablation and no matched direct network.",
         "There is a linear baseline, but until a direct network of matched size is "
         "trained on the same inputs, nothing here shows the architecture earns "
         "its size."),
        ("Historical cross-run comparisons are unreliable.",
         "Epoch-to-epoch evaluation noise was as large as the differences being "
         "compared."),
    ]
    lis = "".join(f"<li><strong>{t}</strong> {d}</li>" for t, d in items)
    return f"""<section id="limits"><div class="wrap">
{_shead("09", "Limitations", "Everything known to be wrong with this.",
        "Stated in full, because a careful reader finds all of it within ten "
        "minutes anyway and it is better coming from me.")}
<div class="sbody"><ol>{lis}</ol></div>
</div></section>"""


def _footer() -> str:
    """Closes the page. The links and the ask live in the code section."""
    return f"""<footer><div class="wrap">
<p class="muted" style="margin:0 0 10px">CosmUFR &middot; Aaditya Rajgor &middot;
released open under MIT. Numbers on this page are labelled reproduced, saved
report or diagnostic, and the faults are listed before the results.</p>
<p><a href="{REPO_URL}">{REPO_URL}</a> &middot; <a href="{HF_URL}">{HF_URL}</a></p>
<p class="dim hashline" style="margin-top:14px">Checkpoint sha256 {SHA256}</p>
</div></footer>"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

def _full(demo_inner: str = "", selected=None) -> str:
    """
    Five questions, in the order a stranger would ask them.

    Everything that used to be a top-level section is still here; the ones that
    are evidence rather than argument now sit inside the question they answer.
    """
    did = _chapter(
        "01", "What I did", "Train a network to map a spectrum to its parameters.",
        "Inferring parameters from a measurement usually means generating and "
        "comparing many candidate universes. I tried training a network on "
        "emulator-generated spectra to map a spectrum to its parameters in one "
        "pass.",
        _sub(_idea(), "The problem, and the bet"),
        _sub(_input_section(), "What the model reads",
             "Four hundred numbers go in. Everything the model will ever know "
             "about a universe has to be in them."))

    observed = _extend(
        _audit(),
        _sub(_evolution(), "The training runs, and what each one taught"),
        _sub(_results(), "How accurate it actually is"),
        _sub(_baseline(), "Is the large model earning its size?"),
        _sub(_ceiling(), "And my explanation for all of it was wrong"))

    conclude = _chapter(
        "03", "What I conclude", "A working baseline, and an untested idea.",
        "The interesting claim has not been tested, because the mechanism that "
        "would test it did not operate. What exists is a checkable starting "
        "point with its faults named.",
        _conclusion(),
        _sub(_roadmap(), "The longer plan, one constraint at a time"),
        _sub(_limits(), "Everything known to be wrong with this"))

    return (_hero() + did + observed + conclude
            + _demo(demo_inner, selected) + _the_code() + _footer())


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
    truth, selected = None, None

    if upload is not None and upload.filename:
        raw = await upload.read()
        if len(raw) > MAX_UPLOAD:
            return _page(_full(_err(
                f"File is {len(raw)/1e6:.1f} MB; the limit is {MAX_UPLOAD/1e6:.0f} MB.")))
        try:
            if upload.filename.lower().endswith(".npy"):
                arr = np.load(io.BytesIO(raw), allow_pickle=False)
            else:
                arr = np.loadtxt(io.StringIO(raw.decode("utf-8", "replace")),
                                 delimiter=",")
        except Exception as e:
            # Escaped: this text can come from an uploaded file on a public endpoint.
            return _page(_full(_err(
                "The file could not be parsed. A .npy must be a plain array of "
                "shape (2, 200); a .csv must be two comma-separated rows of 200 "
                "numbers, with # comment lines ignored.",
                f"{type(e).__name__}: {e}")))
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim != 2 or arr.shape != (2, 200):
            return _page(_full(_err(
                f"Expected shape (2, 200), got {arr.shape}. Row 1 is P(k) at z=0 "
                f"and row 2 is P(k) at z=0.47, each 200 values on the training k "
                f"grid. Download template.csv above for a file that works.")))
        pk0, pk047 = arr[0], arr[1]
        source = f"your uploaded file, {upload.filename}"
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
        return _page(_full(_err("The input is not a usable pair of power spectra.",
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
        "untrained_on_default_path_note": ("modules whose Linear biases are all "
                                           "zero, a clue rather than proof; see "
                                           "reports/results_v1.json for the "
                                           "checkpoint comparison"),
        "code": REPO_URL, "weights": HF_URL,
    })
