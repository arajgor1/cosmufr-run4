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
    "dark_emulator": "Dark Emulator, built from the Quijote N-body suite",
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
N_PARAMS = sum(p.numel() for p in MODEL.parameters())

_reports = Path(__file__).parent / "reports"
REPORT = json.loads((_reports / "honest_eval.json").read_text()) \
    if (_reports / "honest_eval.json").exists() else {}
RIDGE = json.loads((_reports / "ridge_baseline.json").read_text()) \
    if (_reports / "ridge_baseline.json").exists() else {}

EXAMPLE_IDS = list(range(0, min(len(BENCH), 5400), 211))[:24]

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
.navlinks{display:flex; gap:26px; margin-left:auto; align-items:center; overflow-x:auto}
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

/* ── sections ────────────────────────────────────────────────────────── */
section{padding:104px 0; border-bottom:1px solid var(--hair); position:relative}
.shead{display:grid; grid-template-columns:88px 1fr; gap:26px; margin-bottom:44px}
.shead .num{font-family:var(--mono); font-size:12px; color:var(--accent);
  letter-spacing:.1em; padding-top:9px}
.shead .cat{font-family:var(--mono); font-size:10.5px; letter-spacing:.21em;
  text-transform:uppercase; color:var(--dim); margin:0 0 13px}
.shead h2{font-size:clamp(27px,3.9vw,44px); margin:0 0 16px; max-width:20ch}
.shead p{color:var(--mut); max-width:66ch; margin:0; font-size:16.5px}
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
.figcap .grid{display:grid; grid-template-columns:repeat(auto-fit,minmax(215px,1fr)); gap:16px}
.figcap .lbl{font-family:var(--mono); font-size:9.5px; letter-spacing:.17em;
  text-transform:uppercase; color:var(--accent); display:block; margin-bottom:5px}
.figcap p{margin:0; font-size:13.5px; color:var(--mut); line-height:1.6}

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

NAV_LINKS = [("#idea", "The idea"), ("#input", "The input"), ("#demo", "Live demo"),
             ("#output", "The output"), ("#results", "Accuracy"),
             ("#audit", "The audit"), ("#evolution", "History"),
             ("#roadmap", "Roadmap")]


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


def _figblock(key: str, img_src: str) -> str:
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
            f'<div class="figcap"><div class="grid">{inner}</div></div></div>')


def _shead(num: str, cat: str, title: str, sub: str = "") -> str:
    p = f"<p>{sub}</p>" if sub else ""
    return (f'<div class="shead"><div class="num">{num}</div><div>'
            f'<p class="cat">{cat}</p><h2 class="display">{title}</h2>{p}'
            f'</div></div>')


def _page(body: str, title: str = "CosmUFR — cosmology from a power spectrum") -> HTMLResponse:
    links = "".join(f'<a href="{h}">{html.escape(t)}</a>' for h, t in NAV_LINKS)
    return HTMLResponse(f"""<!doctype html><html lang="en"><head>
<meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{html.escape(title)}</title>
<meta name="description" content="A neural network that reads the matter power spectrum and returns eight cosmological parameters in a quarter of a second, with the benchmark and the audit published alongside it.">
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
<script>{PARTICLES_JS}{SCROLL_JS}</script>
</body></html>""")


# ─────────────────────────────────────────────────────────────────────────────
# Narrative sections
# ─────────────────────────────────────────────────────────────────────────────


def _hero_visual() -> str:
    """
    The two input curves and the eight numbers they turn into, drawn from a real
    benchmark spectrum rather than mocked up. It is the whole product in one
    picture: a measurement on the left, a cosmology on the right.
    """
    # Prefer a suite that varies all eight parameters: a fiducial universe
    # shows w0 = -1.000 and mv = 0.000 and reads as placeholder text.
    i = next((j for j in EXAMPLE_IDS if _src_name(j) == "bacco_full8"),
             EXAMPLE_IDS[1] if len(EXAMPLE_IDS) > 1 else 0)
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

    p = BENCH.params[i]
    cells = "".join(
        f'<div><div class="p">{PARAM_TEX[l]}</div>'
        f'<div class="q">{p[j]:+.3f}</div></div>' if l in ("w0", "wa") else
        f'<div><div class="p">{PARAM_TEX[l]}</div>'
        f'<div class="q">{p[j]:.3f}</div></div>'
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
<div class="cap" style="margin:16px 0 0"><span>Output &middot; the cosmology that produced it</span><span>245 ms</span></div>
<div class="hgrid">{cells}</div>
</div></div>"""


def _hero() -> str:
    return f"""<header class="hero">
<canvas id="plexus"></canvas>
<div class="brackets"><i></i><i></i><i></i><i></i></div>
<div class="wrap">
<div>
<div class="eyebrow"><span class="dot"></span>Run 4 &middot; research preview &middot; live model</div>
<h1 class="display">From a sky measurement to a cosmology,<br><em>in a quarter of a second.</em></h1>
<p class="lede">Working out which universe produced a given measurement normally
takes days of compute. CosmUFR is a neural network that learns the inverse map
directly and returns an answer in one forward pass. This page runs the real
model, on real held-out data, while you watch.</p>
<div class="ctas">
  <a class="btn btn-p" href="#demo">Run it on real data &rarr;</a>
  <a class="btn btn-g" href="#idea">What problem is this</a>
</div>
<div class="metrics">
  <div><div class="v">245&thinsp;ms</div><div class="k">to infer, one CPU</div></div>
  <div><div class="v">8</div><div class="k">parameters out</div></div>
  <div><div class="v">84.5M</div><div class="k">training spectra</div></div>
  <div><div class="v">6,000</div><div class="k">checkable test cases</div></div>
</div>
</div>
{_hero_visual()}
</div></header>"""


def _idea() -> str:
    return f"""<section id="idea"><div class="wrap">
{_shead("00", "The idea", "Surveys are fast now. Interpreting them is not.",
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
<p class="muted" style="margin:0"><span class="lead-in">The alternative this
tests.</span> Train a network on millions of simulated universes until it learns
the inverse directly. Then a new measurement is a single forward pass. If that
works, the cost of an analysis drops by orders of magnitude, and questions you
would never run because they are too expensive become routine: sweep the whole
survey, re-run under every systematic, iterate in an afternoon.</p>
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
doing. Plus, honestly, which of those it does not yet get right.</p></div>
</div>

<h3 class="display" style="font-size:21px; margin:44px 0 6px">The path a spectrum takes</h3>
<p class="muted" style="max-width:66ch; margin:0 0 6px">Four hundred numbers go in
on the left. The encoder turns them into a 1024-dimensional "belief" about which
universe this is, sixteen refinement steps are meant to sharpen that belief, and
three read-out heads turn it into answers.</p>
<div class="diagram"><p class="cap">Inference path &middot; one forward pass</p>
{architecture_svg()}</div>

<p class="muted" style="margin-top:26px">This release is an early, working, and
openly flawed attempt at that. Everything below is measured rather than claimed,
the test data ships with the code, and the parts that do not work are named.</p>
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
{_shead("01", "The input", "What the model actually reads.",
        "Two curves. Everything the model knows about a universe comes from "
        "these 400 numbers.")}
<div class="sbody">

<div class="cards">
<div class="card"><div class="n">What P(k) is</div>
<h4>Clumpiness, scale by scale</h4>
<p>Matter is not spread evenly. The power spectrum says how much structure exists
at each size: large scales on the left, small ones on the right. It falls to the
right because the universe is smoother at large scales.</p></div>
<div class="card"><div class="n">Why two of them</div>
<h4>Now, and 4.7 billion years ago</h4>
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

<h3 class="display" style="font-size:21px; margin:44px 0 10px">The examples in the demo</h3>
<p class="muted" style="max-width:66ch">The dropdown is not a set of toy inputs.
Each entry is a real simulated universe from a published cosmology code, held out
of training, with its true parameters recorded. The label shows the suite it came
from and three of its true values, so you can see before you run it what the
right answer is.</p>
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
{_shead("03", "The output", "Reading the eight numbers.",
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
ignored. Section 05 explains why.</p></div>
<div class="card"><div class="n">Expect a spread</div><h4>Not all eight are equal</h4>
<p>The spectrum constrains matter density and clumpiness strongly, expansion rate
and dark energy weakly, and neutrino mass essentially not at all. That ordering
is physics, and it shows in the accuracy table.</p></div>
</div>
</div></div></section>"""



RUNS = [
    ("Run 1", "Mar 2026", "&mdash;",
     "First end-to-end training. 57M samples across 8 sources.",
     "The infrastructure worked. Nothing else did: the sequential loss term was "
     "identically zero and every parameter was out of range."),
    ("Run 2", "9 Apr 2026", "$280",
     "First serious run. 45 epochs, batch 2048, full curriculum.",
     "Matter density and clustering amplitude came in strong. The sequential "
     "loss was still dead, traced to a two-pass forward bug."),
    ("Run 3", "12 Apr 2026", "$54",
     "Fixed the sequential loss, reweighted the objective, enabled torch.compile.",
     "1.64&times; throughput. The expansion rate got worse, not better, because "
     "one shared belief-proposal module was serving two incompatible jobs."),
    ("Run 4", "14 Apr 2026", "$94",
     "Split that module into separate joint and sequential paths. Fixed a state "
     "leak in the settling core.",
     "The released checkpoint. Expansion rate improved most, and calibration hit "
     "a ceiling that later turned out to be the uncertainty head's clamp floor."),
    ("Runs 5&ndash;8", "May 2026", "$145",
     "Seven architectural variants, chasing what looked like a hard ceiling.",
     "All of it inconclusive. The evaluation noise was larger than every effect "
     "being measured, so none of those experiments could have shown anything."),
    ("The audit", "Sep 2026", "$0",
     "Stopped training. Read the weights instead.",
     "The belief pipeline had never received a gradient, and the energy heads had "
     "collapsed to a constant. The ceiling was a defect, not physics."),
]


def _evolution() -> str:
    rows = "".join(
        f'<tr><td><strong>{name}</strong></td>'
        f'<td class="dim" style="white-space:nowrap">{when}</td>'
        f'<td class="num dim">{cost}</td>'
        f'<td class="muted">{changed}</td>'
        f'<td class="muted">{learned}</td></tr>'
        for name, when, cost, changed, learned in RUNS)
    return f"""<section id="evolution"><div class="wrap">
{_shead("06", "How it got here", "Four training runs, then a decision to stop training.",
        "The useful history is not a score table. It is what each run changed "
        "and what that taught, because the scores turned out to be less "
        "trustworthy than they looked.")}
<div class="sbody">
<div class="panel"><div class="tw"><table>
<tr><th>run</th><th>when</th><th class="num">cost</th><th>what changed</th>
<th>what it taught</th></tr>{rows}</table></div></div>

<div class="panel warn">
<p class="muted" style="margin:0"><span class="lead-in">Why there are no
per-run scores here.</span> There were, for months. The audit found that
epoch-to-epoch evaluation noise was &plusmn;0.03 to 0.10, larger than nearly
every difference being compared between runs, and that the validation set had
changed partway through without the earlier numbers being re-measured against
it. So the run-to-run deltas that drove five months of decisions were mostly
noise. Publishing them as a progression would repeat the mistake. What survives
is the architectural lesson from each run, which is what the table records.</p>
</div>

<p class="muted">The most expensive lesson was Run 5: seven changes shipped at
once, a catastrophic regression, and no way to tell which change caused it.
Everything after that is single-change-at-a-time with a threshold written down
before the run starts.</p>
</div></div></section>"""


def _roadmap() -> str:
    return f"""<section id="roadmap"><div class="wrap">
{_shead("07", "Roadmap", "Where this is, and where it goes.",
        "An active research programme, not a finished tool. Where the work stands, what it is aimed at, and the order the rest gets attacked in.")}
<div class="sbody">
<div class="road">

<div class="col now"><div class="st">&#9679; Where it is today</div>
<h4>A working, audited baseline</h4>
<ul>
<li>End-to-end inference in 245&thinsp;ms on a CPU, bit-deterministic.</li>
<li>Trained on 84.5M spectra across 14 simulation suites.</li>
<li>Recovers matter density and clumpiness to R&sup2; 0.98&ndash;0.99 on
sound data.</li>
<li>A 6,000-case benchmark ships with the code, so the numbers are checkable.</li>
<li>A linear baseline is published alongside, including where it wins.</li>
<li>Audited: the belief-settling core never trained, and that is documented
rather than buried.</li>
</ul></div>

<div class="col next"><div class="st">&#9654; Next, and costed</div>
<h4>Make the architecture actually run</h4>
<ul>
<li>Repair the severed gradient path, guarded by the unit test that would have
caught it originally. Free, and verifiable before any training spend.</li>
<li>Fix the flat energy landscape. Harder, and the real blocker: the heads
trained themselves into a constant, so there is nothing to descend.</li>
<li>Give the uncertainty head a floor it can leave, so the error bars mean
something.</li>
<li>Rebalance the training corpus, which pins dark energy at its fiducial value
in 86 percent of samples.</li>
<li>One pre-registered training run with a pass/fail threshold set in advance.
Roughly $15.</li>
</ul></div>

<div class="col later"><div class="st">&#9675; Open questions</div>
<h4>What I want advice on</h4>
<ul>
<li>Is iterative belief refinement worth pursuing at all once the gradient path
works, or does an amortized posterior estimator get there in one pass?</li>
<li>How much of the weakness in expansion rate and dark energy is a real
information limit of P(k), and how much is just training coverage? I do not
know how to separate them cleanly.</li>
<li>Would higher k, more redshifts, or explicit acoustic-scale features make the
expansion rate identifiable?</li>
<li>What does a defensible experimental design for any of that look like?</li>
</ul></div>

</div>
<div class="panel accent" style="margin-top:28px">
<p class="muted" style="margin:0 0 12px"><span class="lead-in">The goal this is
aimed at.</span> A survey measurement should be interpretable in seconds, not
weeks, with a stated uncertainty you can defend. Reach that and the analyses
nobody runs today because they cost too much become ordinary: sweep an entire
survey, re-run under every systematic, close the loop between an observation and
a constraint inside a single working session.</p>
<p class="muted" style="margin:0">Nothing here is there yet. What exists is a
working inference path, a benchmark anyone can check, a baseline that says how
much of the result is really the network, and a precise account of which parts
of the architecture do not yet work. That is the foundation the rest gets built
on.</p></div>
</div></div></section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Downloads. Every example is downloadable in both formats so a visitor can take
# one, upload it straight back, and confirm the answer matches. Nothing here is
# special-cased for the built-ins: the upload path runs identical code.
# ─────────────────────────────────────────────────────────────────────────────

def _src_name(i: int) -> str:
    return cosmufr.SOURCE_NAMES.get(int(BENCH.source_lid[i]), f"lid_{BENCH.source_lid[i]}")


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
<p class="dim" style="margin:14px 0 0"><span class="lead-in">Check us.</span>
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
                  "from the simulation that produced this spectrum. The model "
                  "never saw this example during training.</p>")

    return f"""<div id="result"></div>
<h3 class="display" style="font-size:24px; margin:38px 0 6px">Result</h3>
<p class="muted" style="margin:0 0 4px">Input: {html.escape(source_label)}</p>
<p class="dim" style="margin:0 0 16px">Belief moved {report.belief_movement*100:.3f}%
of its norm during settling; energy changed by {report.energy_drop:.2e}, which is
{report.energy_drop_in_ulps:.1f} float32 resolution steps.</p>
<div class="panel"><div class="tw"><table>{head}{rows}</table></div>
<p class="dim" style="margin:14px 0 0">Every &sigma; marked
<span class="flag">ignore</span> is the uncertainty head's clamp constant rather
than a prediction. It is the same number on every input.</p>{truth_note}{dl}</div>
{_timing_panel(t)}
{_figblock("settling", fig_settle)}
{_figblock("pk", fig_pk)}
<details><summary>Full result as JSON</summary>
<pre><code>{html.escape(json.dumps(payload, indent=2))}</code></pre></details>"""


def _demo(inner: str = "", selected=None) -> str:
    return f"""<section id="demo"><div class="wrap">
{_shead("02", "Live demo", "Run it yourself, right now.",
        "The released checkpoint, loaded in this container, running on whatever "
        "you give it. Nothing is cached and nothing is precomputed.")}
<div class="sbody">{_form(selected)}{inner}</div>
</div></section>"""


# ─────────────────────────────────────────────────────────────────────────────
# Accuracy, baseline, audit, limitations
# ─────────────────────────────────────────────────────────────────────────────

def _results() -> str:
    if not REPORT:
        return ""
    full, vary = REPORT["full_val_metrics"], REPORT["full_val_metrics_varying_only"]
    bench = REPORT["benchmark"]["metrics"]

    def f(x):
        return "&ndash;" if x is None else f"{x:.3f}"

    rows = ""
    for lbl in PARAM_LABELS:
        name, _ = PARAM_MEANING[lbl]
        v = vary[lbl]["r2"]
        verdict = ("<span class='good-t'>recovered</span>" if v and v > 0.6 else
                   "<span class='flag'>partial</span>" if v and v > 0.25 else
                   "<span class='bad-t'>not recovered</span>")
        rows += (f'<tr><td>{PARAM_TEX[lbl]}</td>'
                 f'<td class="dim" style="white-space:nowrap">{html.escape(name)}</td>'
                 f'<td class="num">{f(full[lbl]["r2"])}</td>'
                 f'<td class="num">{f(v)}</td>'
                 f'<td class="num">{f(bench[lbl]["r2"])}</td>'
                 f'<td>{verdict}</td></tr>')

    src = ""
    for name, blk in sorted(REPORT["per_source_metrics"].items(),
                            key=lambda kv: -kv[1]["n"]):
        cells = "".join(f'<td class="num">{f(blk["metrics"][l]["r2"])}</td>'
                        for l in PARAM_LABELS)
        flag = (' <span class="flag">known data defect</span>'
                if name == "bacco_multiz" else "")
        src += (f'<tr><td><code>{html.escape(name)}</code>{flag}</td>'
                f'<td class="num">{blk["n"]:,}</td>{cells}</tr>')
    hdr = "".join(f'<th class="num">{PARAM_TEX[l]}</th>' for l in PARAM_LABELS)

    return f"""<section id="results"><div class="wrap">
{_shead("04", "Accuracy", "How well it actually works.",
        f"Measured on {REPORT['val_split']['n']:,} held-out spectra across "
        f"{REPORT['val_split']['n_sources']} simulation suites, with the "
        f"released code on a split that has no randomness in it.")}
<div class="sbody">
<div class="panel"><div class="tw"><table>
<tr><th>parameter</th><th></th><th class="num">all test data</th>
<th class="num">where it varies</th><th class="num">bundled benchmark</th>
<th>verdict</th></tr>{rows}</table></div></div>

<p class="muted"><span class="lead-in">Read the second number, not the first.</span>
R&sup2; measures how much of the spread in the truth the model explains. On data
where a parameter is held at a fixed value there is no spread, so the score is
meaningless. The "where it varies" column restricts each parameter to the data
that actually varies it. For neutrino mass that is the whole story: it looks
competent at 0.41 and is 0.011 once measured honestly.</p>

<p class="muted"><span class="lead-in">What reproduces.</span> The bundled
6,000-case benchmark ships in the repository and regenerates its own column to
about 1e-6 on any machine. The full-test column came from a private split and
cannot be checked from outside; the benchmark lands within about 0.03 of it and
narrows that gap rather than closing it.</p>

<h3 class="display" style="font-size:21px; margin:40px 0 10px">Why the headline is lower than the model deserves</h3>
<div class="panel tight"><div class="tw"><table>
<tr><th>simulation suite</th><th class="num">n</th>{hdr}</tr>{src}</table></div>
<p class="dim" style="margin:12px 0 0"><code>&ndash;</code> means the parameter is
held fixed in that suite, so R&sup2; is undefined rather than bad.</p></div>
<p class="muted">One suite, <code>bacco_multiz</code>, is 15 percent of the test
set and scores zero on everything, because its two redshift channels are
identical copies of each other and carry no growth information. That is a defect
in how the data was generated. On suites with sound data, matter density comes
back at 0.98 to 0.99.</p>

<h3 class="display" style="font-size:21px; margin:40px 0 10px">Check it yourself</h3>
<pre><code>git clone {REPO_URL}
cd cosmufr-run4
pip install -e ".[demo]"
python -m cosmufr.reproduce</code></pre>
</div></div></section>"""


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
                 f'<td class="dim" style="white-space:nowrap">{html.escape(name)}</td>'
                 f'<td class="num">{rb}</td><td class="num">{cb}</td>'
                 f'<td class="dim">{p["winner"]}</td></tr>')
    wins, n = RIDGE.get("cosmufr_wins", 0), RIDGE.get("n_compared", 0)
    return f"""<section id="baseline"><div class="wrap">
{_shead("05", "Baseline", "Is the big model earning its keep?",
        "The first question anyone should ask about a 136-million-parameter "
        "network is whether a simple method does just as well. So here is the "
        "simple method.")}
<div class="sbody">
<p class="muted" style="max-width:66ch">Ridge regression on exactly the same 400
input numbers, fitted on half the benchmark and scored on the other half.
CosmUFR is scored on that same held-out half, so it is a fair fight.</p>
<div class="panel"><div class="tw"><table>
<tr><th>parameter</th><th></th><th class="num">ridge, 400 features</th>
<th class="num">CosmUFR, 136M</th><th>winner</th></tr>{rows}</table></div></div>
<p class="muted">CosmUFR is ahead on {wins} of {n}. A plain linear fit is
competitive on matter density, which is a real and slightly uncomfortable result:
that parameter is written into the height of the curve, and you do not need a
large network to read it. The network earns its keep on the parameters that are
subtle or that the training data barely varies, where it has learned a prior
ridge cannot get from three thousand rows.</p>
<p class="dim">Both caveats favour ridge, which is fitted on rows from the same
suites it is tested on while CosmUFR has never seen any of these spectra. This
is a generous baseline, not a hostile one. Rerun it with
<code>python scripts/ridge_baseline.py</code>.</p>
</div></div></section>"""


def _audit() -> str:
    img = _png(F.fig_weight_audit(AUDIT))
    rows = ""
    for name, m in AUDIT.modules.items():
        bad = m["verdict"] == "UNTRAINED"
        rows += (f'<tr><td><code>{html.escape(name)}</code></td>'
                 f'<td class="dim">{"used" if m["on_default_path"] else "unused"}</td>'
                 f'<td class="num">{m["n_zero_bias"]}/{m["n_linear"]}</td>'
                 f'<td class="num">{m["max_abs_bias"]:.3e}</td>'
                 f'<td class="{"bad-t" if bad else "good-t"}">'
                 f'{"never trained" if bad else "trained"}</td></tr>')
    return f"""<section id="audit"><div class="wrap">
{_shead("05", "The audit", "I stress-tested my own model. It failed.",
        "Everything above is what the model does. This is what I found when I "
        "went looking for reasons not to trust it, and it is the reason the "
        "roadmap looks the way it does.")}
<div class="sbody">

<div class="panel bad">
<p class="muted" style="margin:0"><span class="lead-in">The short version.</span>
CosmUFR is built around a "belief-settling" core: encode the spectrum into a
belief, then refine it over 16 steps. That refinement is the research idea. The
weights say it never trained. What learned is the read-out heads, reading a fixed
random projection of the input. The numbers in section 04 are real, and they were
produced by a simpler machine than the architecture diagram claims.</p>
</div>

<h3 class="display" style="font-size:20px; margin:34px 0 6px">The same diagram, marked up</h3>
<p class="muted" style="max-width:66ch; margin:0 0 6px">This is the picture from
section 00 again, with each block coloured by what its weights actually show.</p>
<div class="diagram"><p class="cap">Inference path &middot; what the weights say</p>
{architecture_svg(audit=True)}</div>

<p class="muted">A <code>Linear</code> bias is initialised from a random draw, and
any optimizer step moves it. Eighty-four of them are still bit-exactly
<code>0.0</code> after forty epochs of training:</p>
<div class="panel tight"><div class="tw"><table>
<tr><th>module</th><th></th><th class="num">biases = 0</th>
<th class="num">max |bias|</th><th>verdict</th></tr>{rows}</table></div></div>

{_figblock("weight_audit", img)}

<h3 class="display" style="font-size:20px; margin:36px 0 10px">Why, in released source</h3>
<p class="muted">No checkpoint needed. The settling loop detaches the belief on
entry to every step, which cuts everything upstream of it off from the loss:</p>
<pre><code>for step in range(k):
    b = b.detach()                      # &lt;- severs everything upstream
    with torch.enable_grad():
        b_g = b.requires_grad_(True)
        E = energy_fn(b_g, z.detach(), b_prev.detach())
        grad = torch.autograd.grad(E.sum(), b_g)[0]
    b = b - eta * P * grad.detach()</code></pre>
<p class="muted">One synthetic training step on a fresh model confirms it:
<code>modules that received any gradient: ['param_head']</code></p>

<h3 class="display" style="font-size:20px; margin:36px 0 10px">And a second cause, which is worse</h3>
<div class="panel bad"><p class="muted" style="margin:0">The energy heads
<em>did</em> train, through their own optimizer, and converged on a constant. The
energy varies by about one part in seven million across completely different
spectra, and its gradient has norm 0.11 against a belief of norm 16.5. With the
step size capped where it is, sixteen steps could move the belief half a percent
at most, whatever the input.
<strong style="color:var(--fg)">So repairing the gradient path alone would not
make settling work.</strong> There would still be no landscape to descend. That
is a harder problem than the one I first reported, and I do not have a fix
for it yet.</p></div>

<p class="muted">All of this is reproducible from the released weights in about a
minute: <code>cosmufr.weight_audit(model)</code> and
<code>cosmufr.settling_report(...)</code>. The test that would have caught it on
day one now ships in the repository.</p>
</div></div></section>"""


def _limits() -> str:
    items = [
        ("The belief pipeline never trained.",
         "The encoder, the belief proposal and the settling core sit at "
         "initialization. Section 05 has the evidence."),
        ("The energy landscape is flat.",
         "The energy heads collapsed to an input-independent constant, so there "
         "is nothing for the refinement to descend."),
        ("Reported uncertainties are meaningless.",
         "&sigma; = 0.1 for six of eight parameters on every input, because the "
         "uncertainty head sits at its clamp floor. Do not use them."),
        ("The P(k) reconstruction is a constant.",
         "The generative head returns the same value at every scale, for every "
         "input, and for a random belief vector."),
        ("Neutrino mass is not recovered.",
         "R&sup2; = 0.011 measured only on data where it varies."),
        ("The anomaly score is not usable.",
         "The energy subsystem diverged; E_con sits around &minus;4.6e5."),
        ("Two redshifts only.",
         "Multi-redshift generalization is unvalidated, and that corpus has a "
         "documented ordering defect."),
        ("The headline table is not externally reproducible.",
         "It was measured on a private split. The bundled benchmark narrows that "
         "gap to about 0.03 rather than closing it."),
        ("No ablation.",
         "There is a linear baseline now, but no ablation of the architecture's "
         "own components."),
    ]
    lis = "".join(f"<li><strong>{t}</strong> {d}</li>" for t, d in items)
    return f"""<section id="limits"><div class="wrap">
{_shead("08", "Limitations", "Everything known to be wrong with this.",
        "Stated in full, because a careful reader finds all of it within ten "
        "minutes anyway and it is better coming from me.")}
<div class="sbody"><ol>{lis}</ol></div>
</div></section>

<footer><div class="wrap">
<p class="muted" style="margin:0 0 10px">CosmUFR is an active research programme
by Aaditya Rajgor, released open under MIT. If you work on cosmological inference
and any of the open questions above look answerable, I would like to hear from
you.</p>
<p>Code, benchmark and full report &middot; <a href="{REPO_URL}">{REPO_URL}</a><br>
Weights and model card &middot; <a href="{HF_URL}">{HF_URL}</a></p>
<p class="dim" style="margin-top:14px">Checkpoint sha256 {SHA256}</p>
</div></footer>"""


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

def _full(demo_inner: str = "", selected=None) -> str:
    return (_hero() + _idea() + _input_section() + _demo(demo_inner, selected)
            + _output_section() + _results() + _baseline() + _audit()
            + _evolution() + _roadmap() + _limits())


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
        "code": REPO_URL, "weights": HF_URL,
    })
