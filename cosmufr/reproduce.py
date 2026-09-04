"""
cosmufr/reproduce.py — regenerate every number in the README, in one command.

    python -m cosmufr.reproduce
    python -m cosmufr.reproduce --ckpt best.pt --figures out/

Downloads the released checkpoint if you do not pass one, verifies its SHA256,
runs the bundled benchmark, prints the headline and per-source tables, prints
the defect audit, and compares everything against reports/honest_eval.json.

Exit code is 0 when the reproduction matches, 1 when it does not.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import cosmufr
from cosmufr.load import PARAM_LABELS

EXPECTED_SHA256 = "5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1"
REPORT_PATH = Path(__file__).resolve().parent.parent / "reports" / "honest_eval.json"
TOLERANCE = 1e-3


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _rule(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ckpt", default=None,
                    help="local checkpoint; downloads from HuggingFace if omitted")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--figures", default=None,
                    help="directory to write the figure PNGs into")
    args = ap.parse_args(argv)

    _rule("COSMUFR RUN 4 — REPRODUCTION")
    print(f"package version : {cosmufr.__version__}")

    # ── checkpoint ──────────────────────────────────────────────────────────
    ckpt = args.ckpt
    if ckpt is None:
        print("\nNo --ckpt given; downloading the released weights ...")
        from huggingface_hub import hf_hub_download
        ckpt = hf_hub_download(repo_id=cosmufr.HF_REPO_ID, filename="best.pt")
    print(f"checkpoint      : {ckpt}")

    sha = _sha256(ckpt)
    sha_ok = sha == EXPECTED_SHA256
    print(f"sha256          : {sha}")
    print(f"expected        : {EXPECTED_SHA256}")
    print(f"match           : {sha_ok}")
    if not sha_ok:
        print("\nWARNING: this is not the checkpoint the released numbers were "
              "measured from. Results below will not match the report.")

    model = cosmufr.load_model(ckpt_path=ckpt, device=args.device)
    print(f"parameters      : {sum(p.numel() for p in model.parameters()):,}")

    # ── the audit ───────────────────────────────────────────────────────────
    _rule("1. DEFECT AUDIT")
    audit = cosmufr.weight_audit(model)
    print(audit.table())
    print(f"\nUntrained modules on the default inference path: "
          f"{audit.untrained_on_default_path}")

    bench = cosmufr.load_benchmark()
    report = cosmufr.settling_report(model, bench.pk_z0[0], bench.pk_z047[0])
    print()
    print(report.summary())

    # ── benchmark ───────────────────────────────────────────────────────────
    _rule(f"2. BENCHMARK  (N = {len(bench):,})")
    print(f"source mix: {bench.source_counts()}\n")
    t0 = time.time()
    result = cosmufr.evaluate(model, bench, batch_size=args.batch_size)
    print(f"evaluated in {time.time()-t0:.0f}s on {args.device}\n")
    print(result.table())
    print()
    print(result.source_table())
    if result.notes:
        print("\nNotes on flagged sources:")
        for n in result.notes:
            print(f"  - {n}")

    print()
    print(cosmufr.uncertainty_audit(result.sigmas).table())

    # ── comparison against the published report ─────────────────────────────
    _rule("3. DOES THIS MATCH THE PUBLISHED REPORT?")
    if not REPORT_PATH.exists():
        print(f"reports/honest_eval.json not found at {REPORT_PATH}; skipping.")
        return 0

    published = json.loads(REPORT_PATH.read_text())["benchmark"]["metrics"]
    print(f"{'param':<7}{'this run':>12}{'published':>12}{'delta':>12}{'':>8}")
    print("-" * 51)
    failures = []
    for lbl in PARAM_LABELS:
        got = result.overall[lbl]["r2"]
        want = published[lbl]["r2"]
        if got is None or want is None:
            ok = (got is None) == (want is None)
            print(f"{lbl:<7}{'n/a':>12}{'n/a':>12}{'':>12}"
                  f"{'  ok' if ok else '  MISMATCH':>8}")
        else:
            d = got - want
            ok = abs(d) < TOLERANCE
            print(f"{lbl:<7}{got:>12.4f}{want:>12.4f}{d:>12.2e}"
                  f"{'  ok' if ok else '  MISMATCH':>8}")
        if not ok:
            failures.append(lbl)

    # ── figures ─────────────────────────────────────────────────────────────
    if args.figures:
        _rule("4. FIGURES")
        import matplotlib
        matplotlib.use("Agg")
        from cosmufr import figures as F

        outdir = Path(args.figures)
        outdir.mkdir(parents=True, exist_ok=True)
        r0 = cosmufr.infer(bench.pk_z0[0], bench.pk_z047[0], model=model)
        made = {
            "weight_audit.png": F.fig_weight_audit(audit),
            "settling_trajectory.png": F.fig_settling_trajectory(report),
            "parameter_recovery.png": F.fig_parameter_recovery(result),
            "per_source.png": F.fig_per_source(result),
            "uncertainty_audit.png": F.fig_uncertainty_audit(result),
            "pk_reconstruction.png": F.fig_pk_reconstruction(
                bench.k_bins, bench.pk_z0[0], bench.pk_z047[0],
                r0.pk_recon, r0.log_k),
        }
        for name, fig in made.items():
            fig.savefig(outdir / name, dpi=140, bbox_inches="tight")
            print(f"  wrote {outdir / name}")

    _rule("RESULT")
    if failures:
        print(f"MISMATCH on {failures}.")
        print("Either the checkpoint differs or preprocessing has drifted. "
              "Both are worth reporting at "
              "https://github.com/arajgor1/cosmufr-run4/issues")
        return 1
    print("All benchmark numbers reproduce the published report within "
          f"{TOLERANCE}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
