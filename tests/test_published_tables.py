"""
Every published number comes from one results file, and says where it came from.

The public copy of this project once printed two different RMSE conventions for
the same parameters side by side, and quoted a reproduction tolerance tighter
than anything measured. Both happened because tables were typed. These tests
make typed tables fail: README.md and MODEL_CARD.md must match what
reports/results_v1.json renders, and every block in that file must carry a
provenance label.
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = json.loads((ROOT / "reports" / "results_v1.json").read_text(encoding="utf-8"))
LABELS = {"reproduced", "saved_report", "diagnostic", "saved_log", "historical"}


def test_docs_tables_match_results_file():
    p = subprocess.run([sys.executable, str(ROOT / "scripts" / "render_tables.py"), "--check"],
                       capture_output=True, text=True)
    assert p.returncode == 0, p.stdout + p.stderr


def test_every_block_is_labelled():
    for key in ("checkpoint", "benchmark", "validation_report", "ridge_baseline", "modules",
                "energy", "settling", "uncertainty", "reconstruction"):
        assert RESULTS[key]["provenance"] in LABELS, key


def test_benchmark_block_agrees_with_bundled_report():
    """The reproduced benchmark column and the one saved in honest_eval.json agree within 1e-5."""
    saved = json.loads((ROOT / "reports" / "honest_eval.json").read_text())["benchmark"]["metrics"]
    for p, v in RESULTS["benchmark"]["overall"].items():
        if v["r2"] is None:
            assert saved[p]["r2"] is None
        else:
            assert abs(v["r2"] - saved[p]["r2"]) < 1e-5, p


def test_no_survey_ranking_in_public_copy():
    """Point-estimate scatter is not ranked against survey posterior widths (decision of 12 September 2026)."""
    for name in ("README.md", "MODEL_CARD.md", "server.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for phrase in ("times worse", "published constraint", "Planck 2018"):
            assert phrase not in text, f"{name}: {phrase!r}"


def test_withdrawn_claims_do_not_return():
    for name in ("README.md", "MODEL_CARD.md", "server.py"):
        text = (ROOT / name).read_text(encoding="utf-8")
        for phrase in ("in any of the eight runs", "information is unbounded", "no such ceiling can exist",
                       "built from the Quijote", "really a baryon number", "7e29", "small code change"):
            assert phrase not in text, f"{name}: {phrase!r}"
