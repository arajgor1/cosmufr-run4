"""
scripts/render_tables.py — write every numerical table into README.md and
MODEL_CARD.md from reports/results_v1.json.

Each table sits between markers:

    <!-- table:NAME -->
    ...generated...
    <!-- /table:NAME -->

Nothing between the markers is hand-edited. `--check` exits non-zero if either
file differs from what the results file produces; the test suite runs it.

Usage:
    python scripts/render_tables.py          # rewrite the tables in place
    python scripts/render_tables.py --check  # fail if anything is stale
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "reports" / "results_v1.json"
FILES = [ROOT / "README.md", ROOT / "MODEL_CARD.md"]
PARAMS = ["Om", "s8", "h", "ns", "Ob", "w0", "mv", "wa"]
SYMBOL = {"Om": "Ω_m", "s8": "σ₈", "h": "h", "ns": "n_s", "Ob": "Ω_b",
          "w0": "w₀", "mv": "Σm_ν (eV)", "wa": "w_a"}

MODULE_ROLE = {
    "obs_encoder": "reads the two spectra into a 1024-d vector",
    "belief_proposal": "forms the first belief",
    "settling": "refines the belief over 16 steps",
    "param_head": "computes the eight reported parameters",
    "unc_head": "reports a σ per parameter",
    "gen_head": "redraws P(k) from the belief",
    "obs_energy_head": "energy term: fit to the observation",
    "dyn_energy_head": "energy term: movement between beliefs",
    "constraint_head": "energy term: learned constraint score",
    "halo_head": "side output, not used for the parameters",
    "obs_encoder_single": "single-redshift encoder, training-only sequential path",
    "belief_proposal_seq": "sequential proposal, training-only sequential path",
    "attractor_bank": "prototype bank, updated by moving average",
}


def _f(x, nd=3):
    return "undefined" if x is None else f"{x:.{nd}f}"


def _metrics_table(block: dict) -> str:
    rows = ["| Parameter | R², all rows | R², rows where it varies | RMSE, all rows | RMSE, rows where it varies | Rows where it varies |",
            "|---|---:|---:|---:|---:|---:|"]
    for p in PARAMS:
        o, v = block["overall"][p], block["varying_only"][p]
        rows.append(f"| {SYMBOL[p]} | {_f(o['r2'])} | {_f(v['r2'])} | {_f(o['rmse'], 4)} | "
                    f"{_f(v['rmse'], 4)} | {v['n']:,} |")
    return "\n".join(rows)


def _validation(r: dict) -> str:
    v = r["validation_report"]
    body = _metrics_table(v)
    verdicts = ", ".join(f"{SYMBOL[p].replace(' (eV)', '')} {v['verdicts'][p]}" for p in PARAMS)
    return (f"Saved report, not rerun: {v['rows']:,} rows of the deterministic validation split "
            f"({v['source_ids_in_split']} source ids), generated {v['generated_utc']}. "
            f"It needs the master validation rows and `bad_indices.npy` to regenerate.\n\n"
            f"{body}\n\nVerdicts ({v['verdict_rule']}): {verdicts}.")


def _benchmark(r: dict) -> str:
    b = r["benchmark"]
    a = b["agreement"]
    return (f"Reproduced: the bundled {b['rows']:,}-row benchmark, run `{b['run_id']}` "
            f"(torch {b['environment']['torch']}, CPU). Agreement with an independent reproduction: "
            f"largest R² difference {a['max_abs_r2_diff_vs_independent_review']:.1e}, largest prediction "
            f"difference {a['max_abs_prediction_diff_vs_independent_review']:.1e}; repeat run bit-identical: "
            f"{'yes' if a['repeat_bit_identical'] else 'no'}.\n\n{_metrics_table(b)}")


def _per_source(r: dict) -> str:
    v = r["validation_report"]
    rows = ["| Source | Rows | " + " | ".join(SYMBOL[p].replace(" (eV)", "") for p in PARAMS) + " |",
            "|---|---:|" + "---:|" * len(PARAMS)]
    for name, blk in sorted(v["per_source"].items(), key=lambda kv: -kv[1]["n"]):
        cells = " | ".join("fixed" if blk["metrics"][p]["r2"] is None else f"{blk['metrics'][p]['r2']:.2f}"
                           for p in PARAMS)
        rows.append(f"| `{name}` | {blk['n']:,} | {cells} |")
    return ("Saved report, R² per source. \"fixed\" means the parameter does not vary in that source, "
            "so R² is undefined.\n\n" + "\n".join(rows))


def _ridge(r: dict) -> str:
    g = r["ridge_baseline"]
    rows = ["| Parameter | Ridge, 400 inputs | CosmUFR | Higher |", "|---|---:|---:|---|"]
    for p in PARAMS:
        e = g["params"].get(p, {})
        if e.get("pinned") or e.get("ridge_r2") is None:
            continue
        rows.append(f"| {SYMBOL[p].replace(' (eV)', '')} | {e['ridge_r2']:.3f} | {e['cosmufr_r2']:.3f} | {e['winner']} |")
    return (f"Saved report: ridge fitted on {g['n_fit']:,} benchmark rows and both models scored on the other "
            f"{g['n_test']:,} (seed {g['seed']}), R² on the held-out half. CosmUFR is higher on "
            f"{g['cosmufr_wins']} of {g['n_compared']}.\n\n" + "\n".join(rows))


def _modules(r: dict) -> str:
    m = r["modules"]

    def c(x):
        return "—" if x is None else f"{x['grad_batches']}/{x['of']}"

    rows = ["| Module | What it does | Parameters | Identical to Run 2 before any step | Gradient in Run 4 training code | Current code, k_backprop 4 | Current code, k_backprop 16 |",
            "|---|---|---:|---:|---:|---:|---:|"]
    for name, e in m["by_module"].items():
        ident = e["run2_initial_vs_run4"]
        ident_txt = "—" if ident is None else f"{ident['identical']}/{ident['shared']} tensors"
        t = e["training_step"]
        rows.append(f"| `{name}` | {MODULE_ROLE.get(name, '')} | {e['parameters']:,} ({e['share_percent']:.1f}%) | "
                    f"{ident_txt} | {c(t['run4_code_run4_weights'])} | {c(t['current_code_kbp4'])} | "
                    f"{c(t['current_code_kbp16'])} |")
    return (f"Diagnostic, run `{m['diagnosis_run_id']}`. Gradient columns count batches, of 20, in which the module "
            f"received a finite non-zero gradient from the revision's own training step. `attractor_bank` is updated "
            f"by moving average, not by gradient.\n\n" + "\n".join(rows))


TABLES = {
    "validation_metrics": _validation,
    "benchmark_metrics": _benchmark,
    "validation_per_source": _per_source,
    "ridge_baseline": _ridge,
    "modules": _modules,
}


def render(text: str, results: dict, path: Path) -> str:
    def repl(match):
        name = match.group(1)
        if name not in TABLES:
            raise SystemExit(f"{path.name}: unknown table marker {name!r}")
        return f"<!-- table:{name} -->\n{TABLES[name](results)}\n<!-- /table:{name} -->"

    return re.sub(r"<!-- table:([a-z_]+) -->.*?<!-- /table:\1 -->", repl, text, flags=re.S)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    results = json.loads(RESULTS.read_text(encoding="utf-8"))
    stale = []
    for path in FILES:
        old = path.read_text(encoding="utf-8")
        new = render(old, results, path)
        if new != old:
            if args.check:
                stale.append(path.name)
            else:
                path.write_text(new, encoding="utf-8")
                print(f"updated {path.name}")
    if stale:
        print("stale tables in: " + ", ".join(stale) + " — run python scripts/render_tables.py")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
