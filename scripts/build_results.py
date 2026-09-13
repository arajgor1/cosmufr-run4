"""
scripts/build_results.py — assemble reports/results_v1.json, the single source
every published table is rendered from.

Maintainer script. Its inputs are the evidence files from the September 2026
pre-outreach audit, kept outside this repository because the historical
checkpoints and training source they depend on are not public. The output is
committed, and every table in README.md, MODEL_CARD.md and the site is generated
from it rather than typed.

Every block carries a provenance label:

  reproduced    regenerated in the audit from the released checkpoint and the
                bundled benchmark
  saved_report  read from a report saved earlier; not rerun in the audit
  diagnostic    measured in the audit by a run that is not part of the release
  saved_log     summarised in the audit from a log saved at the time
  historical    the author's documentation of past runs; no primary record located

Usage:
    python scripts/build_results.py [--model-work PATH] [--out reports/results_v1.json]
"""
from __future__ import annotations

import argparse
import datetime
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MW = Path("C:/UFR/ChatGPT Work/September_Update/Model_Work")
PARAMS = ["Om", "s8", "h", "ns", "Ob", "w0", "mv", "wa"]
N_PARAMS = 136194617
LABELS = {"reproduced", "saved_report", "diagnostic", "saved_log", "historical"}


def _sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _metric_block(block: dict, keys=("r2", "rmse", "n", "truth_std")) -> dict:
    return {p: {k: block[p].get(k) for k in keys} for p in PARAMS}


def _verdict(r2):
    if r2 is None:
        return "undefined"
    if r2 > 0.7:
        return "recovered"
    if r2 > 0.25:
        return "partial"
    return "not recovered"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model-work", default=str(DEFAULT_MW))
    ap.add_argument("--out", default=str(ROOT / "reports" / "results_v1.json"))
    args = ap.parse_args()
    mw = Path(args.model_work)

    sources = {
        "reference_metrics": mw / "reference_metrics.json",
        "gradient_report": mw / "gradient_report.json",
        "optimizer_membership": mw / "optimizer_membership.json",
        "checkpoint_lineage": mw / "evidence" / "checkpoint_lineage_compare.json",
        "halo_head": mw / "evidence" / "halo_head_weight_change.json",
        "head_movement": mw / "evidence" / "run2_run4_head_movement.json",
        "run8_smoke": mw / "evidence" / "run8_smoke_unfrozen_multiz_grad_summary.json",
        "benchmark_checks": mw / "evidence" / "benchmark_data_checks.json",
        "master_headers": mw / "evidence" / "master_array_headers.json",
        "run4_training_log": mw / "evidence" / "gcs" /
            "results__v10__b200-b4096-run4-20260414_031247__training_results.json",
        "honest_eval": ROOT / "reports" / "honest_eval.json",
        "ridge_baseline": ROOT / "reports" / "ridge_baseline.json",
        "settling_k0_k16": mw / "evidence" / "settling_k0_k16_benchmark.json",
        "independent_k0_k16": mw.parent / "Worker_Review_Evidence" / "independent_K0_K16_check.json",
    }
    d = {k: _load(p) for k, p in sources.items()}
    sk, ik = d["settling_k0_k16"], d["independent_k0_k16"]
    ref, honest, ridge = d["reference_metrics"], d["honest_eval"], d["ridge_baseline"]

    # ── module accounting from the Run 4 revision's optimizer membership ────
    mem = d["optimizer_membership"]["configs"]["run4_launch_384ad38__run4_weights"]["optimizer_membership"]
    params_by_module = {m: e["params"] for m, e in mem.items()}
    params_by_module["settling"] = params_by_module.pop("settling.precond") + params_by_module.pop("settling.eta_net")
    if sum(params_by_module.values()) != N_PARAMS:
        raise SystemExit(f"module parameter counts sum to {sum(params_by_module.values())}, not {N_PARAMS}")

    grad = d["gradient_report"]["configs"]

    def conn(label, module):
        s = grad[label]["summary"]
        if module == "settling":
            a, b = s["settling.precond"], s["settling.eta_net"]
            return {"grad_batches": min(a["batches_with_finite_nonzero_grad"], b["batches_with_finite_nonzero_grad"]),
                    "update_batches": min(a["batches_with_nonzero_update"], b["batches_with_nonzero_update"]),
                    "of": a["batches_completed"]}
        e = s[module]
        return {"grad_batches": e["batches_with_finite_nonzero_grad"],
                "update_batches": e["batches_with_nonzero_update"], "of": e["batches_completed"]}

    lineage = d["checkpoint_lineage"]["pairs"]

    def ident(pair, module):
        e = lineage[pair].get(module)
        return None if e is None else {"identical": e["identical"], "shared": e["shared"]}

    modules = {}
    for m in sorted(params_by_module, key=lambda k: -params_by_module[k]):
        modules[m] = {
            "parameters": params_by_module[m],
            "share_percent": round(100 * params_by_module[m] / N_PARAMS, 2),
            "run2_initial_vs_run4": ident("run2_phase0_final__vs__run4_best", m),
            "training_step": {
                "run4_code_run4_weights": conn("run4_launch_384ad38__run4_weights", m) if m != "attractor_bank" else None,
                "run4_code_fresh_init": conn("run4_launch_384ad38__fresh_init", m) if m != "attractor_bank" else None,
                "current_code_kbp4": conn("head_9053bbe__run4_weights__kbp4", m) if m != "attractor_bank" else None,
                "current_code_kbp16": conn("head_9053bbe__run4_weights__kbp16", m) if m != "attractor_bank" else None,
            },
        }

    unchanged = ["obs_encoder", "belief_proposal", "settling"]
    unchanged_params = sum(params_by_module[m] for m in unchanged)

    log = d["run4_training_log"]
    energy_traj = [{"epoch": e["epoch"], "train_L_energy": e["train_L_energy"]} for e in log]

    ps_val = {name: {"n": blk["n"], "metrics": _metric_block(blk["metrics"])}
              for name, blk in honest["per_source_metrics"].items()}
    ps_bench = {name: {"n": blk["n"], "metrics": _metric_block(blk["metrics"])}
                for name, blk in ref["metrics"]["per_source"].items()}

    vary_val = honest["full_val_metrics_varying_only"]
    results = {
        "schema": "cosmufr-results/1",
        "generated_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
        "provenance_labels": sorted(LABELS),
        "inputs_sha256": {k: _sha(p) for k, p in sources.items()},

        "checkpoint": {
            "provenance": "reproduced",
            "sha256": ref["inputs"]["checkpoint_sha256"],
            "parameters": ref["parameters"],
            "phase": 4, "epoch": 30,
            "run_id": "b200-b4096-run4-20260414_031247",
            "launch_revision": "384ad38 (inferred from run id timestamp; training step unchanged through bc1faab)",
        },

        "benchmark": {
            "provenance": "reproduced",
            "run_id": ref["run_id"],
            "rows": 6000,
            "sha256": ref["inputs"]["benchmark_sha256"],
            "environment": ref["environment"],
            "agreement": {
                "max_abs_r2_diff_vs_independent_review": max(v["abs_diff"] for v in ref["checks"]["overall_r2_vs_review"].values()),
                "max_abs_r2_diff_vs_published_benchmark_block": max(v["abs_diff"] for v in ref["checks"]["overall_r2_vs_published_benchmark_block"].values()),
                "max_abs_prediction_diff_vs_independent_review": ref["prediction_max_abs_diff_vs_review"],
                "repeat_bit_identical": ref["deterministic_repeat_bit_identical"],
                "tolerance": 1e-5,
            },
            "overall": _metric_block(ref["metrics"]["overall"]),
            "varying_only": _metric_block(ref["metrics"]["varying_only"]),
            "per_source": ps_bench,
            "fixed_parameter_fraction": {
                "w0_at_minus_1": d["benchmark_checks"]["benchmark_fraction_w0_at_minus1"],
                "wa_at_0": d["benchmark_checks"]["benchmark_fraction_wa_at_0"],
                "mv_at_0": d["benchmark_checks"]["benchmark_fraction_mv_at_0"],
            },
            "redshift_channel_checks": {
                "window": "median over k < 0.2 h/Mpc of log10 P(z=0) - log10 P(z=0.47), per row, then per source",
                "per_source": d["benchmark_checks"]["per_source"],
                "rows_with_negative_gap": d["benchmark_checks"]["rows_with_negative_lowk_gap_total"],
            },
        },

        "validation_report": {
            "provenance": "saved_report",
            "file": "reports/honest_eval.json",
            "generated_utc": honest["generated_utc"],
            "rows": honest["val_split"]["n"],
            "source_ids_in_split": honest["val_split"]["n_sources"],
            "sources_with_metrics": len(honest["per_source_metrics"]),
            "split_rule": honest["val_split"]["rule"],
            "externally_reproducible": False,
            "needed_to_reproduce": ["master validation rows (data/master)", "bad_indices.npy", "the split rule above"],
            "overall": _metric_block(honest["full_val_metrics"]),
            "varying_only": _metric_block(vary_val),
            "verdict_rule": "R2 where the parameter varies: above 0.7 recovered, 0.25 to 0.7 partial, below 0.25 not recovered",
            "verdicts": {p: _verdict(vary_val[p]["r2"]) for p in PARAMS},
            "per_source": ps_val,
            "diagnostics": honest["diagnostics"],
        },

        "ridge_baseline": {
            "provenance": "saved_report",
            "file": "reports/ridge_baseline.json",
            "note": "independently re-derived with the same seed by the September 2026 review",
            "seed": ridge["seed"], "n_fit": ridge["n_fit"], "n_test": ridge["n_test"],
            "cosmufr_wins": ridge["cosmufr_wins"], "n_compared": ridge["n_compared"],
            "params": ridge["params"],
        },

        "modules": {
            "provenance": "diagnostic",
            "note": ("parameter counts from the Run 4 architecture; identity against Run 2's checkpoint saved "
                     "before any optimizer step; training-step connectivity from the M01 diagnosis "
                     "(batches with a finite non-zero gradient / a non-zero update, of 20)"),
            "diagnosis_run_id": d["gradient_report"]["run_id"],
            "unchanged_from_initialization": unchanged,
            "unchanged_parameters": unchanged_params,
            "unchanged_share_percent": round(100 * unchanged_params / N_PARAMS, 2),
            "by_module": modules,
            "halo_head_weight_rescale_run2_to_run4": sorted(
                {round(v["norm_ratio_run2_run4"], 4) for v in d["halo_head"].values()
                 if v["norm_ratio_run2_run4"] == v["norm_ratio_run2_run4"]}),
            "head_movement_run2_to_run4_package_mean": {
                m: round(v["package_mean_relative_diff_all_tensors"], 3)
                for m, v in d["head_movement"].items() if m in ("param_head", "unc_head", "gen_head")},
        },

        "energy": {
            "provenance": "diagnostic",
            "objective": "mean(E_star) + mean(relu(0.5 + E_star - E_neg))",
            "loss_change_for_shift_minus_1000": {
                label: c["energy_offset_test"]["change_for_shift_1e3"] for label, c in grad.items()},
            "run4_training_log": {"provenance": "saved_log", "trajectory": energy_traj},
            "benchmark_energy_values": {
                "provenance": "reproduced",
                "rows": {k: v["energy_step0"] for k, v in ref["diagnostics"]["settling"].items()},
                "max_abs_change_over_16_steps_in_float32_steps": max(
                    abs(v["energy_drop_in_float32_ulps"]) for v in ref["diagnostics"]["settling"].values()),
            },
        },

        "settling": {
            "provenance": "reproduced",
            "belief_movement_benchmark_rows": {k: v["belief_movement"] for k, v in ref["diagnostics"]["settling"].items()},
            "validation_report_mean": honest["diagnostics"]["settling_rel_movement"]["mean"],
        },

        "uncertainty": {
            "provenance": "reproduced",
            "fraction_at_clamp_floor_benchmark": ref["diagnostics"]["uncertainty_frac_at_floor"],
        },

        "reconstruction": {
            "provenance": "diagnostic",
            "scope": "generative head output on all 6,000 benchmark rows at the default 16 steps",
            "environment": sk["environment"],
            "rows": sk["reconstruction_k16"]["rows"],
            "value_min": sk["reconstruction_k16"]["min"],
            "value_max": sk["reconstruction_k16"]["max"],
            "max_std_across_k": sk["reconstruction_k16"]["max_std_across_k"],
            "max_std_across_rows": sk["reconstruction_k16"]["max_std_across_rows"],
        },

        "settling_effect": {
            "provenance": "diagnostic",
            "scope": sk["scope"],
            "environment": sk["environment"],
            "rows": 6000,
            "k_settle": sk["k_settle_default"],
            "what_the_loop_does": ("each step takes the gradient of the trained energy heads with respect to "
                                   "the belief and applies it through the untrained step-size and "
                                   "preconditioner networks"),
            "relative_belief_movement": sk["relative_belief_movement_k16"],
            "rows_with_any_prediction_change": sk["rows_with_any_prediction_change"],
            "prediction_change": sk["prediction_change_k16_minus_k0"],
            "r2_varying_k0": {p: sk["scores_k0"][p]["r2_varying"] for p in PARAMS},
            "r2_varying_k16": {p: sk["scores_k16"][p]["r2_varying"] for p in PARAMS},
            "r2_varying_difference": {p: sk["score_difference_k16_minus_k0"][p]["r2_varying"] for p in PARAMS},
            "energy_checked_rows": 64,
            "energy_gradient_norm_at_start": sk["energy_gradient_at_b_hat_float32"],
            "first_step_update_relative_to_belief": sk["first_step_update_relative_to_belief"],
            "energy_float32_distinct_values_at_start": sk["energy_float32_at_b_hat"]["unique_values"],
            "energy_float64_change_over_16_steps": sk["energy_float64_change_b_star_minus_b_hat"],
            "independent_check": {
                "provenance": "diagnostic",
                "source": "independent review, 13 September 2026: first 8 benchmark rows, torch " + ik["torch"],
                "relative_belief_movement_range": [min(ik["relative_belief_movement"]), max(ik["relative_belief_movement"])],
                "max_abs_prediction_change": dict(zip(PARAMS, ik["max_abs_prediction_change_by_parameter"])),
                "exact_prediction_equality": ik["exact_prediction_equality"],
            },
            "note": ("The loop executes and changes the belief and every prediction slightly. This is not a "
                     "test of predictive benefit: one benchmark, one checkpoint, no uncertainty on the "
                     "differences, and no matched comparison."),
        },

        "later_training_code_saved_log": {
            "provenance": "saved_log",
            "run": d["run8_smoke"]["source_gcs"],
            "logged_steps": d["run8_smoke"]["grad_summary"]["grad_encoder"]["n"],
            "median_grad_norm": {k.replace("grad_", ""): v["median"] for k, v in d["run8_smoke"]["grad_summary"].items()},
            "fraction_of_steps_zero": {k.replace("grad_", ""): v["frac_zero"] for k, v in d["run8_smoke"]["grad_summary"].items()},
        },

        "data_rows": {
            "historical_run4_training_documentation": {"provenance": "historical", "rows": "84.5 million"},
            "gcs_master": {"provenance": "reproduced", "rows": d["master_headers"]["data/master/"]["params_master.npy"]["shape"][0]},
            "gcs_master_v2": {"provenance": "reproduced", "rows": d["master_headers"]["data/master_v2/"]["params_master.npy"]["shape"][0],
                              "note": "built 2026-06-12, after Run 4; used by no released model"},
            "master_v2_sampled_channel_order": {
                "provenance": "saved_report",
                "source": "independent review, 12 September 2026, section 3 (648 sampled rows, 27 strata)",
                "median_lowk_log10_ratio": {"camb_nl": -0.220, "camb_wa_grid": -0.300},
                "scope": "sample only; no source-wide audit",
            },
        },
    }

    def check_labels(obj):
        if isinstance(obj, dict):
            if "provenance" in obj and obj["provenance"] not in LABELS:
                raise SystemExit(f"unknown provenance label {obj['provenance']!r}")
            for v in obj.values():
                check_labels(v)
        elif isinstance(obj, list):
            for v in obj:
                check_labels(v)

    check_labels(results)
    out = Path(args.out)
    out.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
