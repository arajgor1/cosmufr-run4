"""
cosmufr/benchmark.py — the bundled, reproducible evaluation set.

Why this ships inside the repo
------------------------------
The Run 4 numbers originally published for this model were computed on a
validation split that lived only on the author's training infrastructure. Nobody
could check them. This module removes that problem: the benchmark is a
deterministic 6,000-row subsample of the master validation split, committed to
the repository, and `evaluate()` regenerates the released table from it.

If your numbers differ from `reports/honest_eval.json`, that is a bug worth
reporting, not an expected difference.

Usage
-----
>>> import cosmufr
>>> bench  = cosmufr.load_benchmark()
>>> model  = cosmufr.load_model(ckpt_path="best.pt")
>>> result = cosmufr.evaluate(model, bench)
>>> print(result.table())
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from cosmufr.load import PARAM_LABELS

# Numeric id of each training source, as recorded in the master corpus.
SOURCE_NAMES = {
    0: "camb_nl", 1: "bacco", 2: "bcemu", 3: "dark_emulator", 4: "spk",
    5: "bacco_neutrino", 6: "bacco_full8", 7: "bcemu_neutrino",
    8: "bacco_multiz", 10: "camels_illtng", 11: "camels_simba",
    12: "camels_astrid_x", 13: "camels_astrid", 14: "quijote_lh",
    15: "quijote_neutrino", 16: "camb_wa_grid", 17: "multi_z_grid",
    18: "ns_grid", 19: "w0_fluid_grid",
}

# Sources with a documented data defect. Kept in the benchmark so the
# aggregate stays honest, but flagged so the per-source view explains the gap.
KNOWN_DEFECTIVE = {
    8: "bacco_multiz: the z=0.47 spectra are self-paired copies of the z=0 "
       "spectra, so the two input channels are identical and carry no growth "
       "information. Documented in the training corpus source registry.",
}

# R2 divides by the variance of the truth. Where a parameter is held at a
# fiducial constant that variance is ~0, so R2 stops measuring skill and is
# driven entirely by its denominator: an early pass of this evaluation
# reported R2 = -3.2e8 for h on dark_emulator for exactly this reason. Slices
# below this threshold are reported as undefined instead.
PINNED_STD_THRESHOLD = 1e-4

_BENCHMARK_PATH = Path(__file__).resolve().parent.parent / "benchmark" / "cosmufr_benchmark.npz"


def hf_download(repo_id: str, filename: str, token=None) -> str:
    """Fetch a file from the HuggingFace model repo, with a clear error."""
    try:
        from huggingface_hub import hf_hub_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "huggingface_hub is required to download the benchmark. "
            "Install it, or clone the repository which ships the file."
        ) from e
    return hf_hub_download(repo_id=repo_id, filename=filename, token=token)


@dataclass
class Benchmark:
    """The bundled evaluation set."""
    pk_z0: np.ndarray        # (N, 200) raw P(k) at z=0
    pk_z047: np.ndarray      # (N, 200) raw P(k) at z=0.47
    params: np.ndarray       # (N, 8) true parameters, PARAM_LABELS order
    source_lid: np.ndarray   # (N,) source id, see SOURCE_NAMES
    k_bins: np.ndarray       # (200,) k grid in h/Mpc
    note: str = ""

    def __len__(self) -> int:
        return len(self.params)

    def source_counts(self) -> Dict[str, int]:
        vals, counts = np.unique(self.source_lid, return_counts=True)
        return {SOURCE_NAMES.get(int(v), f"lid_{v}"): int(c)
                for v, c in zip(vals, counts)}


BENCHMARK_FILENAME = "cosmufr_benchmark.npz"


def load_benchmark(path: Optional[str] = None) -> Benchmark:
    """
    Load the benchmark.

    Resolution order: an explicit `path`, then the copy committed in the
    repository, then a download from the HuggingFace model repo. The last
    fallback matters for anyone who ran `pip install git+...` rather than
    cloning, since a wheel does not carry the repository's data directories.
    """
    from cosmufr.load import HF_REPO_ID

    if path:
        p = Path(path)
    elif _BENCHMARK_PATH.exists():
        p = _BENCHMARK_PATH
    else:
        import os
        p = Path(hf_download(HF_REPO_ID, BENCHMARK_FILENAME,
                             token=os.environ.get("HF_TOKEN")))

    if not p.exists():
        raise FileNotFoundError(
            f"Benchmark not found at {p}. It ships with the repository at "
            f"https://github.com/arajgor1/cosmufr-run4 and alongside the "
            f"weights at https://huggingface.co/{HF_REPO_ID}"
        )
    d = np.load(p, allow_pickle=False)
    return Benchmark(
        pk_z0=d["pk_z0"], pk_z047=d["pk_z047"], params=d["params"],
        source_lid=d["source_lid"], k_bins=d["k_bins"], note=str(d["note"]),
    )


def _r2(t: np.ndarray, p: np.ndarray) -> Optional[float]:
    ss_tot = float(((t - t.mean()) ** 2).sum())
    if ss_tot <= 0:
        return None
    return 1.0 - float(((t - p) ** 2).sum()) / ss_tot


def _block(truth: np.ndarray, pred: np.ndarray) -> Dict[str, dict]:
    out = {}
    for i, lbl in enumerate(PARAM_LABELS):
        t, p = truth[:, i], pred[:, i]
        pinned = bool(t.std() < PINNED_STD_THRESHOLD)
        out[lbl] = {
            "r2": None if pinned else _r2(t, p),
            "rmse": float(np.sqrt(((t - p) ** 2).mean())),
            "truth_std": float(t.std()),
            "pinned": pinned,
            "n": int(len(t)),
        }
    return out


@dataclass
class EvalResult:
    """Metrics from a benchmark run."""
    overall: Dict[str, dict]
    varying_only: Dict[str, dict]
    per_source: Dict[str, dict]
    predictions: np.ndarray
    sigmas: np.ndarray
    truth: np.ndarray
    source_lid: np.ndarray
    n: int = 0
    notes: List[str] = field(default_factory=list)

    def table(self) -> str:
        """The released table, as text."""
        def f(x, w=13):
            return f"{'n/a':>{w}}" if x is None else f"{x:>{w}.4f}"

        lines = [
            f"CosmUFR Run 4 on the bundled benchmark (N = {self.n:,})",
            "",
            f"{'param':<7}{'R2 (all)':>12}{'R2 where it varies':>21}{'RMSE':>11}",
            "-" * 51,
        ]
        for lbl in PARAM_LABELS:
            o, v = self.overall[lbl], self.varying_only[lbl]
            lines.append(f"{lbl:<7}{f(o['r2'],12)}{f(v['r2'],21)}"
                         f"{o['rmse']:>11.5f}")
        lines += [
            "",
            "'R2 where it varies' excludes sources that hold the parameter at a",
            "fiducial constant. R2 is a ratio against the variance of the truth,",
            "so on a pinned slice it measures nothing. m_nu is the parameter this",
            "matters most for: it looks competent in the left column only because",
            "most of the corpus has m_nu = 0.",
        ]
        return "\n".join(lines)

    def source_table(self) -> str:
        """Per-source R2. This is where the aggregate is explained."""
        def f(x, w=8):
            return f"{'--':>{w}}" if x is None else f"{x:>{w}.2f}"

        lines = [
            f"{'source':<18}{'n':>7}  " + "".join(f"{p:>8}" for p in PARAM_LABELS),
            "-" * (27 + 8 * len(PARAM_LABELS)),
        ]
        for name, blk in sorted(self.per_source.items(),
                                key=lambda kv: -kv[1]["n"]):
            row = "".join(f(blk["metrics"][p]["r2"]) for p in PARAM_LABELS)
            flag = "  <- known data defect" if blk.get("known_defective") else ""
            lines.append(f"{name:<18}{blk['n']:>7,}  {row}{flag}")
        lines += ["", "'--' means the parameter is pinned in that source, so R2 is undefined."]
        return "\n".join(lines)


def evaluate(model, bench: Optional[Benchmark] = None,
             batch_size: int = 128, verbose: bool = False) -> EvalResult:
    """
    Run `model` over the benchmark and compute the released metrics.

    Parameters
    ----------
    model : CosmUFRLite
        A loaded model, e.g. from `cosmufr.load_model()`.
    bench : Benchmark, optional
        Defaults to the bundled benchmark.
    batch_size : int
        Forward-pass batch size. The settling loop needs autograd with respect
        to the belief, so memory scales with this.
    """
    import torch

    if bench is None:
        bench = load_benchmark()

    device = next(model.parameters()).device
    preds, sigmas = [], []

    for start in range(0, len(bench), batch_size):
        a = bench.pk_z0[start:start + batch_size]
        b = bench.pk_z047[start:start + batch_size]
        obs = np.concatenate(
            [np.log10(np.clip(a, 1e-30, None)), np.log10(np.clip(b, 1e-30, None))],
            axis=1,
        ).astype(np.float32)
        # inference_mode(False) matters: SettlingCore takes autograd gradients
        # with respect to the belief, which inference_mode would forbid.
        with torch.inference_mode(False), torch.no_grad():
            out = model(torch.from_numpy(obs).to(device), return_full=False)
        preds.append(out["params"].detach().cpu().numpy())
        sigmas.append(out["variances"].detach().cpu().numpy() ** 0.5)
        if verbose:
            print(f"  {min(start+batch_size, len(bench)):,} / {len(bench):,}")

    pred = np.concatenate(preds)
    sig = np.concatenate(sigmas)
    truth = bench.params
    lids = bench.source_lid

    per_source, notes = {}, []
    for src in sorted(set(lids.tolist())):
        m = lids == src
        if m.sum() < 50:
            continue
        name = SOURCE_NAMES.get(int(src), f"lid_{src}")
        per_source[name] = {
            "lid": int(src), "n": int(m.sum()),
            "metrics": _block(truth[m], pred[m]),
            "known_defective": int(src) in KNOWN_DEFECTIVE,
        }
        if int(src) in KNOWN_DEFECTIVE:
            notes.append(KNOWN_DEFECTIVE[int(src)])

    varying = {}
    for i, lbl in enumerate(PARAM_LABELS):
        keep = np.zeros(len(truth), dtype=bool)
        for src in sorted(set(lids.tolist())):
            m = lids == src
            if m.sum() >= 50 and truth[m, i].std() >= PINNED_STD_THRESHOLD:
                keep |= m
        if keep.sum() < 50:
            varying[lbl] = {"r2": None, "n": int(keep.sum()), "rmse": None}
        else:
            varying[lbl] = {
                "r2": _r2(truth[keep, i], pred[keep, i]),
                "rmse": float(np.sqrt(((truth[keep, i] - pred[keep, i]) ** 2).mean())),
                "n": int(keep.sum()),
            }

    return EvalResult(
        overall=_block(truth, pred), varying_only=varying, per_source=per_source,
        predictions=pred, sigmas=sig, truth=truth, source_lid=lids,
        n=len(bench), notes=sorted(set(notes)),
    )
