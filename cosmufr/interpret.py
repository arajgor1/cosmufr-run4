"""
cosmufr/interpret.py — say what a particular figure shows, from its own numbers.

A caption that reads "the settling trace is flat" is a claim about the released
checkpoint, not about the run in front of you. Hand the demo an unusual spectrum
and static prose can quietly become false.

Everything here is computed from the result being displayed. Each function
measures first and then chooses its sentences, so the reading is true for
whatever was actually run, including a spectrum the model has never seen and a
future checkpoint where these defects are fixed.

Returns plain strings. No HTML, so the notebook and the site can share them.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

from cosmufr.load import PARAM_LABELS

# Each parameter's prior width, used to express drift in units a reader can
# compare across parameters. Percent-of-value is unstable for w_a and m_nu,
# which sit near zero.
PRIOR_RANGE = {"Om": 0.40, "s8": 0.48, "h": 0.20, "ns": 0.12,
               "Ob": 0.04, "w0": 0.40, "mv": 0.40, "wa": 1.00}

NICE = {"Om": "matter density", "s8": "clustering amplitude",
        "h": "expansion rate", "ns": "spectral index", "Ob": "baryon density",
        "w0": "dark energy equation of state", "mv": "neutrino mass",
        "wa": "dark energy evolution"}


def _fmt(x: float, n: int = 3) -> str:
    return f"{x:.{n}f}"


def read_settling(report) -> List[str]:
    """What this particular settling trace shows."""
    out: List[str] = []
    move = report.belief_movement * 100
    ulp = report.energy_drop_in_ulps
    traj = report.param_trajectory
    drift = {l: abs(float(traj[-1, i] - traj[0, i])) / PRIOR_RANGE[l] * 100
             for i, l in enumerate(PARAM_LABELS)}
    worst, worst_pct = max(drift.items(), key=lambda kv: kv[1])

    if move < 1.0 and abs(ulp) <= 4:
        if abs(ulp) < 0.5:
            # Saying "0 steps, the smallest representable change" is wrong:
            # zero is not the smallest change, it is the absence of one.
            out.append(
                "Read the top panel first. Over all sixteen iterations the "
                "energy did not change at all. Not by a little: the number "
                "that the refinement exists to minimise came out bit-for-bit "
                "identical at every step.")
        else:
            out.append(
                f"Read the top panel first. The energy moved by {abs(ulp):.0f} "
                f"float32 step{'s' if abs(ulp) != 1 else ''} across all sixteen "
                f"iterations, which is the smallest change a number of that "
                f"magnitude can represent. It did not descend; it rounded.")
        out.append(
            f"The bottom panel says the same thing in the answer rather than "
            f"the objective. The parameter that moved most was "
            f"{NICE[worst]}, by {worst_pct:.2f}% of its allowed range. Every "
            f"other one moved less. The answer you get at step 16 is the "
            f"answer you already had at step 0.")
        out.append(
            f"So for this spectrum the refinement is decoration: the belief "
            f"shifted {move:.3f}% of its length and nothing downstream noticed. "
            f"Whatever accuracy the model has, it comes from the read-out head, "
            f"not from iterating.")
    elif move < 1.0:
        out.append(
            f"The energy fell by {abs(ulp):.0f} float32 steps while the belief "
            f"moved only {move:.3f}% of its length. Something is happening to "
            f"the objective, but almost nothing is happening to the state, "
            f"which usually means the step size is far below the scale of the "
            f"landscape.")
        out.append(
            f"The largest parameter movement was {NICE[worst]}, at "
            f"{worst_pct:.2f}% of its range.")
    else:
        out.append(
            f"The belief moved {move:.2f}% of its length over sixteen steps and "
            f"the energy changed by {abs(ulp):.0f} float32 steps. This trace is "
            f"doing real work, which is what the architecture was designed for.")
        out.append(
            f"{NICE[worst].capitalize()} moved most, by {worst_pct:.2f}% of its "
            f"allowed range. Watching which parameter moves during settling is "
            f"the point of this figure: it shows what the refinement is "
            f"actually deciding.")
        out.append(
            "Note that this contradicts the audit in section 05, which is "
            "measured on the released checkpoint. If you are seeing this on the "
            "released weights, please report it.")
    return out


def read_pk(k, pk_z0, pk_z047, pk_recon, log_k=None) -> List[str]:
    """What the reconstruction panel shows for this particular spectrum."""
    k = np.asarray(k, dtype=float)
    a = np.asarray(pk_z0, dtype=float)
    b = np.asarray(pk_z047, dtype=float)
    r = np.asarray(pk_recon, dtype=float)
    if (a > 100).any():
        a = np.log10(np.clip(a, 1e-30, None))
    if (b > 100).any():
        b = np.log10(np.clip(b, 1e-30, None))

    k_r = np.exp(np.asarray(log_k, dtype=float)) if log_k is not None else k
    r_on_k = np.interp(np.log(k), np.log(k_r), r)
    resid = a - r_on_k

    in_span = float(a.max() - a.min())
    rec_span = float(r.max() - r.min())
    growth = float(np.median(a - b))
    out: List[str] = []

    out.append(
        f"Your input falls by {in_span:.2f} decades between the largest and "
        f"smallest scales shown, which is the ordinary shape of a matter power "
        f"spectrum: the universe is smoother on big scales than small ones.")

    if growth > 0.01:
        out.append(
            f"The two input curves are separated by about {growth:.2f} decades, "
            f"with z=0 above z=0.47. That gap is structure growth over the "
            f"intervening 4.7 billion years, and it is the only handle the model "
            f"has on how fast the universe expanded.")
    else:
        out.append(
            f"The two input curves sit almost on top of each other "
            f"({abs(growth):.3f} decades apart). There is essentially no growth "
            f"signal in this pair, so the model has little to separate the "
            f"parameters that depend on it.")

    if rec_span < 1e-3:
        crossings = np.where(np.diff(np.sign(resid)))[0]
        where = (f" It happens to equal your spectrum near k = "
                 f"{k[crossings[0]]:.2f} h/Mpc, which is coincidence, not a fit."
                 if len(crossings) else "")
        out.append(
            f"Now the dashed line. It varies by {rec_span:.1e} decades across "
            f"all 200 scales: it is a horizontal line. The model is not "
            f"reconstructing your spectrum, it is emitting one number and "
            f"ignoring k entirely.{where}")
        out.append(
            f"That is why the residual below is just your input turned upside "
            f"down, running from {resid[0]:+.2f} at the largest scale to "
            f"{resid[-1]:+.2f} at the smallest. A residual that traces the input "
            f"is the signature of a model that has learned the average and "
            f"nothing else.")
    else:
        worst = int(np.argmax(np.abs(resid)))
        out.append(
            f"The dashed reconstruction varies by {rec_span:.2f} decades and "
            f"tracks the input. The residual below is what it is missing: "
            f"largest at k = {k[worst]:.2f} h/Mpc, where it is off by "
            f"{resid[worst]:+.2f} decades.")
        out.append(
            "Structure left in a residual is the interesting part of this "
            "figure: it is where the model's picture of the universe and your "
            "data disagree.")
    return out


def read_parameters(params: np.ndarray, sigmas: np.ndarray,
                    truth: Optional[np.ndarray] = None) -> List[str]:
    """What the parameter table shows for this run."""
    out: List[str] = []
    params = np.asarray(params, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)

    at_floor = int(np.sum(np.abs(sigmas - 0.1) < 1e-6))
    if at_floor >= 5:
        out.append(
            f"{at_floor} of the 8 reported uncertainties are exactly 0.1, the "
            f"smallest value the uncertainty head can emit. They are a clamp "
            f"constant, not a measurement of confidence, and they will be the "
            f"same for any spectrum you give it.")

    if truth is None:
        out.append(
            "There is no truth to compare against here, because this spectrum "
            "came from you rather than from the benchmark. Nothing on this page "
            "can tell you whether these numbers are right. Pick a built-in "
            "example if you want a scored run.")
        return out

    truth = np.asarray(truth, dtype=float)
    frac = {l: abs(params[i] - truth[i]) / PRIOR_RANGE[l]
            for i, l in enumerate(PARAM_LABELS)}
    good = [l for l, v in frac.items() if v < 0.02]
    poor = [l for l, v in frac.items() if v > 0.10]
    best = min(frac.items(), key=lambda kv: kv[1])
    worst = max(frac.items(), key=lambda kv: kv[1])

    out.append(
        f"Against the truth, {len(good)} of 8 parameters landed within 2% of "
        f"their allowed range. The closest was {NICE[best[0]]}, off by "
        f"{best[1]*100:.1f}% of its range; the furthest was {NICE[worst[0]]}, "
        f"off by {worst[1]*100:.1f}%.")

    if poor:
        names = ", ".join(NICE[l] for l in poor)
        out.append(
            f"The ones to distrust on this run are {names}. That ordering is "
            f"not random: the power spectrum constrains the amount and lumpiness "
            f"of matter directly, and says much less about the expansion rate "
            f"and what dark energy is doing.")
    else:
        out.append(
            "Nothing missed badly on this particular universe, which is a good "
            "run rather than a general claim. The accuracy section has the "
            "distribution over 162,795 of them.")
    return out
