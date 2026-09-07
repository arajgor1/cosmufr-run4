"""
cosmufr/interpret.py — say what a figure means, from its own numbers.

Two rules here.

First, everything is computed from the result being displayed. A caption reading
"the settling trace is flat" is a claim about the released checkpoint, not about
the run in front of you: hand the demo an unfamiliar spectrum and fixed prose can
quietly become false. Each function measures, then chooses its sentences.

Second, lead with the conclusion. A reader looking at a chart wants to know what
it means for whether they can use the answer, not what is plotted on each axis.
Every reading opens with a bottom line in plain words, then the evidence, then
what it changes about trusting the numbers. Someone who has never seen a power
spectrum should be able to follow it, without the sentences becoming untrue for
someone who has.

Each function returns a list of (kind, text). `kind` is "bottom" for the
headline, "body" for supporting text, so a caller can style them differently.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from cosmufr.load import PARAM_LABELS

Reading = List[Tuple[str, str]]

# Prior width per parameter. Errors are expressed as a fraction of this, so they
# are comparable across parameters; percent-of-value is unstable for w_a and
# m_nu, which sit near zero.
PRIOR_RANGE = {"Om": 0.40, "s8": 0.48, "h": 0.20, "ns": 0.12,
               "Ob": 0.04, "w0": 0.40, "mv": 0.40, "wa": 1.00}

NICE = {"Om": "how much matter the universe holds",
        "s8": "how lumpy that matter is",
        "h": "how fast the universe is expanding",
        "ns": "the tilt of the primordial ripples",
        "Ob": "how much ordinary matter there is",
        "w0": "what dark energy is doing",
        "mv": "the mass of neutrinos",
        "wa": "whether dark energy is changing over time"}

SHORT = {"Om": "matter density", "s8": "lumpiness", "h": "expansion rate",
         "ns": "spectral tilt", "Ob": "ordinary matter", "w0": "dark energy",
         "mv": "neutrino mass", "wa": "dark energy drift"}

# Values these parameters are held at when a suite does not vary them. Hitting
# one is not a recovery: the model can score it perfectly by always guessing the
# same number. Naming such a parameter as the run's best result is the pinned-
# parameter mistake the accuracy section warns about, made in plain English.
FIDUCIAL = {"w0": -1.0, "wa": 0.0, "mv": 0.0}


def _pinned(label: str, truth_value: float) -> bool:
    f = FIDUCIAL.get(label)
    return f is not None and abs(truth_value - f) < 1e-6


def read_settling(report) -> Reading:
    """What the settling panel means for this run."""
    move = report.belief_movement * 100
    ulp = report.energy_drop_in_ulps
    traj = report.param_trajectory
    drift = {l: abs(float(traj[-1, i] - traj[0, i])) / PRIOR_RANGE[l] * 100
             for i, l in enumerate(PARAM_LABELS)}
    worst, worst_pct = max(drift.items(), key=lambda kv: kv[1])
    working = move >= 1.0

    out: Reading = []
    if not working:
        out.append(("bottom",
                    "The model reached its answer immediately and then spent "
                    "sixteen steps not changing it."))
        out.append(("body",
                    "This model was designed to think in stages: make a rough "
                    "guess about the universe, then sharpen it sixteen times "
                    "before answering. These two panels check whether that "
                    "sharpening happens."))
        if abs(ulp) < 0.5:
            out.append(("body",
                        "The top panel is the model's own score for how wrong "
                        "it currently thinks it is. It is supposed to fall as "
                        "the model improves. Here it never moved at all, not "
                        "even by the smallest amount the computer can store."))
        else:
            out.append(("body",
                        f"The top panel is the model's own score for how wrong "
                        f"it currently thinks it is, and it is supposed to fall "
                        f"as the model improves. Here it changed by the "
                        f"smallest amount a number that size can store, which "
                        f"is rounding rather than progress."))
        out.append(("body",
                    f"The bottom panel is the eight answers themselves while "
                    f"it thinks. Flat again: the one that moved most was "
                    f"{SHORT[worst]}, by {worst_pct:.2f}% of its range, which "
                    f"would not change any conclusion. The answer at the end "
                    f"is the answer it had at the start."))
        out.append(("body",
                    "What this changes: the numbers above are not invalidated, "
                    "but they are not coming from the part of the design that "
                    "was supposed to produce them. They come from the final "
                    "read-out step alone. Section 01 explains why."))
    else:
        out.append(("bottom",
                    f"The refinement is doing real work on this run: the "
                    f"model's answer changed measurably as it thought."))
        out.append(("body",
                    f"The internal score fell over the sixteen steps and the "
                    f"belief moved {move:.2f}% of its length. The answer that "
                    f"shifted most was {SHORT[worst]}, by {worst_pct:.2f}% of "
                    f"its range: that is the model changing its mind, which is "
                    f"what the architecture was built to do."))
        out.append(("body",
                    "Note this contradicts the audit in section 01, which is "
                    "measured on the released checkpoint. If you are seeing "
                    "this on the released weights, please report it."))
    return out


def read_pk(k, pk_z0, pk_z047, pk_recon, log_k=None,
            reference: Optional[np.ndarray] = None,
            has_truth: bool = True) -> Reading:
    """
    What the reconstruction panel means for this spectrum.

    `reference`, if given, is an (N, 200) array of log10 P(k) at z=0 from the
    benchmark. It is used only to say whether the input looks like the data the
    model was trained on, which is the single most useful thing to tell someone
    who uploaded their own file.
    """
    k = np.asarray(k, dtype=float)
    a = np.asarray(pk_z0, dtype=float)
    b = np.asarray(pk_z047, dtype=float)
    r = np.asarray(pk_recon, dtype=float)
    if (a > 100).any():
        a = np.log10(np.clip(a, 1e-30, None))
    if (b > 100).any():
        b = np.log10(np.clip(b, 1e-30, None))

    k_r = np.exp(np.asarray(log_k, dtype=float)) if log_k is not None else k
    resid = a - np.interp(np.log(k), np.log(k_r), r)
    rec_span = float(r.max() - r.min())
    growth = float(np.median(a - b))
    flat = rec_span < 1e-3

    out: Reading = []
    if flat:
        out.append(("bottom",
                    "This is a comprehension check, and the model fails it."))
        out.append(("body",
                    "Having read your data, the model is asked to draw it back "
                    "from memory. If it understood the input, the dashed line "
                    "would follow the solid ones. It is a flat line: the model "
                    "returns a single number for every scale, and the same "
                    "number no matter which spectrum you give it. It is like "
                    "asking someone to sketch a photograph they just studied "
                    "and getting the same blank stroke every time."))
        out.append(("body",
                    f"The panel underneath is the gap between your data and "
                    f"that flat line. Because the line is constant, the gap is "
                    f"just your own curve upside down, running from "
                    f"{resid[0]:+.2f} at the largest scales to {resid[-1]:+.2f} "
                    f"at the smallest. There is no information in it."))
        out.append(("body",
                    "What this changes: it does not prove the parameters above "
                    "are wrong. It does mean one of the two independent ways we "
                    "had of checking them is unavailable, so they rest on the "
                    "accuracy table in section 04 alone."))
    else:
        worst = int(np.argmax(np.abs(resid)))
        out.append(("bottom",
                    "The model can redraw your spectrum, and where it cannot "
                    "is the interesting part."))
        out.append(("body",
                    f"The dashed line is the model's reconstruction from what "
                    f"it understood, and it tracks your data across "
                    f"{rec_span:.2f} decades. The panel underneath is what it "
                    f"missed: largest near k = {k[worst]:.2f}, where it is off "
                    f"by {resid[worst]:+.2f}. Leftover structure there is where "
                    f"the model's picture of the universe and your data "
                    f"disagree, which is exactly where you would look for "
                    f"something new."))

    # Does the input look like what the model was trained on? This is the most
    # useful sentence available when there is no truth to score against.
    if reference is not None and len(reference):
        lo = np.percentile(reference, 1, axis=0)
        hi = np.percentile(reference, 99, axis=0)
        outside = float(np.mean((a < lo) | (a > hi))) * 100
        if outside < 5:
            out.append(("body",
                        f"One thing in the model's favour here: your spectrum "
                        f"sits inside the range it was trained on at "
                        f"{100 - outside:.0f}% of scales. It is being asked "
                        f"about familiar territory."))
        elif outside < 40:
            out.append(("body",
                        f"Worth knowing: {outside:.0f}% of your spectrum falls "
                        f"outside the range the model saw in training. Treat "
                        f"the numbers above with more caution than the accuracy "
                        f"table would suggest."))
        else:
            out.append(("body",
                        f"Important: {outside:.0f}% of your spectrum lies "
                        f"outside anything the model was trained on. It will "
                        f"still return eight confident-looking numbers, and you "
                        f"should not believe them. Models asked about "
                        f"unfamiliar data tend to be confidently wrong."))

    if growth <= 0.01:
        out.append(("body",
                    f"Also note the two input curves sit almost on top of each "
                    f"other, {abs(growth):.3f} decades apart. The difference "
                    f"between them is how much the universe grew over 4.7 "
                    f"billion years, and it is the model's only clue about the "
                    f"expansion rate. With the two this close, that clue is "
                    f"largely absent."))
    return out


def read_parameters(params: np.ndarray, sigmas: np.ndarray,
                    truth: Optional[np.ndarray] = None) -> Reading:
    """What the parameter table means for this run."""
    params = np.asarray(params, dtype=float)
    sigmas = np.asarray(sigmas, dtype=float)
    out: Reading = []

    if truth is None:
        out.append(("bottom",
                    "Nothing on this page can tell you whether these eight "
                    "numbers are right."))
        out.append(("body",
                    "This spectrum came from you, so there is no known answer "
                    "to compare against. With one of the built-in examples the "
                    "true values are recorded, and the table shows you exactly "
                    "how close the model got. Here it cannot."))
    else:
        truth = np.asarray(truth, dtype=float)
        frac = {l: abs(params[i] - truth[i]) / PRIOR_RANGE[l]
                for i, l in enumerate(PARAM_LABELS)}
        good = [l for l, v in frac.items() if v < 0.05]
        poor = [l for l, v in frac.items() if v > 0.10]
        pinned = [l for i, l in enumerate(PARAM_LABELS) if _pinned(l, truth[i])]
        # Score the run on the parameters this universe actually varies.
        live = {l: v for l, v in frac.items() if l not in pinned}
        best = min(live.items(), key=lambda kv: kv[1]) if live else None

        out.append(("bottom",
                    f"On this universe the model got {len(good)} of the 8 "
                    f"close, and {len(poor)} clearly wrong."))
        if best is not None:
            out.append(("body",
                        f"Because this example came from a simulation we know "
                        f"the real answer, so the last column is a score rather "
                        f"than a guess. It did best on {SHORT[best[0]]}, landing "
                        f"within {best[1] * 100:.1f}% of the full range that "
                        f"quantity can take."))
        if pinned:
            names = ", ".join(SHORT[l] for l in pinned[:-1])
            names = (f"{names} and {SHORT[pinned[-1]]}" if len(pinned) > 1
                     else SHORT[pinned[0]])
            out.append(("body",
                        f"Discount {names} on this run. This universe was built "
                        f"with {'them' if len(pinned) > 1 else 'it'} held at the "
                        f"default value, so the model can score perfectly by "
                        f"always guessing the same number. A right answer here "
                        f"is not evidence of anything."))
        if poor:
            # Short labels inside a list; the long descriptive phrases read as
            # a run-on when three of them are strung together.
            names = ", ".join(SHORT[l] for l in poor[:-1])
            names = f"{names} and {SHORT[poor[-1]]}" if len(poor) > 1 else SHORT[poor[0]]
            out.append(("body",
                        f"The ones to distrust here are {names}. That is not "
                        f"bad luck. A power spectrum shows you the amount and "
                        f"the lumpiness of matter almost directly, and says "
                        f"much less about the expansion rate or dark energy, so "
                        f"those are the ones a model of this kind gets wrong "
                        f"first."))
        else:
            out.append(("body",
                        "Nothing missed badly here, which is one good run "
                        "rather than a general claim. Section 04 has the "
                        "distribution across 162,795 of them."))

    at_floor = int(np.sum(np.abs(sigmas - 0.1) < 1e-6))
    if at_floor >= 5:
        out.append(("body",
                    f"Ignore the uncertainty column entirely. {at_floor} of the "
                    f"8 read exactly 0.1, which is the smallest value that part "
                    f"of the model can output. It is a floor it never leaves, "
                    f"not a measure of confidence, and it will read the same for "
                    f"any spectrum you ever give it."))
    return out
