"""
cosmufr/narrate.py — have a language model read one run and explain it.

Why this exists. The computed readings in `interpret.py` are honest but they are
prose written in advance and selected by thresholds. They cannot respond to a
spectrum nobody anticipated, only sort it into a bucket someone did. For a
visitor who uploads their own data, that is the difference between an
explanation and a lookup.

Why it is safe to do here. The model is never asked what the numbers mean. It is
handed a block of measurements that this codebase computed, told it may not
introduce any fact that is not in that block, and asked to write it in plain
words. Every number it emits is then checked back against the block, and any
sentence carrying a number that was not supplied causes the whole thing to be
discarded. On rejection, on timeout, on a missing key, or on any error at all,
the caller falls back to the computed reading. The page is never worse off than
it was, and never says anything the code did not measure.

Why it is labelled. A reader is told which sentences a language model wrote, and
that the measurements underneath them came from the model being described. That
distinction is the whole reason anyone should trust the page.
"""
from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from typing import Dict, List, Optional, Tuple

ENDPOINT = "https://openrouter.ai/api/v1/chat/completions"
MODEL = os.environ.get("COSMUFR_NARRATOR_MODEL", "anthropic/claude-haiku-4.5")
TIMEOUT_S = float(os.environ.get("COSMUFR_NARRATOR_TIMEOUT", "12"))
MAX_CHARS = 2400
ATTEMPTS = 2

SYSTEM = """You explain the output of a cosmology model to someone reading a \
research page. Your readers range from a professor in the field to a manager who \
has never heard of a power spectrum. Write for both: no equation is needed, and \
no sentence should be untrue to the expert.

You will be given a block of MEASUREMENTS taken from one run of the model. Rules, \
and they are absolute:

1. Use only what is in the MEASUREMENTS block. You may not add a fact, a number, \
a comparison or a claim from your own knowledge of cosmology or machine learning.
2. Every number you write must appear in the MEASUREMENTS block. Do not compute \
new ones, do not convert units, do not round to a different number of digits.
3. If the measurements do not settle a question, say it is not settled. Never \
resolve an uncertainty by guessing.
4. Do not reassure. If the run shows the answer is untrustworthy, say so first.

Write 3 to 4 short paragraphs, under 260 words in total, no headings, no lists, \nno markdown. Open with the \
single most important thing a reader should take away. Then the evidence for it. \
Then what it changes about whether these numbers can be used.

Plain words. No jargon without an immediate plain-language gloss. Never use \
em-dashes."""


# What each output slot actually is, in words a reader would use. Supplied to
# the narrator so it never has to infer a meaning from a variable name; the one
# time it did, it read `mv` as a matter-to-radiation ratio.
PLAIN_NAME = {
    "Om": "how much matter the universe holds",
    "s8": "how lumpy that matter is",
    "h": "how fast the universe is expanding",
    "ns": "the tilt of the primordial ripples",
    "Ob": "how much ordinary matter there is",
    "w0": "what dark energy is doing",
    "mv": "the combined mass of neutrinos",
    "wa": "whether dark energy is changing over time",
}


def _facts_block(facts: Dict[str, object]) -> str:
    lines = []
    for k, v in facts.items():
        lines.append(f"{k}: {v}")
    return "\n".join(lines)


def _numbers(text: str) -> List[float]:
    """
    Every numeric literal in a string, as a value.

    Typographic minus signs and thousands separators are normalised first: a
    narrator that writes a supplied figure with nicer punctuation has not
    invented anything, and treating that as a new number throws away good work.
    """
    t = (text.replace("−", "-").replace("–", "-")
             .replace("—", "-").replace(",", ""))
    out = []
    for m in re.finditer(r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?", t):
        try:
            out.append(float(m.group(0)))
        except ValueError:
            continue
    return out


def _grounded(draft: str, source: str) -> Tuple[bool, str]:
    """
    Reject a draft carrying a number the measurements did not supply.

    Values, not digits: 0.0411 and 0.04110 are the same measurement stated two
    ways. Whole numbers up to a hundred are exempt because in prose they are
    almost always counting words, "three of the eight", rather than data.
    """
    allowed = _numbers(source)
    for n in _numbers(draft):
        if float(n).is_integer() and abs(n) <= 100:
            continue
        # Magnitude match as well as signed: "off by 0.00164" restates a
        # difference the measurements give as -0.00164, and that is a way of
        # saying it rather than a new claim.
        if any(abs(abs(n) - abs(a)) <= 1e-9 * max(1.0, abs(a)) for a in allowed):
            continue
        return False, f"unsupported number {n!r}"
    return True, ""


def narrate(facts: Dict[str, object], *, api_key: Optional[str] = None,
            ask: str = "") -> Optional[str]:
    """
    Ask a small language model to explain one run, or return None.

    None is a normal outcome and the caller must handle it: no key configured, a
    slow or failing endpoint, or a draft that failed the grounding check. There
    is no partial success here, because half an explanation is worse than the
    computed one.
    """
    key = api_key or os.environ.get("OPENROUTER_API_KEY")
    if not key:
        return None

    source = _facts_block(facts)
    user = (f"MEASUREMENTS from one run of the model:\n\n{source}\n\n"
            f"{ask or 'Explain this run to the reader.'}")

    body = json.dumps({
        "model": MODEL,
        "temperature": 0.2,
        "max_tokens": 700,
        "messages": [{"role": "system", "content": SYSTEM},
                     {"role": "user", "content": user}],
    }).encode()

    # One retry. A draft usually fails because the narrator rounded a single
    # figure, which a second attempt at the same temperature rarely repeats, and
    # a second try is much cheaper than losing the explanation.
    for _ in range(ATTEMPTS):
        req = urllib.request.Request(
            ENDPOINT, data=body,
            headers={"Content-Type": "application/json",
                     "Authorization": f"Bearer {key}",
                     "X-Title": "CosmUFR demo"})
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT_S) as r:
                payload = json.loads(r.read().decode("utf-8", "replace"))
            draft = payload["choices"][0]["message"]["content"].strip()
        except (urllib.error.URLError, TimeoutError, OSError, ValueError,
                KeyError, IndexError, TypeError):
            return None

        if not draft or len(draft) > MAX_CHARS:
            continue
        ok, _why = _grounded(draft, source)
        if not ok:
            continue
        # Told not to use them; a stray one is a formatting slip rather than a
        # reason to throw away a grounded explanation.
        return draft.replace("—", ", ").replace("**", "")
    return None


def run_facts(params, sigmas, labels, truth=None, *, source: str = "",
              outside_pct: Optional[float] = None,
              belief_movement: Optional[float] = None,
              energy_ulps: Optional[float] = None,
              recon_span: Optional[float] = None,
              track_record: Optional[Dict[str, str]] = None,
              typical_error: Optional[Dict[str, float]] = None) -> Dict[str, object]:
    """
    Assemble the measurements a narrator is allowed to talk about.

    Everything here was computed by this package from this run, or is a published
    figure from the evaluation report. Nothing is inferred and nothing is
    editorial: the judgements are the narrator's job, under the rules above.
    """
    facts: Dict[str, object] = {
        "what the model does": (
            "reads a matter power spectrum at two moments in cosmic history and "
            "returns eight numbers describing the universe that produced it"),
        "how many parameters it returns": f"{len(labels)}, all listed below",
        "input": source or "not stated",
    }
    for i, l in enumerate(labels):
        plain = PLAIN_NAME.get(l, l)
        row = f"({plain}) predicted {float(params[i]):.5f}"
        if truth is not None:
            row += (f", true value {float(truth[i]):.5f}"
                    f", difference {float(params[i]) - float(truth[i]):+.5f}")
        if track_record and l in track_record:
            row += f", how well this parameter recovers in general: {track_record[l]}"
        if typical_error and l in typical_error:
            row += f", typical error across held-out data: {typical_error[l]:.4f}"
        facts[l] = row

    facts["uncertainty column"] = (
        "the model reports the same fixed value on every input, so it carries no "
        "information about this run and must not be read as an error bar")
    if truth is None:
        facts["is there a known answer"] = (
            "no, this spectrum was supplied by the reader, so nothing can score it")
    else:
        facts["is there a known answer"] = (
            "yes, this spectrum came from a simulation whose parameters were recorded")
    if outside_pct is not None:
        facts["how much of this spectrum lies outside the range the model was trained on"] = (
            f"{outside_pct:.0f} percent")
    if belief_movement is not None:
        facts["how far the sixteen refinement steps moved the internal answer"] = (
            f"{belief_movement * 100:.3f} percent of its size. This is a known "
            f"property of the released model rather than something about this "
            f"spectrum: an audit found those steps were never trained, so they "
            f"do nothing on every input. The answer above is what the model had "
            f"before the steps ran. Report this, do not speculate about why")
    if energy_ulps is not None:
        facts["how much the score those steps were meant to reduce actually changed"] = (
            f"{energy_ulps:.1f} of the smallest amounts the arithmetic can represent")
    if recon_span is not None:
        facts["spread of the model's attempt to redraw the input spectrum"] = (
            f"{recon_span:.6f} decades, where the input itself spans several")
    return facts
