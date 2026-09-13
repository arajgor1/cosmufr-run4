"""
Negative findings are scoped as carefully as positive ones.

An earlier correction pass replaced overclaims about the model with overclaims
about its failure: that the refinement loop "does nothing", that the answers
come from "a fixed random projection", that the energy landscape is flat because
its stored value barely changes, and that the reconstruction is constant "for
every input". The released code shows the loop follows the gradient of trained
energy heads and changes every benchmark prediction slightly; an additive offset
changes an energy's value without changing its gradient; and a constant output
was checked on the benchmark, not on every possible input.

These tests fail if that wording returns anywhere a reader meets it: the model
card, README, site source, Gradio app, package strings, or the quickstart
notebook's sources and stored outputs. They check wording only. Numerical
consistency and scientific correctness are separate questions.

The check is deliberately literal. A correct sentence that negates one of these
phrases ("not evidence that the landscape is flat") should be reworded so it does
not contain the phrase, rather than weakening the test.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

BANNED = [
    "does nothing", "do nothing", "did nothing", "no measurable work",
    "nothing to descend", "no landscape to descend", "landscape is flat",
    "fixed random projection", "fixed random function",
    "never operated", "did not operate", "does not operate", "non-operating",
    "carries no information", "carry no information", "learn nothing",
    "compute the reported parameters", "produces the answers",
    "the energy is constant", "energy is flat",
    "for every input spectrum", "at every k and for every", "for every input.",
    "not changing it", "never moved at all", "is the answer it had at the start",
    "the same number no matter which spectrum",
]


def _public_texts():
    files = [ROOT / "README.md", ROOT / "MODEL_CARD.md", ROOT / "server.py", ROOT / "app.py"]
    files += sorted((ROOT / "cosmufr").glob("*.py"))
    for f in files:
        yield f.name, f.read_text(encoding="utf-8")
    nb = json.loads((ROOT / "examples" / "01_quickstart.ipynb").read_text(encoding="utf-8"))
    for i, cell in enumerate(nb["cells"]):
        yield f"notebook cell {i} source", "".join(cell["source"])
        for out in cell.get("outputs", []):
            text = out.get("text", [])
            text = "".join(text) if isinstance(text, list) else str(text)
            plain = out.get("data", {}).get("text/plain", [])
            plain = "".join(plain) if isinstance(plain, list) else str(plain)
            yield f"notebook cell {i} output", text + plain


def _normalise(text: str) -> str:
    # Prose in Python sources is split across adjacent string literals and lines.
    text = re.sub(r'"\s*\n\s*f?"', "", text)
    return re.sub(r"\s+", " ", text).lower()


def test_no_overstated_negative_findings():
    hits = []
    for name, text in _public_texts():
        flat = _normalise(text)
        for phrase in BANNED:
            if phrase in flat:
                hits.append(f"{name}: {phrase!r}")
    assert not hits, "overstated negative findings:\n  " + "\n  ".join(hits)


def test_settling_effect_is_labelled_as_not_a_benefit_test():
    results = json.loads((ROOT / "reports" / "results_v1.json").read_text(encoding="utf-8"))
    se = results["settling_effect"]
    assert se["provenance"] == "diagnostic"
    assert "not a test of predictive benefit" in se["note"]
    assert se["rows_with_any_prediction_change"] > 0


def test_notebook_records_how_it_was_executed():
    nb = json.loads((ROOT / "examples" / "01_quickstart.ipynb").read_text(encoding="utf-8"))
    assert "cosmufr_reexecuted" in nb.get("metadata", {})
