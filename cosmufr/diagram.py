"""
cosmufr/diagram.py — the architecture, drawn.

One function, two modes. `architecture_svg()` draws the data path: two spectra
in, an encoder, a belief, sixteen refinement steps, three read-out heads, eight
parameters out. `architecture_svg(audit=True)` draws the identical diagram with
each block coloured by what the released weights actually show.

Using the same picture twice is deliberate. The second pass is the argument: a
reader who has just followed how the model is supposed to work then sees exactly
which parts of it never did.

The SVG paints its own dark background rather than inheriting the page's. That
costs nothing on the site and means the same file stays readable when GitHub
renders it in a light-themed README.
"""
from __future__ import annotations

BG = "#0B0B0E"
C_EDGE = "rgba(255,255,255,.30)"
C_TEXT = "#E8EAEE"
C_MUT = "#9aa1ad"
C_DIM = "#6f7783"
C_OK = "#3FBF8F"
C_BAD = "#E2643B"
C_WARN = "#E6A23C"
C_ACC = "#60a5fa"

# What the audit found, per block. Kept beside the drawing so the picture and
# the claim cannot drift apart.
VERDICT = {
    "obs": ("never trained", C_BAD),
    "prop": ("never trained", C_BAD),
    "settle": ("never trained", C_BAD),
    "energy": ("trained, went flat", C_WARN),
    "param": ("trained, works", C_OK),
    "unc": ("stuck at its floor", C_WARN),
    "gen": ("returns a constant", C_WARN),
}

W, H = 1020, 366
BOX_H = 54


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def _box(x, y, w, title, sub, edge=C_EDGE, dashed=False, h=BOX_H, glow=False):
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    fill = "rgba(255,255,255,.045)" if not glow else "rgba(96,165,250,.10)"
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" '
        f'stroke="{edge}" stroke-width="1.5"{dash}/>',
        f'<text x="{x + w/2:.0f}" y="{y + 23:.0f}" fill="{C_TEXT}" font-size="12.5" '
        f'font-weight="600" text-anchor="middle" '
        f'font-family="Inter,Segoe UI,system-ui,sans-serif">{_esc(title)}</text>',
    ]
    if sub:
        out.append(
            f'<text x="{x + w/2:.0f}" y="{y + 39:.0f}" fill="{C_DIM}" font-size="9.5" '
            f'text-anchor="middle" font-family="ui-monospace,Menlo,monospace">{sub}</text>')
    return "".join(out)


def _arrow(x1, y1, x2, y2):
    return (f'<line x1="{x1:.0f}" y1="{y1:.0f}" x2="{x2:.0f}" y2="{y2:.0f}" '
            f'stroke="{C_EDGE}" stroke-width="1.5" marker-end="url(#ar)"/>')


def _txt(x, y, text, colour=C_DIM, size=9.5, anchor="middle", mono=True, weight=400):
    fam = ("ui-monospace,Menlo,monospace" if mono
           else "Inter,Segoe UI,system-ui,sans-serif")
    return (f'<text x="{x:.0f}" y="{y:.0f}" fill="{colour}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}" font-family="{fam}">{text}</text>')


def architecture_svg(audit: bool = False) -> str:
    """
    audit=False : neutral, for explaining how the model is meant to work.
    audit=True  : each block coloured and annotated with what its weights show.
    """
    def edge(k):
        return VERDICT[k][1] if audit else C_EDGE

    def dash(k):
        return audit and VERDICT[k][1] == C_BAD

    def note(k, x, y):
        if not audit:
            return ""
        t, c = VERDICT[k]
        return _txt(x, y, t, c, 9)

    MID = 150            # baseline for the main chain
    ty = MID - BOX_H / 2  # 123

    p = [f'<svg viewBox="0 0 {W} {H}" xmlns="http://www.w3.org/2000/svg" '
         f'role="img" aria-label="CosmUFR inference path">',
         '<defs><marker id="ar" viewBox="0 0 10 10" refX="9" refY="5" '
         'markerWidth="7" markerHeight="7" orient="auto-start-reverse">'
         f'<path d="M 0 0 L 10 5 L 0 10 z" fill="{C_EDGE}"/></marker></defs>',
         f'<rect x="0" y="0" width="{W}" height="{H}" rx="12" fill="{BG}"/>']

    # ── inputs ──────────────────────────────────────────────────────────
    p.append(_txt(76, 74, "INPUT", C_ACC, 9, weight=600))
    p.append(_box(20, 90, 112, "P(k) at z = 0", "200 values", C_ACC, glow=True, h=48))
    p.append(_box(20, 148, 112, "P(k) at z = 0.47", "200 values", C_ACC, glow=True, h=48))
    p.append(_arrow(132, 114, 166, 140))
    p.append(_arrow(132, 172, 166, 160))

    # ── encode / propose ────────────────────────────────────────────────
    p.append(_box(168, ty, 124, "ObsEncoder", "400 &#8594; 1024", edge("obs"), dash("obs")))
    p.append(note("obs", 230, ty + BOX_H + 16))
    p.append(_arrow(292, MID, 320, MID))
    p.append(_box(322, ty, 128, "BeliefProposal", "the first guess", edge("prop"), dash("prop")))
    p.append(note("prop", 386, ty + BOX_H + 16))
    p.append(_arrow(450, MID, 478, MID))

    # ── settling ────────────────────────────────────────────────────────
    p.append(_box(480, ty - 8, 160, "SettlingCore", "16 refinement steps",
                  edge("settle"), dash("settle"), h=BOX_H + 16))
    p.append(_txt(560, ty + 58, "the research idea", C_MUT, 9.5))
    p.append(note("settle", 560, ty + BOX_H + 34))

    p.append(_box(480, 268, 160, "Energy heads", "define the landscape",
                  edge("energy"), False, h=46))
    p.append(note("energy", 560, 330))
    # Offset to the right of centre so it does not run through the "never
    # trained" caption that sits under the settling box in audit mode.
    p.append(_arrow(612, 268, 612, ty + BOX_H + 10))

    p.append(_arrow(640, MID, 672, MID))
    p.append(_txt(656, MID - 10, "b*", C_MUT, 10))

    # ── read-out heads ──────────────────────────────────────────────────
    p.append(_txt(748, 40, "READ-OUT", C_ACC, 9, weight=600))
    p.append(_box(674, 50, 148, "ParameterHead", "8 parameters", edge("param")))
    p.append(_box(674, ty, 148, "UncertaintyHead", "8 uncertainties", edge("unc")))
    p.append(_box(674, 214, 148, "GenerativeHead", "rebuild P(k)", edge("gen")))
    p.append(_arrow(672, MID, 674, 77))
    p.append(_arrow(672, MID, 674, MID))
    p.append(_arrow(672, MID, 674, 241))

    # ── outputs ─────────────────────────────────────────────────────────
    p.append(_arrow(822, 77, 852, 77))
    p.append(_box(854, 50, 146, "Cosmology",
                  "&#937;m &#963;8 h ns &#937;b w&#8320; m&#957; wa",
                  C_OK if audit else C_ACC, glow=not audit))

    if audit:
        p.append(_txt(846, MID + 4, "ignore &#8212; a constant", C_WARN, 9, anchor="start"))
        p.append(_txt(846, 241, "ignore &#8212; a constant", C_WARN, 9, anchor="start"))
        p.append(_txt(510, 350,
                      "Of the seven trained blocks, one does useful work.",
                      C_TEXT, 11.5, mono=False, weight=500))
    else:
        p.append(_txt(846, MID + 4, "reported &#963;", C_DIM, 9, anchor="start"))
        p.append(_txt(846, 241, "reconstruction", C_DIM, 9, anchor="start"))
        p.append(_txt(510, 350,
                      "One forward pass. No simulator in the loop, no chain to converge.",
                      C_MUT, 11.5, mono=False))

    p.append("</svg>")
    return "".join(x for x in p if x)


if __name__ == "__main__":  # pragma: no cover
    from pathlib import Path
    out = Path(__file__).resolve().parent.parent / "docs"
    out.mkdir(exist_ok=True)
    (out / "architecture.svg").write_text(architecture_svg(), encoding="utf-8")
    (out / "architecture_audit.svg").write_text(architecture_svg(audit=True),
                                                encoding="utf-8")
    print(f"wrote {out / 'architecture.svg'}")
    print(f"wrote {out / 'architecture_audit.svg'}")
