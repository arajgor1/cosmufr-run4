"""
cosmufr/explain.py — what each figure and number actually means.

Kept out of server.py so the notebook, the demo and any future frontend
describe the same result the same way. A figure a reader has to decode for
themselves is a figure that gets skimmed.
"""
from __future__ import annotations

FIGURE_NOTES = {
    "settling": {
        "title": "Belief settling",
        "input": "The 1024-dimensional belief vector proposed from your spectrum.",
        "output": "The energy at each of the 16 refinement steps, and the eight "
                  "parameters read off the belief at every step.",
        "means": "This is the figure the architecture exists for. The model is "
                 "supposed to refine its belief by descending a learned energy. "
                 "Both panels are flat. The top panel is drawn in units of "
                 "float32 resolution: a trace inside the shaded band is not a "
                 "small descent, it is a change too small for the number to "
                 "represent. The refinement runs and does nothing.",
        "why": "The energy heads collapsed to an input-independent constant "
               "during training, so there is no landscape to descend, and the "
               "networks that would steer the descent never received a "
               "gradient at all.",
    },
    "pk": {
        "title": "Power spectrum",
        "input": "Your two spectra, P(k) at z=0 and z=0.47, on 200 log-spaced "
                 "k bins from 0.1 to 4.5 h/Mpc.",
        "output": "The GenerativeHead's attempt to reconstruct log10 P(k), "
                  "overlaid, with the residual below.",
        "means": "The dashed line is flat. The head returns the same number at "
                 "every k, for every input spectrum, and for a random belief "
                 "vector. It is not reconstructing anything. Its reported "
                 "error of 0.687 is just the variance of log10 P(k) about a "
                 "constant, which is what any predictor that ignores its input "
                 "would score.",
        "why": "The head trained, unlike the encoder, but it converged to "
               "emitting a constant rather than a function of the belief.",
    },
    "weight_audit": {
        "title": "Which modules ever received a gradient",
        "input": "The released checkpoint's weights, nothing else.",
        "output": "For each module, the fraction of its Linear layers whose "
                  "bias is still bit-exactly zero.",
        "means": "PyTorch initialises a Linear bias from a random uniform "
                 "draw, and any optimizer step moves it. A module whose biases "
                 "are all still exactly 0.0 after forty epochs never received "
                 "a gradient. Four modules are in that state, three of them "
                 "the belief pipeline this architecture is named for.",
        "why": "SettlingCore.forward detaches the belief at the start of every "
               "step, which severs everything upstream of it from the loss.",
    },
    "recovery": {
        "title": "Parameter recovery",
        "input": "All 6,000 spectra in the bundled benchmark.",
        "output": "Predicted against true value, one panel per parameter, with "
                  "the diagonal marking a perfect prediction.",
        "means": "Tight scatter along the diagonal means the parameter is "
                 "recovered. A horizontal band means the model is predicting "
                 "roughly the same value regardless of truth, which is what "
                 "you see for w_a and, once restricted to data where it "
                 "varies, for the neutrino mass.",
        "why": "",
    },
    "per_source": {
        "title": "R-squared by training source",
        "input": "The benchmark, split by which simulation suite each spectrum "
                 "came from.",
        "output": "R-squared per parameter per source. '--' means the "
                  "parameter is held at a fixed value in that source, so "
                  "R-squared is undefined rather than bad.",
        "means": "This explains the headline. The aggregate is dragged down by "
                 "bacco_multiz, which is 15 percent of the set and scores zero "
                 "on everything because its two redshift channels are "
                 "identical copies. On sources with sound data the model "
                 "reaches 0.98 to 0.99 on the matter density.",
        "why": "",
    },
    "uncertainty": {
        "title": "Reported uncertainty against actual error",
        "input": "Every prediction on the benchmark, paired with the error it "
                 "actually made.",
        "output": "Reported sigma on the horizontal axis, actual absolute "
                  "error on the vertical.",
        "means": "A working uncertainty head produces a cloud rising to the "
                 "right: bigger claimed uncertainty where the error is bigger. "
                 "This produces a vertical line, because the head reports the "
                 "same number every time.",
        "why": "The uncertainty head's output is driven below its clamp floor, "
               "so what comes out is the floor constant, not a prediction.",
    },
}

PARAM_MEANING = {
    "Om": ("Matter density", "Fraction of the universe's energy budget in "
                             "matter. Sets how much structure forms."),
    "s8": ("Clustering amplitude", "How lumpy matter is on 8 Mpc/h scales. "
                                   "Directly visible in the spectrum's height."),
    "h": ("Hubble parameter", "Expansion rate today, in units of 100 km/s/Mpc. "
                              "Enters the spectrum only through its shape."),
    "ns": ("Spectral index", "Tilt of the primordial spectrum. A small, "
                             "broadband effect."),
    "Ob": ("Baryon density", "Ordinary matter. Its main signature is the "
                             "acoustic wiggles, which are weak at these scales."),
    "w0": ("Dark energy equation of state", "Present-day pressure-to-density "
                                            "ratio. -1 is a cosmological constant."),
    "mv": ("Sum of neutrino masses", "Massive neutrinos suppress small-scale "
                                     "power. The model does not recover this."),
    "wa": ("Dark energy evolution", "How the equation of state changes with "
                                    "time. The hardest parameter here."),
}
