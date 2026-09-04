"""
The test that would have caught this model's central defect on day one.

CosmUFR trained for months with its observation encoder, belief proposal
network and settling core receiving zero gradient. Nothing in the loss curves
showed it: the read-out heads learned to read the frozen random projection
well enough that the headline metrics looked reasonable, and several auxiliary
losses sat at constants that were read as convergence.

A single synthetic training step, checking that every module's gradient norm
is non-zero and within a sane ratio of the others, would have surfaced it
immediately. It costs about a second on CPU.

These tests document the defect as it currently stands. `test_encoder_is_frozen`
asserts the KNOWN-BAD state on purpose, so that if a future checkpoint fixes
the gradient path this test fails loudly and tells you to update the docs.
"""
from __future__ import annotations

import os

import numpy as np
import pytest
import torch

import cosmufr
from cosmufr.config import CosmUFRConfig
from cosmufr.diagnostics import DEFAULT_PATH_MODULES
from cosmufr.model import CosmUFRLite

CKPT = os.environ.get("COSMUFR_CKPT", "_local_ckpt/best.pt")
have_ckpt = os.path.exists(CKPT)
needs_ckpt = pytest.mark.skipif(not have_ckpt, reason=f"checkpoint not found at {CKPT}")


def _one_training_step_grads(model):
    """Run one synthetic forward/backward and return per-module gradient norms."""
    model.train()
    model.zero_grad(set_to_none=True)

    torch.manual_seed(0)
    obs = torch.randn(8, model.cfg.d_obs) * 0.5 + 2.0
    target = torch.rand(8, model.cfg.n_cosmo_params)

    out = model(obs, return_full=False)
    loss = torch.nn.functional.mse_loss(out["params"], target)
    loss.backward()

    norms = {}
    for name, p in model.named_parameters():
        if p.grad is None:
            continue
        top = name.split(".")[0]
        norms[top] = norms.get(top, 0.0) + float(p.grad.norm()) ** 2
    return {k: v ** 0.5 for k, v in norms.items()}


def test_only_the_parameter_head_receives_gradient():
    """
    The defect, demonstrated from the architecture alone in about a second.

    No checkpoint required. One synthetic training step on a freshly
    initialized model shows that `param_head` is the ONLY module that receives
    any gradient. The encoder, the belief proposal and the settling core do not
    appear in the gradient dictionary at all, because `SettlingCore.forward`
    calls `b.detach()` on entry to every step and passes `z.detach()` into the
    energy, severing the belief pipeline from the loss.

    This test passing is the bad news. `test_gradient_reaches_every_module`
    below is the same fact stated as the goal.
    """
    model = CosmUFRLite(CosmUFRConfig())
    grads = _one_training_step_grads(model)

    assert "param_head" in grads and grads["param_head"] > 0
    for severed in ("obs_encoder", "belief_proposal", "settling"):
        assert severed not in grads or grads[severed] == 0.0, (
            f"{severed} now receives gradient. If the detach in "
            f"SettlingCore.forward was removed deliberately, this is the fix "
            f"working: update README.md, MODEL_CARD.md and cosmufr/__init__.py."
        )


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEFECT: SettlingCore.forward detaches the belief every step, "
           "so no gradient reaches obs_encoder, belief_proposal or settling. "
           "This is the target state, not the current one. When the gradient "
           "path is repaired this test XPASSes and the suite fails on purpose, "
           "as a reminder to update the documented defect list.",
)
def test_gradient_reaches_every_module():
    """The state this model should be in. Currently it is not."""
    model = CosmUFRLite(CosmUFRConfig())
    grads = _one_training_step_grads(model)

    for module in ["obs_encoder", "belief_proposal", "settling", "param_head"]:
        assert module in grads, f"{module} received no gradient at all"
        assert grads[module] > 0.0, f"{module} gradient norm is exactly zero"


@pytest.mark.xfail(
    strict=True,
    reason="KNOWN DEFECT: the encoder receives no gradient at all, so the "
           "ratio is infinite. Same target state as above.",
)
def test_gradient_magnitudes_are_within_a_sane_ratio():
    """
    A module receiving gradient five or more orders of magnitude below the
    heads is effectively frozen even though it is technically non-zero. That is
    what the Run 8 "unfreezing" attempt produced: encoder gradients arriving at
    1e-5 against 5.6 for the parameter head, which reads as "learning slowly"
    and is actually "not learning".
    """
    model = CosmUFRLite(CosmUFRConfig())
    grads = _one_training_step_grads(model)

    head = grads["param_head"]
    enc = grads.get("obs_encoder", 0.0)
    assert head > 0, "parameter head received no gradient"
    ratio = head / max(enc, 1e-30)
    assert ratio < 1e4, (
        f"encoder gradient is {ratio:.1e}x below the parameter head. "
        f"At that scale the encoder does not train in any practical number "
        f"of epochs."
    )


@needs_ckpt
def test_encoder_is_frozen_in_the_released_checkpoint():
    """
    Asserts the KNOWN DEFECT of the shipped weights.

    This test passing is not good news. It records that the released
    checkpoint has an untrained encoder, belief proposal and settling core.
    If it ever fails, a checkpoint with a working gradient path has been
    dropped in and README.md, MODEL_CARD.md and cosmufr/__init__.py all need
    updating before release.
    """
    model = cosmufr.load_model(ckpt_path=CKPT)
    audit = cosmufr.weight_audit(model)

    frozen = set(audit.untrained_on_default_path)
    assert {"obs_encoder", "belief_proposal", "settling"} <= frozen, (
        "The released checkpoint no longer has a frozen belief pipeline. "
        "That is a fix, not a failure: update the documented defect list."
    )
    for m in frozen:
        assert m in DEFAULT_PATH_MODULES


@needs_ckpt
def test_settling_does_not_move_the_belief():
    """Records the measured no-op, so a real fix trips this test."""
    model = cosmufr.load_model(ckpt_path=CKPT)
    bench = cosmufr.load_benchmark()
    report = cosmufr.settling_report(model, bench.pk_z0[0], bench.pk_z047[0])

    assert report.belief_movement < 0.01, (
        f"Belief now moves {report.belief_movement*100:.2f}% during settling. "
        f"If this is a genuine fix, update the documented defect list."
    )
    assert abs(report.energy_drop_in_ulps) < 100, (
        "Settling energy now descends by more than float32 noise. Update docs."
    )


@needs_ckpt
def test_uncertainties_are_the_clamp_floor():
    """Records that reported sigmas carry no information."""
    model = cosmufr.load_model(ckpt_path=CKPT)
    bench = cosmufr.load_benchmark()

    sig = []
    for i in range(0, 64, 32):
        r = cosmufr.infer(bench.pk_z0[i], bench.pk_z047[i], model=model)
        sig.append(r.sigmas_array)
    sig = np.stack(sig)

    audit = cosmufr.uncertainty_audit(sig)
    pinned = [p for p, m in audit.per_param.items() if m["frac_at_floor"] == 1.0]
    assert len(pinned) >= 5, (
        f"Only {len(pinned)} parameters sit at the clamp floor now. "
        f"If the uncertainty head was retrained, update the documented defects."
    )
