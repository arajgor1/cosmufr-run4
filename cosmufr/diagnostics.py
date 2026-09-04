"""
cosmufr/diagnostics.py — the audit that found this model's central defect.

Background
----------
CosmUFR was designed around a belief-settling core: encode the spectrum into a
1024-d belief, then refine it through 16 steps of gradient descent on a learned
energy. That refinement was the research claim.

In September 2026 an audit of the released checkpoint found that the encoder,
the belief proposal network and the settling core's own networks had received
**zero gradient** across the entire training programme. They sit at their
initialization. The trained parts of the model are the read-out heads, which
learn to read a fixed random projection of the input.

This module reproduces that audit, so a reader can confirm it in about a minute
rather than taking the claim on trust. Every function returns measurements, not
opinions.

Usage
-----
>>> import cosmufr
>>> model = cosmufr.load_model(ckpt_path="best.pt")
>>> print(cosmufr.weight_audit(model).table())
>>> print(cosmufr.settling_report(model, pk_z0, pk_z047).summary())
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np

from cosmufr.load import PARAM_LABELS


# ─────────────────────────────────────────────────────────────────────────────
# Weight audit
# ─────────────────────────────────────────────────────────────────────────────

# Modules exercised by the default two-redshift forward pass. The `_single`
# and `_seq` variants belong to a single-redshift path this release does not
# use, and they DID train -- so an audit that does not distinguish the two
# invites the fair objection "but the encoder shows movement."
DEFAULT_PATH_MODULES = {
    "obs_encoder", "belief_proposal", "settling", "param_head", "unc_head",
    "gen_head", "halo_head", "obs_energy_head", "constraint_head",
    "dyn_energy_head",
}


@dataclass
class WeightAudit:
    modules: Dict[str, dict]

    def table(self) -> str:
        lines = [
            "Per-module weight audit",
            "",
            "A Linear layer that has taken even one optimizer step essentially",
            "never has a bias of exactly 0.0. Modules below with every bias at",
            "exactly zero never received a gradient.",
            "",
            f"{'module':<22}{'on path':>9}{'n Linear':>10}{'biases == 0':>14}"
            f"{'max |bias|':>13}{'verdict':>12}",
            "-" * 80,
        ]
        for name, m in self.modules.items():
            lines.append(
                f"{name:<22}{('yes' if m['on_default_path'] else 'no'):>9}"
                f"{m['n_linear']:>10}"
                f"{m['n_zero_bias']:>7}/{m['n_linear']:<6}"
                f"{m['max_abs_bias']:>13.3e}{m['verdict']:>12}"
            )
        lines += [
            "",
            "'on path' marks the modules the default two-redshift inference "
            "actually runs.",
            "The obs_encoder_single / belief_proposal_seq pair belongs to an "
            "unused single-redshift",
            "path and did train, which is why this column matters.",
        ]
        return "\n".join(lines)

    @property
    def untrained_modules(self) -> List[str]:
        return [k for k, v in self.modules.items() if v["verdict"] == "UNTRAINED"]

    @property
    def untrained_on_default_path(self) -> List[str]:
        """The finding that matters: untrained modules the model actually uses."""
        return [k for k, v in self.modules.items()
                if v["verdict"] == "UNTRAINED" and v["on_default_path"]]


def weight_audit(model) -> WeightAudit:
    """
    Classify each top-level module as trained or untrained from its biases.

    PyTorch initializes Linear biases from a uniform distribution, and any
    optimizer step moves them off their starting value. A module whose biases
    are all bit-exactly 0.0 in a checkpoint that trained for tens of epochs
    therefore received no gradient at all.
    """
    import torch

    groups: Dict[str, List[torch.nn.Linear]] = {}
    for name, mod in model.named_modules():
        if isinstance(mod, torch.nn.Linear) and mod.bias is not None:
            top = name.split(".")[0]
            groups.setdefault(top, []).append(mod)

    out = {}
    for top, mods in sorted(groups.items()):
        biases = [m.bias.detach() for m in mods]
        n_zero = sum(1 for b in biases if bool((b == 0).all()))
        max_abs = max(float(b.abs().max()) for b in biases)
        out[top] = {
            "n_linear": len(mods),
            "n_zero_bias": n_zero,
            "max_abs_bias": max_abs,
            "verdict": "UNTRAINED" if n_zero == len(mods) else "trained",
            "on_default_path": top in DEFAULT_PATH_MODULES,
        }
    return WeightAudit(modules=out)


# ─────────────────────────────────────────────────────────────────────────────
# Settling report
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class SettlingReport:
    energy_log: List[float]
    param_trajectory: np.ndarray   # (17, 8)
    belief_movement: float         # ||b* - b_hat|| / ||b_hat||
    cosine_similarity: float
    b_hat: np.ndarray
    b_star: np.ndarray

    @property
    def energy_drop(self) -> float:
        return self.energy_log[0] - self.energy_log[-1]

    @property
    def energy_drop_in_ulps(self) -> float:
        """
        The energy drop expressed in float32 resolution steps.

        Energy sits near -9.3e5, where consecutive float32 values are ~0.06
        apart. A "drop" of a few ULP is not a small descent: it is the
        smallest change the number can represent, i.e. no descent at all.
        """
        ulp = float(np.spacing(np.float32(abs(self.energy_log[0]))))
        return self.energy_drop / ulp if ulp > 0 else float("nan")

    def summary(self) -> str:
        d = self.param_trajectory
        lines = [
            "Settling report (16 refinement steps)",
            "",
            f"  belief movement    : {self.belief_movement*100:.4f}% of norm",
            f"  cosine(b_hat, b*)  : {self.cosine_similarity:.6f}",
            f"  energy at step 0   : {self.energy_log[0]:.4f}",
            f"  energy at step 16  : {self.energy_log[-1]:.4f}",
            f"  energy drop        : {self.energy_drop:.6e}"
            f"   ({self.energy_drop_in_ulps:.1f} float32 ULP"
            f" at E = {self.energy_log[0]:.0f})",
            "",
            f"  {'param':<7}{'step 0':>12}{'step 16':>12}{'drift':>12}{'% of value':>13}",
            "  " + "-" * 54,
        ]
        for i, lbl in enumerate(PARAM_LABELS):
            a, b = float(d[0, i]), float(d[-1, i])
            drift = abs(b - a)
            pct = 100 * drift / (abs(a) + 1e-12)
            lines.append(f"  {lbl:<7}{a:>12.5f}{b:>12.5f}{drift:>12.2e}{pct:>12.3f}%")
        lines += [
            "",
            "Interpretation: the settling loop runs, but it moves the belief by a",
            "fraction of a percent and the energy changes by a couple of float32",
            "resolution steps -- the smallest change representable at that",
            "magnitude. The refinement does no measurable work. weight_audit()",
            "gives the reason: its networks never trained.",
        ]
        return "\n".join(lines)


def settling_report(model, pk_z0, pk_z047) -> SettlingReport:
    """
    Trace one inference through all 16 settling steps.

    Records the energy and the parameter read-out at every step, so the claim
    that refinement converges can be checked rather than assumed.
    """
    import torch

    def prep(x):
        t = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        return torch.log10(t.clamp(min=1e-30)) if (t > 100).any() else t

    device = next(model.parameters()).device
    obs = torch.cat([prep(pk_z0), prep(pk_z047)]).unsqueeze(0).to(device)
    cfg = model.cfg

    traj, energies = [], []
    with torch.inference_mode(False), torch.no_grad():
        z = model.obs_encoder(obs)
        b_prev = torch.zeros(1, cfg.d_b, device=device)
        b_hat = model.belief_proposal(z, b_prev)
        b = b_hat
        traj.append(model.param_head(b)[0].clone())
        energies.append(float(model.energy_fn(b, z, b_prev).mean()))

        for step in range(cfg.k_settle):
            b = b.detach()
            # The settling step needs a gradient with respect to the belief.
            # A blanket no_grad around inference would silently disable the
            # refinement and still return plausible numbers.
            with torch.enable_grad():
                b_g = b.requires_grad_(True)
                E = model.energy_fn(b_g, z.detach(), b_prev.detach())
                grad = torch.autograd.grad(E.sum(), b_g, create_graph=False)[0]
            grad = torch.clamp(grad, -cfg.belief_grad_clip, cfg.belief_grad_clip)
            with torch.no_grad():
                P = torch.sigmoid(model.settling.precond[step](b))
                P = cfg.P_min + (cfg.P_max - cfg.P_min) * P
                eta = cfg.eta_min + (cfg.eta_max - cfg.eta_min) * torch.sigmoid(
                    model.settling.eta_net[step](b))
            b = b - eta * P * grad.detach()
            traj.append(model.param_head(b)[0].clone())
            energies.append(float(model.energy_fn(b, z, b_prev).mean()))

    b_star = b
    move = float(torch.norm(b_star - b_hat) / torch.norm(b_hat))
    cos = float(torch.nn.functional.cosine_similarity(b_hat, b_star))
    return SettlingReport(
        energy_log=energies,
        param_trajectory=torch.stack(traj).cpu().numpy(),
        belief_movement=move, cosine_similarity=cos,
        b_hat=b_hat[0].cpu().numpy(), b_star=b_star[0].cpu().numpy(),
    )


# ─────────────────────────────────────────────────────────────────────────────
# Uncertainty audit
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class UncertaintyAudit:
    per_param: Dict[str, dict]
    clamp_floor: float

    def table(self) -> str:
        lines = [
            "Uncertainty audit",
            "",
            f"UncertaintyHead returns clamp(softplus(net(b)) + 1e-2, max=4.0),",
            f"so the smallest sigma it can emit is sqrt(1e-2) = {self.clamp_floor}.",
            "A parameter sitting at that floor for every input is reporting the",
            "clamp constant, not a prediction.",
            "",
            f"{'param':<7}{'mean sigma':>13}{'std':>12}{'at floor':>11}",
            "-" * 43,
        ]
        for lbl in PARAM_LABELS:
            m = self.per_param[lbl]
            lines.append(f"{lbl:<7}{m['mean']:>13.6f}{m['std']:>12.3e}"
                         f"{m['frac_at_floor']*100:>10.1f}%")
        return "\n".join(lines)


def uncertainty_audit(sigmas: np.ndarray, clamp_floor: float = 0.1) -> UncertaintyAudit:
    """Check how much of the uncertainty output is the clamp floor."""
    per = {}
    for i, lbl in enumerate(PARAM_LABELS):
        s = sigmas[:, i]
        per[lbl] = {
            "mean": float(s.mean()), "std": float(s.std()),
            "min": float(s.min()), "max": float(s.max()),
            "frac_at_floor": float((np.abs(s - clamp_floor) < 1e-6).mean()),
        }
    return UncertaintyAudit(per_param=per, clamp_floor=clamp_floor)


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint comparison
# ─────────────────────────────────────────────────────────────────────────────

def compare_checkpoints(path_a: str, path_b: str) -> Dict[str, dict]:
    """
    Compare two checkpoints tensor by tensor, grouped by module.

    This is the measurement that settled the question. Run 4 was warm-started
    from Run 3, itself warm-started from Run 2, then trained 40 further epochs.
    Comparing Run 2 against Run 4 shows the read-out heads moved by 66-79% in
    relative norm while the encoder, belief proposal and settling core were
    bit-identical: 204 of 204 tensors unchanged.
    """
    import torch
    from collections import defaultdict

    sa = torch.load(path_a, map_location="cpu", weights_only=False)
    sb = torch.load(path_b, map_location="cpu", weights_only=False)
    sa = sa["model"] if isinstance(sa, dict) and "model" in sa else sa
    sb = sb["model"] if isinstance(sb, dict) and "model" in sb else sb

    groups = defaultdict(lambda: {"n": 0, "identical": 0, "rel": []})
    for k in sa:
        if k not in sb or sa[k].shape != sb[k].shape:
            continue
        a, b = sa[k].float(), sb[k].float()
        g = k.split(".")[0]
        groups[g]["n"] += 1
        groups[g]["identical"] += int(torch.equal(a, b))
        groups[g]["rel"].append(float((b - a).norm() / (a.norm() + 1e-30)))

    return {
        g: {
            "tensors": d["n"],
            "bit_identical": d["identical"],
            "all_identical": d["identical"] == d["n"],
            "mean_relative_diff": float(np.mean(d["rel"])),
        }
        for g, d in sorted(groups.items(), key=lambda kv: -kv[1]["n"])
    }
