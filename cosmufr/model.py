"""
cosmufr/model.py — CosmUFR Run 4 architecture (inference-only minimal copy).

This is a trimmed copy of the training-time `model/cosmufr_arch.py` with:
  * Run 4's exact architecture (matches checkpoint state_dict)
  * No training-only branches (gradient checkpointing, optimizer factory removed)
  * Source-aware / multi-z / explicit-z code paths removed (Run 4 used none)

The forward pass is identical to training: encode P(k) -> belief proposal ->
16-step energy settling -> task heads (params, uncertainties, P(k) reconstruction).

For full architectural commentary see the original `model/cosmufr_arch.py` in
the calybre-ufr training repo.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from cosmufr.config import CosmUFRConfig


# ─────────────────────────────────────────────────────────────────────────────
# Building blocks
# ─────────────────────────────────────────────────────────────────────────────

class ResidualBlock(nn.Module):
    """Linear(d,d) -> GELU -> LayerNorm -> Linear(d,d) + skip."""
    def __init__(self, d: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, d),
            nn.GELU(),
            nn.LayerNorm(d),
            nn.Linear(d, d),
        )
        self.norm = nn.LayerNorm(d)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.norm(x + self.drop(self.net(x)))


class ResidualMLP(nn.Module):
    """in_dim -> Linear(in_dim, d_h) -> [n_blocks x ResidualBlock] -> Linear(d_h, out_dim)"""
    def __init__(self, in_dim: int, out_dim: int, d_h: int, n_blocks: int,
                 dropout: float = 0.0):
        super().__init__()
        self.proj_in  = nn.Sequential(nn.Linear(in_dim, d_h), nn.LayerNorm(d_h))
        self.blocks   = nn.ModuleList([ResidualBlock(d_h, dropout) for _ in range(n_blocks)])
        self.proj_out = nn.Linear(d_h, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.proj_in(x)
        for block in self.blocks:
            h = block(h)
        return self.proj_out(h)


# ─────────────────────────────────────────────────────────────────────────────
# Observation encoders
# ─────────────────────────────────────────────────────────────────────────────

class ObsEncoder(nn.Module):
    """log10 P(k) [400d: 200 k-bins x 2 redshifts] -> z_t [1024d]."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_obs, cfg.d_b, cfg.d_h_enc, cfg.n_blocks_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ObsEncoderSingle(nn.Module):
    """log10 P(k) [200d: single redshift] -> z_t [1024d]."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_obs_single, cfg.d_b, cfg.d_h_enc, cfg.n_blocks_enc)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
# Belief proposal
# ─────────────────────────────────────────────────────────────────────────────

class BeliefProposal(nn.Module):
    """(z_t, b_prev) -> b_hat. Outputs delta; b_hat = b_prev + delta."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_b + cfg.d_b, cfg.d_b, cfg.d_h_enc, cfg.n_blocks_enc)

    def forward(self, z: torch.Tensor, b_prev: torch.Tensor) -> torch.Tensor:
        delta = self.net(torch.cat([z, b_prev], dim=-1))
        return delta + b_prev


# ─────────────────────────────────────────────────────────────────────────────
# Energy heads
# ─────────────────────────────────────────────────────────────────────────────

class ObsEnergyHead(nn.Module):
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_b + cfg.d_b, 1, cfg.d_h_energy, cfg.n_blocks_energy)

    def forward(self, b: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([b, z], dim=-1)).squeeze(-1)


class ConstraintHead(nn.Module):
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_b, 1, cfg.d_h_energy, cfg.n_blocks_energy)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return self.net(b).squeeze(-1)


class DynEnergyHead(nn.Module):
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_b + cfg.d_b, 1, cfg.d_h_energy, cfg.n_blocks_energy)

    def forward(self, b: torch.Tensor, b_prev: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([b, b_prev], dim=-1)).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# Settling core — 16-step energy gradient descent with learned preconditioner
# ─────────────────────────────────────────────────────────────────────────────

class SettlingCore(nn.Module):
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.cfg = cfg
        self.precond = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg.d_b, cfg.d_h_settle),
                          nn.GELU(),
                          nn.Linear(cfg.d_h_settle, cfg.d_b))
            for _ in range(cfg.k_settle)
        ])
        self.eta_net = nn.ModuleList([
            nn.Sequential(nn.Linear(cfg.d_b, 64), nn.GELU(), nn.Linear(64, 1))
            for _ in range(cfg.k_settle)
        ])

    def forward(self, b_hat: torch.Tensor, energy_fn, z: torch.Tensor,
                b_prev: torch.Tensor, k_settle_override: int = None):
        cfg = self.cfg
        k   = k_settle_override if k_settle_override is not None else cfg.k_settle
        b   = b_hat
        energy_log = []

        # Inference-only path: log first energy for monitoring, then iterate.
        with torch.no_grad():
            energy_log.append(energy_fn(b, z, b_prev).mean().item())

        for step in range(k):
            b = b.detach()
            with torch.enable_grad():
                b_g = b.requires_grad_(True)
                E = energy_fn(b_g, z.detach(), b_prev.detach())
                grad = torch.autograd.grad(E.sum(), b_g, create_graph=False)[0]
            grad = torch.clamp(grad, -cfg.belief_grad_clip, cfg.belief_grad_clip)

            with torch.no_grad():
                P = torch.sigmoid(self.precond[step](b))
                P = cfg.P_min + (cfg.P_max - cfg.P_min) * P
                eta_raw = self.eta_net[step](b)
                eta = cfg.eta_min + (cfg.eta_max - cfg.eta_min) * torch.sigmoid(eta_raw)

            b = b - eta * P * grad.detach()

            with torch.no_grad():
                energy_log.append(energy_fn(b.detach(), z.detach(), b_prev.detach()).mean().item())

        return b, energy_log


# ─────────────────────────────────────────────────────────────────────────────
# Attractor bank
# ─────────────────────────────────────────────────────────────────────────────

class AttractorBank(nn.Module):
    """4096 prototype belief vectors, EMA-updated during training (frozen at inference)."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.cfg = cfg
        self.attractors = nn.Parameter(
            torch.randn(cfg.n_attractors, cfg.d_b) * 0.02,
            requires_grad=False,
        )

    def prototype_loss(self, b_star: torch.Tensor) -> torch.Tensor:
        a_norm = F.normalize(self.attractors.detach(), dim=-1)
        b_norm = F.normalize(b_star, dim=-1)
        sim = b_norm @ a_norm.T
        max_sim = sim.max(dim=-1).values
        return -max_sim.mean()


# ─────────────────────────────────────────────────────────────────────────────
# k-continuous Generative Head
# ─────────────────────────────────────────────────────────────────────────────

class GenerativeHead(nn.Module):
    """(b_star, log_k) -> (log_pk_mean, log_pk_logvar). k-continuous implicit field."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.cfg = cfg
        self.net = ResidualMLP(cfg.d_b + 1, 2, cfg.d_h_gen, cfg.n_blocks_gen)

    def forward(self, b_star: torch.Tensor, log_k: torch.Tensor) -> tuple:
        B = b_star.shape[0]
        if log_k.dim() == 1:
            K = log_k.shape[0]
            b_exp = b_star.unsqueeze(1).expand(B, K, -1)
            k_exp = log_k.unsqueeze(0).expand(B, K).unsqueeze(-1)
        else:
            K = log_k.shape[1]
            b_exp = b_star.unsqueeze(1).expand(B, K, -1)
            k_exp = log_k.unsqueeze(-1)

        inp = torch.cat([b_exp, k_exp], dim=-1)
        out = self.net(inp.reshape(B * K, -1)).reshape(B, K, 2)
        return out[..., 0], out[..., 1]


# ─────────────────────────────────────────────────────────────────────────────
# Task heads
# ─────────────────────────────────────────────────────────────────────────────

class ParameterHead(nn.Module):
    """b_star -> 8 cosmology params clamped to physical ranges via sigmoid."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.cfg = cfg
        self.net = ResidualMLP(cfg.d_b, cfg.n_cosmo_params, cfg.d_h_task, cfg.n_blocks_task)
        self.register_buffer('p_min', torch.tensor([
            cfg.om_min, cfg.s8_min, cfg.h_min,
            cfg.ns_min, cfg.ob_min, cfg.w0_min,
            cfg.mv_min, cfg.wa_min]))
        self.register_buffer('p_max', torch.tensor([
            cfg.om_max, cfg.s8_max, cfg.h_max,
            cfg.ns_max, cfg.ob_max, cfg.w0_max,
            cfg.mv_max, cfg.wa_max]))

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        raw = self.net(b)
        return torch.sigmoid(raw) * (self.p_max - self.p_min) + self.p_min


class UncertaintyHead(nn.Module):
    """b_star -> 8 per-parameter variances (NLL-calibrated)."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_b, cfg.n_cosmo_params, cfg.d_h_task, cfg.n_blocks_task)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return torch.clamp(F.softplus(self.net(b)) + 1e-2, max=4.0)


class HaloMassHead(nn.Module):
    """b_star -> n(M) at n_mass_bins bins."""
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.net = ResidualMLP(cfg.d_b, cfg.n_mass_bins, cfg.d_h_task, cfg.n_blocks_task)

    def forward(self, b: torch.Tensor) -> torch.Tensor:
        return self.net(b)


# ─────────────────────────────────────────────────────────────────────────────
# Full model
# ─────────────────────────────────────────────────────────────────────────────

class CosmUFRLite(nn.Module):
    """
    CosmUFR Run 4 — 137M-parameter belief-settling cosmology emulator.

    Forward pass (joint, default):
        obs [B, 400] -> ObsEncoder -> z [B, 1024]
        (z, b_prev)  -> BeliefProposal -> b_hat [B, 1024]
        b_hat        -> SettlingCore   -> b_star [B, 1024] (16-step energy GD)
        b_star       -> ParameterHead  -> params [B, 8]
        b_star       -> UncertaintyHead -> variances [B, 8]
        (b_star, k)  -> GenerativeHead -> pk_mean, pk_logvar [B, K]
    """
    def __init__(self, cfg: CosmUFRConfig):
        super().__init__()
        self.cfg = cfg

        # Encoders
        self.obs_encoder        = ObsEncoder(cfg)
        self.obs_encoder_single = ObsEncoderSingle(cfg)

        # Belief
        self.belief_proposal     = BeliefProposal(cfg)
        self.belief_proposal_seq = BeliefProposal(cfg)
        self.settling            = SettlingCore(cfg)
        self.attractor_bank      = AttractorBank(cfg)

        # Energy heads
        self.obs_energy_head = ObsEnergyHead(cfg)
        self.constraint_head = ConstraintHead(cfg)
        self.dyn_energy_head = DynEnergyHead(cfg)

        # Task heads
        self.param_head = ParameterHead(cfg)
        self.unc_head   = UncertaintyHead(cfg)
        self.gen_head   = GenerativeHead(cfg)
        self.halo_head  = HaloMassHead(cfg)

        # Training k-grid (200 log-spaced k-values from k_min to k_max)
        log_k = torch.linspace(math.log(cfg.k_min), math.log(cfg.k_max), cfg.n_k_modes)
        self.register_buffer('log_k_train', log_k)

    # ── Energy functional ────────────────────────────────────────────────────
    def energy_fn(self, b: torch.Tensor, z: torch.Tensor,
                  b_prev: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        E_obs = self.obs_energy_head(b, z)
        E_con = self.constraint_head(b)
        E_dyn = self.dyn_energy_head(b, b_prev)
        return cfg.l_obs * E_obs + cfg.l_con * E_con + cfg.l_dyn * E_dyn

    # ── Forward ──────────────────────────────────────────────────────────────
    def forward(self, obs: torch.Tensor,
                b_prev: torch.Tensor = None,
                single_z: bool = False,
                log_k: torch.Tensor = None,
                return_full: bool = True,
                k_settle_override: int = None) -> dict:
        B      = obs.shape[0]
        device = obs.device
        cfg    = self.cfg

        if b_prev is None:
            b_prev = torch.zeros(B, cfg.d_b, device=device)

        # 1. Encode observation
        if single_z:
            z = self.obs_encoder_single(obs)
        else:
            z = self.obs_encoder(obs)

        # 2. Belief proposal
        if single_z:
            b_hat = self.belief_proposal_seq(z, b_prev)
        else:
            b_hat = self.belief_proposal(z, b_prev)

        # 3. Energy settling -> b_star
        b_star, energy_log = self.settling(
            b_hat, self.energy_fn, z, b_prev,
            k_settle_override=k_settle_override,
        )

        # 4. Task heads
        params    = self.param_head(b_star)
        variances = self.unc_head(b_star)

        out = dict(
            z=z, b_hat=b_hat, b_star=b_star,
            params=params, variances=variances,
            energy_log=energy_log,
        )

        if return_full:
            if log_k is None:
                log_k = self.log_k_train
            pk_mean, pk_logvar = self.gen_head(b_star, log_k)
            nmass = self.halo_head(b_star)
            out.update(pk_mean=pk_mean, pk_logvar=pk_logvar, nmass=nmass)

        return out
