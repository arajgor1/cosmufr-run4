"""
cosmufr/config.py — CosmUFR Run 4 release configuration.

Frozen architecture matching the published `best.pt` checkpoint
(b200-b4096-run4-20260414_031247, Phase 4 epoch 30).

This is a slimmed-down copy of the training-time `CosmUFRConfig`. Run-time
flags that were used by later experimental runs (source-aware embedding,
multi-z encoder, explicit-z input augmentation, L2-SP anchoring) are all
hard-coded OFF here because the released checkpoint was trained without them.

Do NOT change the architecture fields (d_obs, d_b, n_blocks_*, n_attractors,
k_settle, etc.) — they must match the checkpoint or the state_dict won't load.
"""
from dataclasses import dataclass


@dataclass
class CosmUFRConfig:
    # ── Observation ───────────────────────────────────────────────────────────
    d_obs:        int = 400    # log10 P(k): 200 k-modes x 2 redshifts (z=0, z=0.47)
    d_obs_single: int = 200    # single redshift (sequential mode)
    n_k_modes:    int = 200    # wavenumber bins, k in [0.1, 4.5] h/Mpc
    n_k_train:    int = 100    # k-modes sampled per training step (training-only)
    n_mass_bins:  int = 20     # halo mass function bins

    # ── Belief state  b_t = [s_t | c_t | u_t | p_t] ─────────────────────────
    d_b: int = 1024   # total belief dim
    d_s: int = 512    # world-state
    d_c: int = 256    # context
    d_u: int = 128    # uncertainty
    d_p: int = 128    # prototype proximity

    # ── Per-component hidden dims and block counts ───────────────────────────
    d_h_enc:        int = 1024
    n_blocks_enc:   int = 4
    d_h_gen:        int = 2048
    n_blocks_gen:   int = 8
    d_h_energy:     int = 512
    n_blocks_energy: int = 2
    d_h_task:       int = 512
    n_blocks_task:  int = 2
    d_h_settle:     int = 256

    # ── Attractor bank ────────────────────────────────────────────────────────
    n_attractors:  int   = 4096
    attractor_top: int   = 16
    attractor_ema: float = 0.99

    # ── Settling ──────────────────────────────────────────────────────────────
    k_settle:         int   = 16
    k_settle_seq:     int   = 8
    k_backprop:       int   = 4
    P_min:            float = 0.01
    P_max:            float = 1.0
    eta_min:          float = 0.001
    eta_max:          float = 0.05
    belief_grad_clip: float = 5.0

    # ── Energy weights ────────────────────────────────────────────────────────
    l_obs:         float = 1.0
    l_con:         float = 0.5
    l_dyn:         float = 0.5
    energy_margin: float = 0.5

    # ── Cosmological parameter ranges [Om, s8, h, ns, Ob, w0, mv, wa] ────────
    n_cosmo_params: int   = 8
    om_min: float = 0.10;  om_max: float = 0.50
    s8_min: float = 0.52;  s8_max: float = 1.00
    h_min:  float = 0.60;  h_max:  float = 0.80
    ns_min: float = 0.90;  ns_max: float = 1.02
    ob_min: float = 0.03;  ob_max: float = 0.07
    w0_min: float = -1.20; w0_max: float = -0.80
    mv_min: float = 0.00;  mv_max: float = 0.40   # neutrino mass sum [eV]
    wa_min: float = -0.50; wa_max: float = 0.50   # dark energy CPL wa

    # ── k grid ────────────────────────────────────────────────────────────────
    k_min: float = 0.1
    k_max: float = 4.5

    # ── Architecture flags — frozen for Run 4 release ────────────────────────
    # Run 4 was trained without any of these. Do not flip; the checkpoint
    # state_dict shapes will mismatch. See architecture comment at top.
    use_explicit_z:      bool = False
    use_source_aware:    bool = False
    use_multi_z_encoder: bool = False
    n_sources:           int  = 32
    d_source_emb:        int  = 64
