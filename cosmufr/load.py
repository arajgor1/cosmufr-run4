"""
cosmufr/load.py — Checkpoint loading from local path or HuggingFace Hub.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Union

import torch

from cosmufr.config import CosmUFRConfig
from cosmufr.model import CosmUFRLite


HF_REPO_ID = "arajgor1/cosmufr-run4"
CKPT_FILENAME = "best.pt"

PARAM_LABELS = ["Om", "s8", "h", "ns", "Ob", "w0", "mv", "wa"]


def load_model(
    ckpt_path: Optional[Union[str, Path]] = None,
    repo_id: str = HF_REPO_ID,
    filename: str = CKPT_FILENAME,
    device: Union[str, torch.device] = "cpu",
    cfg: Optional[CosmUFRConfig] = None,
) -> CosmUFRLite:
    """
    Load CosmUFR Run 4 from a local checkpoint or HuggingFace Hub.

    Parameters
    ----------
    ckpt_path : str | Path, optional
        If given, load from this local file. Otherwise download from `repo_id`.
    repo_id : str
        HuggingFace repo, default ``arajgor1/cosmufr-run4``.
    filename : str
        Checkpoint filename inside the repo, default ``best.pt``.
    device : str | torch.device
        Target device. Inference works on CPU; GPU recommended only for batch eval.
    cfg : CosmUFRConfig, optional
        Override config. Defaults match the released checkpoint — do not change
        architecture fields or the state_dict load will fail.

    Returns
    -------
    CosmUFRLite (in eval mode, on `device`).
    """
    if cfg is None:
        cfg = CosmUFRConfig()

    if ckpt_path is None:
        from huggingface_hub import hf_hub_download
        token = os.environ.get("HF_TOKEN")  # private repo requires this
        ckpt_path = hf_hub_download(
            repo_id=repo_id, filename=filename, token=token,
        )

    ckpt_path = str(ckpt_path)
    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    sd = state["model"] if isinstance(state, dict) and "model" in state else state

    model = CosmUFRLite(cfg)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # Run 4 ckpt does not include the legacy `attractor_bank._initialized`
    # buffer flag from older trainer versions; missing buffers are harmless
    # because attractor_bank is only used at training time.
    real_missing = [k for k in missing if "attractor_bank" not in k]
    if real_missing:
        raise RuntimeError(
            f"Missing keys when loading checkpoint:\n  {real_missing[:10]}"
        )
    if unexpected:
        # Newer training-only buffers (e.g. source_bias from Run 6+) — log silently.
        pass

    model.to(device).eval()
    return model
