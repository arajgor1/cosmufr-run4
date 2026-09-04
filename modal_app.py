"""
modal_app.py — serve the CosmUFR demo as a public URL on Modal.

Why Modal and not a HuggingFace Space: hosting a Gradio Space, even on free CPU
hardware, now requires a paid PRO subscription.

Why `server.py` and not the Gradio `app.py`: behind Modal's proxy, Gradio's
Server-Sent Events stream is aborted, which leaves the client in a failed state
where the page renders but no button submits anything. `server.py` renders every
page server-side with no streaming and no client framework, so it works behind
any proxy. `app.py` is kept for local use and for anyone with a PRO Space.

Deploy:
    python -m modal deploy modal_app.py

The returned URL is public and needs no token, because the weights repo is
public.

Cost control: CPU only, 2 cores, 4 GiB, and the container is torn down after
5 minutes idle. There is no GPU anywhere in this file.
"""
from __future__ import annotations

from pathlib import Path

import modal

APP_NAME = "cosmufr-demo"
HF_REPO = "arajgor1/cosmufr-run4"
CKPT_FILE = "best.pt"

_release_root = Path(__file__).resolve().parent

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.7.0", "numpy", "matplotlib", "pillow", "requests",
        "fastapi", "python-multipart", "uvicorn",
        "huggingface_hub>=1.9.0", "hf_xet",
        extra_index_url="https://download.pytorch.org/whl/cpu",
    )
    .add_local_dir(
        str(_release_root),
        remote_path="/pkg",
        copy=True,
        ignore=["**/__pycache__", "**/.git", "**/_local_ckpt", "**/*.pyc",
                "**/figs", "**/.pytest_cache"],
    )
    .run_commands("pip install -e /pkg -q")
    .env({"HF_HOME": "/cache/hf", "MPLBACKEND": "Agg"})
)

# The 545 MB checkpoint lives on a Volume rather than inside the image, so a
# code change does not force a re-download.
cache_vol = modal.Volume.from_name("cosmufr-demo-cache", create_if_missing=True)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    volumes={"/cache": cache_vol},
    cpu=2.0,
    memory=4096,
    min_containers=0,        # scale to zero: an idle demo costs nothing
    scaledown_window=300,    # tear down 5 minutes after the last request
    timeout=60 * 10,
    max_containers=4,        # cap concurrent spend
)
@modal.concurrent(max_inputs=4)
@modal.asgi_app()
def serve():
    import os
    import sys

    from huggingface_hub import hf_hub_download

    os.chdir("/pkg")
    sys.path.insert(0, "/pkg")

    # Public repo, so no token is needed.
    ckpt = hf_hub_download(repo_id=HF_REPO, filename=CKPT_FILE,
                           cache_dir="/cache/hf")
    cache_vol.commit()
    os.environ["COSMUFR_CKPT"] = ckpt

    import server

    return server.app
