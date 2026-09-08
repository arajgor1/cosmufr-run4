"""Run the site on localhost with the checkpoint that is already on disk.

`python -m uvicorn server:app` works too, but only once COSMUFR_CKPT points at a
weights file. This sets it to the local copy so `python scripts/serve_local.py`
is the whole instruction.
"""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
os.environ.setdefault("COSMUFR_CKPT", str(ROOT / "_local_ckpt" / "best.pt"))
os.environ.setdefault("PYTHONUTF8", "1")
sys.path.insert(0, str(ROOT))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server:app", host="127.0.0.1",
                port=int(os.environ.get("PORT", "8000")), log_level="info")
