# CosmUFR Run 4 demo — Cloud Run image.
#
# Two decisions worth stating:
#
# 1. CPU-only torch, from PyTorch's CPU wheel index. The CUDA build pulls in
#    well over a gigabyte of libraries this service will never call.
# 2. The 545 MB checkpoint is baked into the image rather than fetched at
#    container start. It costs a few cents a month of Artifact Registry storage
#    and makes the image larger, but it removes a network call from the startup
#    path. This demo's whole job is to work on the first click, and a cold start
#    that depends on a third-party download is a failure mode we do not need.

FROM python:3.11-slim AS weights

# The weights repo is public, so no credentials are needed here.
ARG CKPT_URL=https://huggingface.co/arajgor1/cosmufr-run4/resolve/main/best.pt
ARG CKPT_SHA256=5db09d4ff02316c60a43e08fa242223d3243f4f224b625798eaf385151150fc1

RUN apt-get update -qq \
 && apt-get install -y --no-install-recommends curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

# Fail the build rather than ship a container serving weights we did not verify.
RUN curl -fsSL "${CKPT_URL}" -o /best.pt \
 && echo "${CKPT_SHA256}  /best.pt" | sha256sum -c - \
 && ls -l /best.pt


FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    MPLBACKEND=Agg \
    COSMUFR_CKPT=/opt/cosmufr/best.pt \
    MPLCONFIGDIR=/tmp/mpl

WORKDIR /app

COPY requirements-cloudrun.txt .
RUN pip install --no-cache-dir -r requirements-cloudrun.txt

COPY . /app
RUN pip install --no-cache-dir -e . \
 && rm -rf /root/.cache

COPY --from=weights /best.pt /opt/cosmufr/best.pt

# Cloud Run sends SIGTERM and provides $PORT. One worker: the model is ~1.1 GB
# resident and a second worker would double that for no benefit on a demo.
EXPOSE 8080
CMD exec uvicorn server:app --host 0.0.0.0 --port ${PORT:-8080} --workers 1 --timeout-keep-alive 65
