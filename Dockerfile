# Speech Model Toolkit: wake word (openWakeWord) + voice (Piper) training.
#
# The two trainers need incompatible PyTorch stacks, so they live in separate
# Python environments inside this one image:
#   system Python     web app + openWakeWord training (torch 2.2, numpy 1.26)
#   /opt/piper-venv   piper1-gpl training, run as a subprocess ($PIPER_PYTHON)
FROM python:3.11-slim-bookworm

# git (clones), build-essential/cmake/ninja (piper1-gpl + piper-phonemize
# builds), libsndfile1 + ffmpeg (audio decoding), wget/curl (downloads)
RUN apt-get update && apt-get install -y --no-install-recommends \
        git \
        wget \
        curl \
        build-essential \
        cmake \
        ninja-build \
        libsndfile1 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Several multi-hundred-MB wheels (torch, CUDA libs): be patient with slow mirrors
ENV PIP_DEFAULT_TIMEOUT=120 \
    PIP_RETRIES=10

# ── Voice: piper1-gpl training ──────────────────────────────────────────────
# Its torch wheel includes CUDA; the GPU is used when the container is started
# with GPU access (see docker-compose.yml).
COPY scripts/install_piper /tmp/install_piper
RUN python3 -m venv /opt/piper-venv \
    && /tmp/install_piper /opt/piper-venv/bin/python3 /opt/piper1-gpl \
    && rm -rf /root/.cache

# ── Wake word: openWakeWord training ────────────────────────────────────────
# Clone into /app/oww-src, NOT /app/openwakeword: a directory named
# "openwakeword" under the working directory would shadow the installed
# package as an empty namespace package.
RUN git clone --depth 1 --branch v0.6.0 \
        https://github.com/dscripka/openWakeWord /app/oww-src \
    && git clone --depth 1 --branch v2.0.0 \
        https://github.com/rhasspy/piper-sample-generator /app/piper-sample-generator

# Pin multiprocess/dill and pyarrow/fsspec/huggingface-hub before
# datasets==2.14.6 so pip neither backtracks nor upgrades pyarrow past 13.
RUN pip install --no-cache-dir multiprocess==0.70.15 dill==0.3.7 \
    && pip install --no-cache-dir \
        "pyarrow==13.0.0" \
        "fsspec==2023.10.0" \
        "huggingface-hub==0.19.4"

# Web app + training stack (torch, speechbrain, onnx, fastapi, ...)
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt \
    && pip install --no-cache-dir -e /app/oww-src

# DataLoader num_workers=0 avoids the >2 GB IPC pipe limit when the ACAV
# features tensor is shared across worker subprocesses.
RUN python3 -c "\
path='/app/oww-src/openwakeword/train.py'; \
c=open(path).read(); \
old='batch_size=None, num_workers=n_cpus, prefetch_factor=16'; \
assert old in c, 'openWakeWord train.py changed, update this patch'; \
open(path,'w').write(c.replace(old, 'batch_size=None, num_workers=0')); \
print('Patched train.py num_workers=0')"

# TFLite export (separate because it pulls keras/tf, which can conflict with torch)
RUN pip install --no-cache-dir onnx2tf

# ── App ─────────────────────────────────────────────────────────────────────
COPY export_dataset/ /app/export_dataset/
COPY prompts/ /app/prompts/
COPY scripts/ /app/scripts/
COPY app/ /app/app/

ENV PYTHONUNBUFFERED=1 \
    WAKEWORD_DATA_DIR=/data/wakeword \
    WAKEWORD_OUTPUT_DIR=/data/wakeword-models \
    VOICE_DATA_DIR=/data/voice \
    PROMPTS_DIR=/app/prompts \
    OPENWAKEWORD_DIR=/app/oww-src \
    PIPER_GENERATOR_DIR=/app/piper-sample-generator \
    PIPER_PYTHON=/opt/piper-venv/bin/python3 \
    SETTINGS_FILE=/data/settings.json \
    STT_MODELS_DIR=/data/stt-models

VOLUME ["/data"]
EXPOSE 8765

CMD ["python3", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8765"]
