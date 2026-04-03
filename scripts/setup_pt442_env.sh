#!/usr/bin/env bash
set -euo pipefail

# Clean environment for 70B PT runs with 4-bit quantization compatibility.
# This pins transformers to 4.44.2 to avoid the 4-bit `.to()` guard introduced later.
VENV_PATH="${VENV_PATH:-/home/${USER}/.venvs/pt442}"
PY_BIN="${PY_BIN:-python3}"

echo "[SETUP] VENV_PATH=${VENV_PATH}"
"${PY_BIN}" -m venv "${VENV_PATH}"

"${VENV_PATH}/bin/python" -m pip install --upgrade pip setuptools wheel
"${VENV_PATH}/bin/pip" install \
  torch \
  "transformers==4.44.2" \
  "accelerate==0.33.0" \
  "bitsandbytes==0.42.0" \
  numpy pandas matplotlib scipy einops python-dotenv rich scikit-learn torchvision sentencepiece

# baukit is hosted on GitHub; bypass cluster pip config just for this install.
PIP_CONFIG_FILE=/dev/null "${VENV_PATH}/bin/pip" install --no-deps git+https://github.com/davidbau/baukit.git

"${VENV_PATH}/bin/python" - <<'PY'
import torch, transformers, accelerate, bitsandbytes
print("torch", torch.__version__, "cuda", torch.version.cuda, "cuda_available", torch.cuda.is_available())
print("transformers", transformers.__version__)
print("accelerate", accelerate.__version__)
print("bitsandbytes", bitsandbytes.__version__)
PY

echo "[SETUP] done"
echo "[SETUP] use with:"
echo "  export BNB_CUDA_VERSION=122"
echo "  # avoid mixing CUDA 12.2 runtime paths with torch CUDA 12.9"
echo "  unset CUDA_HOME"
echo "  unset LD_LIBRARY_PATH"
echo "  PY=${VENV_PATH}/bin/python scripts/run_pt_llama70b.sh"
