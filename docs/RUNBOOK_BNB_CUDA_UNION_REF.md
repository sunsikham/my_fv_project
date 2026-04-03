# RUNBOOK: bitsandbytes + CUDA for union_ref PCA

## Purpose

Use this runbook when union-ref vector extraction or PCA fails because of GPU, CUDA, or bitsandbytes issues.

Typical symptoms:
- `No GPU found. A GPU is needed for quantization.`
- `CUDA Setup failed despite GPU being available`
- `Failed to import transformers.integrations.bitsandbytes`

## Active-Root Assumption

This runbook assumes:
- repo root: `/home/sunsik/my_fv_project`
- model root: `/scratch/sunsik/models`
- artifact roots may be either home or scratch depending on the run

## 1. Use The Correct Python And Q Directory

Always run on a GPU compute node, not a login node.

```bash
ROOT=/home/sunsik/my_fv_project
VENV=/home/sunsik/.venvs/pt442
PY=$VENV/bin/python
QDIR=<set-per-q-directory>

cd "$ROOT"
source "$VENV/bin/activate"
```

`QDIR` should point to an actual condition-qwise q directory, for example:
- `/home/sunsik/my_fv_project/results_fv/relation_condition_qwise/<relation_name>/Q1`
- or `/scratch/sunsik/my_fv_project/results_fv/relation_condition_qwise/<relation_name>/Q1`

If your cluster needs modules, load CUDA first:

```bash
module load cuda/12.9
```

## 2. Required Environment Variables

This project commonly uses bitsandbytes CUDA `122`.

```bash
export BNB_CUDA_VERSION=122
unset CUDA_HOME
```

If old CUDA libraries are polluting `LD_LIBRARY_PATH`, sanitize it:

```bash
if [ -n "${LD_LIBRARY_PATH:-}" ]; then
  export LD_LIBRARY_PATH="$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | rg -v 'cudacore/12\.2|cuda/12\.2' | paste -sd: -)"
fi
```

## 3. Preflight Checks

### 3.1 Torch CUDA

```bash
"$PY" -c "import torch; print('cuda_ok', torch.cuda.is_available(), 'count', torch.cuda.device_count(), 'torch_cuda', torch.version.cuda)"
```

Expected:
- `cuda_ok True`
- `count >= 1`

### 3.2 bitsandbytes

```bash
"$PY" -m bitsandbytes
```

Expected:
- no CUDA setup traceback

### 3.3 Optional binary existence check

```bash
python - <<'PY'
from pathlib import Path
import bitsandbytes as bnb

pkg = Path(bnb.__file__).resolve().parent
target = pkg / "libbitsandbytes_cuda122.so"
print(target)
print("exists", target.exists())
PY
```

## 4. Run union_ref PCA

If vectors are already present and you only need union-ref PCA:

```bash
"$PY" scripts/run_union_ref_pca.py \
  --q_dir "$QDIR"
```

If vectors must be rebuilt, pass an explicit model profile:

```bash
MODEL=/scratch/sunsik/models/Llama-3.1-70B

"$PY" scripts/run_union_ref_pca.py \
  --q_dir "$QDIR" \
  --rebuild_vectors 1 \
  --model "$MODEL" \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit
```

Expected outputs:
- `$QDIR/_top_heads/sets/top_heads_ref_union.json`
- `$QDIR/_vectors/trial_vectors_union_ref_AAA.npy`
- `$QDIR/_vectors/trial_vectors_union_ref_BBB.npy`
- `$QDIR/_vectors/trial_vectors_union_ref_BABA.npy`
- `$QDIR/_pca_common/union_ref/`

## 5. Optional Flags

Force vector rebuild:

```bash
"$PY" scripts/run_union_ref_pca.py \
  --q_dir "$QDIR" \
  --rebuild_vectors 1 \
  --model "$MODEL" \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit
```

Skip PCA and only update heads/vectors:

```bash
"$PY" scripts/run_union_ref_pca.py \
  --q_dir "$QDIR" \
  --skip_pca 1
```

## 6. Troubleshooting Matrix

### A. `No GPU found. A GPU is needed for quantization.`

Cause:
- not on a GPU node
- CUDA runtime unavailable in current shell

Fix:
- verify allocation with `nvidia-smi`
- rerun the preflight checks

### B. `CUDA Setup failed despite GPU being available`

Cause:
- bitsandbytes CUDA mismatch
- bad CUDA library ordering

Fix:
- set `BNB_CUDA_VERSION=122`
- unset `CUDA_HOME`
- sanitize `LD_LIBRARY_PATH`
- rerun `"$PY" -m bitsandbytes`

### C. `installed version of bitsandbytes was compiled without GPU support`

Cause:
- login-node execution
- or CUDA runtime failed to load and bnb fell back

Fix:
- confirm GPU-node execution
- rerun preflight after env setup

### D. vectors still fail to rebuild

Collect:

```bash
"$PY" -c "import torch, transformers, bitsandbytes, accelerate; print(torch.__version__, torch.version.cuda, torch.cuda.is_available()); print(transformers.__version__, bitsandbytes.__version__, accelerate.__version__)"
"$PY" -m bitsandbytes
env | rg 'BNB|CUDA|LD_LIBRARY_PATH' | sort
```

## 7. Practical Rule

For this runbook, the q directory may live under either home or scratch.

What matters is:
- use the real q directory for the run you are repairing
- do not assume the home copy is canonical if the run was scratch-first

