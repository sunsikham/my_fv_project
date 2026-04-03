#!/bin/bash
#SBATCH -A def-twwebb_gpu
#SBATCH -p gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=3-00:00:00
#SBATCH -J fv_q1_dext_pca5
#SBATCH -o /home/sunsik/my_fv_project/logs/%x_%j.out

set -euo pipefail

cd /home/sunsik/my_fv_project
mkdir -p logs

source /home/sunsik/.venvs/pt442/bin/activate
module load cuda/12.9
export LD_LIBRARY_PATH="$EBROOTCUDA/lib64:${LD_LIBRARY_PATH:-}"
export BNB_CUDA_VERSION=122
unset CUDA_HOME
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

echo "[slurm] job_id=${SLURM_JOB_ID:-unknown}"
scontrol show job "${SLURM_JOB_ID:-0}" | egrep 'ReqTRES|AllocTRES|NodeList|StartTime|TimeLimit' || true

# Defaults:
#   BASE_ROOT : existing Q1 root
#   Q_LIST    : Q1 only
# Override examples:
#   Q_LIST_OVERRIDE="Q1" sbatch scripts/slurm/run_q1_d_extension_pca5_3d.sbatch
#   BASE_ROOT_OVERRIDE="/path/to/another/root" sbatch scripts/slurm/run_q1_d_extension_pca5_3d.sbatch
BASE_ROOT="${BASE_ROOT_OVERRIDE:-/scratch/${USER}/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc}"
HOME_MIRROR_ROOT="${HOME_MIRROR_ROOT_OVERRIDE:-/home/sunsik/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc}"
SYNC_MODE="${SYNC_MODE:-per_q}"
HOME_ARTIFACT_PROFILE="${HOME_ARTIFACT_PROFILE:-core}"
Q_LIST="${Q_LIST_OVERRIDE:-Q1}"

echo "[slurm] base_root=${BASE_ROOT}"
echo "[slurm] home_mirror_root=${HOME_MIRROR_ROOT}"
echo "[slurm] sync_mode=${SYNC_MODE} profile=${HOME_ARTIFACT_PROFILE}"
echo "[slurm] q_list=${Q_LIST}"

python scripts/run_d_extension_stepd_pca.py \
  --base_root "${BASE_ROOT}" \
  --relation_a_csv /home/sunsik/my_fv_project/datasets/relation/relationA_ex.csv \
  --relation_d_csv /home/sunsik/my_fv_project/datasets/relation/relationD_ex.csv \
  --q_list "${Q_LIST}" \
  --d_conditions DDD,DADA \
  --pca_conditions AAA,BBB,BABA,DDD,DADA \
  --model /scratch/sunsik/models/Llama-3.1-70B \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit \
  --n_trials_per_q 25 \
  --n_demos 9 \
  --topk 20 \
  --score_key mean_delta_p \
  --stepd_layers 16-55 \
  --pca_ref_mode AAA_ref \
  --pca_out_subdir AAA_ref_with_D \
  --resume 1 \
  --stop_on_error 1 \
  --sync_home_root "${HOME_MIRROR_ROOT}" \
  --sync_mode "${SYNC_MODE}" \
  --home_artifact_profile "${HOME_ARTIFACT_PROFILE}" \
  --sync_on_failure 1
