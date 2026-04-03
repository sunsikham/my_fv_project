#!/bin/bash
#SBATCH -A def-twwebb_gpu
#SBATCH -p gpubase_bygpu_b4
#SBATCH --gres=gpu:h100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --time=1-00:00:00
#SBATCH -J q1_bd_pca
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

Q_DIR="${Q_DIR_OVERRIDE:-/scratch/${USER}/my_fv_project/results_fv/relation_condition_qwise/relA_relationA_ex__relB_relationB_ex__hd011861bcc/Q1}"
TOPK_VALUE="${TOPK_OVERRIDE:-20}"
REF_NAME_VALUE="${REF_NAME_OVERRIDE:-BD_ref}"
PCA_OUT_SUBDIR_VALUE="${PCA_OUT_SUBDIR_OVERRIDE:-BD_ref_BD_compare}"
RUN_LABEL_VALUE="${RUN_LABEL_OVERRIDE:-}"

echo "[slurm] q_dir=${Q_DIR}"
echo "[slurm] topk=${TOPK_VALUE} ref_name=${REF_NAME_VALUE} pca_out_subdir=${PCA_OUT_SUBDIR_VALUE} run_label=${RUN_LABEL_VALUE}"

python scripts/run_bd_interleave_pca.py \
  --q_dir "${Q_DIR}" \
  --relation_b_csv /home/sunsik/my_fv_project/datasets/relation/relationB_ex.csv \
  --relation_d_csv /home/sunsik/my_fv_project/datasets/relation/relationD_ex.csv \
  --model /scratch/sunsik/models/Llama-3.1-70B \
  --model_spec llama3 \
  --device cuda \
  --dtype bf16 \
  --quant 4bit \
  --topk "${TOPK_VALUE}" \
  --score_key mean_delta_p \
  --stepd_layers 16-55 \
  --resume 1 \
  --stop_on_error 1 \
  --ref_name "${REF_NAME_VALUE}" \
  --run_label "${RUN_LABEL_VALUE}" \
  --pca_out_subdir "${PCA_OUT_SUBDIR_VALUE}"
