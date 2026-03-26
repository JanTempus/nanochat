#!/bin/bash
#SBATCH --job-name=nanochat-train
#SBATCH --account=a139
#SBATCH --nodes=1                    # Number of nodes
#SBATCH --ntasks-per-node=4          # 4 tasks per node (one per GPU — nodes always have exactly 4 GPUs)
#SBATCH --gpus-per-node=4            # 4 GPUs per node (fixed: cannot request fewer)
#SBATCH --cpus-per-task=8            # CPUs per task (32 / 4 = 8 per GPU)
#SBATCH --time=00:15:00              # Maximum runtime
#SBATCH --partition=normal
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ============================================================================
# SLURM Multi-Node Distributed Training Script for nanochat
# One srun task per GPU; RANK/LOCAL_RANK/WORLD_SIZE are passed via env vars
# directly to the Python process (no torchrun needed — nanochat reads them
# from os.environ in nanochat/common.py:get_dist_info()).
#
# Usage (from the repo root directory):
#   sbatch runs/slurm_train.sbatch
#   sbatch --export=ALL,DEPTH=24,WANDB_RUN=my_run runs/slurm_train.sbatch
#
# Overridable environment variables:
#   DEPTH             - transformer depth (default 24, as in speedrun)
#   DEVICE_BATCH_SIZE - per-GPU batch size (default 16)
#   TARGET_PDR        - target param:data ratio, Chinchilla-style (default 8)
#   WANDB_RUN         - wandb run name ('dummy' disables logging)
#   NANOCHAT_BASE_DIR      - data/cache/checkpoint directory
#   NANOCHAT_TOKENIZER_DIR - tokenizer directory (default: $NANOCHAT_BASE_DIR/tokenizer)
#   PYTHON_PATH         - path to Python executable
# ============================================================================

# ── Job info ─────────────────────────────────────────────────────────────────
echo "======================================================================"
echo "SLURM Job Information"
echo "======================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NNODES"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "GPUs per node:  $SLURM_GPUS_ON_NODE"
echo "CPUs per task:  $SLURM_CPUS_PER_TASK"
echo "======================================================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p logs

# ── Networking ───────────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# ── Experiment identity (override via --export) ───────────────────────────────
DEPTH=${DEPTH:-12}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}
TARGET_PDR=${TARGET_PDR:-8}          # param:data ratio (8 = speedrun setting)
WANDB_RUN=${WANDB_RUN:-"dummy"}
NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"/iopsstor/scratch/cscs/jtempus/nanochat"}
export NANOCHAT_BASE_DIR
mkdir -p "$NANOCHAT_BASE_DIR"

NANOCHAT_TOKENIZER_DIR=${NANOCHAT_TOKENIZER_DIR:-"/iopsstor/scratch/cscs/jtempus/tokenisation_lp/rounded_tokenizers/basic_finewebedu20B_n180000_seed42/vocab_8192/lp_8192_bias"}
export NANOCHAT_TOKENIZER_DIR

# ── Python path ───────────────────────────────────────────────────────────────
PYTHON_PATH=${PYTHON_PATH:-"/iopsstor/scratch/cscs/jtempus/nanochat/.venv/bin/python"}
echo "Using Python: $PYTHON_PATH"

# ── Derived: total GPUs ───────────────────────────────────────────────────────
TOTAL_GPUS=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export WORLD_SIZE=$TOTAL_GPUS

echo "Total GPUs:        $TOTAL_GPUS (${SLURM_NNODES} nodes x ${SLURM_GPUS_ON_NODE} GPUs/node)"
echo "Depth:             $DEPTH"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Target PDR:        $TARGET_PDR"
echo "WANDB_RUN:         $WANDB_RUN"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "======================================================================"

# ── Load secrets ──────────────────────────────────────────────────────────────
ENV_FILE="${HOME}/.env.local"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    echo "Loaded secrets from $ENV_FILE"
else
    echo "WARNING: $ENV_FILE not found — WANDB_API_KEY may be unset"
fi

# ── Training ──────────────────────────────────────────────────────────────────
# srun spawns one task per GPU (ntasks-per-node=4).
# SLURM_PROCID  = global rank across all tasks  => RANK
# SLURM_LOCALID = local rank within the node    => LOCAL_RANK
# nanochat/common.py:get_dist_info() reads RANK, LOCAL_RANK, WORLD_SIZE
# and dist.init_process_group uses MASTER_ADDR + MASTER_PORT for rendezvous.
echo "Starting distributed training..."
srun --environment=nanochat bash -c "
    echo '[Rank \$SLURM_PROCID / LocalRank \$SLURM_LOCALID] Starting...' && \
    cd $SLURM_SUBMIT_DIR || { echo 'ERROR: cd to $SLURM_SUBMIT_DIR failed'; exit 1; } && \
    WANDB_API_KEY=$WANDB_API_KEY \
    NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR \
    NANOCHAT_TOKENIZER_DIR=$NANOCHAT_TOKENIZER_DIR \
    RANK=\$SLURM_PROCID \
    LOCAL_RANK=\$SLURM_LOCALID \
    $PYTHON_PATH -m scripts.base_train \
        --depth=$DEPTH \
        --device-batch-size=$DEVICE_BATCH_SIZE \
        --target-param-data-ratio=$TARGET_PDR \
        --fp8 \
        --run=$WANDB_RUN
"

echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
