#!/bin/bash
#SBATCH --job-name=nanochat-single-train
#SBATCH --account=a0229
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --partition=normal
#SBATCH --output=logs/single-train-%j.out
#SBATCH --error=logs/single-train-%j.err

set -euo pipefail

# Usage:
#   sbatch --time=HH:MM:SS sbatch_files/single_train.sbatch /path/to/tokenizer SEED

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

export REPO_DIR="/capstor/store/cscs/swissai/a0229/jtempus/nanochat"
export VENV_DIR="$REPO_DIR/.venv"
export NANOCHAT_BASE_DIR="/iopsstor/scratch/cscs/jtempus/.cache/nanochat"
export NANOCHAT_TOKENIZER_DIR="$1"

export DEPTH=24
export SEED="$2"
export DEVICE_BATCH_SIZE=16
export TARGET_PDR=10.5
export DATASET_SHARDS=170

export MATRIX_LR=0.020
export WEIGHT_DECAY=0.28
export EMBEDDING_LR=0.50
export UNEMBEDDING_LR=0.016
export SCALAR_LR=0.50
export WARMUP_STEPS=40
export WARMDOWN_RATIO=0.65
export FINAL_LR_FRAC=0.05

export WANDB_ENTITY="tempus-apertus"
export WANDB_PROJECT="baseline_runs"

set -a
source "$HOME/.env.local"
set +a

TOKENIZER_NAME=$(basename "${NANOCHAT_TOKENIZER_DIR%/}")
export RUN_NAME="${TOKENIZER_NAME}_d${DEPTH}_s${SEED}"

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Job:       $SLURM_JOB_ID"
echo "Nodes:     $SLURM_NNODES"
echo "GPUs:      $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Seed:      $SEED"
echo "Tokenizer: $NANOCHAT_TOKENIZER_DIR"
echo "W&B run:   $RUN_NAME"
echo "Model tag: $RUN_NAME"

srun \
    --nodes=1 \
    --ntasks=1 \
    --environment=nano_distributed \
    bash -c '
        source "$VENV_DIR/bin/activate"
        cd "$REPO_DIR"
        python -m scripts.generate_token_bytes
        python -m nanochat.dataset -n "$DATASET_SHARDS"
    '

srun \
    --ntasks="$SLURM_NNODES" \
    --ntasks-per-node=1 \
    --environment=nano_distributed \
    bash -c '
        source "$VENV_DIR/bin/activate"
        cd "$REPO_DIR"

        exec python -m torch.distributed.run \
            --nproc_per_node="$GPUS_PER_NODE" \
            --nnodes="$SLURM_NNODES" \
            --node_rank="$SLURM_PROCID" \
            --rdzv_backend=c10d \
            --rdzv_endpoint="$MASTER_ADDR:$MASTER_PORT" \
            --rdzv_id="$SLURM_JOB_ID" \
            -m scripts.base_train -- \
            --depth="$DEPTH" \
            --seed="$SEED" \
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --target-param-data-ratio="$TARGET_PDR" \
            --run="$RUN_NAME" \
            --model-tag="$RUN_NAME" \
            --fix-vocab-size-compute \
            --matrix-lr="$MATRIX_LR" \
            --weight-decay="$WEIGHT_DECAY" \
            --embedding-lr="$EMBEDDING_LR" \
            --unembedding-lr="$UNEMBEDDING_LR" \
            --scalar-lr="$SCALAR_LR" \
            --warmup-steps="$WARMUP_STEPS" \
            --warmdown-ratio="$WARMDOWN_RATIO" \
            --final-lr-frac="$FINAL_LR_FRAC" \
            --eval-every=200 \
            --sample-every=-1 \
            --save-every=-1
    '
