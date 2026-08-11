#!/bin/bash
#SBATCH --job-name=nanochat-train
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:40:00
#SBATCH --partition=normal
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# ── Paths ────────────────────────────────────────────────────────────────────

export REPO_DIR="/capstor/store/cscs/swissai/a0229/jtempus/nanochat"
export VENV_DIR="$REPO_DIR/.venv"

export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"/iopsstor/scratch/cscs/jtempus/.cache/nanochat"}

export NANOCHAT_TOKENIZER_DIR=${NANOCHAT_TOKENIZER_DIR:-"/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/bpe_tokenizers_climbmix/nano_bpe_climb_mix/32768"}

# ── Experiment ───────────────────────────────────────────────────────────────

export DEPTH=${DEPTH:-12}
export DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}
export TARGET_PDR=${TARGET_PDR:-10.5}
export DATASET_SHARDS=${DATASET_SHARDS:-170}

export WANDB_RUN=${WANDB_RUN:-"bpe_8192_d12"}
export WANDB_ENTITY=${WANDB_ENTITY:-"tempus-apertus"}
export WANDB_PROJECT=${WANDB_PROJECT:-"hyperparameter_search"}

# ── W&B credentials ─────────────────────────────────────────────────────────

set -a
source "$HOME/.env.local"
set +a

# ── Distributed setup ────────────────────────────────────────────────────────

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Job:        $SLURM_JOB_ID"
echo "Nodes:      $SLURM_NNODES"
echo "GPUs:       $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Master:     $MASTER_ADDR:$MASTER_PORT"
echo "Depth:      $DEPTH"
echo "Batch size: $DEVICE_BATCH_SIZE"
echo "Run:        $WANDB_RUN"

# ── Generate token_bytes.pt ──────────────────────────────────────────────────

TOKEN_BYTES_PATH="$NANOCHAT_TOKENIZER_DIR/token_bytes.pt"

if [[ ! -f "$TOKEN_BYTES_PATH" ]]; then
    srun \
        --nodes=1 \
        --ntasks=1 \
        --environment=nano_distributed \
        bash -c '
            source "$VENV_DIR/bin/activate"
            cd "$REPO_DIR"
            python -m scripts.generate_token_bytes
        '
fi

# ── Download pretraining dataset ─────────────────────────────────────────────

srun \
    --nodes=1 \
    --ntasks=1 \
    --environment=nano_distributed \
    bash -c '
        source "$VENV_DIR/bin/activate"
        cd "$REPO_DIR"
        python -m nanochat.dataset -n "$DATASET_SHARDS"
    '

# ── Train ────────────────────────────────────────────────────────────────────

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
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --target-param-data-ratio="$TARGET_PDR" \
            --run="$WANDB_RUN" \
            --fix-vocab-size-compute \
            --eval-every=200 \
            --fp8
    '
