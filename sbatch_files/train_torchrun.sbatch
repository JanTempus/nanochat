#!/bin/bash
#SBATCH --job-name=nanochat-train
#SBATCH --account=a139
#SBATCH --nodes=1                    # Number of nodes (scale up as needed)
#SBATCH --ntasks-per-node=1          # 1 task per node — torchrun spawns GPU workers
#SBATCH --gpus-per-node=4            # 4 GPUs per node (fixed: cannot request fewer)
#SBATCH --cpus-per-task=32           # All CPUs to the single task (torchrun distributes them)
#SBATCH --time=00:30:00              # Maximum runtime
#SBATCH --partition=normal
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

# ============================================================================
# SLURM Multi-Node Distributed Training Script for nanochat (torchrun variant)
#
# Uses torchrun for process spawning and rendezvous, matching how speedrun.sh
# launches training. srun launches one torchrun per node; torchrun spawns
# --nproc_per_node workers and sets RANK/LOCAL_RANK/WORLD_SIZE automatically.
#
# Usage (from the repo root directory):
#   sbatch sbatch_files/train_torchrun.sbatch
#   sbatch --export=ALL,DEPTH=24,WANDB_RUN=my_run sbatch_files/train_torchrun.sbatch
#
# Overridable environment variables:
#   DEPTH             - transformer depth (default 12)
#   DEVICE_BATCH_SIZE - per-GPU batch size (default 16)
#   TARGET_PDR        - target param:data ratio (default 12)
#   WANDB_RUN         - wandb run name ('dummy' disables logging)
#   NANOCHAT_BASE_DIR      - data/cache/checkpoint directory
#   NANOCHAT_TOKENIZER_DIR - tokenizer directory
#   PYTHON_PATH            - path to Python executable
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

# ── Experiment identity (override via --export) ──────────────────────────────
DEPTH=${DEPTH:-12}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-32}
TARGET_PDR=${TARGET_PDR:-10.5}
WANDB_RUN=${WANDB_RUN:-"nanochat_32768_12_nano_bpe_climbmix"}
WANDB_ENTITY=${WANDB_ENTITY:-"tempus-apertus"}
NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"/iopsstor/scratch/cscs/jtempus/.cache/nanochat"}
export NANOCHAT_BASE_DIR

NANOCHAT_TOKENIZER_DIR=${NANOCHAT_TOKENIZER_DIR:-"/iopsstor/scratch/cscs/jtempus/tokenisation_lp/bpe_tokenizer/nano_bpe_climb_mix/32768/"}
export NANOCHAT_TOKENIZER_DIR

# ── Python path ──────────────────────────────────────────────────────────────
# Activate venv so all venv packages are importable, then add system
# site-packages (where torch lives) to PYTHONPATH as well.
VENV_DIR=${VENV_DIR:-"/iopsstor/scratch/cscs/jtempus/nanochat/.venv"}
source "$VENV_DIR/bin/activate"
VENV_PACKAGES=$(python -c "import site; print(site.getsitepackages()[0])")
export PYTHONPATH="$VENV_PACKAGES:/usr/local/lib/python3.12/dist-packages:$PYTHONPATH"
echo "Using Python: $(which python)"
echo "PYTHONPATH:    $PYTHONPATH"

GPUS_PER_NODE=${SLURM_GPUS_ON_NODE:-4}
TOTAL_GPUS=$((SLURM_NNODES * GPUS_PER_NODE))

echo "Total GPUs:        $TOTAL_GPUS (${SLURM_NNODES} nodes x ${GPUS_PER_NODE} GPUs/node)"
echo "Depth:             $DEPTH"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "Target PDR:        $TARGET_PDR"
echo "WANDB_RUN:         $WANDB_RUN"
echo "NANOCHAT_BASE_DIR: $NANOCHAT_BASE_DIR"
echo "======================================================================"

# ── Generate token_bytes.pt if not present ───────────────────────────────────
# Runs inside the nano_distributed container so torch is available.
TOKEN_BYTES_PATH="$NANOCHAT_TOKENIZER_DIR/token_bytes.pt"
if [[ ! -f "$TOKEN_BYTES_PATH" ]]; then
    echo "token_bytes.pt not found at $TOKEN_BYTES_PATH, generating..."
    srun --environment=nano_distributed --ntasks=1 bash -c "
        source $VENV_DIR/bin/activate && \
        VENV_PKG=\$(python -c \"import site; print(site.getsitepackages()[0])\") && \
        export PYTHONPATH=\$VENV_PKG:/usr/local/lib/python3.12/dist-packages:\$PYTHONPATH && \
        export NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR \
               NANOCHAT_TOKENIZER_DIR=$NANOCHAT_TOKENIZER_DIR && \
        cd $SLURM_SUBMIT_DIR && \
        python -m scripts.generate_token_bytes
    "
    echo "token_bytes.pt generated."
else
    echo "token_bytes.pt already exists, skipping generation."
fi

# ── Load secrets ─────────────────────────────────────────────────────────────
ENV_FILE="${HOME}/.env.local"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
    echo "Loaded secrets from $ENV_FILE"
else
    echo "WARNING: $ENV_FILE not found — WANDB_API_KEY may be unset"
fi

# ── Training ─────────────────────────────────────────────────────────────────
# srun launches 1 task per node. torchrun on each node spawns nproc_per_node
# workers and handles RANK/LOCAL_RANK/WORLD_SIZE + NCCL rendezvous.
# This matches how speedrun.sh launches training (torch.distributed.run).
echo "Starting distributed training..."

srun --environment=nano_distributed bash -c "
    source $VENV_DIR/bin/activate && \
    VENV_PKG=\$(python -c \"import site; print(site.getsitepackages()[0])\") && \
    export PYTHONPATH=\$VENV_PKG:/usr/local/lib/python3.12/dist-packages:\$PYTHONPATH && \
    export WANDB_API_KEY=$WANDB_API_KEY \
           WANDB_ENTITY=$WANDB_ENTITY \
           NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR \
           NANOCHAT_TOKENIZER_DIR=$NANOCHAT_TOKENIZER_DIR && \
    cd $SLURM_SUBMIT_DIR && \
    python -m torch.distributed.run \
        --nproc_per_node=$GPUS_PER_NODE \
        --nnodes=$SLURM_NNODES \
        --node-rank=\$SLURM_NODEID \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m scripts.base_train -- \
            --depth=$DEPTH \
            --device-batch-size=$DEVICE_BATCH_SIZE \
            --target-param-data-ratio=$TARGET_PDR \
            --run=$WANDB_RUN
"

echo "======================================================================"
echo "Training completed!"
echo "======================================================================"
