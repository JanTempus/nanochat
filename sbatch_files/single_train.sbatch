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
#   sbatch --time=HH:MM:SS sbatch_files/single_train.sbatch /path/to/tokenizer SEED TAG

if (( $# != 3 )); then
    echo "Usage: sbatch $0 /path/to/tokenizer SEED TAG" >&2
    exit 2
fi

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

export REPO_DIR="/capstor/store/cscs/swissai/a0229/jtempus/nanochat"
export VENV_DIR="$REPO_DIR/.venv"
export NANOCHAT_BASE_DIR="/iopsstor/scratch/cscs/jtempus/.cache/nanochat"

export DEPTH=24
export DEVICE_BATCH_SIZE=16
export TARGET_PDR=10.5
export DATASET_SHARDS=170

export MATRIX_LR=0.020
export WEIGHT_DECAY=0.28
export EMBEDDING_LR=0.30
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

# Resolve the submitted path after loading environment defaults so the command-
# line selection wins over any NANOCHAT_TOKENIZER_DIR value in .env.local. The
# directory leaf is the tokenizer name (for example, lp_8192_bias).
TOKENIZER_PATH=${1%/}
TOKENIZER_NAME=$(basename "$TOKENIZER_PATH")
TOKENIZER_PARENT=$(basename "$(dirname "$TOKENIZER_PATH")")
export SEED="$2"
RUN_TAG=$3

if [[ ! "$RUN_TAG" =~ ^[[:alnum:]_.-]+$ ]]; then
    echo "TAG must contain only letters, numbers, dots, underscores, or hyphens: $RUN_TAG" >&2
    exit 2
fi

# Rounded tokenizers use .../vocab_8192/lp_8192_bias, while baseline
# tokenizers can use .../basic_bpe/8192. Support both layouts.
if [[ "$TOKENIZER_PARENT" =~ ^vocab_([0-9]+)$ ]]; then
    VOCAB_SIZE=${BASH_REMATCH[1]}
elif [[ "$TOKENIZER_NAME" =~ (^|_)([0-9]+)($|_) ]]; then
    VOCAB_SIZE=${BASH_REMATCH[2]}
else
    echo "Could not infer vocabulary size from tokenizer path: $TOKENIZER_PATH" >&2
    exit 2
fi

export NANOCHAT_TOKENIZER_DIR="$TOKENIZER_PATH"
export RUN_NAME="${RUN_TAG}_${VOCAB_SIZE}_d${DEPTH}_s${SEED}"

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Job:       $SLURM_JOB_ID"
echo "Nodes:     $SLURM_NNODES"
echo "GPUs:      $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Seed:      $SEED"
echo "Tokenizer: $NANOCHAT_TOKENIZER_DIR"
echo "Tok name:  $TOKENIZER_NAME"
echo "Run tag:   $RUN_TAG"
echo "Vocab:     $VOCAB_SIZE"
echo "W&B run:   $RUN_NAME"
echo "Model tag: $RUN_NAME"

TOKEN_BYTES_PATH="$NANOCHAT_TOKENIZER_DIR/token_bytes.pt"
if [[ ! -f "$TOKEN_BYTES_PATH" ]]; then
    echo "token_bytes.pt not found; generating it now"
    srun \
        --nodes=1 \
        --ntasks=1 \
        --environment=nano_distributed \
        bash -c '
            source "$VENV_DIR/bin/activate"
            cd "$REPO_DIR"
            python -m scripts.generate_token_bytes
        '
else
    echo "token_bytes.pt already exists; skipping generation"
fi

DATASET_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix"
if [[ ! -d "$DATASET_DIR" ]]; then
    echo "Dataset directory not found; downloading to $DATASET_DIR"
    srun \
        --nodes=1 \
        --ntasks=1 \
        --environment=nano_distributed \
        bash -c '
            source "$VENV_DIR/bin/activate"
            cd "$REPO_DIR"
            python -m nanochat.dataset -n "$DATASET_SHARDS"
        '
else
    echo "Dataset directory already exists; skipping download: $DATASET_DIR"
fi


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
            --eval-every=200 \
            --sample-every=-1 \
            --save-every=-1
    '
