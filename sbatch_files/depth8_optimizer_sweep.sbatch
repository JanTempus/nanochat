#!/bin/bash
#SBATCH --job-name=d8-opt-sweep
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=03:00:00
#SBATCH --partition=normal
#SBATCH --array=0-49%4
#SBATCH --output=logs/d8-sweep-%A_%a.out
#SBATCH --error=logs/d8-sweep-%A_%a.err

# This file deliberately keeps depth, batch sizing, and the LR schedule fixed.
# It sweeps Muon and AdamW independently and does not enable FP8.

# ── Fixed training configuration ─────────────────────────────────────────────

DEPTH=8
SEED=42
DEVICE_BATCH_SIZE=32
TARGET_PDR=10.5
WARMUP_STEPS=40
WARMDOWN_RATIO=0.65
FINAL_LR_FRAC=0.05

BASE_MATRIX_LR=0.020
BASE_WEIGHT_DECAY=0.28
BASE_EMBEDDING_LR=0.30
BASE_UNEMBEDDING_LR=0.008
BASE_SCALAR_LR=0.50

# ── Build the 50 non-baseline configurations ─────────────────────────────────

CONFIGS=()

MATRIX_LRS=(0.010 0.015 0.020 0.030 0.040)
WEIGHT_DECAYS=(0.14 0.21 0.28 0.42 0.56)

for matrix_lr in "${MATRIX_LRS[@]}"; do
    for weight_decay in "${WEIGHT_DECAYS[@]}"; do
        if [[ "$matrix_lr" == "$BASE_MATRIX_LR" && "$weight_decay" == "$BASE_WEIGHT_DECAY" ]]; then
            continue
        fi
        CONFIGS+=("muon|$matrix_lr|$weight_decay|$BASE_EMBEDDING_LR|$BASE_UNEMBEDDING_LR|$BASE_SCALAR_LR")
    done
done

EMBEDDING_LRS=(0.15 0.30 0.60)
UNEMBEDDING_LRS=(0.004 0.008 0.016)
SCALAR_LRS=(0.20 0.50 1.00)

for embedding_lr in "${EMBEDDING_LRS[@]}"; do
    for unembedding_lr in "${UNEMBEDDING_LRS[@]}"; do
        for scalar_lr in "${SCALAR_LRS[@]}"; do
            if [[ "$embedding_lr" == "$BASE_EMBEDDING_LR" && \
                  "$unembedding_lr" == "$BASE_UNEMBEDDING_LR" && \
                  "$scalar_lr" == "$BASE_SCALAR_LR" ]]; then
                continue
            fi
            CONFIGS+=("adamw|$BASE_MATRIX_LR|$BASE_WEIGHT_DECAY|$embedding_lr|$unembedding_lr|$scalar_lr")
        done
    done
done

# ── Select a sweep trial or an explicit confirmation configuration ───────────

SWEEP_MODE=${SWEEP_MODE:-sweep}

if [[ "$SWEEP_MODE" == "sweep" ]]; then
    TASK_ID=$SLURM_ARRAY_TASK_ID
    IFS='|' read -r OPTIMIZER_FAMILY MATRIX_LR WEIGHT_DECAY EMBEDDING_LR UNEMBEDDING_LR SCALAR_LR <<< "${CONFIGS[$TASK_ID]}"
    TRIAL_LABEL="${OPTIMIZER_FAMILY}_${TASK_ID}"
else
    OPTIMIZER_FAMILY=confirm
    TASK_ID=$SLURM_ARRAY_TASK_ID
    TRIAL_LABEL=confirm
fi

print_configuration() {
    echo "CONFIG|$TASK_ID|$OPTIMIZER_FAMILY|$MATRIX_LR|$WEIGHT_DECAY|$EMBEDDING_LR|$UNEMBEDDING_LR|$SCALAR_LR|$WARMUP_STEPS|$WARMDOWN_RATIO|$FINAL_LR_FRAC"
}

print_configuration

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

# ── Paths and credentials ────────────────────────────────────────────────────

export REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/nanochat}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/iopsstor/scratch/cscs/jtempus/.cache/nanochat}"
export NANOCHAT_TOKENIZER_DIR="${NANOCHAT_TOKENIZER_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/rounded_tokenizers/cross_over_climbmix400b_s7/vocab_32768/lp_32768_det}"

ENV_FILE="$HOME/.env.local"
set -a
source "$ENV_FILE"
set +a

export WANDB_ENTITY="tempus-apertus"
export WANDB_PROJECT="hyperparameter_search"

# ── Download the dataset once when its directory is absent ───────────────────

DATASET_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix"
mkdir -p "$NANOCHAT_BASE_DIR"

exec 9>"$NANOCHAT_BASE_DIR/.climbmix-download.lock"
flock 9
if [[ -d "$DATASET_DIR" ]]; then
    echo "Dataset directory already exists; skipping download: $DATASET_DIR"
else
    echo "Dataset directory is missing; downloading 170 training shards to $DATASET_DIR"
    srun \
        --nodes=1 \
        --ntasks=1 \
        --environment=nano_distributed \
        bash -c '
            source "$VENV_DIR/bin/activate"
            cd "$REPO_DIR"
            python -m nanochat.dataset -n 170
        '
fi
flock -u 9

# ── Unique experiment identity ───────────────────────────────────────────────

JOB_ID=$SLURM_JOB_ID
ARRAY_JOB_ID=$SLURM_ARRAY_JOB_ID
RUN_SUFFIX="${ARRAY_JOB_ID}_${TASK_ID}_${JOB_ID}"

export WANDB_RUN="d8_${TRIAL_LABEL}_${RUN_SUFFIX}"
export MODEL_TAG="d8_${TRIAL_LABEL}_${RUN_SUFFIX}"

export MATRIX_LR
export WEIGHT_DECAY
export EMBEDDING_LR
export UNEMBEDDING_LR
export SCALAR_LR
export WARMUP_STEPS
export WARMDOWN_RATIO
export FINAL_LR_FRAC
export SEED

echo "W&B:        $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN"
echo "Model tag:  $MODEL_TAG"
echo "Tokenizer:  $NANOCHAT_TOKENIZER_DIR"
echo "Dataset:    $DATASET_DIR"

# ── Train on one node with four BF16 workers ─────────────────────────────────

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

srun \
    --nodes=1 \
    --ntasks=1 \
    --environment=nano_distributed \
    bash -c '
        source "$VENV_DIR/bin/activate"
        cd "$REPO_DIR"

        exec python -m torch.distributed.run \
            --standalone \
            --nproc_per_node=4 \
            -m scripts.base_train -- \
            --depth=8 \
            --seed="$SEED" \
            --device-batch-size=32 \
            --target-param-data-ratio=10.5 \
            --run="$WANDB_RUN" \
            --model-tag="$MODEL_TAG" \
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
            --core-metric-every=-1 \
            --sample-every=-1 \
            --save-every=-1
    '
