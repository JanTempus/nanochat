#!/bin/bash
#SBATCH --job-name=d8-lr-boundary
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --array=0-8%4
#SBATCH --output=logs/d8-lr-boundary-%A_%a.out
#SBATCH --error=logs/d8-lr-boundary-%A_%a.err

# This sweep extends d8_blend_m19_a48_7_3065610_7_3065797 while keeping
# every setting except matrix_lr and unembedding_lr fixed.
#
# Tasks 0-10 are the boundary search. Task 11 combines the best marginal
# values and should normally be submitted after inspecting tasks 0-10:
#
#   sbatch --array=0-10 sbatch_files/depth8_lr_boundary_sweep.sbatch
#   sbatch --array=11 \
#     --export=ALL,BEST_MATRIX_LR=0.180,BEST_UNEMBEDDING_LR=0.054 \
#     sbatch_files/depth8_lr_boundary_sweep.sbatch

DEPTH=8
SEED=42
DEVICE_BATCH_SIZE=32
TARGET_PDR=10.5
WARMUP_STEPS=40
WARMDOWN_RATIO=0.80
FINAL_LR_FRAC=0.05

WEIGHT_DECAY=0.14
EMBEDDING_LR=0.60
SCALAR_LR=0.50

ANCHOR_MATRIX_LR=0.080
ANCHOR_UNEMBEDDING_LR=0.024
BEST_MATRIX_LR_WAS_SET=${BEST_MATRIX_LR+x}
BEST_UNEMBEDDING_LR_WAS_SET=${BEST_UNEMBEDDING_LR+x}
BEST_MATRIX_LR=${BEST_MATRIX_LR:-$ANCHOR_MATRIX_LR}
BEST_UNEMBEDDING_LR=${BEST_UNEMBEDDING_LR:-$ANCHOR_UNEMBEDDING_LR}

# CORE_METRIC_EVERY is deliberately larger than the training horizon. The
# trainer still evaluates CORE on the final step, yielding one CORE result.
CORE_METRIC_EVERY=1000000
CORE_METRIC_MAX_PER_TASK=500

CONFIGS=(
    "anchor|$ANCHOR_MATRIX_LR|$ANCHOR_UNEMBEDDING_LR"
    "matrix_0p120|0.120|$ANCHOR_UNEMBEDDING_LR"
    "matrix_0p180|0.180|$ANCHOR_UNEMBEDDING_LR"
    "matrix_0p270|0.270|$ANCHOR_UNEMBEDDING_LR"
    "matrix_0p405|0.405|$ANCHOR_UNEMBEDDING_LR"
    "matrix_0p608|0.608|$ANCHOR_UNEMBEDDING_LR"
    "unembedding_0p036|$ANCHOR_MATRIX_LR|0.036"
    "unembedding_0p054|$ANCHOR_MATRIX_LR|0.054"
    "unembedding_0p081|$ANCHOR_MATRIX_LR|0.081"
    "unembedding_0p122|$ANCHOR_MATRIX_LR|0.122"
    "unembedding_0p182|$ANCHOR_MATRIX_LR|0.182"
    "best_pair|$BEST_MATRIX_LR|$BEST_UNEMBEDDING_LR"
)

TASK_ID=${SLURM_ARRAY_TASK_ID:?SLURM_ARRAY_TASK_ID must be set}
if (( TASK_ID < 0 || TASK_ID >= ${#CONFIGS[@]} )); then
    echo "Invalid task ID $TASK_ID; expected 0-$(( ${#CONFIGS[@]} - 1 ))" >&2
    exit 2
fi

IFS='|' read -r TRIAL_LABEL MATRIX_LR UNEMBEDDING_LR <<< "${CONFIGS[$TASK_ID]}"

echo "CONFIG|$TASK_ID|$TRIAL_LABEL|$MATRIX_LR|$WEIGHT_DECAY|$EMBEDDING_LR|$UNEMBEDDING_LR|$SCALAR_LR|$WARMUP_STEPS|$WARMDOWN_RATIO|$FINAL_LR_FRAC|$CORE_METRIC_EVERY|$CORE_METRIC_MAX_PER_TASK"

if [[ "$TRIAL_LABEL" == "best_pair" && \
      ( -z "$BEST_MATRIX_LR_WAS_SET" || -z "$BEST_UNEMBEDDING_LR_WAS_SET" ) ]]; then
    echo "Warning: task 11 is using the anchor defaults; set BEST_MATRIX_LR and BEST_UNEMBEDDING_LR after the boundary search." >&2
fi

if [[ "${DRY_RUN:-0}" == "1" ]]; then
    exit 0
fi

export REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/nanochat}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/iopsstor/scratch/cscs/jtempus/.cache/nanochat}"
export NANOCHAT_TOKENIZER_DIR="${NANOCHAT_TOKENIZER_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/rounded_tokenizers/cross_over_climbmix400b_s7/vocab_32768/lp_32768_det}"

set -a
source "$HOME/.env.local"
set +a

export WANDB_ENTITY="tempus-apertus"
export WANDB_PROJECT="hyperparameter_search"

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

RUN_SUFFIX="${SLURM_ARRAY_JOB_ID}_${TASK_ID}_${SLURM_JOB_ID}"
export WANDB_RUN="d8_lr_boundary_${TRIAL_LABEL}_${RUN_SUFFIX}"
export MODEL_TAG="d8_lr_boundary_${TRIAL_LABEL}_${RUN_SUFFIX}"

export DEPTH
export SEED
export DEVICE_BATCH_SIZE
export TARGET_PDR
export MATRIX_LR
export WEIGHT_DECAY
export EMBEDDING_LR
export UNEMBEDDING_LR
export SCALAR_LR
export WARMUP_STEPS
export WARMDOWN_RATIO
export FINAL_LR_FRAC
export CORE_METRIC_EVERY
export CORE_METRIC_MAX_PER_TASK

echo "W&B:        $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN"
echo "Model tag:  $MODEL_TAG"
echo "Trial:      $TRIAL_LABEL"
echo "Tokenizer:  $NANOCHAT_TOKENIZER_DIR"
echo "Dataset:    $DATASET_DIR"

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
            --depth="$DEPTH" \
            --seed="$SEED" \
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --target-param-data-ratio="$TARGET_PDR" \
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
            --core-metric-every="$CORE_METRIC_EVERY" \
            --core-metric-max-per-task="$CORE_METRIC_MAX_PER_TASK" \
            --sample-every=-1 \
            --save-every=-1
    '
