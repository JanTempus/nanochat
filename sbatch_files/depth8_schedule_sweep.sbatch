#!/bin/bash
#SBATCH --job-name=d8-schedule-sweep
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:12:00
#SBATCH --partition=normal
#SBATCH --array=0-25%4
#SBATCH --output=logs/d8-schedule-%A_%a.out
#SBATCH --error=logs/d8-schedule-%A_%a.err

# This sweep varies only the shared learning-rate schedule.

DEPTH=8
SEED=42
DEVICE_BATCH_SIZE=32
TARGET_PDR=10.5

MATRIX_LR=0.020
WEIGHT_DECAY=0.28
EMBEDDING_LR=0.30
UNEMBEDDING_LR=0.016
SCALAR_LR=0.50

CONFIGS=()

WARMUP_STEPS_GRID=(20 40 80)
WARMDOWN_RATIO_GRID=(0.45 0.65 0.80)
FINAL_LR_FRAC_GRID=(0.00 0.05 0.10)

for warmup_steps in "${WARMUP_STEPS_GRID[@]}"; do
    for warmdown_ratio in "${WARMDOWN_RATIO_GRID[@]}"; do
        for final_lr_frac in "${FINAL_LR_FRAC_GRID[@]}"; do
            if [[ "$warmup_steps" == "40" && "$warmdown_ratio" == "0.65" && "$final_lr_frac" == "0.05" ]]; then
                continue
            fi
            CONFIGS+=("$warmup_steps|$warmdown_ratio|$final_lr_frac")
        done
    done
done

TASK_ID=$SLURM_ARRAY_TASK_ID
IFS='|' read -r WARMUP_STEPS WARMDOWN_RATIO FINAL_LR_FRAC <<< "${CONFIGS[$TASK_ID]}"

echo "CONFIG|$TASK_ID|$WARMUP_STEPS|$WARMDOWN_RATIO|$FINAL_LR_FRAC|$MATRIX_LR|$WEIGHT_DECAY|$EMBEDDING_LR|$UNEMBEDDING_LR|$SCALAR_LR|$DEPTH|$SEED|$DEVICE_BATCH_SIZE|$TARGET_PDR"

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
export WANDB_PROJECT="Schedule_search"

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
export WANDB_RUN="d8_schedule_${TASK_ID}_${RUN_SUFFIX}"
export MODEL_TAG="d8_schedule_${TASK_ID}_${RUN_SUFFIX}"

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
