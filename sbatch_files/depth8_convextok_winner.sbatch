#!/bin/bash
#SBATCH --job-name=d8-convextok-winner
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:20:00
#SBATCH --partition=normal
#SBATCH --output=logs/d8-convextok-winner-%j.out
#SBATCH --error=logs/d8-convextok-winner-%j.err

# Frozen ConvexTok winner configuration.
DEPTH=8
SEED=42
DEVICE_BATCH_SIZE=32
TARGET_PDR=10.5

MATRIX_LR=0.080
WEIGHT_DECAY=0.28
EMBEDDING_LR=0.60
UNEMBEDDING_LR=0.024
SCALAR_LR=0.50
WARMUP_STEPS=40
WARMDOWN_RATIO=0.80
FINAL_LR_FRAC=0.05
LR_SCHEDULE=linear

# scripts.base_train currently implements the linear schedule directly and does
# not expose an --lr-schedule argument. Fail loudly if this file is edited to
# request a schedule the trainer cannot select.
if [[ "$LR_SCHEDULE" != "linear" ]]; then
    echo "Unsupported LR schedule: $LR_SCHEDULE (expected linear)" >&2
    exit 2
fi

export REPO_DIR="${REPO_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/nanochat}"
export VENV_DIR="${VENV_DIR:-$REPO_DIR/.venv}"
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-/iopsstor/scratch/cscs/jtempus/.cache/nanochat}"
export NANOCHAT_TOKENIZER_DIR="${NANOCHAT_TOKENIZER_DIR:-/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/rounded_tokenizers/cross_over_climbmix400b_s7/vocab_32768/lp_32768_det}"

set -a
source "$HOME/.env.local"
set +a

export WANDB_ENTITY="${WANDB_ENTITY:-tempus-apertus}"
export WANDB_PROJECT="${WANDB_PROJECT:-hyperparameter_search}"

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

RUN_SUFFIX="${SLURM_JOB_ID}"
export WANDB_RUN="${WANDB_RUN:-d8_convextok_winner_${RUN_SUFFIX}}"
export MODEL_TAG="${MODEL_TAG:-d8_convextok_winner_${RUN_SUFFIX}}"

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

echo "W&B:         $WANDB_ENTITY/$WANDB_PROJECT/$WANDB_RUN"
echo "Model tag:   $MODEL_TAG"
echo "Tokenizer:   $NANOCHAT_TOKENIZER_DIR"
echo "Dataset:     $DATASET_DIR"
echo "Winner:      matrix_lr=$MATRIX_LR weight_decay=$WEIGHT_DECAY embedding_lr=$EMBEDDING_LR"
echo "             unembedding_lr=$UNEMBEDDING_LR scalar_lr=$SCALAR_LR warmup_steps=$WARMUP_STEPS"
echo "             warmdown_ratio=$WARMDOWN_RATIO final_lr_frac=$FINAL_LR_FRAC lr_schedule=$LR_SCHEDULE"

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
            --core-metric-every=-1 \
            --sample-every=-1 \
            --save-every=-1
    '
