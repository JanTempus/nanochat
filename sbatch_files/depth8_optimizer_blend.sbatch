#!/bin/bash
#SBATCH --job-name=d8-opt-blend
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:10:00
#SBATCH --partition=normal
#SBATCH --array=0-11%4
#SBATCH --output=logs/d8-blend-%A_%a.out
#SBATCH --error=logs/d8-blend-%A_%a.err

DEPTH=8
SEED=42
DEVICE_BATCH_SIZE=32
TARGET_PDR=10.5
WARMUP_STEPS=40
WARMDOWN_RATIO=0.8
FINAL_LR_FRAC=0.05

MUON_CANDIDATES=(
    "15|0.030|0.21"
    "14|0.030|0.14"
    "19|0.040|0.14"
    "16|0.030|0.28"
)

ADAMW_CANDIDATES=(
    "39|0.30|0.016|0.50"
    "48|0.60|0.016|0.50"
    "38|0.30|0.016|0.20"
)

CONFIGS=()

for muon_candidate in "${MUON_CANDIDATES[@]}"; do
    IFS='|' read -r muon_run matrix_lr weight_decay <<< "$muon_candidate"
    for adamw_candidate in "${ADAMW_CANDIDATES[@]}"; do
        IFS='|' read -r adamw_run embedding_lr unembedding_lr scalar_lr <<< "$adamw_candidate"
        CONFIGS+=("$muon_run|$adamw_run|$matrix_lr|$weight_decay|$embedding_lr|$unembedding_lr|$scalar_lr")
    done
done

TASK_ID=$SLURM_ARRAY_TASK_ID
IFS='|' read -r MUON_RUN ADAMW_RUN MATRIX_LR WEIGHT_DECAY EMBEDDING_LR UNEMBEDDING_LR SCALAR_LR <<< "${CONFIGS[$TASK_ID]}"

echo "CONFIG|$TASK_ID|$MUON_RUN|$ADAMW_RUN|$MATRIX_LR|$WEIGHT_DECAY|$EMBEDDING_LR|$UNEMBEDDING_LR|$SCALAR_LR|$WARMUP_STEPS|$WARMDOWN_RATIO|$FINAL_LR_FRAC"

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
export WANDB_RUN="d8_blend_m${MUON_RUN}_a${ADAMW_RUN}_${TASK_ID}_${RUN_SUFFIX}"
export MODEL_TAG="d8_blend_m${MUON_RUN}_a${ADAMW_RUN}_${TASK_ID}_${RUN_SUFFIX}"

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
echo "Muon run:   $MUON_RUN"
echo "AdamW run:  $ADAMW_RUN"
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
