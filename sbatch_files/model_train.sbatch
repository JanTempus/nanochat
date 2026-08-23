#!/bin/bash
#SBATCH --job-name=nanochat-train
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:35:00
#SBATCH --partition=normal
#SBATCH --array=0-8
#SBATCH --output=logs/model-train-%A_%a.out
#SBATCH --error=logs/model-train-%A_%a.err

set -euo pipefail

# Submit one independent job per vocabulary/seed combination. There is
# intentionally no "%N" limit on --array, so all jobs may run simultaneously:
#   sbatch sbatch_files/model_train.sbatch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# ── Paths ────────────────────────────────────────────────────────────────────

export REPO_DIR="/capstor/store/cscs/swissai/a0229/jtempus/nanochat"
export VENV_DIR="$REPO_DIR/.venv"

export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"/iopsstor/scratch/cscs/jtempus/.cache/nanochat"}


export ROUNDING_SCHEME="det"
export TOKENIZER_DIR_TEMPLATE="/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/rounded_tokenizers/cross_over_climbmix400b_s7/vocab_{vocab}/lp_{vocab}_{rounding_scheme}"
RESOLVED_TOKENIZER_DIR_TEMPLATE=${TOKENIZER_DIR_TEMPLATE//\{rounding_scheme\}/$ROUNDING_SCHEME}

# Keep the --array upper bound above equal to the number of
# vocabulary/seed combinations minus one.
VOCAB_SIZE_LIST=(8192 16384 32768)
SEED_LIST=(42 43 44)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This script must be submitted with sbatch as a Slurm job array." >&2
    exit 2
fi

NUM_SEEDS=${#SEED_LIST[@]}
TOTAL_RUNS=$((${#VOCAB_SIZE_LIST[@]} * NUM_SEEDS))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_RUNS )); then
    echo "Array task $SLURM_ARRAY_TASK_ID has no matching vocabulary/seed combination." >&2
    exit 2
fi

VOCAB_INDEX=$((SLURM_ARRAY_TASK_ID / NUM_SEEDS))
SEED_INDEX=$((SLURM_ARRAY_TASK_ID % NUM_SEEDS))
VOCAB_SIZE=${VOCAB_SIZE_LIST[$VOCAB_INDEX]}
export SEED=${SEED_LIST[$SEED_INDEX]}

# ── Experiment ───────────────────────────────────────────────────────────────

# Edit this block to change the shared training configuration.
export DEPTH=12
export DEVICE_BATCH_SIZE=32
export TARGET_PDR=10.5
export DATASET_SHARDS=170

export MATRIX_LR=0.040 # yes
export WEIGHT_DECAY=0.14 # yes
export EMBEDDING_LR=0.60 # yes
export UNEMBEDDING_LR=0.016 # yes
export SCALAR_LR=0.50 # yes
export WARMUP_STEPS=40 # yes
export WARMDOWN_RATIO=0.80 # yes
export FINAL_LR_FRAC=0.05 # yes

export RUN_NAME_TEMPLATE="vocab_{vocab}_${ROUNDING_SCHEME}_d${DEPTH}_s${SEED}"
export WANDB_ENTITY=${WANDB_ENTITY:-"tempus-apertus"}
export WANDB_PROJECT=${WANDB_PROJECT:-"different_tokenisers"}

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
echo "Seed:       $SEED"
echo "Batch size: $DEVICE_BATCH_SIZE"
echo "Array task: $SLURM_ARRAY_TASK_ID"
echo "Vocab:      $VOCAB_SIZE"
echo "Rounding:   $ROUNDING_SCHEME"
echo "Run pattern: $RUN_NAME_TEMPLATE"
echo "Tuned HPs:  matrix_lr=$MATRIX_LR weight_decay=$WEIGHT_DECAY embedding_lr=$EMBEDDING_LR"
echo "            unembedding_lr=$UNEMBEDDING_LR scalar_lr=$SCALAR_LR warmup_steps=$WARMUP_STEPS"
echo "            warmdown_ratio=$WARMDOWN_RATIO final_lr_frac=$FINAL_LR_FRAC"

# ── Prepare this job's tokenizer ─────────────────────────────────────────────

tokenizer_dir=${RESOLVED_TOKENIZER_DIR_TEMPLATE//\{vocab\}/$VOCAB_SIZE}

if [[ ! -d "$tokenizer_dir" ]]; then
    echo "Tokenizer directory for vocabulary $VOCAB_SIZE does not exist: $tokenizer_dir" >&2
    exit 2
fi

export NANOCHAT_TOKENIZER_DIR=$tokenizer_dir
token_bytes_path="$tokenizer_dir/token_bytes.pt"

# The three seed jobs for a vocabulary share this generated tokenizer file.
exec 8>"$tokenizer_dir/.token-bytes.lock"
flock 8
if [[ ! -f "$token_bytes_path" ]]; then
    echo "Generating token_bytes.pt for vocabulary $VOCAB_SIZE in $tokenizer_dir"
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
flock -u 8

# ── Download pretraining dataset ─────────────────────────────────────────────

DATASET_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix"
mkdir -p "$NANOCHAT_BASE_DIR"

# Prevent concurrently starting array tasks from racing on the first download.
exec 9>"$NANOCHAT_BASE_DIR/.climbmix-download.lock"
flock 9
if [[ -d "$DATASET_DIR" ]]; then
    echo "Dataset already exists; skipping download: $DATASET_DIR"
else
    echo "Dataset not found; downloading $DATASET_SHARDS shards to $DATASET_DIR"
    srun \
        --nodes=1 \
        --ntasks=1 \
        --environment=nano_distributed \
        bash -c '
            source "$VENV_DIR/bin/activate"
            cd "$REPO_DIR"
            python -m nanochat.dataset -n "$DATASET_SHARDS"
        '
fi
flock -u 9

# ── Train this array task's vocabulary and seed ──────────────────────────

run_name=${RUN_NAME_TEMPLATE//\{vocab\}/$VOCAB_SIZE}

export NANOCHAT_TOKENIZER_DIR=$tokenizer_dir
export CURRENT_WANDB_RUN=$run_name
export CURRENT_MODEL_TAG=$run_name
export CURRENT_VOCAB_SIZE=$VOCAB_SIZE

echo "========================================================================"
echo "Training vocabulary: $CURRENT_VOCAB_SIZE"
echo "Tokenizer:           $NANOCHAT_TOKENIZER_DIR"
echo "W&B run:             $CURRENT_WANDB_RUN"
echo "Model tag:           $CURRENT_MODEL_TAG"
echo "========================================================================"

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
            --rdzv_id="${SLURM_JOB_ID}_${CURRENT_VOCAB_SIZE}" \
            -m scripts.base_train -- \
            --depth="$DEPTH" \
            --seed="$SEED" \
            --device-batch-size="$DEVICE_BATCH_SIZE" \
            --target-param-data-ratio="$TARGET_PDR" \
            --run="$CURRENT_WANDB_RUN" \
            --model-tag="$CURRENT_MODEL_TAG" \
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
