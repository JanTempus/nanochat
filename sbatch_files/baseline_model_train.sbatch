#!/bin/bash
#SBATCH --job-name=baseline-model-train
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:35:00
#SBATCH --partition=normal
#SBATCH --array=0-35
#SBATCH --output=logs/baseline-model-train-%A_%a.out
#SBATCH --error=logs/baseline-model-train-%A_%a.err

set -euo pipefail

# Submit one independent job per tokenizer-family/vocabulary/seed combination.
# There is intentionally no "%N" array limit, so all jobs may run concurrently:
#   sbatch sbatch_files/baseline_model_train.sbatch

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

# ── Paths and experiment grid ──────────────────────────────────────────────

export REPO_DIR="/capstor/store/cscs/swissai/a0229/jtempus/nanochat"
export VENV_DIR="$REPO_DIR/.venv"
export NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"/iopsstor/scratch/cscs/jtempus/.cache/nanochat"}

export TOKENIZER_BASE_DIR="/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/baseline_tokenisers"
TOKENIZER_FAMILY_LIST=(basic_bpe bpe_vocab_unigram frequent_pretoken unigram)
VOCAB_SIZE_LIST=(8192 16384 32768)
SEED_LIST=(42 43 44)

if [[ -z "${SLURM_ARRAY_TASK_ID:-}" ]]; then
    echo "This script must be submitted with sbatch as a Slurm job array." >&2
    exit 2
fi

NUM_FAMILIES=${#TOKENIZER_FAMILY_LIST[@]}
NUM_VOCABS=${#VOCAB_SIZE_LIST[@]}
NUM_SEEDS=${#SEED_LIST[@]}
RUNS_PER_FAMILY=$((NUM_VOCABS * NUM_SEEDS))
TOTAL_RUNS=$((NUM_FAMILIES * RUNS_PER_FAMILY))

if (( SLURM_ARRAY_TASK_ID < 0 || SLURM_ARRAY_TASK_ID >= TOTAL_RUNS )); then
    echo "Array task $SLURM_ARRAY_TASK_ID has no matching tokenizer/vocabulary/seed combination." >&2
    exit 2
fi

FAMILY_INDEX=$((SLURM_ARRAY_TASK_ID / RUNS_PER_FAMILY))
FAMILY_TASK_INDEX=$((SLURM_ARRAY_TASK_ID % RUNS_PER_FAMILY))
VOCAB_INDEX=$((FAMILY_TASK_INDEX / NUM_SEEDS))
SEED_INDEX=$((FAMILY_TASK_INDEX % NUM_SEEDS))

TOKENIZER_FAMILY=${TOKENIZER_FAMILY_LIST[$FAMILY_INDEX]}
VOCAB_SIZE=${VOCAB_SIZE_LIST[$VOCAB_INDEX]}
export SEED=${SEED_LIST[$SEED_INDEX]}

# Keep --array's upper bound equal to TOTAL_RUNS - 1 when editing these lists.
if (( TOTAL_RUNS != 36 )); then
    echo "The grid contains $TOTAL_RUNS runs, but #SBATCH --array is configured for 36." >&2
    exit 2
fi

# ── Shared training configuration ──────────────────────────────────────────

export DEPTH=12
export DEVICE_BATCH_SIZE=32
export TARGET_PDR=10.5
export DATASET_SHARDS=170

export MATRIX_LR=0.020 # yes
export WEIGHT_DECAY=0.28 # yes
export EMBEDDING_LR=0.50 # yes
export UNEMBEDDING_LR=0.016 # yes
export SCALAR_LR=0.50 # yes
export WARMUP_STEPS=40 # yes
export WARMDOWN_RATIO=0.65 # yes
export FINAL_LR_FRAC=0.05 # yes


export WANDB_ENTITY=${WANDB_ENTITY:-"tempus-apertus"}
export WANDB_PROJECT=${WANDB_PROJECT:-"different_tokenisers"}

# ── W&B credentials ──────────────────────────────────────────────────────────

set -a
source "$HOME/.env.local"
set +a

# ── Distributed setup ─────────────────────────────────────────────────────

export GPUS_PER_NODE=4
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500

echo "Job:              $SLURM_JOB_ID"
echo "Array task:       $SLURM_ARRAY_TASK_ID"
echo "Nodes:            $SLURM_NNODES"
echo "GPUs:             $((SLURM_NNODES * GPUS_PER_NODE))"
echo "Master:           $MASTER_ADDR:$MASTER_PORT"
echo "Tokenizer family: $TOKENIZER_FAMILY"
echo "Vocabulary:       $VOCAB_SIZE"
echo "Depth:            $DEPTH"
echo "Seed:             $SEED"
echo "Batch size:       $DEVICE_BATCH_SIZE"

# ── Prepare this job's tokenizer ─────────────────────────────────────────────

tokenizer_dir="$TOKENIZER_BASE_DIR/$TOKENIZER_FAMILY/$VOCAB_SIZE"

if [[ ! -d "$tokenizer_dir" ]]; then
    echo "Tokenizer directory does not exist: $tokenizer_dir" >&2
    exit 2
fi

export NANOCHAT_TOKENIZER_DIR=$tokenizer_dir
token_bytes_path="$tokenizer_dir/token_bytes.pt"

# Seed jobs for the same tokenizer share this generated file.
exec 8>"$tokenizer_dir/.token-bytes.lock"
flock 8
if [[ ! -f "$token_bytes_path" ]]; then
    echo "Generating token_bytes.pt in $tokenizer_dir"
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

# ── Download pretraining dataset ───────────────────────────────────────────

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

# ── Train this tokenizer/vocabulary/seed combination ─────────────────────────

run_name="baseline_${TOKENIZER_FAMILY}_vocab_${VOCAB_SIZE}_d${DEPTH}_s${SEED}"

export CURRENT_WANDB_RUN=$run_name
export CURRENT_MODEL_TAG=$run_name
export CURRENT_VOCAB_SIZE=$VOCAB_SIZE
export CURRENT_TOKENIZER_FAMILY=$TOKENIZER_FAMILY

echo "========================================================================"
echo "Tokenizer: $NANOCHAT_TOKENIZER_DIR"
echo "W&B run:   $CURRENT_WANDB_RUN"
echo "Model tag: $CURRENT_MODEL_TAG"
echo "========================================================================"

srun \
    --nodes=1 \
    --ntasks=1 \
    --environment=nano_distributed \
    bash -c '
        source "$VENV_DIR/bin/activate"
        cd "$REPO_DIR"

        exec python -m torch.distributed.run \
            --standalone \
            --nproc_per_node="$GPUS_PER_NODE" \
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
