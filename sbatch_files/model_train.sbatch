#!/bin/bash
#SBATCH --job-name=nanochat-train
#SBATCH --account=a0229
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=32
#SBATCH --time=00:25:00
#SBATCH --partition=normal
#SBATCH --output=logs/slurm-%j.out
#SBATCH --error=logs/slurm-%j.err

set -euo pipefail

# Train all vocabulary sizes below sequentially in the same allocation:
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

# Edit this list to select the tokenizer vocabulary sizes to train.
VOCAB_SIZE_LIST=(8192 16384 32768)

# ── Experiment ───────────────────────────────────────────────────────────────

# Edit this block to change the shared training configuration.
export DEPTH=12
export SEED=42
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
export WANDB_PROJECT=${WANDB_PROJECT:-"Tuned_runs"}

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
echo "Vocabs:     ${VOCAB_SIZE_LIST[*]}"
echo "Rounding:   $ROUNDING_SCHEME"
echo "Run pattern: $RUN_NAME_TEMPLATE"
echo "Tuned HPs:  matrix_lr=$MATRIX_LR weight_decay=$WEIGHT_DECAY embedding_lr=$EMBEDDING_LR"
echo "            unembedding_lr=$UNEMBEDDING_LR scalar_lr=$SCALAR_LR warmup_steps=$WARMUP_STEPS"
echo "            warmdown_ratio=$WARMDOWN_RATIO final_lr_frac=$FINAL_LR_FRAC"

# ── Generate token_bytes.pt for every tokenizer ─────────────────────────────

for vocab_size in "${VOCAB_SIZE_LIST[@]}"; do
    tokenizer_dir=${RESOLVED_TOKENIZER_DIR_TEMPLATE//\{vocab\}/$vocab_size}

    if [[ ! -d "$tokenizer_dir" ]]; then
        echo "Tokenizer directory for vocabulary $vocab_size does not exist: $tokenizer_dir" >&2
        exit 2
    fi

    token_bytes_path="$tokenizer_dir/token_bytes.pt"
    if [[ ! -f "$token_bytes_path" ]]; then
        echo "Generating token_bytes.pt for vocabulary $vocab_size in $tokenizer_dir"
        export NANOCHAT_TOKENIZER_DIR=$tokenizer_dir
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
done

# ── Download pretraining dataset ─────────────────────────────────────────────

DATASET_DIR="$NANOCHAT_BASE_DIR/base_data_climbmix"

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

# ── Train every vocabulary size sequentially ────────────────────────────────

for vocab_size in "${VOCAB_SIZE_LIST[@]}"; do
    tokenizer_dir=${RESOLVED_TOKENIZER_DIR_TEMPLATE//\{vocab\}/$vocab_size}
    run_name=${RUN_NAME_TEMPLATE//\{vocab\}/$vocab_size}
    model_tag=$run_name

    export NANOCHAT_TOKENIZER_DIR=$tokenizer_dir
    export CURRENT_WANDB_RUN=$run_name
    export CURRENT_MODEL_TAG=$model_tag
    export CURRENT_VOCAB_SIZE=$vocab_size

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
done
