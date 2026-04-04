#!/bin/bash
#SBATCH --job-name=nanochat-benchmark
#SBATCH --account=a139
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=4          # 4 tasks per node (one per GPU — nodes always have exactly 4 GPUs)
#SBATCH --gpus-per-node=4            # 4 GPUs per node (fixed: cannot request fewer)
#SBATCH --cpus-per-task=8            # CPUs per task (32 / 4 = 8 per GPU)
#SBATCH --time=00:15:00              # Should be plenty for 1k steps
#SBATCH --partition=normal
#SBATCH --output=logs/slurm-benchmark-%j.out
#SBATCH --error=logs/slurm-benchmark-%j.err

# ============================================================================
# Short benchmark run to estimate training throughput.
# Runs 1000 steps and reports tokens/sec + ETA for the full run.
# ============================================================================

echo "======================================================================"
echo "SLURM Benchmark Job"
echo "======================================================================"
echo "Job ID:         $SLURM_JOB_ID"
echo "Nodes:          $SLURM_NNODES"
echo "Node List:      $SLURM_JOB_NODELIST"
echo "GPUs per node:  $SLURM_GPUS_ON_NODE"
echo "======================================================================"

export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1
mkdir -p logs

# ── Networking ───────────────────────────────────────────────────────────────
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
echo "Master: $MASTER_ADDR:$MASTER_PORT"

# ── Config (match your real training run) ────────────────────────────────────
DEPTH=${DEPTH:-12}
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-16}
TARGET_PDR=${TARGET_PDR:-8}
NANOCHAT_BASE_DIR=${NANOCHAT_BASE_DIR:-"/iopsstor/scratch/cscs/jtempus/nanochat"}
export NANOCHAT_BASE_DIR


NANOCHAT_TOKENIZER_DIR=${NANOCHAT_TOKENIZER_DIR:-"/iopsstor/scratch/cscs/jtempus/tokenisation_lp/rounded_tokenizers/basic_finewebedu20B_n180000_seed42/vocab_8192/lp_8192_bias"}
export NANOCHAT_TOKENIZER_DIR

PYTHON_PATH=${PYTHON_PATH:-"/iopsstor/scratch/cscs/jtempus/nanochat/.venv/bin/python"}
echo "Using Python: $PYTHON_PATH"

TOTAL_GPUS=$((SLURM_NNODES * SLURM_GPUS_ON_NODE))
export WORLD_SIZE=$TOTAL_GPUS

echo "Total GPUs:        $TOTAL_GPUS"
echo "Depth:             $DEPTH"
echo "Device batch size: $DEVICE_BATCH_SIZE"
echo "======================================================================"

# ── Generate token_bytes.pt if not present ───────────────────────────────────
TOKEN_BYTES_PATH="$NANOCHAT_TOKENIZER_DIR/token_bytes.pt"
if [[ ! -f "$TOKEN_BYTES_PATH" ]]; then
    echo "token_bytes.pt not found, generating..."
    NANOCHAT_TOKENIZER_DIR=$NANOCHAT_TOKENIZER_DIR $PYTHON_PATH -m scripts.generate_token_bytes
    echo "token_bytes.pt generated."
else
    echo "token_bytes.pt already exists, skipping generation."
fi

# ── Load secrets ──────────────────────────────────────────────────────────────
ENV_FILE="${HOME}/.env.local"
if [[ -f "$ENV_FILE" ]]; then
    set -a; source "$ENV_FILE"; set +a
fi

# ── Benchmark run (1k steps, no wandb logging) ────────────────────────────────
echo "Starting benchmark (1000 steps)..."
srun --environment=flashnano bash -c "
    echo '[Rank \$SLURM_PROCID / LocalRank \$SLURM_LOCALID] Starting...' && \
    cd $SLURM_SUBMIT_DIR || { echo 'ERROR: cd to $SLURM_SUBMIT_DIR failed'; exit 1; } && \
    export PYTHONPATH=/usr/local/lib/python3.12/dist-packages:\$PYTHONPATH && \
    export NCCL_DEBUG=INFO && \
    export NCCL_DEBUG_SUBSYS=INIT,NET && \
    NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR \
    NANOCHAT_TOKENIZER_DIR=$NANOCHAT_TOKENIZER_DIR \
    RANK=\$SLURM_PROCID \
    LOCAL_RANK=\$SLURM_LOCALID \
    $PYTHON_PATH -m scripts.base_train \
        --depth=$DEPTH \
        --device-batch-size=$DEVICE_BATCH_SIZE \
        --target-param-data-ratio=$TARGET_PDR \
        --fp8 \
        --num-iterations=1000 \
        --eval-every=-1 \
        --core-metric-every=-1 \
        --sample-every=-1 \
        --save-every=-1 \
        --run=dummy
"

echo "======================================================================"
echo "Benchmark completed!"
echo "======================================================================"
