#!/bin/bash

# Speedrun adapted for the "satay" machine: RTX A6000 (Ampere, sm_86, 48GB) GPUs.
# Differences from runs/speedrun.sh (which targets 8XH100):
# - Uses only 2 GPUs (pick which via SATAY_GPUS, default "0,1"). Gradient accumulation
#   kicks in automatically, so results match the 8-GPU run, it just takes longer.
# - Flash Attention 2 instead of FA3 (auto-selected by nanochat/flash_attention.py on sm8x).
# - No --fp8: Ampere has no FP8 tensor cores.
# - Smaller --device-batch-size to fit in 48GB of VRAM.
# Measured on 2x A6000: d24 pretraining runs at ~37.6k tok/sec (58% MFU, 42GB peak at
# device-batch-size 8; 16 OOMs), i.e. ~43 hours of pretraining, ~2 days end to end.

# 1) Example launch:
# bash runs/satay.sh
# 2) In a screen session, on GPUs 2 and 3, with wandb logging:
# SATAY_GPUS=2,3 WANDB_RUN=satay screen -L -Logfile runs/satay.log -S satay bash runs/satay.sh

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$HOME/.cache/nanochat"
mkdir -p $NANOCHAT_BASE_DIR

# Only ever use 2 GPUs on this machine
export CUDA_VISIBLE_DEVICES="${SATAY_GPUS:-0,1}"
NPROC=2
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-8}"

# -----------------------------------------------------------------------------
# Python venv setup with uv
command -v uv &> /dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
[ -d ".venv" ] || uv venv
uv sync --extra gpu
source .venv/bin/activate

# -----------------------------------------------------------------------------
# wandb setup (see runs/speedrun.sh)
if [ -z "$WANDB_RUN" ]; then
    WANDB_RUN=dummy
fi

python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer
python -m nanochat.dataset -n 8
python -m nanochat.dataset -n 170 &
DATASET_DOWNLOAD_PID=$!
python -m scripts.tok_train
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_train -- --depth=24 --target-param-data-ratio=8 --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC -m scripts.base_eval -- --device-batch-size=$DEVICE_BATCH_SIZE

# -----------------------------------------------------------------------------
# SFT
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- --device-batch-size=$DEVICE_BATCH_SIZE --run=$WANDB_RUN
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_eval -- -i sft

# python -m scripts.chat_cli -p "Why is the sky blue?"
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
python -m nanochat.report generate
