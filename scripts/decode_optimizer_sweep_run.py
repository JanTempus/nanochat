#!/usr/bin/env python3

import sys


optimizer_configs = []

for matrix_lr in ("0.010", "0.015", "0.020", "0.030", "0.040"):
    for weight_decay in ("0.14", "0.21", "0.28", "0.42", "0.56"):
        if matrix_lr == "0.020" and weight_decay == "0.28":
            continue
        optimizer_configs.append((
            "muon",
            matrix_lr,
            weight_decay,
            "0.30",
            "0.008",
            "0.50",
        ))

for embedding_lr in ("0.15", "0.30", "0.60"):
    for unembedding_lr in ("0.004", "0.008", "0.016"):
        for scalar_lr in ("0.20", "0.50", "1.00"):
            if embedding_lr == "0.30" and unembedding_lr == "0.008" and scalar_lr == "0.50":
                continue
            optimizer_configs.append((
                "adamw",
                "0.020",
                "0.28",
                embedding_lr,
                unembedding_lr,
                scalar_lr,
            ))

schedule_configs = []

for warmup_steps in ("20", "40", "80"):
    for warmdown_ratio in ("0.45", "0.65", "0.80"):
        for final_lr_frac in ("0.00", "0.05", "0.10"):
            if warmup_steps == "40" and warmdown_ratio == "0.65" and final_lr_frac == "0.05":
                continue
            schedule_configs.append((warmup_steps, warmdown_ratio, final_lr_frac))

schedule_mode = sys.argv[1] == "schedule"
task_ids = sys.argv[2:] if schedule_mode else sys.argv[1:]

for position, task_id_text in enumerate(task_ids):
    task_id = int(task_id_text)

    if position:
        print()
    print(f"[{task_id}]")

    if schedule_mode:
        warmup_steps, warmdown_ratio, final_lr_frac = schedule_configs[task_id]
        print(f"WARMUP_STEPS={warmup_steps}")
        print(f"WARMDOWN_RATIO={warmdown_ratio}")
        print(f"FINAL_LR_FRAC={final_lr_frac}")
    else:
        family, matrix_lr, weight_decay, embedding_lr, unembedding_lr, scalar_lr = optimizer_configs[task_id]
        print(f"MATRIX_LR={matrix_lr}")
        print(f"WEIGHT_DECAY={weight_decay}")
        print(f"EMBEDDING_LR={embedding_lr}")
        print(f"UNEMBEDDING_LR={unembedding_lr}")
        print(f"SCALAR_LR={scalar_lr}")
