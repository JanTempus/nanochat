import subprocess
from pathlib import Path


TOKENIZER_PATHS = [
    "/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/rounded_tokenizers/cross_over_climbmix400b_s7/vocab_8192/lp_8192_det",
]
TIME_LIMIT = "02:30:00"
SEEDS = [43, 44]

repo_dir = Path(__file__).resolve().parents[1]
sbatch_file = repo_dir / "sbatch_files" / "single_train.sbatch"

for tokenizer in TOKENIZER_PATHS:
    for seed in SEEDS:
        subprocess.run(
            [
                "sbatch",
                f"--time={TIME_LIMIT}",
                str(sbatch_file),
                tokenizer,
                str(seed),
            ],
            cwd=repo_dir,
            check=True,
        )
