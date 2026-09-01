import argparse
import re
import subprocess
from pathlib import Path


TOKENIZER_PATHS = [
    "/capstor/store/cscs/swissai/a0229/jtempus/tokenisation_lp/rounded_tokenizers/cross_over_climbmix400b_s7/vocab_8192/lp_8192_bias",
]
TIME_LIMIT = "03:00:00"
SEEDS = [43, 44]

repo_dir = Path(__file__).resolve().parents[1]
sbatch_file = repo_dir / "sbatch_files" / "single_train.sbatch"


def parse_tag(value):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", value):
        raise argparse.ArgumentTypeError(
            "tag must contain only letters, numbers, dots, underscores, or hyphens"
        )
    return value


def parse_args():
    parser = argparse.ArgumentParser(description="Submit nanochat training runs")
    parser.add_argument(
        "--tag",
        required=True,
        type=parse_tag,
        help="Run-name prefix, for example BPE",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    for tokenizer in TOKENIZER_PATHS:
        for seed in SEEDS:
            subprocess.run(
                [
                    "sbatch",
                    f"--time={TIME_LIMIT}",
                    str(sbatch_file),
                    tokenizer,
                    str(seed),
                    args.tag,
                ],
                cwd=repo_dir,
                check=True,
            )


if __name__ == "__main__":
    main()
