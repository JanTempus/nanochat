#!/usr/bin/env python3
"""Compare tokenizer results with paired, two-sided permutation tests.

Runs are paired by (depth, vocab_size, seed). The reported difference is the
comparator score minus the baseline score, averaged over matched runs. Thus a
negative difference favors the comparator for val/bpb, while a positive one
favors it for core_metric.

By default, compare BPE with every other tokenizer for val/bpb and core_metric:

    python3 model_results/paired_permutation_test.py
    python3 model_results/paired_permutation_test.py --compare PathPiece

Results are written as CSV to standard output. Holm adjustment covers every
comparison in that invocation. Each matched run is treated as one independent
paired observation in the sign-flip test.
"""

import argparse
import csv
import itertools
import math
import random
import sys
from pathlib import Path
from statistics import fmean


DEFAULT_INPUT = Path(__file__).resolve().with_name("all_results.csv")
DEFAULT_METRICS = ("val/bpb", "core_metric")
PAIR_COLUMNS = ("depth", "vocab_size", "seed")
EXACT_MAX_PAIRS = 16
OUTPUT_COLUMNS = (
    "metric",
    "baseline",
    "comparator",
    "n_pairs",
    "mean_baseline",
    "mean_comparator",
    "mean_difference",
    "p_value",
    "holm_p_value",
    "method",
    "permutations_evaluated",
)


def load_results(path: Path, metrics: tuple[str, ...]) -> dict:
    """Return tokenizer -> (depth, vocab, seed) -> metric values."""
    results = {}
    seen = {}
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        required = {"tokenizer_name", *PAIR_COLUMNS, *metrics}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"missing CSV columns: {', '.join(sorted(missing))}")

        for row in reader:
            if not any(row.values()):
                continue
            depth_text = (row["depth"] or "").strip()
            if depth_text.startswith("---") and all(
                not value for column, value in row.items() if column != "depth"
            ):
                continue
            try:
                if None in row:
                    raise ValueError("too many CSV fields")
                depth, vocab, seed = (int(row[column]) for column in PAIR_COLUMNS)
                if depth <= 0 or vocab <= 0:
                    raise ValueError("depth and vocab_size must be positive")
                tokenizer = (row["tokenizer_name"] or "").strip().casefold()
                if not tokenizer:
                    raise ValueError("empty tokenizer_name")
                values = {}
                for metric in metrics:
                    raw = (row[metric] or "").strip()
                    if raw:
                        value = float(raw)
                        if not math.isfinite(value):
                            raise ValueError(f"non-finite {metric}: {raw!r}")
                        values[metric] = value
                key = (depth, vocab, seed)
                identity = (tokenizer, key)
                signature = ((row.get("run_id") or "").strip(), values)
                if identity in seen:
                    if seen[identity] != signature:
                        raise ValueError(f"conflicting duplicate run for {identity}")
                    continue
                seen[identity] = signature
                results.setdefault(tokenizer, {})[key] = values
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{path}:{reader.line_num}: {exc}") from exc

    if not results:
        raise ValueError(f"no runs found in {path}")
    return results


def permutation_p_value(
    differences: list[float], permutations: int, rng: random.Random
) -> tuple[float, str, int]:
    """Two-sided sign-flip test of zero mean paired difference."""
    if not differences:
        raise ValueError("at least one paired difference is required")
    observed = abs(math.fsum(differences))
    tolerance = 1e-12 * max(observed, max(map(abs, differences)))

    def is_extreme(signed_sum: float) -> bool:
        return abs(signed_sum) >= observed - tolerance

    count = len(differences)
    if count <= EXACT_MAX_PAIRS:
        total = 1 << count
        extreme = sum(
            is_extreme(math.fsum(sign * difference for sign, difference in zip(signs, differences)))
            for signs in itertools.product((-1, 1), repeat=count)
        )
        return extreme / total, "exact", total

    extreme = 0
    for _ in range(permutations):
        signs = rng.getrandbits(count)
        signed_sum = math.fsum(
            difference if signs & (1 << index) else -difference
            for index, difference in enumerate(differences)
        )
        extreme += is_extreme(signed_sum)
    # Include the observed assignment for a valid finite-sample Monte Carlo p-value.
    return (extreme + 1) / (permutations + 1), "monte_carlo", permutations


def holm_adjust(p_values: list[float]) -> list[float]:
    """Holm step-down family-wise adjustment in original result order."""
    adjusted = [0.0] * len(p_values)
    running_max = 0.0
    for rank, index in enumerate(sorted(range(len(p_values)), key=p_values.__getitem__)):
        running_max = max(running_max, (len(p_values) - rank) * p_values[index])
        adjusted[index] = min(1.0, running_max)
    return adjusted


def compare(
    results: dict,
    baseline: str,
    comparators: list[str],
    metrics: tuple[str, ...],
    permutations: int,
    seed: int,
) -> list[dict]:
    rng = random.Random(seed)
    output = []
    for metric in metrics:
        for comparator in comparators:
            baseline_runs = results[baseline]
            comparator_runs = results[comparator]
            keys = sorted(baseline_runs.keys() & comparator_runs.keys())
            pairs = [
                (baseline_runs[key][metric], comparator_runs[key][metric])
                for key in keys
                if metric in baseline_runs[key] and metric in comparator_runs[key]
            ]
            if not pairs:
                raise ValueError(f"no matched {metric} results for {baseline} and {comparator}")
            baseline_values, comparator_values = zip(*pairs)
            differences = [other - base for base, other in pairs]
            p_value, method, evaluated = permutation_p_value(differences, permutations, rng)
            output.append({
                "metric": metric,
                "baseline": baseline,
                "comparator": comparator,
                "n_pairs": len(pairs),
                "mean_baseline": fmean(baseline_values),
                "mean_comparator": fmean(comparator_values),
                "mean_difference": fmean(differences),
                "p_value": p_value,
                "method": method,
                "permutations_evaluated": evaluated,
            })

    for row, adjusted in zip(output, holm_adjust([row["p_value"] for row in output])):
        row["holm_p_value"] = adjusted
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("input", nargs="?", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--baseline", default="bpe", help="baseline tokenizer (default: bpe)")
    parser.add_argument("--compare", help="compare only this tokenizer (default: all others)")
    parser.add_argument(
        "--metric", action="append", help="metric column to test; repeat for multiple metrics"
    )
    parser.add_argument(
        "--permutations", type=int, default=100_000,
        help="random sign flips when there are more than 16 pairs (default: 100000)",
    )
    parser.add_argument("--seed", type=int, default=0, help="random seed (default: 0)")
    args = parser.parse_args()
    if args.permutations <= 0:
        parser.error("--permutations must be positive")
    metrics = tuple(dict.fromkeys(metric.strip() for metric in (args.metric or DEFAULT_METRICS)))
    if any(not metric for metric in metrics):
        parser.error("--metric must be a nonempty CSV column name")
    baseline = args.baseline.strip().casefold()
    comparator = args.compare.strip().casefold() if args.compare is not None else None
    if not baseline:
        parser.error("--baseline must be nonempty")
    if comparator == "":
        parser.error("--compare must be nonempty")
    if comparator == baseline:
        parser.error("--compare must differ from --baseline")

    try:
        results = load_results(args.input, metrics)
        if baseline not in results:
            raise ValueError(f"baseline tokenizer {baseline!r} not found")
        if comparator and comparator not in results:
            raise ValueError(f"comparator tokenizer {comparator!r} not found")
        comparators = [comparator] if comparator else sorted(set(results) - {baseline})
        if not comparators:
            raise ValueError("no other tokenizer found for comparison")
        output = compare(results, baseline, comparators, metrics, args.permutations, args.seed)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))

    writer = csv.DictWriter(sys.stdout, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(output)


if __name__ == "__main__":
    main()
