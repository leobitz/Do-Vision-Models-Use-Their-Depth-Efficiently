"""Validate and combine per-model Modal experiment results."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable


EXPECTED_EXPERIMENTS = ("exp1", "exp2", "exp3", "exp4", "exp5")
EXPECTED_MODELS = {
    "DeiT-tiny",
    "DeiT-small",
    "DeiT-base",
    "DeiT-tiny distilled",
    "DeiT-small distilled",
    "DeiT-base distilled",
    "DeiT-base 384",
    "DeiT-base distilled 384",
}

AGGREGATES: dict[str, dict[str, str]] = {
    "exp1": {
        "contribution_raw_all_models.csv": "*_contribution_raw.csv",
        "contribution_summary_all_models.csv": "*_contribution_summary.csv",
        "noise_robustness_all_models.csv": "*_noise_robustness.csv",
        "fragility_all_models.csv": "*_fragility.csv",
    },
    "exp2": {
        "layer_approximation_all_models_mean_non_cls.csv": (
            "layer_approximation_all_models_mean_non_cls.csv"
        ),
        "layer_approximation_best_surrogates_mean_non_cls.csv": (
            "layer_approximation_best_surrogates_mean_non_cls.csv"
        ),
    },
    "exp3": {"component_noise_results.csv": "component_noise_results.csv"},
    "exp4": {
        "dla_summary_all_models.csv": "dla_summary_all_models.csv",
        "logit_lens_all_models.csv": "logit_lens_all_models.csv",
    },
    "exp5": {"linear_probe_all_models.csv": "linear_probe_all_models.csv"},
}


def load_successful_runs(root: Path) -> dict[str, list[dict]]:
    """Load full-run manifests and resolve their result directories."""
    runs: dict[str, list[dict]] = defaultdict(list)
    for success_path in sorted((root / "full").glob("*/*/SUCCESS.json")):
        manifest = json.loads(success_path.read_text())
        experiment = manifest["experiment"]
        attempt_dir = success_path.parent / "attempts" / manifest["attempt_id"]
        manifest["result_dir"] = attempt_dir / "results"
        manifest["success_path"] = success_path
        runs[experiment].append(manifest)
    return dict(runs)


def validate_runs(runs: dict[str, list[dict]]) -> list[str]:
    """Return completeness errors for a downloaded full suite."""
    errors = []
    for experiment in EXPECTED_EXPERIMENTS:
        experiment_runs = runs.get(experiment, [])
        models = {run["model_name"] for run in experiment_runs}
        missing = sorted(EXPECTED_MODELS - models)
        extra = sorted(models - EXPECTED_MODELS)
        if missing:
            errors.append(f"{experiment} is missing models: {missing}")
        if extra:
            errors.append(f"{experiment} has unknown models: {extra}")
        failed = [run["model_name"] for run in experiment_runs if run["status"] != "completed"]
        if failed:
            errors.append(f"{experiment} has non-completed success manifests: {failed}")
    return errors


def find_inputs(runs: Iterable[dict], pattern: str) -> list[Path]:
    """Find one matching CSV in each model result directory."""
    inputs = []
    for run in sorted(runs, key=lambda item: item["model_name"]):
        matches = sorted(run["result_dir"].glob(pattern))
        if len(matches) != 1:
            raise ValueError(
                f"Expected one {pattern!r} file for {run['experiment']} / "
                f"{run['model_name']}, found {len(matches)}."
            )
        inputs.append(matches[0])
    return inputs


def combine_csv_files(inputs: list[Path], output: Path) -> int:
    """Concatenate CSV rows while checking that every header matches."""
    expected_header = None
    row_count = 0
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as output_file:
        writer = csv.writer(output_file)
        for input_path in inputs:
            with input_path.open(newline="") as input_file:
                reader = csv.reader(input_file)
                try:
                    header = next(reader)
                except StopIteration as exc:
                    raise ValueError(f"CSV is empty: {input_path}") from exc
                if expected_header is None:
                    expected_header = header
                    writer.writerow(header)
                elif header != expected_header:
                    raise ValueError(f"CSV header mismatch in {input_path}")
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    return row_count


def collate(root: Path, allow_incomplete: bool = False) -> dict:
    """Validate manifests and write combined CSVs and a run summary."""
    runs = load_successful_runs(root)
    errors = validate_runs(runs)
    if errors and not allow_incomplete:
        raise ValueError("\n".join(errors))

    output_root = root / "collated"
    aggregate_rows = {}
    for experiment, outputs in AGGREGATES.items():
        experiment_runs = runs.get(experiment, [])
        if not experiment_runs:
            continue
        for output_name, input_pattern in outputs.items():
            inputs = find_inputs(experiment_runs, input_pattern)
            output_path = output_root / experiment / output_name
            aggregate_rows[str(output_path.relative_to(root))] = combine_csv_files(
                inputs, output_path
            )

    run_rows = []
    for experiment in sorted(runs):
        for run in sorted(runs[experiment], key=lambda item: item["model_name"]):
            run_rows.append(
                {
                    "experiment": experiment,
                    "model_name": run["model_name"],
                    "duration_seconds": run["duration_seconds"],
                    "gpu": run["gpu"],
                    "attempt_id": run["attempt_id"],
                    "source_sha256": run["source_sha256"],
                }
            )

    summary = {
        "errors": errors,
        "completed_runs": len(run_rows),
        "total_duration_seconds": round(
            sum(float(row["duration_seconds"]) for row in run_rows), 3
        ),
        "aggregate_rows": aggregate_rows,
        "runs": run_rows,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "run_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, nargs="?", default=Path("cloud_results"))
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="collate the successful subset instead of rejecting missing models",
    )
    args = parser.parse_args()
    summary = collate(args.root, allow_incomplete=args.allow_incomplete)
    print(
        f"Collated {summary['completed_runs']} runs into "
        f"{len(summary['aggregate_rows'])} CSV files."
    )


if __name__ == "__main__":
    main()
