"""Run the experiment notebooks as isolated, resumable Modal jobs."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import modal


APP_NAME = "vision-depth-experiments"
GPU_TYPE = "T4"
PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_RESULTS_DIR = Path("/outputs")

NOTEBOOKS = {
    "exp1": "exp1.ipynb",
    "exp2": "exp2_layer_approximation.ipynb",
    "exp3": "exp3_component_noise.ipynb",
    "exp4": "exp4_direct_logit_attribution.ipynb",
    "exp5": "exp5_linear_probing.ipynb",
}

NOTEBOOK_SUPPORT_FILES = {}

EXPERIMENT_ALIASES = {
    "exp1": "exp1",
    "exp1.ipynb": "exp1",
    "exp2": "exp2",
    "exp2_layer_approximation": "exp2",
    "exp2_layer_approximation.ipynb": "exp2",
    "exp3": "exp3",
    "exp3_component_noise": "exp3",
    "exp3_component_noise.ipynb": "exp3",
    "exp4": "exp4",
    "exp4_direct_logit_attribution": "exp4",
    "exp4_direct_logit_attribution.ipynb": "exp4",
    "exp5": "exp5",
    "exp5_linear_probing": "exp5",
    "exp5_linear_probing.ipynb": "exp5",
}

MODEL_NAMES = (
    "DeiT-tiny",
    "DeiT-small",
    "DeiT-base",
    "DeiT-tiny distilled",
    "DeiT-small distilled",
    "DeiT-base distilled",
    "DeiT-base 384",
    "DeiT-base distilled 384",
)

PILOT_OVERRIDES: dict[str, dict[str, Any]] = {
    "exp1": {
        "contribution_max_samples": 16,
        "noise_eval_samples": 16,
        "alphas": [0.0, 1.0],
        "noise_target_layers": [8],
        "fragility_target_layers": [8],
    },
    "exp2": {
        "max_samples": 64,
        "surrogate_epochs": 1,
        "width_multipliers": [2.0],
    },
    "exp3": {
        "noise_eval_samples": 16,
        "alphas": [0.0, 1.0],
        "component_noise_components": ["residual", "attention", "ffn"],
        "component_noise_target_layers": [0],
    },
    "exp4": {"max_samples": 16},
    "exp5": {
        "max_train_samples": 128,
        "max_test_samples": 64,
        "probe_max_iter": 50,
    },
}


def slugify(value: str) -> str:
    """Convert a display name into a stable path segment."""
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_experiments(selection: str) -> list[str]:
    """Resolve an experiment selection from the CLI."""
    if selection.strip().lower() == "all":
        return list(NOTEBOOKS)

    resolved = []
    for item in selection.split(","):
        key = item.strip()
        try:
            experiment = EXPERIMENT_ALIASES[key]
        except KeyError as exc:
            choices = ", ".join(NOTEBOOKS)
            raise ValueError(f"Unknown experiment {key!r}. Choose from {choices}.") from exc
        if experiment not in resolved:
            resolved.append(experiment)
    return resolved


def parse_models(selection: str) -> list[str]:
    """Resolve a model selection from the CLI."""
    if selection.strip().lower() == "all":
        return list(MODEL_NAMES)

    requested = [item.strip() for item in selection.split(",")]
    unknown = sorted(set(requested) - set(MODEL_NAMES))
    if unknown:
        raise ValueError(f"Unknown model names: {unknown}")
    return list(dict.fromkeys(requested))


def build_overrides(
    experiment: str,
    model_name: str,
    results_dir: str,
    pilot: bool,
) -> dict[str, Any]:
    """Build the notebook config overrides for one remote job."""
    overrides: dict[str, Any] = {
        "selected_model_names": [model_name],
        "results_dir": results_dir,
        "show_progress": False,
    }
    if pilot:
        overrides.update(PILOT_OVERRIDES[experiment])
    return overrides


def build_override_cell(overrides: dict[str, Any]) -> str:
    """Create the notebook cell that applies remote execution settings."""
    serialized = json.dumps(overrides, sort_keys=True)
    return f'''# Injected by modal_runner.py
import json as _cloud_json
import random as _cloud_random
import numpy as _cloud_numpy
import torch as _cloud_torch

_cloud_overrides = _cloud_json.loads({serialized!r})
EXPERIMENT_CONFIG.update(_cloud_overrides)
_cloud_random.seed(0)
_cloud_numpy.random.seed(0)
_cloud_torch.manual_seed(0)
if _cloud_torch.cuda.is_available():
    _cloud_torch.cuda.manual_seed_all(0)
print("Cloud overrides:", _cloud_overrides)
'''


cache_volume = modal.Volume.from_name("vision-depth-cache", create_if_missing=True)
results_volume = modal.Volume.from_name("vision-depth-results", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements(str(PROJECT_DIR / "requirements.txt"))
    .env(
        {
            "HF_HOME": "/cache/huggingface",
            "HF_DATASETS_CACHE": "/cache/huggingface/datasets",
            "HF_HUB_CACHE": "/cache/huggingface/hub",
            "HF_HUB_DISABLE_TELEMETRY": "1",
            "MPLBACKEND": "Agg",
            "TOKENIZERS_PARALLELISM": "false",
        }
    )
    .add_local_dir(
        PROJECT_DIR,
        str(REMOTE_PROJECT_DIR),
        copy=True,
        ignore=[
            ".git",
            ".venv",
            ".modal_calls*.json",
            "__pycache__",
            "cloud_results",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(
    gpu=GPU_TYPE,
    cpu=8,
    memory=32_768,
    timeout=6 * 60 * 60,
    max_containers=1,
    retries=0,
    volumes={"/cache": cache_volume, "/outputs": results_volume},
)
def execute_notebook(
    experiment: str,
    model_name: str,
    pilot: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Execute one notebook for one model and persist every artifact."""
    import importlib.metadata
    import platform
    import shutil
    import subprocess
    import time
    import traceback
    from datetime import datetime, timezone

    import nbformat
    import torch
    from nbclient import NotebookClient

    experiment = parse_experiments(experiment)[0]
    model_name = parse_models(model_name)[0]
    model_slug = slugify(model_name)
    run_kind = "pilot" if pilot else "full"
    run_root = REMOTE_RESULTS_DIR / run_kind / experiment / model_slug
    success_path = run_root / "SUCCESS.json"

    if success_path.exists() and not force:
        previous = json.loads(success_path.read_text())
        print(f"Skipping completed run at {run_root}", flush=True)
        return {"status": "skipped", **previous}

    started_at = datetime.now(timezone.utc)
    attempt_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    attempt_dir = run_root / "attempts" / attempt_id
    artifact_dir = attempt_dir / "results"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True)

    notebook_name = NOTEBOOKS[experiment]
    source_path = REMOTE_PROJECT_DIR / notebook_name
    source_bytes = source_path.read_bytes()
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    working_dir = Path("/tmp") / f"{experiment}_{model_slug}_{attempt_id}"
    working_dir.mkdir(parents=True, exist_ok=False)
    working_notebook = working_dir / notebook_name
    shutil.copy2(source_path, working_notebook)
    support_sha256 = {}
    for support_name in NOTEBOOK_SUPPORT_FILES.get(experiment, ()):
        support_source = REMOTE_PROJECT_DIR / support_name
        support_bytes = support_source.read_bytes()
        support_sha256[support_name] = hashlib.sha256(support_bytes).hexdigest()
        shutil.copy2(support_source, working_dir / support_name)
        shutil.copy2(support_source, attempt_dir / support_name)

    overrides = build_overrides(experiment, model_name, str(artifact_dir), pilot)
    notebook = nbformat.reads(source_bytes.decode("utf-8"), as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None

    config_cell_index = next(
        (
            index
            for index, cell in enumerate(notebook.cells)
            if cell.cell_type == "code" and "EXPERIMENT_CONFIG = {" in cell.source
        ),
        None,
    )
    if config_cell_index is None:
        raise RuntimeError(f"Could not find EXPERIMENT_CONFIG in {notebook_name}")
    notebook.cells.insert(
        config_cell_index + 1,
        nbformat.v4.new_code_cell(build_override_cell(overrides)),
    )

    prepared_path = attempt_dir / "prepared.ipynb"
    executed_path = attempt_dir / "executed.ipynb"
    nbformat.write(notebook, prepared_path)

    def report_cell_start(cell, cell_index):
        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else "<empty>"
        print(
            f"Cell {cell_index + 1}/{len(notebook.cells)}: {first_line[:100]}",
            flush=True,
        )

    started_monotonic = time.monotonic()
    failure: dict[str, Any] | None = None
    print(
        f"Starting {run_kind} {experiment} for {model_name} on {torch.cuda.get_device_name(0)}",
        flush=True,
    )
    try:
        client = NotebookClient(
            notebook,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(working_dir)}},
            on_cell_start=report_cell_start,
        )
        client.execute()
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
        print(f"Notebook failed: {failure['error_type']}: {failure['error']}", flush=True)
    finally:
        nbformat.write(notebook, executed_path)

    duration_seconds = time.monotonic() - started_monotonic
    package_names = (
        "datasets",
        "matplotlib",
        "nbclient",
        "numpy",
        "pandas",
        "scikit-learn",
        "seaborn",
        "torch",
        "torchvision",
        "transformers",
    )
    package_versions = {
        name: importlib.metadata.version(name) for name in package_names
    }
    gpu_details = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    artifacts = sorted(
        (
            {
                "path": str(path.relative_to(attempt_dir)),
                "bytes": path.stat().st_size,
            }
            for path in attempt_dir.rglob("*")
            if path.is_file()
        ),
        key=lambda artifact: artifact["path"],
    )
    manifest = {
        "status": "failed" if failure else "completed",
        "experiment": experiment,
        "notebook": notebook_name,
        "model_name": model_name,
        "run_kind": run_kind,
        "attempt_id": attempt_id,
        "started_at": started_at.isoformat(),
        "duration_seconds": round(duration_seconds, 3),
        "gpu": gpu_details,
        "python": platform.python_version(),
        "packages": package_versions,
        "source_sha256": source_sha256,
        "source_files_sha256": {
            notebook_name: source_sha256,
            **support_sha256,
        },
        "overrides": overrides,
        "artifacts": artifacts,
    }
    if failure:
        manifest["failure"] = failure

    manifest_path = attempt_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    if failure:
        (run_root / "LATEST_FAILURE.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
    else:
        success_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    results_volume.commit()
    cache_volume.commit()
    shutil.rmtree(working_dir, ignore_errors=True)

    if failure:
        raise RuntimeError(
            f"{experiment} failed for {model_name}: "
            f"{failure['error_type']}: {failure['error']}"
        )
    print(
        f"Completed {experiment} for {model_name} in {duration_seconds / 60:.1f} minutes",
        flush=True,
    )
    return manifest


@app.local_entrypoint()
def main(
    experiment: str = "all",
    model: str = "all",
    pilot: bool = False,
    force: bool = False,
) -> None:
    """Run the requested notebook and model combinations sequentially."""
    experiments = parse_experiments(experiment)
    models = parse_models(model)
    total = len(experiments) * len(models)
    print(f"Submitting {total} {GPU_TYPE} job(s) sequentially.")

    completed = []
    failures = []
    for index, experiment_name in enumerate(experiments, start=1):
        for model_index, model_name in enumerate(models, start=1):
            ordinal = (index - 1) * len(models) + model_index
            print(f"[{ordinal}/{total}] {experiment_name} / {model_name}")
            try:
                result = execute_notebook.remote(
                    experiment_name,
                    model_name,
                    pilot=pilot,
                    force=force,
                )
            except Exception as exc:
                failure = {
                    "experiment": experiment_name,
                    "model_name": model_name,
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }
                failures.append(failure)
                print(f"Failed {experiment_name} / {model_name}: {exc}")
            else:
                completed.append(result)

    summary = {"completed": completed, "failures": failures}
    print(json.dumps(summary, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"{len(failures)} remote job(s) failed. See the summary above.")
