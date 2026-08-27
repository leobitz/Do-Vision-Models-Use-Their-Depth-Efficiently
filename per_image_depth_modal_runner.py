"""Run the per-image depth and layer-contribution experiment on Modal."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import modal


APP_NAME = "vision-depth-per-image-analysis"
PROJECT_DIR = Path(__file__).resolve().parent
REMOTE_PROJECT_DIR = Path("/root/project")
REMOTE_RESULTS_DIR = Path("/outputs")
NOTEBOOK_NAME = "exp7_per_image_depth_analysis.ipynb"
SUPPORT_FILES = ("individual_image_analysis.py", "per_image_depth_analysis.py")

cache_volume = modal.Volume.from_name("vision-depth-cache", create_if_missing=True)
results_volume = modal.Volume.from_name(
    "vision-depth-per-image-results", create_if_missing=True
)

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
            "__pycache__",
            "cloud_results",
            "per_image_depth_results",
        ],
    )
)

app = modal.App(APP_NAME, image=image)


@app.function(
    cpu=2,
    memory=4_096,
    timeout=15 * 60,
    volumes={"/cache": cache_volume},
)
def validate_validation_stream(limit: int = 200):
    from collections import Counter
    import gc
    import json
    import time

    from datasets import load_dataset

    source = (
        "hf://datasets/benjamin-paine/imagenet-1k/"
        "data/validation-*.parquet"
    )
    dataset = load_dataset(
        "parquet",
        data_files={"validation": source},
        split="validation",
        streaming=True,
    )
    labels = []
    image_sizes = []
    for index, example in enumerate(dataset):
        if index >= limit:
            break
        labels.append(int(example["label"]))
        image_sizes.append(tuple(example["image"].size))
    if len(labels) != limit:
        raise AssertionError(f"Expected {limit} rows, streamed {len(labels)}.")
    del dataset
    gc.collect()
    time.sleep(2)
    cache_volume.commit()
    result = {
        "source": source,
        "rows": len(labels),
        "unique_labels": len(set(labels)),
        "most_common_labels": Counter(labels).most_common(5),
        "first_image_size": image_sizes[0],
    }
    print(json.dumps(result, sort_keys=True), flush=True)
    return result


def override_cell(overrides: dict[str, Any]) -> str:
    serialized = json.dumps(overrides, sort_keys=True)
    return f'''# Injected by per_image_depth_modal_runner.py
import json as _runner_json
import random as _runner_random
import numpy as _runner_numpy
import torch as _runner_torch

_runner_overrides = _runner_json.loads({serialized!r})
EXPERIMENT_CONFIG.update(_runner_overrides)
_runner_random.seed(0)
_runner_numpy.random.seed(0)
_runner_torch.manual_seed(0)
if _runner_torch.cuda.is_available():
    _runner_torch.cuda.manual_seed_all(0)
print("Per-image depth experiment overrides:", _runner_overrides)
'''


@app.function(
    gpu="T4",
    cpu=8,
    memory=32_768,
    timeout=6 * 60 * 60,
    max_containers=1,
    retries=0,
    volumes={"/cache": cache_volume, "/outputs": results_volume},
)
def execute_per_image_depth_analysis(pilot: bool = False, force: bool = False):
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

    run_kind = "pilot" if pilot else "full"
    run_root = REMOTE_RESULTS_DIR / run_kind
    success_path = run_root / "SUCCESS.json"
    if success_path.exists() and not force:
        return {"status": "skipped", **json.loads(success_path.read_text())}

    started_at = datetime.now(timezone.utc)
    attempt_id = started_at.strftime("%Y%m%dT%H%M%SZ")
    attempt_dir = run_root / "attempts" / attempt_id
    artifact_dir = attempt_dir / "results"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    artifact_dir.mkdir(parents=True)

    source_files = (NOTEBOOK_NAME, *SUPPORT_FILES)
    source_hashes = {}
    working_dir = Path("/tmp") / f"per_image_depth_analysis_{attempt_id}"
    working_dir.mkdir(parents=True, exist_ok=False)
    for source_name in source_files:
        source = REMOTE_PROJECT_DIR / source_name
        source_hashes[source_name] = hashlib.sha256(source.read_bytes()).hexdigest()
        shutil.copy2(source, working_dir / source_name)
        if source_name in SUPPORT_FILES:
            shutil.copy2(source, attempt_dir / source_name)

    notebook = nbformat.read(working_dir / NOTEBOOK_NAME, as_version=4)
    for cell in notebook.cells:
        if cell.cell_type == "code":
            cell.outputs = []
            cell.execution_count = None
    config_index = next(
        index
        for index, cell in enumerate(notebook.cells)
        if cell.cell_type == "code" and "EXPERIMENT_CONFIG = {" in cell.source
    )
    overrides = {
        "results_dir": str(artifact_dir),
        "log_every_batches": 1 if pilot else 25,
    }
    if pilot:
        overrides.update(
            {
                "stream_limit": 200,
                "samples_per_class": 1,
                "max_classes": 2,
                "batch_size": 2,
            }
        )
    notebook.cells.insert(
        config_index + 1,
        nbformat.v4.new_code_cell(override_cell(overrides)),
    )
    nbformat.write(notebook, attempt_dir / "prepared.ipynb")

    def report_cell_start(cell, cell_index):
        first_line = cell.source.strip().splitlines()[0] if cell.source.strip() else "<empty>"
        print(f"Cell {cell_index + 1}/{len(notebook.cells)}: {first_line[:100]}", flush=True)

    started = time.monotonic()
    failure = None
    try:
        NotebookClient(
            notebook,
            timeout=None,
            kernel_name="python3",
            resources={"metadata": {"path": str(working_dir)}},
            on_cell_start=report_cell_start,
        ).execute()
    except Exception as exc:
        failure = {
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
        }
    finally:
        nbformat.write(notebook, attempt_dir / "executed.ipynb")

    duration = time.monotonic() - started
    package_names = (
        "datasets",
        "matplotlib",
        "nbclient",
        "numpy",
        "pandas",
        "seaborn",
        "torch",
        "transformers",
    )
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
    manifest = {
        "status": "failed" if failure else "completed",
        "run_kind": run_kind,
        "attempt_id": attempt_id,
        "started_at": started_at.isoformat(),
        "duration_seconds": round(duration, 3),
        "gpu": gpu_details,
        "python": platform.python_version(),
        "packages": {name: importlib.metadata.version(name) for name in package_names},
        "source_files_sha256": source_hashes,
        "overrides": overrides,
    }
    if failure:
        manifest["failure"] = failure
    (attempt_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
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
            "Per-image depth experiment failed: "
            f"{failure['error_type']}: {failure['error']}"
        )
    return manifest


@app.local_entrypoint()
def main(pilot: bool = False, force: bool = False):
    result = execute_per_image_depth_analysis.remote(pilot=pilot, force=force)
    print(json.dumps(result, indent=2, sort_keys=True))
