"""Submit and monitor durable calls to the deployed Modal experiment runner."""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import modal

from modal_runner import APP_NAME, GPU_TYPE, parse_experiments, parse_models


DEFAULT_CALLS_FILE = Path(".modal_calls.json")


def load_calls(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text())


def save_calls(path: Path, calls: list[dict]) -> None:
    path.write_text(json.dumps(calls, indent=2, sort_keys=True) + "\n")


def submit(
    experiment_selection: str,
    model_selection: str,
    pilot: bool,
    force: bool,
    calls_file: Path,
) -> None:
    function = modal.Function.from_name(APP_NAME, "execute_notebook")
    calls = load_calls(calls_file)
    submitted_at = datetime.now(timezone.utc).isoformat()

    for experiment in parse_experiments(experiment_selection):
        for model_name in parse_models(model_selection):
            function_call = function.spawn(
                experiment,
                model_name,
                pilot=pilot,
                force=force,
            )
            record = {
                "call_id": function_call.object_id,
                "experiment": experiment,
                "model_name": model_name,
                "pilot": pilot,
                "force": force,
                "submitted_at": submitted_at,
            }
            calls.append(record)
            save_calls(calls_file, calls)
            print(
                f"Submitted {experiment} / {model_name} as "
                f"{function_call.object_id} on {GPU_TYPE}"
            )


def get_status(record: dict) -> tuple[str, str | None]:
    function_call = modal.FunctionCall.from_id(record["call_id"])
    try:
        result = function_call.get(timeout=0)
    except (TimeoutError, modal.exception.TimeoutError):
        return "pending", None
    except (ConnectionError, modal.exception.ConnectionError) as exc:
        # A function can survive a Modal worker preemption while the status RPC
        # briefly returns a connection deadline. Keep waiting instead of
        # misreporting a healthy restarted call as failed.
        return "unknown", f"transient status error: {exc}"
    except Exception as exc:
        return "failed", f"{type(exc).__name__}: {exc}"
    return result.get("status", "completed"), None


def report(calls_file: Path, wait: bool, poll_seconds: float) -> None:
    calls = load_calls(calls_file)
    if not calls:
        raise ValueError(f"No calls recorded in {calls_file}")

    while True:
        pending = 0
        failed = 0
        for record in calls:
            status, error = get_status(record)
            label = f"{record['experiment']} / {record['model_name']}"
            if status in {"pending", "unknown"}:
                pending += 1
            elif status == "failed":
                failed += 1
            suffix = f": {error}" if error else ""
            print(f"{record['call_id']}\t{status}\t{label}{suffix}")

        if not wait or pending == 0:
            if failed:
                raise RuntimeError(f"{failed} Modal call(s) failed")
            return
        print(f"Waiting for {pending} call(s)...", flush=True)
        time.sleep(poll_seconds)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="queue remote jobs")
    submit_parser.add_argument("--experiment", default="all")
    submit_parser.add_argument("--model", default="all")
    submit_parser.add_argument("--pilot", action="store_true")
    submit_parser.add_argument("--force", action="store_true")
    submit_parser.add_argument("--calls-file", type=Path, default=DEFAULT_CALLS_FILE)

    for command in ("status", "wait"):
        report_parser = subparsers.add_parser(command, help=f"{command} queued jobs")
        report_parser.add_argument("--calls-file", type=Path, default=DEFAULT_CALLS_FILE)
        report_parser.add_argument("--poll-seconds", type=float, default=30.0)

    args = parser.parse_args()
    if args.command == "submit":
        submit(
            args.experiment,
            args.model,
            args.pilot,
            args.force,
            args.calls_file,
        )
    else:
        report(
            args.calls_file,
            wait=args.command == "wait",
            poll_seconds=args.poll_seconds,
        )


if __name__ == "__main__":
    main()
