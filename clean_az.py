"""Command line for the isolated clean-start AlphaZero run."""

from __future__ import annotations

import argparse
import json

from az.config import RunConfig
from az.iteration import initialize_clean_run, run_iteration, train_candidate_worker
from az.preflight import run_preflight
from az.provenance import load_manifest


def _config(path: str) -> RunConfig:
    return RunConfig.load_json(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for command in ("init", "preflight", "train-candidate", "iteration", "status"):
        command_parser = subparsers.add_parser(command)
        command_parser.add_argument("--config", required=True, help="Path to clean run JSON config.")
    preflight = subparsers.choices["preflight"]
    preflight.add_argument("--require-two-gpus", action="store_true")
    preflight.add_argument("--require-cpp-engine", action="store_true")
    args = parser.parse_args()
    config = _config(args.config)

    if args.command == "init":
        manifest, root = initialize_clean_run(config)
        print(json.dumps({"run_root": str(root), "run_id": manifest.run_id}, indent=2))
        return
    if args.command == "preflight":
        report = run_preflight(
            config=config,
            root=config.root,
            require_two_gpus=args.require_two_gpus,
            require_cpp_engine=args.require_cpp_engine,
        )
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        if not report.passed:
            raise SystemExit(1)
        return
    if args.command == "status":
        manifest = load_manifest(config.root)
        with (config.root / "run_state.json").open("r", encoding="utf-8") as handle:
            state = json.load(handle)
        print(json.dumps({"manifest": manifest.to_dict(), "state": state}, indent=2, sort_keys=True))
        return

    # Training is deliberately guarded: the requested system must not start a
    # clean RL run on the unvalidated Python reference engine.
    report = run_preflight(
        config=config,
        root=config.root,
        require_two_gpus=(config.runtime.num_gpus or 1) >= 2,
        require_cpp_engine=True,
    )
    if not report.passed:
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        raise SystemExit("Training blocked by clean-run preflight.")
    if args.command == "train-candidate":
        print(json.dumps(train_candidate_worker(config), indent=2, sort_keys=True))
        return
    print(json.dumps(run_iteration(config, entry_script=__file__), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
