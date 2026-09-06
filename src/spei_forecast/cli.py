from __future__ import annotations

import argparse
from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path

from .config import load_config
from .data import load_all, write_clean_dataset
from .experiment import run_benchmark
from .robustness import load_robustness_config, run_robustness


DEFAULT_CONFIG = Path("configs/benchmark_v1.toml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate recovered SPEI data and run temporally controlled benchmarks."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    validate = subcommands.add_parser(
        "validate", help="Validate all recovered station CSVs"
    )
    validate.add_argument("--config", default=str(DEFAULT_CONFIG))

    build = subcommands.add_parser(
        "build", help="Write a canonical tidy data derivative"
    )
    build.add_argument("--config", default=str(DEFAULT_CONFIG))
    build.add_argument(
        "--output",
        help="Output CSV (defaults inside the configured repository)",
    )

    benchmark = subcommands.add_parser(
        "benchmark", help="Reproduce benchmark v1 on its fixed evaluation period"
    )
    benchmark.add_argument("--config", default=str(DEFAULT_CONFIG))
    benchmark.add_argument("--output-dir")
    benchmark.add_argument("--bootstrap-resamples", type=int)
    robustness = subcommands.add_parser(
        "robustness",
        help="Run retrospective rolling-origin and sensitivity diagnostics",
    )
    robustness.add_argument("--config", default=str(DEFAULT_CONFIG))
    robustness.add_argument(
        "--robustness-config",
        help="Defaults to configs/robustness_v1.toml beside --config",
    )
    robustness.add_argument("--output-dir")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    config = load_config(args.config)
    if args.command == "validate":
        _, quality = load_all(config)
        print(json.dumps(quality, indent=2, sort_keys=True))
        return 0
    if args.command == "build":
        frame, quality = load_all(config)
        output = args.output or (
            config.repository_root
            / "data"
            / "processed"
            / "clean"
            / "observations.csv"
        )
        path = write_clean_dataset(frame, output)
        quality_path = path.with_name("data_quality.json")
        quality_path.write_text(
            json.dumps(quality, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"Wrote {len(frame):,} rows to {path}")
        print(f"Wrote validation report to {quality_path}")
        return 0
    if args.command == "robustness":
        robustness_path = args.robustness_config or (
            config.repository_root / "configs" / "robustness_v1.toml"
        )
        robustness_config = load_robustness_config(robustness_path)
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        output = (
            Path(args.output_dir)
            if args.output_dir
            else config.repository_root / "artifacts" / "robustness_v1" / run_id
        )
        result = run_robustness(
            config,
            robustness_config,
            output,
        )
        print(f"Robustness artifacts: {result['output_dir']}")
        print(result["rolling_metrics_macro"].to_string(index=False))
        return 0
    if args.bootstrap_resamples is not None:
        config = replace(config, bootstrap_resamples=args.bootstrap_resamples)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output_dir)
        if args.output_dir
        else config.repository_root / "artifacts" / "benchmark_v1" / run_id
    )
    result = run_benchmark(config, output)
    print(f"Benchmark artifacts: {result['output_dir']}")
    print(result["metrics_macro"].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
