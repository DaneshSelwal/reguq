"""Command-line entry point ``reguq-run``."""

from __future__ import annotations

import logging

from .config import PipelineConfig
from .runner import run_pipeline


def main(argv: list[str] | None = None) -> None:
    """CLI: ``reguq-run config.yaml``."""
    import argparse

    ap = argparse.ArgumentParser(
        prog="reguq-run", description="Run a reguq uncertainty-quantification pipeline."
    )
    ap.add_argument("config", help="YAML or JSON PipelineConfig")
    ap.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    args = ap.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(message)s"
    )
    res = run_pipeline(PipelineConfig.from_file(args.config))
    cols = [
        c
        for c in ["method", "model", "coverage", "mean_width", "interval_score", "rmse", "status"]
        if c in res.metrics
    ]
    print(res.metrics[cols].to_string(index=False))


if __name__ == "__main__":  # pragma: no cover
    main()
