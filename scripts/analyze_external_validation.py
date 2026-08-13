#!/usr/bin/env python3
"""Analyze a checksum-bound held-out instrument-validation manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.validation.external_validation import analyze_external_validation, write_report

ROOT = Path(__file__).resolve().parents[1]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--readiness-protocol", type=Path, default=ROOT / "configs/step6_readiness.yaml"
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = analyze_external_validation(
        args.manifest, readiness_protocol_path=args.readiness_protocol
    )
    write_report(args.output, report)
    decision = report["decision"]
    print(
        "External-validation analysis: "
        f"thresholds_evaluable={decision['technical_thresholds_evaluable']} "
        f"thresholds_met={decision['technical_thresholds_met']} output={args.output}"
    )
    print("This technical report is not external validation approval or collection approval.")
    return 0 if decision["technical_thresholds_met"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
