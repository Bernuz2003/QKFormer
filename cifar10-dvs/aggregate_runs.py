from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


FIELDS = [
    "run_id",
    "model",
    "encoder_name",
    "source_T",
    "encoded_T",
    "source_C",
    "encoded_C",
    "best_test_acc1",
    "checkpoint_max_test_acc1",
    "total_estimated_sops_per_sample",
    "weighted_output_firing_rate",
    "input_density",
    "temporal_burstiness",
    "patch_embed_sops_share_pct",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate already analyzed QKFormer runs.")
    parser.add_argument("--runs-root", required=True, help="Directory containing one subdirectory per run.")
    parser.add_argument("--output-dir", default="", help="Defaults to <runs-root>/aggregate_analysis.")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def collect_rows(runs_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run_dir in sorted(path for path in runs_root.iterdir() if path.is_dir()):
        if run_dir.name.startswith("_") or run_dir.name in {"aggregate_analysis", "analysis"}:
            continue
        analysis = load_json(run_dir / "profile" / "run_analysis.json")
        if analysis:
            rows.append(analysis)
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, rows: list[dict[str, Any]]) -> None:
    lines = ["# Encoder V2 Aggregate Report", ""]
    if not rows:
        lines.append("No analyzed runs found. Run `analyze_runs.py --run-dir <run>` for each run first.")
    else:
        lines.append("| " + " | ".join(FIELDS) + " |")
        lines.append("| " + " | ".join(["---"] * len(FIELDS)) + " |")
        for row in rows:
            lines.append("| " + " | ".join(str(row.get(field, "-")) for field in FIELDS) + " |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    runs_root = Path(args.runs_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else runs_root / "aggregate_analysis"
    rows = collect_rows(runs_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "aggregate_summary.csv", rows)
    write_report(output_dir / "aggregate_report.md", rows)
    print(f"Aggregated {len(rows)} analyzed runs.")
    print(f"Summary CSV: {output_dir / 'aggregate_summary.csv'}")
    print(f"Report: {output_dir / 'aggregate_report.md'}")


if __name__ == "__main__":
    main()
