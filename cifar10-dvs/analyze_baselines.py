from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class RunPaths:
    run_dir: Path
    lr_dir: Path
    logs_dir: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and plot QKFormer CIFAR10-DVS baseline runs.")
    parser.add_argument("--run-root", required=True, help="Root directory containing QKFormer baseline runs.")
    parser.add_argument("--output-dir", default="", help="Analysis output directory. Defaults to <run-root>/analysis.")
    parser.add_argument("--activity-root", default="", help="Optional root containing activity_profile.py outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def discover_runs(run_root: Path) -> list[RunPaths]:
    runs: list[RunPaths] = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_") or run_dir.name == "analysis":
            continue
        for lr_dir in sorted(run_dir.glob("lr*")):
            if not lr_dir.is_dir() or lr_dir.name.endswith("_logs"):
                continue
            logs_dir = run_dir / f"{lr_dir.name}_logs"
            runs.append(RunPaths(run_dir=run_dir, lr_dir=lr_dir, logs_dir=logs_dir))
    return runs


def parse_namespace_text(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8", errors="replace")
    values: dict[str, Any] = {}
    for key, raw_value in re.findall(r"(\w+)=('[^']*'|None|True|False|[-+0-9.eE]+)", text):
        values[key] = parse_value(raw_value)
    return values


def parse_value(raw_value: str) -> Any:
    if raw_value.startswith("'") and raw_value.endswith("'"):
        return raw_value[1:-1]
    if raw_value == "None":
        return None
    if raw_value == "True":
        return True
    if raw_value == "False":
        return False
    try:
        if any(ch in raw_value for ch in ".eE"):
            return float(raw_value)
        return int(raw_value)
    except ValueError:
        return raw_value


def infer_model_name(run_name: str, args_values: dict[str, Any]) -> str:
    if args_values.get("model"):
        return str(args_values["model"])
    return run_name.split("_b", 1)[0]


def optional_torch():
    try:
        import torch

        return torch
    except Exception:
        return None


def load_checkpoint(path: Path) -> dict[str, Any]:
    torch = optional_torch()
    if torch is None or not path.exists():
        return {}
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def newest_checkpoint(lr_dir: Path) -> Path | None:
    checkpoints = list(lr_dir.glob("checkpoint_*.pth"))
    numbered: list[tuple[int, Path]] = []
    for path in checkpoints:
        match = re.search(r"checkpoint_(\d+)\.pth$", path.name)
        if match:
            numbered.append((int(match.group(1)), path))
    if numbered:
        return sorted(numbered)[-1][1]
    return checkpoints[0] if checkpoints else None


def model_params(model_name: str) -> int | None:
    try:
        from timm.models import create_model
        import model as qkformer_models  # noqa: F401 - registers local model names.

        net = create_model(model_name, pretrained=False, drop_rate=0.0, drop_path_rate=0.1)
        return int(sum(param.numel() for param in net.parameters() if param.requires_grad))
    except Exception:
        return None


def read_scalars(logs_dir: Path, run_name: str) -> list[dict[str, Any]]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return []

    rows: list[dict[str, Any]] = []
    for split_dir in [logs_dir / "train", logs_dir / "te"]:
        if not split_dir.exists():
            continue
        split_name = "test" if split_dir.name == "te" else split_dir.name
        for event_file in sorted(split_dir.glob("events.out.tfevents.*")):
            accumulator = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
            try:
                accumulator.Reload()
            except Exception:
                continue
            for tag in accumulator.Tags().get("scalars", []):
                for event in accumulator.Scalars(tag):
                    rows.append(
                        {
                            "run": run_name,
                            "split": split_name,
                            "tag": tag,
                            "step": int(event.step),
                            "value": float(event.value),
                            "event_file": str(event_file),
                        }
                    )
    return rows


def best_scalar(rows: list[dict[str, Any]], tag: str) -> tuple[int | None, float | None]:
    selected = [row for row in rows if row["tag"] == tag]
    if not selected:
        return None, None
    best = max(selected, key=lambda row: row["value"])
    return int(best["step"]), float(best["value"])


def last_scalar(rows: list[dict[str, Any]], tag: str) -> tuple[int | None, float | None]:
    selected = [row for row in rows if row["tag"] == tag]
    if not selected:
        return None, None
    last = max(selected, key=lambda row: row["step"])
    return int(last["step"]), float(last["value"])


def load_activity_summary(activity_root: Path | None, model_name: str) -> dict[str, Any]:
    if activity_root is None or not activity_root.exists():
        return {}
    candidates = [
        activity_root / model_name / "summary_metrics.json",
        activity_root / f"{model_name}_activity" / "summary_metrics.json",
    ]
    candidates.extend(activity_root.glob(f"{model_name}*/summary_metrics.json"))
    for path in candidates:
        if path.exists():
            try:
                return json.loads(path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                return {}
    return {}


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(path: Path, summary_rows: list[dict[str, Any]]) -> None:
    headers = [
        "model",
        "params",
        "best_test_acc1",
        "best_epoch",
        "final_test_acc1",
        "checkpoint_max_test_acc1",
        "estimated_sops_per_batch",
    ]
    lines = ["# QKFormer CIFAR10-DVS Baseline Report", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary_rows:
        lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
    lines.append("")
    lines.append("Notes:")
    lines.append("- `best_test_acc1` comes from TensorBoard when available.")
    lines.append("- `checkpoint_max_test_acc1` comes from `checkpoint_max_test_acc1.pth`.")
    lines.append("- Activity/SOP columns are filled only after running `activity_profile.py`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.6g}"
    return str(value)


def make_plots(output_dir: Path, scalar_rows: list[dict[str, Any]]) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return

    for metric_name, title, ylabel in [
        ("acc1", "Accuracy@1", "Accuracy (%)"),
        ("loss", "Loss", "Loss"),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5.5))
        plotted = False
        for run_name in sorted({row["run"] for row in scalar_rows}):
            for split in ["train", "test"]:
                tag = f"{split}_{metric_name}"
                rows = sorted(
                    [row for row in scalar_rows if row["run"] == run_name and row["tag"] == tag],
                    key=lambda row: row["step"],
                )
                if not rows:
                    continue
                ax.plot(
                    [row["step"] for row in rows],
                    [row["value"] for row in rows],
                    label=f"{run_name} {split}",
                )
                plotted = True
        if plotted:
            ax.set_title(title)
            ax.set_xlabel("Epoch")
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
            ax.legend()
            fig.tight_layout()
            fig.savefig(output_dir / f"{metric_name}_curves.png", dpi=180)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_root / "analysis"
    activity_root = Path(args.activity_root).expanduser().resolve() if args.activity_root else None
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(run_root)
    summary_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    param_cache: dict[str, int | None] = {}

    for paths in runs:
        args_values = parse_namespace_text(paths.logs_dir / "args.txt")
        model_name = infer_model_name(paths.run_dir.name, args_values)
        if model_name not in param_cache:
            param_cache[model_name] = model_params(model_name)

        run_scalars = read_scalars(paths.logs_dir, paths.run_dir.name)
        scalar_rows.extend(run_scalars)

        best_epoch, best_test_acc1 = best_scalar(run_scalars, "test_acc1")
        _, best_test_acc5 = best_scalar(run_scalars, "test_acc5")
        final_epoch, final_test_acc1 = last_scalar(run_scalars, "test_acc1")
        _, final_test_acc5 = last_scalar(run_scalars, "test_acc5")
        _, final_train_acc1 = last_scalar(run_scalars, "train_acc1")

        max_checkpoint = paths.lr_dir / "checkpoint_max_test_acc1.pth"
        max_ckpt = load_checkpoint(max_checkpoint)
        final_checkpoint = newest_checkpoint(paths.lr_dir)
        activity = load_activity_summary(activity_root, model_name)

        summary_rows.append(
            {
                "run": paths.run_dir.name,
                "model": model_name,
                "lr": args_values.get("lr"),
                "batch_size": args_values.get("batch_size"),
                "T": args_values.get("T"),
                "epochs_arg": args_values.get("epochs"),
                "best_epoch": best_epoch,
                "best_test_acc1": best_test_acc1,
                "best_test_acc5": best_test_acc5,
                "final_epoch": final_epoch,
                "final_train_acc1": final_train_acc1,
                "final_test_acc1": final_test_acc1,
                "final_test_acc5": final_test_acc5,
                "checkpoint_epoch": max_ckpt.get("epoch"),
                "checkpoint_max_test_acc1": max_ckpt.get("max_test_acc1"),
                "checkpoint_test_acc5_at_max": max_ckpt.get("test_acc5_at_max_test_acc1"),
                "final_checkpoint": str(final_checkpoint) if final_checkpoint else "",
                "max_checkpoint": str(max_checkpoint) if max_checkpoint.exists() else "",
                "params": param_cache[model_name],
                "activity_profiled_layers": activity.get("profiled_layers"),
                "activity_weighted_output_firing_rate": activity.get("weighted_output_firing_rate"),
                "dense_macs_per_batch": activity.get("total_dense_macs_per_batch"),
                "attention_ops_per_batch": activity.get("total_attention_ops_per_batch"),
                "estimated_sops_per_batch": activity.get("total_estimated_sops_per_batch"),
            }
        )

    summary_fields = [
        "run",
        "model",
        "params",
        "lr",
        "batch_size",
        "T",
        "epochs_arg",
        "best_epoch",
        "best_test_acc1",
        "best_test_acc5",
        "final_epoch",
        "final_train_acc1",
        "final_test_acc1",
        "final_test_acc5",
        "checkpoint_epoch",
        "checkpoint_max_test_acc1",
        "checkpoint_test_acc5_at_max",
        "activity_profiled_layers",
        "activity_weighted_output_firing_rate",
        "dense_macs_per_batch",
        "attention_ops_per_batch",
        "estimated_sops_per_batch",
        "max_checkpoint",
        "final_checkpoint",
    ]
    scalar_fields = ["run", "split", "tag", "step", "value", "event_file"]
    write_csv(output_dir / "baseline_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "tensorboard_scalars.csv", scalar_rows, scalar_fields)
    write_report(output_dir / "baseline_report.md", summary_rows)
    if not args.no_plots:
        make_plots(output_dir, scalar_rows)

    print(f"Discovered {len(summary_rows)} runs.")
    print(f"Summary CSV: {output_dir / 'baseline_summary.csv'}")
    print(f"Scalars CSV: {output_dir / 'tensorboard_scalars.csv'}")
    print(f"Report: {output_dir / 'baseline_report.md'}")
    if not args.no_plots:
        print(f"Plots: {output_dir / 'acc1_curves.png'} and {output_dir / 'loss_curves.png'}")


if __name__ == "__main__":
    main()
