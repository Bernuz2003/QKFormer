from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


@dataclass
class RunPaths:
    run_dir: Path
    lr_dir: Path
    logs_dir: Path


@dataclass
class ActivityOutputs:
    summary: dict[str, Any]
    summary_path: Path | None
    layerwise_path: Path | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze QKFormer CIFAR10-DVS runs and activity profiles.")
    parser.add_argument("--run-root", required=True, help="Root directory containing training runs.")
    parser.add_argument("--output-dir", default="", help="Analysis output directory. Defaults to <run-root>/analysis.")
    parser.add_argument("--activity-root", default="", help="Optional root containing activity_profile.py outputs.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plot generation.")
    return parser.parse_args()


def discover_runs(run_root: Path) -> list[RunPaths]:
    runs: list[RunPaths] = []
    for run_dir in sorted(run_root.iterdir()):
        if not run_dir.is_dir() or run_dir.name.startswith("_") or run_dir.name in {"analysis", "activity"}:
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


def load_activity_outputs(activity_root: Path | None, run_name: str, model_name: str) -> ActivityOutputs:
    if activity_root is None or not activity_root.exists():
        return ActivityOutputs({}, None, None)

    candidates: list[Path] = []
    for key in unique_strings([run_name, model_name]):
        candidates.extend(
            [
                activity_root / key,
                activity_root / f"{key}_activity",
            ]
        )
        candidates.extend(sorted(activity_root.glob(f"{key}*")))

    for directory in unique_paths(candidates):
        summary_path = directory / "summary_metrics.json"
        layerwise_path = directory / "layerwise_activity.csv"
        if not summary_path.exists():
            continue
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            summary = {}
        return ActivityOutputs(
            summary=summary,
            summary_path=summary_path,
            layerwise_path=layerwise_path if layerwise_path.exists() else None,
        )

    return ActivityOutputs({}, None, None)


def unique_strings(values: Iterable[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value and value not in seen:
            seen.add(value)
            result.append(value)
    return result


def unique_paths(paths: Iterable[Path]) -> list[Path]:
    seen: set[Path] = set()
    result: list[Path] = []
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(path)
    return result


def read_csv_rows(path: Path | None) -> list[dict[str, str]]:
    if path is None or not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_layerwise_activity(rows: list[dict[str, str]], run_name: str, model_name: str) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row.get("layer_name", "")].append(row)

    summary_rows: list[dict[str, Any]] = []
    for layer_name, layer_rows in grouped.items():
        if not layer_name:
            continue
        first = layer_rows[0]
        summary_rows.append(
            {
                "run": run_name,
                "model": model_name,
                "layer_name": layer_name,
                "module_type": first.get("module_type", ""),
                "category": first.get("category", ""),
                "network_stage": first.get("network_stage", ""),
                "params": to_int(first.get("params")),
                "is_binary_input": first.get("is_binary_input", ""),
                "is_binary_output": first.get("is_binary_output", ""),
                "input_firing_rate": mean(to_float(row.get("input_firing_rate")) for row in layer_rows),
                "output_firing_rate": mean(to_float(row.get("output_firing_rate")) for row in layer_rows),
                "spike_density_mean": mean(to_float(row.get("spike_density_mean")) for row in layer_rows),
                "burstiness": mean(to_float(row.get("burstiness")) for row in layer_rows),
                "dense_macs": mean(to_float(row.get("dense_macs")) for row in layer_rows),
                "attention_ops": mean(to_float(row.get("attention_ops")) for row in layer_rows),
                "estimated_sops": mean(to_float(row.get("estimated_sops")) for row in layer_rows),
                "records": len(layer_rows),
            }
        )
    return sorted(summary_rows, key=lambda row: float(row.get("estimated_sops") or 0.0), reverse=True)


def to_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def to_int(value: Any) -> int:
    try:
        if value in (None, ""):
            return 0
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def add_reference_deltas(summary_rows: list[dict[str, Any]]) -> None:
    if not summary_rows:
        return
    reference = next((row for row in summary_rows if row.get("model") == "QKFormer"), summary_rows[0])
    reference_params = to_float(reference.get("params"))
    reference_acc = to_float(reference.get("best_test_acc1"))
    reference_sops = to_float(reference.get("estimated_sops_per_batch"))
    for row in summary_rows:
        params = to_float(row.get("params"))
        acc = to_float(row.get("best_test_acc1"))
        sops = to_float(row.get("estimated_sops_per_batch"))
        row["reference_model"] = reference.get("model")
        row["param_reduction_vs_reference_pct"] = percent_reduction(reference_params, params)
        row["acc1_delta_vs_reference"] = acc - reference_acc if reference_acc else None
        row["sops_reduction_vs_reference_pct"] = percent_reduction(reference_sops, sops)


def percent_reduction(reference: float, value: float) -> float | None:
    if reference <= 0.0 or value <= 0.0:
        return None
    return 100.0 * (1.0 - value / reference)


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_report(
    path: Path,
    summary_rows: list[dict[str, Any]],
    stage_rows: list[dict[str, Any]],
    layer_rows: list[dict[str, Any]],
) -> None:
    headers = [
        "model",
        "params",
        "param_reduction_vs_reference_pct",
        "best_test_acc1",
        "acc1_delta_vs_reference",
        "best_epoch",
        "checkpoint_max_test_acc1",
        "estimated_sops_per_batch",
        "sops_reduction_vs_reference_pct",
    ]
    lines = ["# QKFormer CIFAR10-DVS Run Analysis", ""]
    lines.append("## Run Summary")
    lines.append("")
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("| " + " | ".join(["---"] * len(headers)) + " |")
    for row in summary_rows:
        lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")

    if stage_rows:
        lines.extend(["", "## Stage Activity", ""])
        stage_headers = [
            "model",
            "network_stage",
            "params",
            "estimated_sops_per_batch",
            "weighted_output_firing_rate",
        ]
        lines.append("| " + " | ".join(stage_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(stage_headers)) + " |")
        for row in stage_rows:
            lines.append("| " + " | ".join(format_cell(row.get(header)) for header in stage_headers) + " |")

    if layer_rows:
        lines.extend(["", "## Top Activity Layers", ""])
        top_headers = [
            "model",
            "layer_name",
            "network_stage",
            "module_type",
            "estimated_sops",
            "output_firing_rate",
        ]
        lines.append("| " + " | ".join(top_headers) + " |")
        lines.append("| " + " | ".join(["---"] * len(top_headers)) + " |")
        for row in sorted(layer_rows, key=lambda item: to_float(item.get("estimated_sops")), reverse=True)[:20]:
            lines.append("| " + " | ".join(format_cell(row.get(header)) for header in top_headers) + " |")

    lines.extend(
        [
            "",
            "Notes:",
            "- `best_test_acc1` comes from TensorBoard when available.",
            "- `checkpoint_max_test_acc1` comes from `checkpoint_max_test_acc1.pth`.",
            "- SOP values are estimates from observed activity, not hardware timing measurements.",
            "- Reference deltas use `QKFormer` when present; otherwise the first discovered run.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def format_cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.6g}"
    return str(value)


def make_training_plots(output_dir: Path, scalar_rows: list[dict[str, Any]]) -> None:
    plt = optional_matplotlib()
    if plt is None:
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
                    label=f"{short_run_name(run_name)} {split}",
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


def make_tradeoff_plots(output_dir: Path, summary_rows: list[dict[str, Any]]) -> None:
    plt = optional_matplotlib()
    if plt is None:
        return
    plot_scatter(
        plt,
        output_dir / "params_vs_accuracy.png",
        summary_rows,
        x_key="params",
        y_key="best_test_acc1",
        title="Accuracy vs Parameters",
        xlabel="Trainable parameters",
        ylabel="Best test Acc@1 (%)",
    )
    plot_scatter(
        plt,
        output_dir / "sops_vs_accuracy.png",
        summary_rows,
        x_key="estimated_sops_per_batch",
        y_key="best_test_acc1",
        title="Accuracy vs Estimated SOPs",
        xlabel="Estimated SOPs per profiled batch",
        ylabel="Best test Acc@1 (%)",
    )


def plot_scatter(
    plt,
    path: Path,
    rows: list[dict[str, Any]],
    x_key: str,
    y_key: str,
    title: str,
    xlabel: str,
    ylabel: str,
) -> None:
    points = [(row, to_float(row.get(x_key)), to_float(row.get(y_key))) for row in rows]
    points = [(row, x, y) for row, x, y in points if x > 0.0 and y > 0.0]
    if not points:
        return
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    for row, x, y in points:
        ax.scatter([x], [y], s=70)
        ax.annotate(str(row.get("model", row.get("run", ""))), (x, y), xytext=(5, 5), textcoords="offset points")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def make_stage_plots(output_dir: Path, stage_rows: list[dict[str, Any]]) -> None:
    plt = optional_matplotlib()
    if plt is None or not stage_rows:
        return
    plot_grouped_bar(
        plt,
        output_dir / "stage_sops.png",
        stage_rows,
        value_key="estimated_sops_per_batch",
        title="Estimated SOPs by Stage",
        ylabel="Estimated SOPs per profiled batch",
    )
    plot_grouped_bar(
        plt,
        output_dir / "stage_firing_rate.png",
        stage_rows,
        value_key="weighted_output_firing_rate",
        title="Weighted Output Firing Rate by Stage",
        ylabel="Weighted output firing rate",
    )


def plot_grouped_bar(
    plt,
    path: Path,
    rows: list[dict[str, Any]],
    value_key: str,
    title: str,
    ylabel: str,
) -> None:
    models = sorted({str(row.get("model", "")) for row in rows})
    stages = sorted({str(row.get("network_stage", "")) for row in rows})
    if not models or not stages:
        return

    values_by_key = {
        (str(row.get("model", "")), str(row.get("network_stage", ""))): to_float(row.get(value_key))
        for row in rows
    }
    x_positions = list(range(len(stages)))
    width = 0.8 / max(1, len(models))
    fig, ax = plt.subplots(figsize=(max(9, len(stages) * 1.15), 5.5))
    for index, model_name in enumerate(models):
        offset = (index - (len(models) - 1) / 2) * width
        values = [values_by_key.get((model_name, stage), 0.0) for stage in stages]
        ax.bar([x + offset for x in x_positions], values, width=width, label=model_name)
    ax.set_title(title)
    ax.set_xlabel("Network stage")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(stages, rotation=35, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def optional_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def short_run_name(run_name: str) -> str:
    return run_name.split("_b", 1)[0]


def main() -> None:
    args = parse_args()
    run_root = Path(args.run_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else run_root / "analysis"
    activity_root = Path(args.activity_root).expanduser().resolve() if args.activity_root else None
    output_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(run_root)
    summary_rows: list[dict[str, Any]] = []
    scalar_rows: list[dict[str, Any]] = []
    stage_rows: list[dict[str, Any]] = []
    layer_summary_rows: list[dict[str, Any]] = []
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
        activity = load_activity_outputs(activity_root, paths.run_dir.name, model_name)

        for stage in activity.summary.get("stage_summary", []):
            stage_rows.append(
                {
                    "run": paths.run_dir.name,
                    "model": model_name,
                    **stage,
                    "activity_summary_path": str(activity.summary_path) if activity.summary_path else "",
                }
            )
        layer_summary_rows.extend(
            summarize_layerwise_activity(read_csv_rows(activity.layerwise_path), paths.run_dir.name, model_name)
        )

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
                "activity_profiled_layers": activity.summary.get("profiled_layers"),
                "activity_profiled_binary_layers": activity.summary.get("profiled_binary_layers"),
                "activity_weighted_output_firing_rate": activity.summary.get("weighted_output_firing_rate"),
                "activity_mean_layer_output_firing_rate": activity.summary.get("mean_layer_output_firing_rate"),
                "activity_max_layer_output_firing_rate": activity.summary.get("max_layer_output_firing_rate"),
                "dense_macs_per_batch": activity.summary.get("total_dense_macs_per_batch"),
                "attention_ops_per_batch": activity.summary.get("total_attention_ops_per_batch"),
                "estimated_sops_per_batch": activity.summary.get("total_estimated_sops_per_batch"),
                "activity_summary_path": str(activity.summary_path) if activity.summary_path else "",
                "activity_layerwise_path": str(activity.layerwise_path) if activity.layerwise_path else "",
            }
        )

    add_reference_deltas(summary_rows)

    summary_fields = [
        "run",
        "model",
        "reference_model",
        "params",
        "param_reduction_vs_reference_pct",
        "lr",
        "batch_size",
        "T",
        "epochs_arg",
        "best_epoch",
        "best_test_acc1",
        "acc1_delta_vs_reference",
        "best_test_acc5",
        "final_epoch",
        "final_train_acc1",
        "final_test_acc1",
        "final_test_acc5",
        "checkpoint_epoch",
        "checkpoint_max_test_acc1",
        "checkpoint_test_acc5_at_max",
        "activity_profiled_layers",
        "activity_profiled_binary_layers",
        "activity_weighted_output_firing_rate",
        "activity_mean_layer_output_firing_rate",
        "activity_max_layer_output_firing_rate",
        "dense_macs_per_batch",
        "attention_ops_per_batch",
        "estimated_sops_per_batch",
        "sops_reduction_vs_reference_pct",
        "max_checkpoint",
        "final_checkpoint",
        "activity_summary_path",
        "activity_layerwise_path",
    ]
    scalar_fields = ["run", "split", "tag", "step", "value", "event_file"]
    stage_fields = [
        "run",
        "model",
        "network_stage",
        "layers",
        "params",
        "dense_macs_per_batch",
        "attention_ops_per_batch",
        "estimated_sops_per_batch",
        "weighted_output_firing_rate",
        "activity_summary_path",
    ]
    layer_fields = [
        "run",
        "model",
        "layer_name",
        "module_type",
        "category",
        "network_stage",
        "params",
        "is_binary_input",
        "is_binary_output",
        "input_firing_rate",
        "output_firing_rate",
        "spike_density_mean",
        "burstiness",
        "dense_macs",
        "attention_ops",
        "estimated_sops",
        "records",
    ]

    write_csv(output_dir / "run_summary.csv", summary_rows, summary_fields)
    write_csv(output_dir / "tensorboard_scalars.csv", scalar_rows, scalar_fields)
    write_csv(output_dir / "activity_stage_summary.csv", stage_rows, stage_fields)
    write_csv(output_dir / "activity_layer_summary.csv", layer_summary_rows, layer_fields)
    write_report(output_dir / "run_report.md", summary_rows, stage_rows, layer_summary_rows)

    if not args.no_plots:
        make_training_plots(output_dir, scalar_rows)
        make_tradeoff_plots(output_dir, summary_rows)
        make_stage_plots(output_dir, stage_rows)

    print(f"Discovered {len(summary_rows)} runs.")
    print(f"Run summary CSV: {output_dir / 'run_summary.csv'}")
    print(f"TensorBoard scalars CSV: {output_dir / 'tensorboard_scalars.csv'}")
    print(f"Activity stage CSV: {output_dir / 'activity_stage_summary.csv'}")
    print(f"Activity layer CSV: {output_dir / 'activity_layer_summary.csv'}")
    print(f"Report: {output_dir / 'run_report.md'}")
    if not args.no_plots:
        print(f"Plots directory: {output_dir}")


if __name__ == "__main__":
    main()
