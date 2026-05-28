from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze one QKFormer CIFAR10-DVS run.")
    parser.add_argument("--run-dir", required=True, help="Run directory, e.g. runs_encoder_v2/E0_count_T16")
    parser.add_argument("--lr-dir", default="", help="Optional lr directory inside the run. Defaults to first lr* dir.")
    parser.add_argument("--profile-dir", default="", help="Optional profile directory. Defaults to <run-dir>/profile.")
    parser.add_argument("--output-dir", default="", help="Optional output directory. Defaults to profile dir.")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots.")
    return parser.parse_args()


def resolve_paths(args: argparse.Namespace) -> dict[str, Path]:
    input_path = Path(args.run_dir).expanduser().resolve()
    if input_path.name.startswith("lr") and input_path.is_dir():
        run_dir = input_path.parent
        lr_dir = input_path
    else:
        run_dir = input_path
        lr_dir = Path(args.lr_dir).expanduser().resolve() if args.lr_dir else first_lr_dir(run_dir)

    logs_dir = run_dir / f"{lr_dir.name}_logs"
    profile_dir = Path(args.profile_dir).expanduser().resolve() if args.profile_dir else run_dir / "profile"
    output_dir = Path(args.output_dir).expanduser().resolve() if args.output_dir else profile_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "run_dir": run_dir,
        "lr_dir": lr_dir,
        "logs_dir": logs_dir,
        "profile_dir": profile_dir,
        "output_dir": output_dir,
    }


def first_lr_dir(run_dir: Path) -> Path:
    lr_dirs = sorted(path for path in run_dir.glob("lr*") if path.is_dir() and not path.name.endswith("_logs"))
    if not lr_dirs:
        raise FileNotFoundError(f"No lr* directory found in {run_dir}")
    return lr_dirs[0]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


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


def summarize_metrics(metrics_rows: list[dict[str, str]]) -> dict[str, Any]:
    if not metrics_rows:
        return {}
    parsed = [{key: parse_number(value) for key, value in row.items()} for row in metrics_rows]
    best = max(parsed, key=lambda row: to_float(row.get("test_acc1")))
    final = max(parsed, key=lambda row: to_float(row.get("epoch")))
    return {
        "best_epoch": best.get("epoch"),
        "best_test_acc1": best.get("test_acc1"),
        "best_test_acc5": best.get("test_acc5"),
        "best_test_loss": best.get("test_loss"),
        "final_epoch": final.get("epoch"),
        "final_train_acc1": final.get("train_acc1"),
        "final_test_acc1": final.get("test_acc1"),
        "final_test_acc5": final.get("test_acc5"),
        "final_test_loss": final.get("test_loss"),
    }


def parse_number(value: Any) -> Any:
    try:
        if value in (None, ""):
            return None
        text = str(value)
        if any(ch in text for ch in ".eE"):
            return float(text)
        return int(text)
    except (TypeError, ValueError):
        return value


def to_float(value: Any) -> float:
    try:
        if value in (None, ""):
            return 0.0
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def compact_summary(paths: dict[str, Path]) -> dict[str, Any]:
    logs_dir = paths["logs_dir"]
    profile_dir = paths["profile_dir"]
    lr_dir = paths["lr_dir"]

    config = load_json(logs_dir / "config.json")
    model_summary = load_json(logs_dir / "model_summary.json")
    encoder_meta = load_json(profile_dir / "encoder_meta.json")
    input_profile = load_json(profile_dir / "input_encoder_profile.json")
    profile_summary = load_json(profile_dir / "profile_summary.json") or load_json(profile_dir / "summary_metrics.json")
    metrics_rows = read_csv_rows(logs_dir / "metrics.csv")
    metrics_summary = summarize_metrics(metrics_rows)

    max_checkpoint = lr_dir / "checkpoint_max_test_acc1.pth"
    max_ckpt = load_checkpoint(max_checkpoint)
    final_checkpoint = newest_checkpoint(lr_dir)

    encoder = profile_summary.get("encoder") or encoder_meta or config.get("encoder_meta", {})
    input_activity = profile_summary.get("input_activity") or input_profile.get("summary", {})
    model_activity = profile_summary.get("model_activity", {})

    return {
        "run_id": paths["run_dir"].name,
        "run_dir": str(paths["run_dir"]),
        "lr_dir": str(lr_dir),
        "logs_dir": str(logs_dir),
        "profile_dir": str(profile_dir),
        "model": config.get("args", {}).get("model") or profile_summary.get("model"),
        "encoder_name": encoder.get("encoder_name"),
        "source_T": encoder.get("source_T"),
        "encoded_T": encoder.get("encoded_T"),
        "source_C": encoder.get("source_C"),
        "encoded_C": encoder.get("encoded_C"),
        "time_compression_ratio": encoder.get("time_compression_ratio"),
        "channel_expansion_ratio": encoder.get("channel_expansion_ratio"),
        "numel_compression_ratio": encoder.get("numel_compression_ratio"),
        "effective_snn_steps": encoder.get("effective_snn_steps"),
        "model_params": model_summary.get("model_params") or profile_summary.get("total_params_model"),
        "checkpoint_epoch": max_ckpt.get("epoch"),
        "checkpoint_max_test_acc1": max_ckpt.get("max_test_acc1"),
        "checkpoint_test_acc5_at_max": max_ckpt.get("test_acc5_at_max_test_acc1"),
        "max_checkpoint": str(max_checkpoint) if max_checkpoint.exists() else "",
        "final_checkpoint": str(final_checkpoint) if final_checkpoint else "",
        "input_density": input_activity.get("input_density"),
        "input_is_binary": input_activity.get("input_is_binary"),
        "input_mean": input_activity.get("input_mean"),
        "input_max": input_activity.get("input_max"),
        "temporal_burstiness": input_activity.get("temporal_burstiness"),
        "active_pixel_ratio": input_activity.get("active_pixel_ratio"),
        "active_patch_ratio_8x8": input_activity.get("active_patch_ratio_8x8"),
        "active_patch_ratio_16x16": input_activity.get("active_patch_ratio_16x16"),
        "total_estimated_sops_per_batch": profile_summary.get("total_estimated_sops_per_batch"),
        "total_estimated_sops_per_sample": profile_summary.get("total_estimated_sops_per_sample"),
        "weighted_output_firing_rate": profile_summary.get("weighted_output_firing_rate"),
        "patch_embed_sops_share_pct": profile_summary.get("patch_embed_sops_share_pct"),
        "patch_embed_params_share_pct": profile_summary.get("patch_embed_params_share_pct"),
        **metrics_summary,
    }


def write_report(path: Path, summary: dict[str, Any], stage_rows: list[dict[str, str]], layer_rows: list[dict[str, str]]) -> None:
    lines = [f"# Run Report: {summary.get('run_id', '')}", ""]
    lines.extend(
        [
            "## Summary",
            "",
            f"- Model: `{summary.get('model')}`",
            f"- Encoder: `{summary.get('encoder_name')}`",
            f"- Source/encoded T: `{summary.get('source_T')}` -> `{summary.get('encoded_T')}`",
            f"- Source/encoded C: `{summary.get('source_C')}` -> `{summary.get('encoded_C')}`",
            f"- Best Acc@1: `{format_cell(summary.get('best_test_acc1') or summary.get('checkpoint_max_test_acc1'))}`",
            f"- Estimated SOPs/sample: `{format_cell(summary.get('total_estimated_sops_per_sample'))}`",
            f"- Weighted firing rate: `{format_cell(summary.get('weighted_output_firing_rate'))}`",
            f"- Input density: `{format_cell(summary.get('input_density'))}`",
            f"- Patch embed SOP share: `{format_cell(summary.get('patch_embed_sops_share_pct'))}%`",
            "",
        ]
    )

    lines.extend(["## Encoder/Input", ""])
    encoder_keys = [
        "time_compression_ratio",
        "channel_expansion_ratio",
        "numel_compression_ratio",
        "input_is_binary",
        "input_mean",
        "input_max",
        "temporal_burstiness",
        "active_pixel_ratio",
        "active_patch_ratio_8x8",
        "active_patch_ratio_16x16",
    ]
    for key in encoder_keys:
        lines.append(f"- `{key}`: `{format_cell(summary.get(key))}`")

    if stage_rows:
        lines.extend(["", "## Stage Activity", ""])
        headers = ["network_stage", "estimated_sops_per_batch", "sops_share_pct", "weighted_output_firing_rate", "params"]
        lines.append(markdown_table(headers, stage_rows))

    if layer_rows:
        lines.extend(["", "## Top Activity Layers", ""])
        top_rows = sorted(layer_rows, key=lambda row: to_float(row.get("estimated_sops")), reverse=True)[:20]
        headers = ["layer_name", "network_stage", "module_type", "estimated_sops", "output_firing_rate"]
        lines.append(markdown_table(headers, top_rows))

    lines.extend(
        [
            "",
            "Notes:",
            "- SOP values are estimates from observed activity, not hardware timing measurements.",
            "- This report describes one run only; use `aggregate_runs.py` for cross-run comparisons.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def markdown_table(headers: list[str], rows: list[dict[str, Any]]) -> str:
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(format_cell(row.get(header)) for header in headers) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if value is None or value == "":
        return "-"
    if isinstance(value, float):
        if math.isnan(value):
            return "-"
        return f"{value:.6g}"
    return str(value)


def make_plots(output_dir: Path, paths: dict[str, Path]) -> None:
    plt = optional_matplotlib()
    if plt is None:
        return

    metrics_rows = read_csv_rows(paths["logs_dir"] / "metrics.csv")
    if metrics_rows:
        epochs = [to_float(row.get("epoch")) for row in metrics_rows]
        fig, ax = plt.subplots(figsize=(8, 4.8))
        for key in ["train_acc1", "test_acc1"]:
            values = [to_float(row.get(key)) for row in metrics_rows]
            if any(values):
                ax.plot(epochs, values, label=key)
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Acc@1")
        ax.set_title("Accuracy Curves")
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_dir / "accuracy_curves.png", dpi=180)
        plt.close(fig)

    input_profile = load_json(paths["profile_dir"] / "input_encoder_profile.json")
    input_summary = input_profile.get("summary", {})
    for key, filename, ylabel in [
        ("density_per_timestep", "input_density_per_timestep.png", "Density"),
        ("density_per_channel", "input_density_per_channel.png", "Density"),
    ]:
        values = input_summary.get(key, [])
        if not values:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.8))
        ax.bar(list(range(len(values))), values)
        ax.set_xlabel(key.replace("density_per_", "").title())
        ax.set_ylabel(ylabel)
        ax.set_title(key)
        ax.grid(True, axis="y", alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / filename, dpi=180)
        plt.close(fig)


def optional_matplotlib():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception:
        return None


def main() -> None:
    args = parse_args()
    paths = resolve_paths(args)
    summary = compact_summary(paths)
    stage_rows = read_csv_rows(paths["profile_dir"] / "stage_summary.csv")
    layer_rows = read_csv_rows(paths["profile_dir"] / "layer_summary.csv")

    output_dir = paths["output_dir"]
    with (output_dir / "run_analysis.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    write_report(output_dir / "run_report.md", summary, stage_rows, layer_rows)
    if not args.no_plots:
        make_plots(output_dir, paths)

    print(f"Run: {summary.get('run_id')}")
    print(f"Report: {output_dir / 'run_report.md'}")
    print(f"Analysis JSON: {output_dir / 'run_analysis.json'}")


if __name__ == "__main__":
    main()
