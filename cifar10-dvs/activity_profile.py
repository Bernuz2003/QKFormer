from __future__ import annotations

import argparse
import csv
import datetime
import json
from collections import defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.utils.data
from spikingjelly.clock_driven import functional
from spikingjelly.datasets import cifar10_dvs
from timm.models import create_model
from torch import nn

import model as qkformer_models  # noqa: F401 - registers timm model names.


@dataclass
class LayerActivity:
    run_id: str
    batch_index: int
    layer_name: str
    module_type: str
    profile_scope: str
    category: str
    network_stage: str
    input_shape: str
    output_shape: str
    input_numel: int
    output_numel: int
    params: int
    is_binary_input: bool
    is_binary_output: bool
    input_spike_count: float
    output_spike_count: float
    input_firing_rate: float
    output_firing_rate: float
    spike_density_mean: float
    spike_density_std: float
    spike_density_timestep: str
    burstiness: float
    kernel_size: str
    stride: str
    padding: str
    dilation: str
    groups: int | None
    in_channels: int | None
    out_channels: int | None
    in_features: int | None
    out_features: int | None
    time_steps: int
    dense_macs: float
    attention_ops: float
    estimated_sops: float


class QKFormerActivityProfiler:
    """Layer-wise activity and operation profiler for CIFAR10-DVS QKFormer."""

    TRACKED_BLOCKS = {
        "PatchEmbedInit",
        "PatchEmbeddingStage",
        "TokenSpikingTransformer",
        "SpikingTransformer",
        "Token_QK_Attention",
        "Spiking_Self_Attention",
        "MLP",
    }
    TRACKED_SPIKING = {
        "MultiStepLIFNode",
        "MultiStepParametricLIFNode",
    }

    def __init__(self, model: nn.Module, run_id: str, time_steps: int) -> None:
        self.model = model
        self.run_id = run_id
        self.time_steps = int(time_steps)
        self.batch_index = 0
        self.enabled = False
        self.handles: list[Any] = []
        self.records: list[LayerActivity] = []

    def attach(self) -> None:
        for name, module in self.model.named_modules():
            if name and self._should_track(module):
                self.handles.append(module.register_forward_hook(self._hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def enable(self) -> None:
        self.enabled = True

    def disable(self) -> None:
        self.enabled = False

    def set_batch_index(self, batch_index: int) -> None:
        self.batch_index = int(batch_index)

    def save(self, output_dir: str | Path, metadata: dict[str, Any]) -> dict[str, Any]:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        layerwise_path = output_dir / "layerwise_activity.csv"
        fieldnames = [field.name for field in fields(LayerActivity)]
        with layerwise_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for record in self.records:
                writer.writerow(asdict(record))

        summary = summarize_activity(self.records)
        summary.update(metadata)
        summary_path = output_dir / "summary_metrics.json"
        with summary_path.open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        return {
            "layerwise_path": str(layerwise_path),
            "summary_path": str(summary_path),
            "summary": summary,
        }

    def _should_track(self, module: nn.Module) -> bool:
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Linear, nn.BatchNorm1d, nn.BatchNorm2d)):
            return True
        if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d)):
            return True
        cls = module.__class__.__name__
        return cls in self.TRACKED_BLOCKS or cls in self.TRACKED_SPIKING

    def _hook(self, name: str):
        def fn(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
            if not self.enabled:
                return
            record = self._make_record(name, module, inputs, output)
            if record is not None:
                self.records.append(record)

        return fn

    def _make_record(
        self,
        name: str,
        module: nn.Module,
        inputs: tuple[Any, ...],
        output: Any,
    ) -> LayerActivity | None:
        inp = first_tensor(inputs)
        out = first_tensor(output)
        if out is None:
            return None

        is_binary_input, input_spikes, input_fr, input_numel = binary_stats(inp)
        is_binary_output, output_spikes, output_fr, output_numel = binary_stats(out)
        density_mean, density_std, burstiness, density_timestep = self._temporal_density(
            module,
            out,
            is_binary_output,
        )
        structure = module_structure(module)
        dense_macs = dense_macs(module, out)
        attention_ops = self._attention_ops(module, inp)
        estimated_sops = self._estimate_sops(dense_macs, attention_ops, is_binary_input, input_fr)
        category, network_stage = categorize_layer(name, module)

        return LayerActivity(
            run_id=self.run_id,
            batch_index=self.batch_index,
            layer_name=name,
            module_type=module.__class__.__name__,
            profile_scope=profile_scope(module),
            category=category,
            network_stage=network_stage,
            input_shape=json.dumps(list(inp.shape)) if inp is not None else "[]",
            output_shape=json.dumps(list(out.shape)),
            input_numel=input_numel,
            output_numel=output_numel,
            params=sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad),
            is_binary_input=is_binary_input,
            is_binary_output=is_binary_output,
            input_spike_count=input_spikes,
            output_spike_count=output_spikes,
            input_firing_rate=input_fr,
            output_firing_rate=output_fr,
            spike_density_mean=density_mean,
            spike_density_std=density_std,
            spike_density_timestep=json.dumps(density_timestep),
            burstiness=burstiness,
            kernel_size=structure.get("kernel_size", ""),
            stride=structure.get("stride", ""),
            padding=structure.get("padding", ""),
            dilation=structure.get("dilation", ""),
            groups=structure.get("groups"),
            in_channels=structure.get("in_channels"),
            out_channels=structure.get("out_channels"),
            in_features=structure.get("in_features"),
            out_features=structure.get("out_features"),
            time_steps=self.time_steps,
            dense_macs=float(dense_macs),
            attention_ops=float(attention_ops),
            estimated_sops=float(estimated_sops),
        )

    def _temporal_density(
        self,
        module: nn.Module,
        tensor: torch.Tensor,
        is_binary: bool,
    ) -> tuple[float, float, float, list[float]]:
        if not is_binary:
            return 0.0, 0.0, 0.0, []

        timed = self._as_time_first(module, tensor.detach())
        if timed is None or timed.dim() < 2:
            return 0.0, 0.0, 0.0, []

        reduce_dims = tuple(dim for dim in range(timed.dim()) if dim != 0)
        density = timed.float().mean(dim=reduce_dims)
        mean = float(density.mean().cpu())
        std = float(density.std(unbiased=False).cpu())
        var = float(density.var(unbiased=False).cpu())
        burstiness = var / (mean + 1e-12)
        return mean, std, burstiness, [float(v) for v in density.cpu().flatten()]

    def _as_time_first(self, module: nn.Module, tensor: torch.Tensor) -> torch.Tensor | None:
        if self.time_steps <= 0:
            return None
        if tensor.dim() >= 3 and tensor.shape[0] == self.time_steps:
            return tensor
        flattened_time_modules = (nn.Conv1d, nn.Conv2d, nn.BatchNorm1d, nn.BatchNorm2d, nn.MaxPool1d, nn.MaxPool2d)
        if isinstance(module, flattened_time_modules) and tensor.dim() >= 2 and tensor.shape[0] % self.time_steps == 0:
            batch = tensor.shape[0] // self.time_steps
            return tensor.reshape(self.time_steps, batch, *tensor.shape[1:])
        return None

    def _attention_ops(self, module: nn.Module, inp: torch.Tensor | None) -> float:
        if inp is None or inp.dim() != 5:
            return 0.0
        cls = module.__class__.__name__
        if cls not in {"Token_QK_Attention", "Spiking_Self_Attention"}:
            return 0.0

        t, batch, channels, height, width = [int(v) for v in inp.shape]
        tokens = height * width
        heads = int(getattr(module, "num_heads", 1))
        head_dim = channels // max(1, heads)

        if cls == "Token_QK_Attention":
            q_reduce_adds = t * batch * heads * tokens * max(0, head_dim - 1)
            gated_k_mults = t * batch * heads * tokens * head_dim
            return float(q_reduce_adds + gated_k_mults)

        kv_macs = t * batch * heads * head_dim * tokens * head_dim
        q_kv_macs = t * batch * heads * tokens * head_dim * head_dim
        return float(kv_macs + q_kv_macs)

    def _estimate_sops(
        self,
        dense_macs_value: float,
        attention_ops_value: float,
        is_binary_input: bool,
        input_firing_rate: float,
    ) -> float:
        if is_binary_input:
            return (dense_macs_value + attention_ops_value) * input_firing_rate
        return dense_macs_value + attention_ops_value


def first_tensor(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            tensor = first_tensor(item)
            if tensor is not None:
                return tensor
    return None


def binary_stats(tensor: torch.Tensor | None) -> tuple[bool, float, float, int]:
    if tensor is None:
        return False, 0.0, 0.0, 0
    detached = tensor.detach()
    numel = int(detached.numel())
    if numel == 0:
        return False, 0.0, 0.0, 0
    is_binary = bool(((detached == 0) | (detached == 1)).all().item())
    if not is_binary:
        return False, 0.0, 0.0, numel
    spike_count = float(detached.float().sum().item())
    return True, spike_count, spike_count / max(1, numel), numel


def profile_scope(module: nn.Module) -> str:
    return "leaf" if not any(True for _ in module.children()) else "block"


def categorize_layer(name: str, module: nn.Module) -> tuple[str, str]:
    cls = module.__class__.__name__
    if name.startswith("patch_embed1"):
        return "feature_extractor", "patch_embed1"
    if name.startswith("patch_embed2"):
        return "feature_extractor", "patch_embed2"
    if name == "head" or name.startswith("head."):
        return "classification_head", "head"
    if name.startswith("stage1"):
        if ".tssa" in name or cls == "Token_QK_Attention":
            return "attention", "stage1_attention"
        if ".mlp" in name or cls == "MLP":
            return "mlp", "stage1_mlp"
        return "transformer_block", "stage1"
    if name.startswith("stage2"):
        if ".ssa" in name or cls == "Spiking_Self_Attention":
            return "attention", "stage2_attention"
        if ".mlp" in name or cls == "MLP":
            return "mlp", "stage2_mlp"
        return "transformer_block", "stage2"
    if cls in QKFormerActivityProfiler.TRACKED_SPIKING:
        return "spiking_neuron", "other"
    return "other", "other"


def module_structure(module: nn.Module) -> dict[str, Any]:
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        return {
            "kernel_size": json.dumps(as_int_list(module.kernel_size)),
            "stride": json.dumps(as_int_list(module.stride)),
            "padding": json.dumps(as_int_list(module.padding)),
            "dilation": json.dumps(as_int_list(module.dilation)),
            "groups": int(module.groups),
            "in_channels": int(module.in_channels),
            "out_channels": int(module.out_channels),
        }
    if isinstance(module, nn.Linear):
        return {
            "in_features": int(module.in_features),
            "out_features": int(module.out_features),
        }
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d)):
        return {
            "kernel_size": json.dumps(as_int_list(module.kernel_size)),
            "stride": json.dumps(as_int_list(module.stride)),
            "padding": json.dumps(as_int_list(module.padding)),
            "dilation": json.dumps(as_int_list(module.dilation)),
        }
    return {}


def as_int_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, tuple):
        return [int(v) for v in value]
    return [int(value)]


def dense_macs(module: nn.Module, out: torch.Tensor) -> float:
    if isinstance(module, (nn.Conv1d, nn.Conv2d)):
        kernel_ops = int(module.in_channels // module.groups)
        for kernel_size in module.kernel_size:
            kernel_ops *= int(kernel_size)
        return float(out.numel() * kernel_ops)
    if isinstance(module, nn.Linear):
        return float(out.numel() * int(module.in_features))
    return 0.0


def summarize_activity(records: list[LayerActivity]) -> dict[str, Any]:
    if not records:
        return {
            "profiled_layers": 0,
            "profiled_binary_layers": 0,
            "total_params_profiled": 0,
            "total_dense_macs_per_batch": 0.0,
            "total_attention_ops_per_batch": 0.0,
            "total_estimated_sops_per_batch": 0.0,
            "mean_layer_output_firing_rate": 0.0,
            "weighted_output_firing_rate": 0.0,
            "max_layer_output_firing_rate": 0.0,
            "stage_summary": [],
        }

    grouped: dict[str, list[LayerActivity]] = defaultdict(list)
    for record in records:
        grouped[record.layer_name].append(record)

    layer_rows = []
    for layer_name, layer_records in grouped.items():
        first = layer_records[0]
        layer_rows.append(
            {
                "layer_name": layer_name,
                "module_type": first.module_type,
                "category": first.category,
                "network_stage": first.network_stage,
                "params": first.params,
                "is_binary_output": first.is_binary_output,
                "output_spike_count": mean(r.output_spike_count for r in layer_records),
                "output_numel": mean(r.output_numel for r in layer_records),
                "output_firing_rate": mean(r.output_firing_rate for r in layer_records),
                "dense_macs": mean(r.dense_macs for r in layer_records),
                "attention_ops": mean(r.attention_ops for r in layer_records),
                "estimated_sops": mean(r.estimated_sops for r in layer_records),
            }
        )

    binary_layers = [row for row in layer_rows if row["is_binary_output"]]
    total_spikes = sum(float(row["output_spike_count"]) for row in binary_layers)
    total_numel = sum(float(row["output_numel"]) for row in binary_layers)

    stage_rows = []
    for stage, rows in group_rows(layer_rows, "network_stage").items():
        binary_stage_rows = [row for row in rows if row["is_binary_output"]]
        stage_spikes = sum(float(row["output_spike_count"]) for row in binary_stage_rows)
        stage_numel = sum(float(row["output_numel"]) for row in binary_stage_rows)
        stage_rows.append(
            {
                "network_stage": stage,
                "layers": len(rows),
                "params": int(sum(int(row["params"]) for row in rows)),
                "dense_macs_per_batch": sum(float(row["dense_macs"]) for row in rows),
                "attention_ops_per_batch": sum(float(row["attention_ops"]) for row in rows),
                "estimated_sops_per_batch": sum(float(row["estimated_sops"]) for row in rows),
                "weighted_output_firing_rate": stage_spikes / max(1.0, stage_numel),
            }
        )

    return {
        "profiled_layers": len(layer_rows),
        "profiled_binary_layers": len(binary_layers),
        "total_params_profiled": int(sum(int(row["params"]) for row in layer_rows)),
        "total_dense_macs_per_batch": sum(float(row["dense_macs"]) for row in layer_rows),
        "total_attention_ops_per_batch": sum(float(row["attention_ops"]) for row in layer_rows),
        "total_estimated_sops_per_batch": sum(float(row["estimated_sops"]) for row in layer_rows),
        "mean_layer_output_firing_rate": mean(row["output_firing_rate"] for row in binary_layers),
        "weighted_output_firing_rate": total_spikes / max(1.0, total_numel),
        "max_layer_output_firing_rate": max((float(row["output_firing_rate"]) for row in binary_layers), default=0.0),
        "stage_summary": sorted(stage_rows, key=lambda row: row["network_stage"]),
    }


def mean(values: Iterable[float]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return grouped


def split_to_train_test_set(
    train_ratio: float,
    origin_dataset: torch.utils.data.Dataset,
    num_classes: int,
) -> tuple[torch.utils.data.Subset, torch.utils.data.Subset]:
    label_idx: list[list[int]] = [[] for _ in range(num_classes)]
    for index, item in enumerate(origin_dataset):
        label = item[1]
        if isinstance(label, (np.ndarray, torch.Tensor)):
            label = int(label.item())
        label_idx[int(label)].append(index)

    train_idx: list[int] = []
    test_idx: list[int] = []
    for class_indices in label_idx:
        split_pos = int(np.ceil(len(class_indices) * train_ratio))
        train_idx.extend(class_indices[:split_pos])
        test_idx.extend(class_indices[split_pos:])
    return torch.utils.data.Subset(origin_dataset, train_idx), torch.utils.data.Subset(origin_dataset, test_idx)


def build_data_loader(args: argparse.Namespace) -> Iterable[tuple[torch.Tensor, torch.Tensor]]:
    if args.synthetic:
        return synthetic_loader(args.batch_size, args.T, args.input_size, args.max_batches)

    if not args.data_path:
        raise ValueError("Set --data-path for CIFAR10-DVS profiling, or use --synthetic for a dry run.")

    origin_set = cifar10_dvs.CIFAR10DVS(
        root=args.data_path,
        data_type="frame",
        frames_number=args.T,
        split_by="number",
    )
    dataset_train, dataset_test = split_to_train_test_set(args.train_ratio, origin_set, args.num_classes)
    dataset = dataset_train if args.split == "train" else dataset_test
    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=args.batch_size,
        shuffle=args.split == "train",
        num_workers=args.workers,
        drop_last=False,
        pin_memory=args.device.startswith("cuda"),
    )


def synthetic_loader(batch_size: int, time_steps: int, input_size: int, batches: int):
    for _ in range(batches):
        image = torch.randint(0, 2, (batch_size, time_steps, 2, input_size, input_size)).float()
        target = torch.zeros(batch_size, dtype=torch.long)
        yield image, target


def load_checkpoint(model: nn.Module, checkpoint_path: str, strict: bool) -> None:
    if not checkpoint_path:
        return
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing:
        print(f"Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"Unexpected checkpoint keys: {unexpected}")


def set_spiking_backend(model: nn.Module, backend: str) -> None:
    for module in model.modules():
        if hasattr(module, "backend"):
            try:
                module.backend = backend
            except Exception:
                pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Profile QKFormer activity on CIFAR10-DVS.")
    parser.add_argument("--model", default="QKFormer", help="timm model name, e.g. QKFormer or Mini_QKFormer_128")
    parser.add_argument("--checkpoint", default="", help="optional checkpoint path")
    parser.add_argument("--data-path", default="", help="CIFAR10-DVS root directory")
    parser.add_argument("--output-dir", default="./activity_logs", help="directory for profiler outputs")
    parser.add_argument("--run-id", default="", help="run id used as output subdirectory")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", default=16, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--T", default=16, type=int, help="simulation steps")
    parser.add_argument("--max-batches", default=8, type=int, help="number of batches to profile")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--train-ratio", default=0.9, type=float)
    parser.add_argument("--num-classes", default=10, type=int)
    parser.add_argument("--drop-path-rate", default=0.1, type=float)
    parser.add_argument("--strict-load", action="store_true", help="use strict checkpoint loading")
    parser.add_argument("--synthetic", action="store_true", help="profile random binary inputs instead of CIFAR10-DVS")
    parser.add_argument("--input-size", default=128, type=int, help="synthetic input height/width")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_id = args.run_id or f"{args.model}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / run_id
    device = torch.device(args.device)

    model = create_model(
        args.model,
        pretrained=False,
        drop_rate=0.0,
        drop_path_rate=args.drop_path_rate,
    )
    load_checkpoint(model, args.checkpoint, args.strict_load)
    if device.type == "cpu":
        set_spiking_backend(model, "torch")
    model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    data_loader = build_data_loader(args)
    profiler = QKFormerActivityProfiler(model, run_id=run_id, time_steps=args.T)
    profiler.attach()

    profiled_batches = 0
    try:
        with torch.no_grad():
            for batch_index, (image, _) in enumerate(data_loader):
                if batch_index >= args.max_batches:
                    break
                image = image.to(device, non_blocking=True).float()
                profiler.set_batch_index(batch_index)
                profiler.enable()
                model(image)
                profiler.disable()
                functional.reset_net(model)
                profiled_batches += 1
    finally:
        profiler.disable()
        profiler.close()

    metadata = {
        "run_id": run_id,
        "model": args.model,
        "checkpoint": args.checkpoint,
        "device": str(device),
        "time_steps": args.T,
        "batch_size": args.batch_size,
        "profiled_batches": profiled_batches,
        "total_params_model": int(total_params),
        "synthetic": bool(args.synthetic),
        "split": args.split,
    }
    saved = profiler.save(output_dir, metadata)
    print(f"Profiled {profiled_batches} batches with {len(profiler.records)} layer records.")
    print(f"Layerwise CSV: {saved['layerwise_path']}")
    print(f"Summary JSON: {saved['summary_path']}")
    print(json.dumps(saved["summary"], indent=2))


if __name__ == "__main__":
    main()
