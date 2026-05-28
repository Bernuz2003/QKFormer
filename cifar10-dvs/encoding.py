from __future__ import annotations

import csv
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import torch


ENCODERS = ("count_number", "occupancy_number", "temporal_pack_2")


@dataclass(frozen=True)
class EncoderMeta:
    encoder_name: str
    source_T: int
    encoded_T: int
    source_C: int
    encoded_C: int
    source_H: int
    source_W: int
    encoded_H: int
    encoded_W: int

    @property
    def time_compression_ratio(self) -> float:
        return self.encoded_T / max(1, self.source_T)

    @property
    def channel_expansion_ratio(self) -> float:
        return self.encoded_C / max(1, self.source_C)

    @property
    def spatial_compression_ratio(self) -> float:
        source = self.source_H * self.source_W
        encoded = self.encoded_H * self.encoded_W
        return encoded / max(1, source)

    @property
    def numel_compression_ratio(self) -> float:
        source = self.source_T * self.source_C * self.source_H * self.source_W
        encoded = self.encoded_T * self.encoded_C * self.encoded_H * self.encoded_W
        return encoded / max(1, source)

    @property
    def effective_snn_steps(self) -> int:
        return self.encoded_T

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload.update(
            {
                "time_compression_ratio": self.time_compression_ratio,
                "channel_expansion_ratio": self.channel_expansion_ratio,
                "spatial_compression_ratio": self.spatial_compression_ratio,
                "numel_compression_ratio": self.numel_compression_ratio,
                "effective_snn_steps": self.effective_snn_steps,
            }
        )
        return payload


@dataclass(frozen=True)
class EncoderConfig:
    encoder: str
    source_time_steps: int
    time_steps: int
    split_by: str = "number"
    in_channels: int = 2

    def __post_init__(self) -> None:
        if self.encoder not in ENCODERS:
            raise ValueError(f"Unknown encoder: {self.encoder}")
        if self.split_by not in {"number", "time"}:
            raise ValueError(f"Unknown split_by: {self.split_by}")
        if self.source_time_steps <= 0 or self.time_steps <= 0:
            raise ValueError("source_time_steps and time_steps must be positive")
        if self.in_channels <= 0:
            raise ValueError("in_channels must be positive")

        if self.encoder in {"count_number", "occupancy_number"}:
            if self.source_time_steps != self.time_steps:
                raise ValueError(f"{self.encoder} requires source_time_steps == time_steps")
            if self.in_channels != 2:
                raise ValueError(f"{self.encoder} requires in_channels=2")
        elif self.encoder == "temporal_pack_2":
            if self.source_time_steps != self.time_steps * 2:
                raise ValueError("temporal_pack_2 requires source_time_steps == 2 * time_steps")
            if self.in_channels != 4:
                raise ValueError("temporal_pack_2 requires in_channels=4")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def add_encoder_args(parser: Any) -> None:
    parser.add_argument("--encoder", default="count_number", choices=ENCODERS)
    parser.add_argument("--source-time-steps", default=None, type=int)
    parser.add_argument("--time-steps", default=None, type=int)
    parser.add_argument("--split-by", default="number", choices=("number", "time"))
    parser.add_argument("--in-channels", default=None, type=int)
    parser.add_argument("--profile-input-activity", action="store_true")
    parser.add_argument("--profile-activity", action="store_true")
    parser.add_argument("--input-profile-batches", default=16, type=int)
    parser.add_argument("--experiment-tag", default="")


def resolve_encoder_config(args: Any) -> EncoderConfig:
    time_steps = args.time_steps if args.time_steps is not None else getattr(args, "T", 16)
    if args.source_time_steps is None:
        source_time_steps = time_steps * 2 if args.encoder == "temporal_pack_2" else time_steps
    else:
        source_time_steps = args.source_time_steps

    if args.in_channels is None:
        in_channels = 4 if args.encoder == "temporal_pack_2" else 2
    else:
        in_channels = args.in_channels

    return EncoderConfig(
        encoder=args.encoder,
        source_time_steps=int(source_time_steps),
        time_steps=int(time_steps),
        split_by=args.split_by,
        in_channels=int(in_channels),
    )


class EncodedFrameDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset: torch.utils.data.Dataset, config: EncoderConfig) -> None:
        self.base_dataset = base_dataset
        self.config = config
        self._meta: EncoderMeta | None = None

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, Any]:
        x, y = self.base_dataset[index]
        encoded, meta = encode_frame_tensor(torch.as_tensor(x), self.config.encoder)
        if self._meta is None:
            self._meta = meta
        return encoded, y

    @property
    def encoder_meta(self) -> EncoderMeta | None:
        return self._meta


def encode_frame_tensor(x: torch.Tensor, encoder: str) -> tuple[torch.Tensor, EncoderMeta]:
    if x.dim() != 4:
        raise ValueError(f"Expected [T,C,H,W] tensor, got shape {list(x.shape)}")
    if encoder == "count_number":
        return encode_count_number(x)
    if encoder == "occupancy_number":
        return encode_occupancy_number(x)
    if encoder == "temporal_pack_2":
        return encode_temporal_pack_2(x)
    raise ValueError(f"Unknown encoder: {encoder}")


def encode_count_number(x: torch.Tensor) -> tuple[torch.Tensor, EncoderMeta]:
    encoded = x.float()
    return encoded, make_meta("count_number", x, encoded)


def encode_occupancy_number(x: torch.Tensor) -> tuple[torch.Tensor, EncoderMeta]:
    encoded = (x > 0).to(dtype=torch.float32)
    return encoded, make_meta("occupancy_number", x, encoded)


def encode_temporal_pack_2(x: torch.Tensor) -> tuple[torch.Tensor, EncoderMeta]:
    t, c, h, w = x.shape
    if t % 2 != 0:
        raise ValueError("temporal_pack_2 requires an even source T")
    if c != 2:
        raise ValueError("temporal_pack_2 assumes two polarity channels")
    x_pair = x.float().reshape(t // 2, 2, c, h, w)
    encoded = x_pair.reshape(t // 2, 2 * c, h, w)
    return encoded, make_meta("temporal_pack_2", x, encoded)


def make_meta(encoder_name: str, source: torch.Tensor, encoded: torch.Tensor) -> EncoderMeta:
    source_t, source_c, source_h, source_w = [int(v) for v in source.shape]
    encoded_t, encoded_c, encoded_h, encoded_w = [int(v) for v in encoded.shape]
    return EncoderMeta(
        encoder_name=encoder_name,
        source_T=source_t,
        encoded_T=encoded_t,
        source_C=source_c,
        encoded_C=encoded_c,
        source_H=source_h,
        source_W=source_w,
        encoded_H=encoded_h,
        encoded_W=encoded_w,
    )


class InputActivityCollector:
    def __init__(self, patch_sizes: tuple[int, ...] = (8, 16)) -> None:
        self.patch_sizes = patch_sizes
        self.records: list[dict[str, Any]] = []

    @torch.no_grad()
    def update(self, x: torch.Tensor, batch_idx: int) -> None:
        self.records.append(compute_input_activity_stats(x.detach(), batch_idx, self.patch_sizes))

    def summarize(self) -> dict[str, Any]:
        return aggregate_input_activity_records(self.records)


def compute_input_activity_stats(
    x: torch.Tensor,
    batch_idx: int,
    patch_sizes: tuple[int, ...] = (8, 16),
) -> dict[str, Any]:
    if x.dim() != 5:
        raise ValueError(f"Expected [B,T,C,H,W] tensor, got shape {list(x.shape)}")

    x = x.detach().float().cpu()
    nonzero = x != 0
    numel = int(x.numel())
    input_nonzero = int(nonzero.sum().item())
    density_per_timestep = density_by_dim(nonzero, keep_dim=1)
    sum_per_timestep = sum_by_dim(x, keep_dim=1)
    mean_per_timestep = mean_by_dim(x, keep_dim=1)
    max_per_timestep = max_by_dim(x, keep_dim=1)
    density_per_channel = density_by_dim(nonzero, keep_dim=2)
    sum_per_channel = sum_by_dim(x, keep_dim=2)
    mean_per_channel = mean_by_dim(x, keep_dim=2)
    max_per_channel = max_by_dim(x, keep_dim=2)
    density_tensor = torch.tensor(density_per_timestep, dtype=torch.float32)

    record: dict[str, Any] = {
        "batch_idx": int(batch_idx),
        "shape": list(x.shape),
        "input_numel": numel,
        "input_nonzero": input_nonzero,
        "input_density": input_nonzero / max(1, numel),
        "input_sum": float(x.sum().item()),
        "input_mean": float(x.mean().item()) if numel else 0.0,
        "input_min": float(x.min().item()) if numel else 0.0,
        "input_max": float(x.max().item()) if numel else 0.0,
        "input_is_binary": bool(((x == 0) | (x == 1)).all().item()) if numel else False,
        "input_unique_values_sample": unique_values_sample(x),
        "density_per_timestep": density_per_timestep,
        "sum_per_timestep": sum_per_timestep,
        "mean_per_timestep": mean_per_timestep,
        "max_per_timestep": max_per_timestep,
        "temporal_density_mean": float(density_tensor.mean().item()) if density_tensor.numel() else 0.0,
        "temporal_density_std": float(density_tensor.std(unbiased=False).item()) if density_tensor.numel() else 0.0,
        "temporal_density_min": float(density_tensor.min().item()) if density_tensor.numel() else 0.0,
        "temporal_density_max": float(density_tensor.max().item()) if density_tensor.numel() else 0.0,
        "temporal_burstiness": float(density_tensor.var(unbiased=False).item() / (density_tensor.mean().item() + 1e-8))
        if density_tensor.numel()
        else 0.0,
        "density_per_channel": density_per_channel,
        "sum_per_channel": sum_per_channel,
        "mean_per_channel": mean_per_channel,
        "max_per_channel": max_per_channel,
        "active_pixel_ratio": active_pixel_ratio(x),
        "active_pixels_mean": active_pixel_ratio(x),
    }
    for patch_size in patch_sizes:
        record[f"active_patch_ratio_{patch_size}x{patch_size}"] = active_patch_ratio(x, patch_size)
    return record


def aggregate_input_activity_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {}

    total_numel = sum(int(record["input_numel"]) for record in records)
    total_nonzero = sum(int(record["input_nonzero"]) for record in records)
    summary: dict[str, Any] = {
        "batches": len(records),
        "shape": records[0].get("shape", []),
        "input_numel": total_numel,
        "input_nonzero": total_nonzero,
        "input_density": total_nonzero / max(1, total_numel),
        "input_sum": sum(float(record["input_sum"]) for record in records),
        "input_mean": mean(record["input_mean"] for record in records),
        "input_min": min(float(record["input_min"]) for record in records),
        "input_max": max(float(record["input_max"]) for record in records),
        "input_is_binary": all(bool(record["input_is_binary"]) for record in records),
        "input_unique_values_sample": merged_unique_sample(records),
    }

    for key in [
        "density_per_timestep",
        "sum_per_timestep",
        "mean_per_timestep",
        "max_per_timestep",
        "density_per_channel",
        "sum_per_channel",
        "mean_per_channel",
        "max_per_channel",
    ]:
        summary[key] = average_lists(record[key] for record in records)

    for key in [
        "temporal_density_mean",
        "temporal_density_std",
        "temporal_density_min",
        "temporal_density_max",
        "temporal_burstiness",
        "active_pixel_ratio",
        "active_pixels_mean",
        "active_patch_ratio_8x8",
        "active_patch_ratio_16x16",
    ]:
        if key in records[0]:
            summary[key] = mean(record[key] for record in records)
    return summary


def save_input_activity_profile(output_dir: str | Path, collector: InputActivityCollector) -> dict[str, Any]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = collector.summarize()
    json_path = output_dir / "input_encoder_profile.json"
    csv_path = output_dir / "input_encoder_profile.csv"

    with json_path.open("w", encoding="utf-8") as f:
        json.dump({"summary": summary, "records": collector.records}, f, indent=2)

    scalar_keys = [
        "batch_idx",
        "input_numel",
        "input_nonzero",
        "input_density",
        "input_sum",
        "input_mean",
        "input_min",
        "input_max",
        "input_is_binary",
        "temporal_density_mean",
        "temporal_density_std",
        "temporal_burstiness",
        "active_pixel_ratio",
        "active_patch_ratio_8x8",
        "active_patch_ratio_16x16",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=scalar_keys)
        writer.writeheader()
        for record in collector.records:
            writer.writerow({key: record.get(key, "") for key in scalar_keys})

    return {"summary": summary, "json_path": str(json_path), "csv_path": str(csv_path)}


def save_encoder_meta(output_dir: str | Path, meta: EncoderMeta) -> str:
    path = Path(output_dir) / "encoder_meta.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(meta.to_dict(), f, indent=2)
    return str(path)


def build_run_id(args: Any, config: EncoderConfig) -> str:
    if getattr(args, "experiment_tag", ""):
        return sanitize_token(args.experiment_tag)
    return sanitize_token(f"{config.encoder}_srcT{config.source_time_steps}_T{config.time_steps}_C{config.in_channels}")


def sanitize_token(value: Any) -> str:
    return str(value).strip().replace("/", "-").replace(" ", "_")


def density_by_dim(nonzero: torch.Tensor, keep_dim: int) -> list[float]:
    dims = tuple(dim for dim in range(nonzero.dim()) if dim != keep_dim)
    return [float(value) for value in nonzero.float().mean(dim=dims).flatten()]


def sum_by_dim(x: torch.Tensor, keep_dim: int) -> list[float]:
    dims = tuple(dim for dim in range(x.dim()) if dim != keep_dim)
    return [float(value) for value in x.sum(dim=dims).flatten()]


def mean_by_dim(x: torch.Tensor, keep_dim: int) -> list[float]:
    dims = tuple(dim for dim in range(x.dim()) if dim != keep_dim)
    return [float(value) for value in x.mean(dim=dims).flatten()]


def max_by_dim(x: torch.Tensor, keep_dim: int) -> list[float]:
    dims = tuple(dim for dim in range(x.dim()) if dim != keep_dim)
    return [float(value) for value in x.amax(dim=dims).flatten()]


def active_pixel_ratio(x: torch.Tensor) -> float:
    active_pixels = (x != 0).any(dim=(0, 1, 2))
    return float(active_pixels.float().mean().item())


def active_patch_ratio(x: torch.Tensor, patch_size: int) -> float:
    active = (x != 0).any(dim=(1, 2))
    batch, height, width = active.shape
    if height % patch_size != 0 or width % patch_size != 0:
        return 0.0
    active = active.reshape(batch, height // patch_size, patch_size, width // patch_size, patch_size)
    active_patch = active.any(dim=(2, 4))
    return float(active_patch.float().mean().item())


def unique_values_sample(x: torch.Tensor, limit: int = 32) -> list[float]:
    values = torch.unique(x.flatten())
    values = values[:limit]
    return [float(value) for value in values]


def merged_unique_sample(records: list[dict[str, Any]], limit: int = 32) -> list[float]:
    values: set[float] = set()
    for record in records:
        values.update(float(value) for value in record.get("input_unique_values_sample", []))
        if len(values) >= limit:
            break
    return sorted(values)[:limit]


def average_lists(values: Iterable[list[float]]) -> list[float]:
    values = [list(value) for value in values]
    if not values:
        return []
    length = min(len(value) for value in values)
    if length == 0:
        return []
    return [mean(value[index] for value in values) for index in range(length)]


def mean(values: Iterable[Any]) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0
