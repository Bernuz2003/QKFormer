from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from spikingjelly import configure
from spikingjelly.datasets import cifar10_dvs


def count_npz(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(1 for _ in path.rglob("*.npz"))


def maybe_remove_incomplete(path: Path, expected_files: int, clean: bool, label: str) -> None:
    if not path.exists():
        return

    files = count_npz(path)
    if files == expected_files:
        print(f"{label}: found {files} .npz files, keeping [{path}].")
        return

    message = f"{label}: found {files} .npz files, expected {expected_files}, path=[{path}]"
    if not clean:
        raise RuntimeError(f"{message}. Re-run with --clean-incomplete to rebuild generated files.")

    print(f"{message}. Removing incomplete generated directory.")
    shutil.rmtree(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare and validate CIFAR10-DVS frames for QKFormer.")
    parser.add_argument("--data-path", required=True, help="CIFAR10-DVS root directory")
    parser.add_argument("--T", default=16, type=int, help="number of frames")
    parser.add_argument("--split-by", default="number", choices=("number", "time"))
    parser.add_argument("--threads", default=4, type=int, help="SpikingJelly preprocessing threads")
    parser.add_argument("--expected-files", default=10000, type=int)
    parser.add_argument("--clean-incomplete", action="store_true", help="remove partial generated dirs before rebuild")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(args.data_path).expanduser().resolve()
    events_dir = root / "events_np"
    frames_dir = root / f"frames_number_{args.T}_split_by_{args.split_by}"

    configure.max_threads_number_for_datasets_preprocess = args.threads
    maybe_remove_incomplete(events_dir, args.expected_files, args.clean_incomplete, "events_np")
    maybe_remove_incomplete(frames_dir, args.expected_files, args.clean_incomplete, "frames")

    dataset = cifar10_dvs.CIFAR10DVS(
        root=str(root),
        data_type="frame",
        frames_number=args.T,
        split_by=args.split_by,
    )

    events_count = count_npz(events_dir)
    frames_count = count_npz(frames_dir)
    sample, label = dataset[0]

    print(f"events_np files: {events_count}")
    print(f"frame files: {frames_count}")
    print(f"dataset length: {len(dataset)}")
    print(f"sample shape: {sample.shape}")
    print(f"sample label: {label}")

    if events_count != args.expected_files or frames_count != args.expected_files or len(dataset) != args.expected_files:
        raise RuntimeError("CIFAR10-DVS preprocessing did not complete correctly.")


if __name__ == "__main__":
    main()
