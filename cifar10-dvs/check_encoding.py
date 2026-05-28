from __future__ import annotations

import torch

import encoding


def test_shapes() -> None:
    x = torch.zeros(16, 2, 128, 128)
    y, meta = encoding.encode_frame_tensor(x, "count_number")
    assert y.shape == (16, 2, 128, 128)
    assert meta.encoded_T == 16
    assert meta.encoded_C == 2

    y, meta = encoding.encode_frame_tensor(x, "occupancy_number")
    assert y.shape == (16, 2, 128, 128)
    assert meta.encoded_T == 16
    assert meta.encoded_C == 2

    y, meta = encoding.encode_frame_tensor(x, "temporal_pack_2")
    assert y.shape == (8, 4, 128, 128)
    assert meta.encoded_T == 8
    assert meta.encoded_C == 4


def test_occupancy_binary() -> None:
    x = torch.tensor([0, 1, 2, 5]).view(1, 1, 2, 2)
    y, _ = encoding.encode_frame_tensor(x, "occupancy_number")
    assert set(y.flatten().tolist()) <= {0.0, 1.0}


def test_temporal_pack_order() -> None:
    x = torch.zeros(16, 2, 1, 1)
    x[0, 0, 0, 0] = 10
    x[0, 1, 0, 0] = 20
    x[1, 0, 0, 0] = 30
    x[1, 1, 0, 0] = 40

    z, meta = encoding.encode_frame_tensor(x, "temporal_pack_2")
    assert z[0, :, 0, 0].tolist() == [10, 20, 30, 40]
    assert meta.time_compression_ratio == 0.5
    assert meta.channel_expansion_ratio == 2.0
    assert meta.numel_compression_ratio == 1.0


def test_config_resolution() -> None:
    class Args:
        encoder = "temporal_pack_2"
        source_time_steps = 16
        time_steps = 8
        split_by = "number"
        in_channels = 4
        T = 16

    config = encoding.resolve_encoder_config(Args())
    assert config.source_time_steps == 16
    assert config.time_steps == 8
    assert config.in_channels == 4


def main() -> None:
    test_shapes()
    test_occupancy_binary()
    test_temporal_pack_order()
    test_config_resolution()
    print("Encoding sanity checks passed.")


if __name__ == "__main__":
    main()
