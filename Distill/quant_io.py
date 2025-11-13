import io
from typing import Tuple, Dict, Any

import numpy as np
import torch

try:
    import zstandard as zstd
except Exception:
    zstd = None


def _compute_per_channel_scale_symmetric(x: np.ndarray, percentile: float = 99.5, eps: float = 1e-8) -> np.ndarray:
    """
    x: shape [C, H, W] (float32/float16)
    returns: scale per channel [C] (float32)
    """
    assert x.ndim == 3, f"expect [C,H,W], got {x.shape}"
    C = x.shape[0]
    x_reshaped = x.reshape(C, -1)
    # percentile clip per-channel for robustness
    p = np.clip(percentile, 50.0, 100.0)
    pos = np.percentile(np.abs(x_reshaped), p, axis=1)
    scale = pos / 127.0
    scale = np.maximum(scale, eps).astype(np.float32)
    return scale


def quantize_per_channel_symmetric_int8(x: np.ndarray, percentile: float = 99.5) -> Tuple[np.ndarray, np.ndarray]:
    """
    Quantize per-channel symmetric INT8.
    x: [C,H,W], float32/float16
    returns: (q:int8[C,H,W], scale:float16[C])
    """
    if x.dtype != np.float32:
        x = x.astype(np.float32, copy=False)
    scale = _compute_per_channel_scale_symmetric(x, percentile=percentile)  # float32
    C = x.shape[0]
    x_flat = x.reshape(C, -1)
    q_flat = np.round(x_flat / scale[:, None])
    q_flat = np.clip(q_flat, -127, 127).astype(np.int8, copy=False)
    q = q_flat.reshape(x.shape)
    return q, scale.astype(np.float16)


def dequantize_per_channel_symmetric_int8(q: np.ndarray, scale_f16: np.ndarray, dtype=np.float32) -> np.ndarray:
    """
    q: int8 [C,H,W]
    scale_f16: float16 [C]
    return: float32/float16 [C,H,W]
    """
    assert q.ndim == 3
    C = q.shape[0]
    scale = scale_f16.astype(np.float32, copy=False)
    out = (q.reshape(C, -1).astype(np.float32) * scale[:, None]).reshape(q.shape)
    if dtype == np.float16:
        out = out.astype(np.float16)
    return out


def save_sample_npz_to_zst(path: str, arrays: Dict[str, np.ndarray], level: int = 10) -> None:
    """
    Save a dict of numpy arrays into a single zstd-compressed .npz container.
    """
    if zstd is None:
        raise ImportError("zstandard is required for .zst compression. Please install 'zstandard'.")
    # serialize to npz bytes (may already be zipped internally; acceptable for simplicity)
    buf = io.BytesIO()
    np.savez(buf, **arrays)
    raw = buf.getvalue()
    cctx = zstd.ZstdCompressor(level=level)
    with open(path, "wb") as f:
        with cctx.stream_writer(f) as compressor:
            compressor.write(raw)


def load_sample_npz_from_zst(path: str) -> Dict[str, np.ndarray]:
    """
    Load a zstd-compressed .npz container and return dict of numpy arrays.
    """
    if zstd is None:
        raise ImportError("zstandard is required for .zst decompression. Please install 'zstandard'.")
    dctx = zstd.ZstdDecompressor()
    with open(path, "rb") as f:
        with dctx.stream_reader(f) as reader:
            raw = reader.read()
    buf = io.BytesIO(raw)
    data = np.load(buf)
    out: Dict[str, np.ndarray] = {}
    for k in data.files:
        out[k] = data[k]
    return out


def torch_from_numpy_channels_last(x: np.ndarray, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    t = torch.from_numpy(x).to(device=device, dtype=dtype)
    return t.to(memory_format=torch.channels_last)


