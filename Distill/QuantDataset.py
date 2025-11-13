import os
from typing import List, Dict, Any, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from Distill.quant_io import (
    load_sample_npz_from_zst,
    dequantize_per_channel_symmetric_int8,
)


class QuantChunkDataset(Dataset):
    """
    Dataset reading per-sample .zst (zstd-compressed .npz) files.
    Each sample file contains:
      - x_T_f16: float16 [3,H,W]
      - rin0_q: int8 [C,H,W]
      - rin0_scale_f16: float16 [C]
      - r_chunks_q: int8 [num_chunks,C,H,W]
      - r_chunks_scale_f16: float16 [num_chunks,C]
      - x0_teacher_f16: float16 [3,H,W]
      - meta_* (optional)
    """
    def __init__(self, dataset_dir: str, num_chunks: int = 10, dtype: torch.dtype = torch.float16, device: torch.device | None = None):
        super().__init__()
        self.dataset_dir = os.path.abspath(dataset_dir)
        self.num_chunks = int(num_chunks)
        self.dtype = dtype
        # Always keep dataset tensors on CPU; device transfer is handled in the training loop
        self.device = torch.device('cpu')
        self.files: List[str] = [
            os.path.join(self.dataset_dir, f)
            for f in os.listdir(self.dataset_dir)
            if f.endswith(".zst")
        ]
        self.files.sort()
        if len(self.files) == 0:
            raise RuntimeError(f"No .zst files found in {self.dataset_dir}")

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        path = self.files[idx]
        data = load_sample_npz_from_zst(path)

        x_T = data["x_T_f16"].astype(np.float16)
        x0_teacher = data["x0_teacher_f16"].astype(np.float16)

        rin0_q = data["rin0_q"].astype(np.int8, copy=False)
        rin0_scale = data["rin0_scale_f16"].astype(np.float16, copy=False)
        r_chunks_q = data["r_chunks_q"].astype(np.int8, copy=False)
        r_chunks_scale = data["r_chunks_scale_f16"].astype(np.float16, copy=False)

        # dequantize to float16 by default (can be cast to float32 if needed)
        rin0 = dequantize_per_channel_symmetric_int8(rin0_q, rin0_scale, dtype=np.float16)
        # r_chunks: loop over chunks to dequantize channel-wise
        num_chunks = r_chunks_q.shape[0]
        r_list = []
        for c in range(num_chunks):
            r_list.append(dequantize_per_channel_symmetric_int8(r_chunks_q[c], r_chunks_scale[c], dtype=np.float16))
        r_chunks = np.stack(r_list, axis=0)

        # torch tensors (CPU). Move to CUDA in training loop.
        x_T_t = torch.from_numpy(x_T).to(dtype=self.dtype)
        x0_t = torch.from_numpy(x0_teacher).to(dtype=self.dtype)
        rin0_t = torch.from_numpy(rin0).to(dtype=self.dtype)
        r_chunks_t = torch.from_numpy(r_chunks).to(dtype=self.dtype)

        return {
            "x_T": x_T_t,  # [3,H,W]
            "x0_teacher": x0_t,  # [3,H,W]
            "rin0": rin0_t,  # [C,H,W]
            "r_chunks": r_chunks_t,  # [num_chunks,C,H,W]
        }


