import os
import math
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from Diffusion.Train import EMA  # reuse EMA helper
from Diffusion.ModelR import RNet, X0out
from Distill.ModelD import ModelD
from Distill.quant_io import (
    quantize_per_channel_symmetric_int8,
    save_sample_npz_to_zst,
)
from Distill.QuantDataset import QuantChunkDataset


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def default_config() -> Dict:
    return {
        # modes: "generate" | "train_d" | "eval_d"
        "mode": "generate",
        # teacher
        "teacher_ckpt": "./Checkpoints/diffusion/s1-3_cons5_ckpt_0050_ema.pt",
        # dataset (offline, int8+zstd)
        "dataset_dir": "./DistillData/",
        "samples_total": 10000,
        "gen_batch_size": 200,
        "chunks": 10,
        "percentile_clip": 99.5,
        # D training
        "epoch": 500,
        "batch_size": 128,
        "lr": 2e-4,
        "lambda_rin": 1.0,
        "lambda_chunks": 1.0,
        "lambda_x0": 0.1,
        "save_weight_dir_d": "./Checkpoints/distill/",
        "test_load_weight_d": "ckpt_d_0100_ema.pt",
        "use_compile": True,
        # model / data common
        "T": 1000,
        "channel": 128,
        "img_size": 32,
        "ema_decay": 0.999,
        "grad_clip": 1.0,
        "device": "cuda:0",
        "sampled_dir": "./SampledImgs/distill/",
        "nrow": 8,
    }


@torch.no_grad()
def generate_offline_samples(cfg: Dict) -> None:
    device = torch.device(cfg["device"])
    T = int(cfg["T"])
    C_chunk = int(cfg["chunks"])
    chunk_size = T // C_chunk
    percentile = float(cfg["percentile_clip"])
    batch_size = int(cfg["gen_batch_size"])
    total = int(cfg["samples_total"])
    img_size = int(cfg["img_size"])

    # dirs
    _ensure_dir(cfg["dataset_dir"])

    # teacher RNet (EMA)
    model = RNet(T=T, ch=cfg["channel"])
    ckpt_path = cfg["teacher_ckpt"]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(ckpt, strict=True)
    model = model.to(device).to(memory_format=torch.channels_last)
    model.eval()

    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    num_written = 0
    next_index = 0
    print(f"[generate] Starting: {total} samples, batch_size={batch_size}")
    while num_written < total:
        B = min(batch_size, total - num_written)
        # x_T
        x_T = torch.randn(size=[B, 3, img_size, img_size], device=device).to(memory_format=torch.channels_last)
        t_init = x_T.new_ones([B, ], dtype=torch.long) * (T - 1)
        with autocast(dtype=torch.float16):
            z = model.rin(x_T, t_init)  # [B,C,H,W]
            rin0 = z.detach().float()  # preserve as float32 for quant

            # chunk accumulation
            z_chunk_start = z
            r_chunks_list = []
            for time_step in reversed(range(T)):
                t = x_T.new_ones([B, ], dtype=torch.long) * time_step
                z = model.r.forward_one_step(z, t)  # one residual step
                if (time_step % chunk_size) == 0:
                    r_c = (z - z_chunk_start)
                    r_chunks_list.append(r_c)
                    z_chunk_start = z
            # z is z_T
            x0_teacher = model.x0out(z)  # [B,3,H,W], float16

        # stack chunks -> [Cchunk,B,C,H,W]
        r_chunks = torch.stack(r_chunks_list, dim=0).detach().float()

        # save per-sample
        for bi in range(B):
            # move small tensors to cpu
            x_T_f16 = x_T[bi].detach().cpu().to(torch.float16).numpy()
            x0_teacher_f16 = x0_teacher[bi].detach().cpu().to(torch.float16).numpy()
            rin0_np = rin0[bi].detach().cpu().numpy()  # [C,H,W] float32

            # quantize rin0 (per-channel int8)
            rin0_q, rin0_scale_f16 = quantize_per_channel_symmetric_int8(rin0_np, percentile=percentile)

            # r_chunks: [Cchunk,B,C,H,W] -> [Cchunk,C,H,W]
            r_list_np = []
            r_scale_list = []
            for ck in range(C_chunk):
                r_ck = r_chunks[ck, bi].cpu().numpy()
                r_ck_q, r_ck_scale = quantize_per_channel_symmetric_int8(r_ck, percentile=percentile)
                r_list_np.append(r_ck_q)
                r_scale_list.append(r_ck_scale)
            r_chunks_q = np.stack(r_list_np, axis=0).astype(np.int8, copy=False)  # [Cchunk,C,H,W]
            r_chunks_scale_f16 = np.stack(r_scale_list, axis=0).astype(np.float16, copy=False)  # [Cchunk,C]

            arrays = {
                "x_T_f16": x_T_f16.astype(np.float16, copy=False),
                "x0_teacher_f16": x0_teacher_f16.astype(np.float16, copy=False),
                "rin0_q": rin0_q.astype(np.int8, copy=False),
                "rin0_scale_f16": rin0_scale_f16.astype(np.float16, copy=False),
                "r_chunks_q": r_chunks_q,
                "r_chunks_scale_f16": r_chunks_scale_f16,
                "T": np.array([T], dtype=np.int16),
                "chunk_size": np.array([chunk_size], dtype=np.int16),
            }
            out_name = f"sample_{next_index:08d}.zst"
            out_path = os.path.join(cfg["dataset_dir"], out_name)
            save_sample_npz_to_zst(out_path, arrays, level=10)
            next_index += 1
            num_written += 1

        progress_pct = 100.0 * num_written / total
        print(f"[generate] {num_written}/{total} ({progress_pct:.1f}%)")


def split_outputs(y: torch.Tensor, latent_ch: int, num_chunks: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    y: [B, (1+num_chunks)*latent_ch, H, W]
    returns: (rin_pred [B,C,H,W], r_chunks_pred [B,num_chunks,C,H,W])
    """
    B, C_all, H, W = y.shape
    assert C_all == (1 + num_chunks) * latent_ch
    rin = y[:, :latent_ch]
    r_rest = y[:, latent_ch:]
    r_chunks = r_rest.view(B, num_chunks, latent_ch, H, W)
    return rin, r_chunks


def train_d(cfg: Dict) -> None:
    device = torch.device(cfg["device"])
    _ensure_dir(cfg["dataset_dir"])
    _ensure_dir(cfg["save_weight_dir_d"])
    torch.backends.cudnn.benchmark = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass

    # dataset
    dataset = QuantChunkDataset(dataset_dir=cfg["dataset_dir"], num_chunks=cfg["chunks"], dtype=torch.float16)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    # model D
    model_d = ModelD(latent_ch=cfg["channel"], num_chunks=cfg["chunks"])
    model_d = model_d.to(device).to(memory_format=torch.channels_last)
    if cfg.get("use_compile", True):
        try:
            model_d = torch.compile(model_d, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass

    # teacher x0out head (frozen)
    teacher = RNet(T=cfg["T"], ch=cfg["channel"])
    ckpt_path = cfg["teacher_ckpt"]
    ckpt = torch.load(ckpt_path, map_location='cpu')
    teacher.load_state_dict(ckpt, strict=True)
    teacher = teacher.to(device).to(memory_format=torch.channels_last)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad = False

    optimizer = torch.optim.AdamW(model_d.parameters(), lr=cfg["lr"], weight_decay=1e-4, fused=True)
    total_steps = int(cfg["epoch"]) * math.ceil(len(dataset) / cfg["batch_size"])
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer=optimizer,
        max_lr=cfg["lr"],
        total_steps=total_steps,
        pct_start=0.05,
        anneal_strategy='cos',
        div_factor=10000.0,
        final_div_factor=10000.0
    )
    ema = EMA(model_d, decay=cfg["ema_decay"])
    scaler = GradScaler()
    amp_dtype = torch.float16

    num_epochs = int(cfg["epoch"])
    step = 0
    for e in range(num_epochs):
        epoch_l_rin = 0.0
        epoch_l_chunks = 0.0
        epoch_l_x0 = 0.0
        nb = 0
        with tqdm(loader, dynamic_ncols=True) as tqdm_loader:
            for batch in tqdm_loader:
                optimizer.zero_grad(set_to_none=True)
                x_T = batch["x_T"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
                x0_teacher = batch["x0_teacher"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
                rin0_tgt = batch["rin0"].to(device, non_blocking=True).to(memory_format=torch.channels_last)
                # r_chunks is 5D [B,num_chunks,C,H,W]; channels_last is only for 4D tensors
                r_chunks_tgt = batch["r_chunks"].to(device, non_blocking=True)

                with autocast(dtype=amp_dtype):
                    y = model_d(x_T)
                    rin_pred, r_chunks_pred = split_outputs(y, latent_ch=cfg["channel"], num_chunks=cfg["chunks"])
                    # losses
                    l_rin = F.mse_loss(rin_pred, rin0_tgt)
                    l_chunks = F.mse_loss(r_chunks_pred, r_chunks_tgt)
                    z_hat_T = rin_pred + r_chunks_pred.sum(dim=1)
                    x0_pred = teacher.x0out(z_hat_T)
                    l_x0 = F.mse_loss(x0_pred, x0_teacher)
                    loss = cfg["lambda_rin"] * l_rin + cfg["lambda_chunks"] * l_chunks + cfg["lambda_x0"] * l_x0

                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model_d.parameters(), cfg["grad_clip"])
                scaler.step(optimizer)
                scaler.update()
                ema.update()
                scheduler.step()

                epoch_l_rin += l_rin.item()
                epoch_l_chunks += l_chunks.item()
                epoch_l_x0 += l_x0.item()
                nb += 1
                step += 1

                # update tqdm postfix
                postfix = {
                    "epoch": f"{e+1}/{num_epochs}",
                    "L_rin": f"{1e3*epoch_l_rin/nb:.4f}",
                    "L_chunks": f"{1e3*epoch_l_chunks/nb:.4f}",
                    "L_x0": f"{1e3*epoch_l_x0/nb:.4f}",
                    "LR": optimizer.state_dict()['param_groups'][0]["lr"],
                }
                tqdm_loader.set_postfix(ordered_dict=postfix)
        
        # save every 100 epochs and end
        should_save = ((e + 1) % 100 == 0) or ((e + 1) == num_epochs)
        if should_save:
            model_to_save = getattr(model_d, '_orig_mod', model_d)
            torch.save(model_to_save.state_dict(), os.path.join(cfg["save_weight_dir_d"], f"ckpt_d_{e+1:04d}.pt"))
            ema.apply_shadow()
            torch.save(model_to_save.state_dict(), os.path.join(cfg["save_weight_dir_d"], f"ckpt_d_{e+1:04d}_ema.pt"))
            ema.restore()


@torch.no_grad()
def eval_d(cfg: Dict) -> None:
    device = torch.device(cfg["device"])
    _ensure_dir(cfg["save_weight_dir_d"])
    _ensure_dir(cfg["sampled_dir"])
    # load D
    model_d = ModelD(latent_ch=cfg["channel"], num_chunks=cfg["chunks"])
    ckpt_path = cfg["test_load_weight_d"]
    if not os.path.isabs(ckpt_path):
        ckpt_path = os.path.join(cfg["save_weight_dir_d"], ckpt_path)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    model_d.load_state_dict(ckpt, strict=True)
    model_d = model_d.to(device).to(memory_format=torch.channels_last)
    model_d.eval()
    # teacher x0out
    teacher = RNet(T=cfg["T"], ch=cfg["channel"])
    ckpt_t = cfg["teacher_ckpt"]
    st = torch.load(ckpt_t, map_location='cpu')
    teacher.load_state_dict(st, strict=True)
    teacher = teacher.to(device).to(memory_format=torch.channels_last)
    teacher.eval()
    # sample
    B = cfg["batch_size"]
    H = W = cfg["img_size"]
    x_T = torch.randn([B, 3, H, W], device=device).to(memory_format=torch.channels_last)
    with autocast(dtype=torch.float16):
        y = model_d(x_T)
        rin_pred, r_chunks_pred = split_outputs(y, latent_ch=cfg["channel"], num_chunks=cfg["chunks"])
        z_hat_T = rin_pred + r_chunks_pred.sum(dim=1)
        x0_pred = teacher.x0out(z_hat_T)
    # save a grid
    imgs = (x0_pred.float().clamp(-1, 1) * 0.5 + 0.5).cpu()
    save_image(imgs, os.path.join(cfg["sampled_dir"], "sampled_d.png"), nrow=cfg["nrow"])
    print("[eval_d] saved sampled_d.png")


def main(config: Dict | None = None) -> None:
    cfg = default_config()
    if config is not None:
        cfg.update(config)
    os.makedirs(cfg["save_weight_dir_d"], exist_ok=True)
    os.makedirs(cfg["dataset_dir"], exist_ok=True)
    os.makedirs(cfg["sampled_dir"], exist_ok=True)

    mode = cfg["mode"]
    if mode == "generate":
        generate_offline_samples(cfg)
    elif mode == "train_d":
        train_d(cfg)
    elif mode == "eval_d":
        eval_d(cfg)
    else:
        raise ValueError(f"Unknown mode: {mode}")


if __name__ == "__main__":
    main()


