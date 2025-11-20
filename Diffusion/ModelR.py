import math
import os
import torch
from torch import nn
from torch.nn import init
from diffusers import UNet2DModel
from torch.nn.utils.parametrizations import spectral_norm


### Helpers

def init_conv_relu_(conv: nn.Conv2d) -> None:
    if tuple(conv.kernel_size) == (1, 1):
        init.xavier_uniform_(conv.weight)
    else:
        init.kaiming_normal_(conv.weight, mode='fan_in', nonlinearity='relu')
    if conv.bias is not None:
        init.zeros_(conv.bias)


def _set_requires_grad(mod: nn.Module, flag: bool) -> None:
    for p in mod.parameters():
        p.requires_grad = flag


def _freeze_all_params(mod: nn.Module) -> None:
    _set_requires_grad(mod, False)


### Diffusers UNet helpers (DDPM)

def _load_diffusers_unet(
    device: torch.device | None = None,
    remove_conv_in: bool = False,
    remove_conv_out: bool = False,
) -> UNet2DModel:
    """
    Load DDPM-CIFAR10 UNet helper.
    - Priority: Local EMA model at `Diffusion/ddpm_ema_cifar10/unet`
    - Fallback: Hugging Face Hub model `google/ddpm-cifar10-32`
    """
    # Search for EMA model using relative path from this file
    ema_root = os.path.join(os.path.dirname(__file__), "ddpm_ema_cifar10")
    ema_unet_dir = os.path.join(ema_root, "unet")

    if os.path.isdir(ema_unet_dir):
        print(f"Loading local EMA UNet from: {ema_unet_dir}")
        unet = UNet2DModel.from_pretrained(
            pretrained_model_name_or_path=ema_root,
            subfolder="unet",
            local_files_only=True,
        )
    else:
        print(f"Local EMA UNet not found at {ema_unet_dir}, fallback to: google/ddpm-cifar10-32")
        unet = UNet2DModel.from_pretrained('google/ddpm-cifar10-32')

    if device is not None:
        unet.to(device)
    if remove_conv_in:
        if hasattr(unet, 'conv_in'):
            unet.conv_in = nn.Identity()
    if remove_conv_out:
        if hasattr(unet, 'conv_out'):
            unet.conv_out = nn.Identity()
    return unet


def _set_trainable_first_layers_unet(unet: UNet2DModel, flag: bool = True) -> None:
    if hasattr(unet, 'conv_in'):
        _set_requires_grad(unet.conv_in, flag)


def _set_trainable_last_layers_unet(unet: UNet2DModel, flag: bool = True) -> None:
    if hasattr(unet, 'conv_out'):
        _set_requires_grad(unet.conv_out, flag)


def _set_trainable_time_embedding(unet: UNet2DModel, flag: bool = True) -> None:
    if hasattr(unet, 'time_embedding'):
        _set_requires_grad(unet.time_embedding, flag)


# LoRA configuration (for attention projections q/k/v and proj_out)
ENABLE_LORA: bool = True
LORA_RANK: int = 8
LORA_ALPHA: int = 16


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        _set_requires_grad(self.base, False)
        in_f = base.in_features
        out_f = base.out_features
        self.lora_down = nn.Linear(in_f, r, bias=False)
        self.lora_up = nn.Linear(r, out_f, bias=False)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_up.weight)
        self.scaling = float(alpha) / float(max(r, 1))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        scale = kwargs.pop("scale", 1.0)
        y = self.base(x)
        y = y + self.lora_up(self.lora_down(x)) * self.scaling
        if scale != 1.0:
            y = y * scale
        return y


class LoRAConv2d(nn.Module):
    """
    LoRA for Conv2d. Supports 1x1 kernels (common in attention blocks). For other kernels we keep base only.
    """
    def __init__(self, base: nn.Conv2d, r: int, alpha: int):
        super().__init__()
        self.base = base
        _set_requires_grad(self.base, False)
        ks = base.kernel_size
        if isinstance(ks, tuple):
            kh, kw = ks
        else:
            kh = kw = ks
        if (kh, kw) != (1, 1) or base.groups != 1:
            self.disabled = True
        else:
            self.disabled = False
            in_ch = base.in_channels
            out_ch = base.out_channels
            self.lora_down = nn.Conv2d(in_ch, r, kernel_size=1, stride=1, padding=0, bias=False)
            self.lora_up = nn.Conv2d(r, out_ch, kernel_size=1, stride=1, padding=0, bias=False)
            nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
            nn.init.zeros_(self.lora_up.weight)
            self.scaling = float(alpha) / float(max(r, 1))

    def forward(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        scale = kwargs.pop("scale", 1.0)
        y = self.base(x)
        if getattr(self, "disabled", False):
            if scale != 1.0:
                y = y * scale
            return y
        y = y + self.lora_up(self.lora_down(x)) * self.scaling
        if scale != 1.0:
            y = y * scale
        return y


def _set_module_by_path(root: nn.Module, path: str, new_module: nn.Module) -> bool:
    """
    Replace a leaf submodule at dot-separated path under root. Returns True on success.
    """
    parts = path.split(".")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            return False
        parent = getattr(parent, p)
    leaf = parts[-1]
    if not hasattr(parent, leaf):
        return False
    setattr(parent, leaf, new_module)
    return True


def _enable_unet_lora(unet: UNet2DModel, rank: int = 8, alpha: int = 16) -> None:
    """
    Inject LoRA into attention projections (Q/K/V and output) for legacy Attention blocks.
    Targets:
      - leaf names: to_q, to_k, to_v, q, k, v
      - exact names: ...to_out.0, ...proj_out
    """
    target_leaf = {"to_q", "to_k", "to_v", "q", "k", "v"}
    out_exact = "to_out.0"
    out_alt = "proj_out"
    replaced = 0
    for name, module in list(unet.named_modules()):
        # restrict to attention-related modules only
        if ("attentions" not in name) and ("attn" not in name) and ("attention" not in name):
            continue
        if name.endswith(out_exact) or name.endswith(out_alt):
            if isinstance(module, nn.Linear):
                wrapped = LoRALinear(module, r=rank, alpha=alpha)
                # align device/dtype to base
                wrapped.lora_down.to(device=module.weight.device, dtype=module.weight.dtype)
                wrapped.lora_up.to(device=module.weight.device, dtype=module.weight.dtype)
                if _set_module_by_path(unet, name, wrapped):
                    replaced += 1
            elif isinstance(module, nn.Conv2d):
                wrapped = LoRAConv2d(module, r=rank, alpha=alpha)
                if not getattr(wrapped, "disabled", False):
                    wrapped.lora_down.to(device=module.weight.device, dtype=module.weight.dtype)
                    wrapped.lora_up.to(device=module.weight.device, dtype=module.weight.dtype)
                if _set_module_by_path(unet, name, wrapped):
                    replaced += 1
            continue
        leaf = name.split(".")[-1]
        if leaf in target_leaf:
            if isinstance(module, nn.Linear):
                wrapped = LoRALinear(module, r=rank, alpha=alpha)
                wrapped.lora_down.to(device=module.weight.device, dtype=module.weight.dtype)
                wrapped.lora_up.to(device=module.weight.device, dtype=module.weight.dtype)
                if _set_module_by_path(unet, name, wrapped):
                    replaced += 1
            elif isinstance(module, nn.Conv2d):
                wrapped = LoRAConv2d(module, r=rank, alpha=alpha)
                if not getattr(wrapped, "disabled", False):
                    wrapped.lora_down.to(device=module.weight.device, dtype=module.weight.dtype)
                    wrapped.lora_up.to(device=module.weight.device, dtype=module.weight.dtype)
                if _set_module_by_path(unet, name, wrapped):
                    replaced += 1
    if replaced == 0:
        print("Warning: lora_conv could not find attention projections to patch")
    else:
        setattr(unet, "_lora_attn_enabled", True)
        print(f"LoRA (legacy attention) enabled: patched {replaced} projections (rank={rank}, alpha={alpha})")


### Models

class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
            nn.GroupNorm(32, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                init_conv_relu_(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)

        
class FiLMResBlock(nn.Module):
    def __init__(self, ch: int, T_embed: int):
        super().__init__()
        self.ch = ch
        self.T = int(T_embed)
        # Residual path layers
        self.gn1 = nn.GroupNorm(32, ch, affine=False)
        self.gn2 = nn.GroupNorm(32, ch, affine=False)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)
        init_conv_relu_(self.conv1)
        # Zero-init final conv to start as identity
        init.zeros_(self.conv2.weight)
        if self.conv2.bias is not None:
            init.zeros_(self.conv2.bias)
        # Time embedding: sinusoidal -> 2-layer MLP -> 4*ch (gamma1,beta1,gamma2,beta2)
        self.time_mlp = nn.Sequential(
            nn.Linear(ch, 4 * ch),
            nn.SiLU(),
            nn.Linear(4 * ch, 4 * ch),
        )
        # Initialize: first layer Xavier, second layer zeros (identity-near FiLM)
        init.xavier_uniform_(self.time_mlp[0].weight)
        init.zeros_(self.time_mlp[0].bias)
        init.zeros_(self.time_mlp[2].weight)
        init.zeros_(self.time_mlp[2].bias)

    def _sinusoidal_time_embed(self, t: torch.Tensor) -> torch.Tensor:
        # Produce [B, ch] sinusoidal embedding from integer t in [0, T)
        device = t.device
        half = self.ch // 2
        if half == 0:
            return torch.zeros(t.shape[0], self.ch, device=device)
        # Normalize t to [0,1]
        t_norm = t.float() / max(self.T - 1, 1)
        freqs = torch.exp(
            -math.log(10000.0) * torch.arange(half, device=device) / max(half - 1, 1)
        )
        args = t_norm.unsqueeze(1) * (2.0 * math.pi) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.ch:
            emb = torch.nn.functional.pad(emb, (0, self.ch - emb.shape[1]))
        return emb

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # Time FiLM parameters for both blocks
        emb_all = self.time_mlp(self._sinusoidal_time_embed(t))  # [B, 4*ch]
        gamma1, beta1, gamma2, beta2 = torch.chunk(emb_all, 4, dim=1)
        gamma1 = 0.1 * gamma1
        gamma2 = 0.1 * gamma2
        # Block 1: GN -> FiLM -> Swish -> Conv
        h = self.gn1(x)
        g1 = gamma1.view([gamma1.shape[0], self.ch] + [1] * (x.ndim - 2))
        b1 = beta1.view([beta1.shape[0], self.ch] + [1] * (x.ndim - 2))
        h = h * (1.0 + g1) + b1
        h = self.act(h)
        h = self.conv1(h)
        # Block 2: GN -> FiLM -> Swish -> Conv
        h = self.gn2(h)
        g2 = gamma2.view([gamma2.shape[0], self.ch] + [1] * (x.ndim - 2))
        b2 = beta2.view([beta2.shape[0], self.ch] + [1] * (x.ndim - 2))
        h = h * (1.0 + g2) + b2
        h = self.act(h)
        h = self.conv2(h)
        return x + h


class Adapter(nn.Module):
    def __init__(self, ch_in: int, ch_mid: int, ch_out: int, T_embed: int, num_blocks: int = 2):
        super().__init__()
        #self.conv_in = nn.Conv2d(ch_in, ch_mid, 3, 1, 1)
        self.conv_in = spectral_norm(nn.Conv2d(ch_in, ch_mid, 3, 1, 1), n_power_iterations=1)
        self.film_blocks = nn.ModuleList([
            FiLMResBlock(ch=ch_mid, T_embed=T_embed) for _ in range(num_blocks)
        ])
        self.tail = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(0.0),
            #nn.Conv2d(ch_mid, ch_out, 3, 1, 1),
            spectral_norm(nn.Conv2d(ch_mid, ch_out, 3, 1, 1), n_power_iterations=1),
        )
        init_conv_relu_(self.conv_in)
        for m in self.tail:
            if isinstance(m, nn.Conv2d):
                init_conv_relu_(m)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for film_block in self.film_blocks:
            h = film_block(h, t)
        return self.tail(h)


class Rin(nn.Module):
    """
    Diffusers UNet backbone per Rin. Input 3x32x32 -> UNet2DModel (no conv_out) -> feat -> proj(feat->ch).
    """
    def __init__(self, ch: int, T: int, device: torch.device | None = None):
        super().__init__()
        self.T = int(T)
        self.device = device or torch.device('cpu')
        # feature channels (fixed to 128 for google/ddpm-cifar10-32)
        feat_ch = 128

        # diffusers UNet (DDPM-CIFAR10) without final conv_out
        self.unet = _load_diffusers_unet(device=self.device, remove_conv_in=False, remove_conv_out=True)
        self.unet.eval()
        _freeze_all_params(self.unet)
        _set_trainable_time_embedding(self.unet, flag=True)  # adapt to different prediction-modes
        _set_trainable_first_layers_unet(self.unet, flag=True)
        # enable attention LoRA for legacy attention blocks (q/k/v/proj_out)
        if ENABLE_LORA:
            _enable_unet_lora(self.unet, rank=LORA_RANK, alpha=LORA_ALPHA)
        # output adapter
        self.out_adapter = Adapter(ch_in=feat_ch, ch_mid=feat_ch, ch_out=ch, T_embed=self.T, num_blocks=4)
        # GroupNorm to avoid drift (affine=False to make groupnorm idempotent)
        self.gn_out = nn.GroupNorm(32, ch, affine=False)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feat = self.unet(sample=x, timestep=t.long())
        out =self.out_adapter(feat.sample, t)
        return self.gn_out(out)


class Rout(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        feat_ch = 128
        self.conv_in = nn.Conv2d(ch, feat_ch, 3, 1, 1)
        init_conv_relu_(self.conv_in)
        self.blocks = nn.ModuleList([ResBlock(ch=feat_ch) for _ in range(4)])
        self.in_proj = nn.Conv2d(ch, feat_ch, 1, 1, 0, bias=False)
        self.tail = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(feat_ch, 3, 3, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.tail[-1]:
                    init.xavier_uniform_(m.weight, gain=1e-5)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif m is self.in_proj:
                    # start as zero to avoid changing behavior at init
                    init.zeros_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                else:
                    init_conv_relu_(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h)
        h = h + self.in_proj(x)
        return self.tail(h)


class X0out(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        feat_ch = 128
        self.conv_in = nn.Conv2d(ch, feat_ch, 3, 1, 1)
        init_conv_relu_(self.conv_in)
        self.blocks = nn.ModuleList([ResBlock(ch=feat_ch) for _ in range(4)])
        self.in_proj = nn.Conv2d(ch, feat_ch, 1, 1, 0, bias=False)
        self.tail = nn.Sequential(
            nn.SiLU(),
            nn.Conv2d(feat_ch, 3, 3, 1, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is self.tail[-1]:
                    init.xavier_uniform_(m.weight, gain=1e-5)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                elif m is self.in_proj:
                    # start as zero to avoid changing behavior at init
                    init.zeros_(m.weight)
                    if m.bias is not None:
                        init.zeros_(m.bias)
                else:
                    init_conv_relu_(m)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.conv_in(x)
        for blk in self.blocks:
            h = blk(h)
        h = h + self.in_proj(x)
        return self.tail(h)


class R(nn.Module):
    """
    Diffusers UNet backbone per R with residual add.
    Input ch -> conv(ch->3) -> UNet2DModel (no conv_out) -> conv(feat->ch) -> residual add.
    """
    def __init__(self, ch: int, T: int, device: torch.device | None = None):
        super().__init__()
        self.T = int(T)
        self.device = device or torch.device('cpu')
        # Feature channels (fixed to 128 for google/ddpm-cifar10-32)
        feat_ch = 128

        # diffusers UNet without first and final conv
        self.unet = _load_diffusers_unet(device=self.device, remove_conv_in=True, remove_conv_out=True)
        self.unet.eval()
        _freeze_all_params(self.unet)
        _set_trainable_time_embedding(self.unet, flag=True)
        # enable attention LoRA for legacy attention blocks (q/k/v/proj_out)
        if ENABLE_LORA:
            _enable_unet_lora(self.unet, rank=LORA_RANK, alpha=LORA_ALPHA)
        # input/output adapters
        self.in_adapter = Adapter(ch_in=ch, ch_mid=feat_ch, ch_out=feat_ch, T_embed=self.T, num_blocks=2)
        self.out_adapter = Adapter(ch_in=feat_ch + ch, ch_mid=feat_ch + ch, ch_out=ch, T_embed=self.T, num_blocks=2)
        # learnable time-dependent residual scale: Embedding(T)->sigmoid->cap
        self.res_scale_logits = nn.Embedding(self.T, 1)
        nn.init.zeros_(self.res_scale_logits.weight)  # start near 0.5 * cap
        # GroupNorm to avoid drift (affine=False to make groupnorm idempotent)
        self.gn_out = nn.GroupNorm(32, ch, affine=False)

    def forward_one_step(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feat = self.in_adapter(z, t)
        feat = self.unet(sample=feat, timestep=t.long())
        dy = self.out_adapter(torch.cat([feat.sample, z], dim=1), t)
        # time-dependent learnable residual scale with cap
        cap = 0.3
        s = torch.sigmoid(self.res_scale_logits(t.long())) * cap  # [B,1]
        s = s.view([-1] + [1] * (z.ndim - 1)).to(z.device)
        out = z + dy * s
        return self.gn_out(out)


class RNet(nn.Module):
    def __init__(self, T: int, ch: int):
        super().__init__()
        self.T = int(T)
        self.rin = Rin(ch=ch, T=self.T)
        self.r = R(ch=ch, T=self.T)
        self.rout = Rout(ch=ch)
        self.x0out = X0out(ch=ch)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        z = self.rin(x_t, t.long())
        # r is not used for normal inference (only for multi-step training and inference)
        return self.rout(z)


