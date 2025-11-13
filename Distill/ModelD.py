import torch
from torch import nn
from torch.nn import init
from Diffusion.ModelR import _load_diffusers_unet, init_conv_relu_, ResBlock


class Up2x(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.gn = nn.GroupNorm(32, in_ch)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        init_conv_relu_(self.conv)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gn(x)
        h = self.act(h)
        h = nn.functional.interpolate(h, scale_factor=2, mode="nearest")
        return self.conv(h)


class ReadoutUp(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, upsample_factor: int):
        super().__init__()
        assert upsample_factor in [2, 4], "upsample_factor must be 2 or 4"
        self.upsample_factor = upsample_factor
        self.up1 = Up2x(in_ch, in_ch // 2)
        if upsample_factor == 4:
            self.up2 = Up2x(in_ch // 2, in_ch // 4)
        self.gn = nn.GroupNorm(32, in_ch // upsample_factor)
        self.act = nn.SiLU()
        self.conv = nn.Conv2d(in_ch // upsample_factor, out_ch, 3, 1, 1)
        init_conv_relu_(self.conv)
        if self.conv.bias is not None:
            init.zeros_(self.conv.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.up1(x)
        if self.upsample_factor == 4:
            h = self.up2(h)
        h = self.gn(h)
        h = self.act(h)
        return self.conv(h)


class ReadoutHead(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, use_gn_out: bool = False):
        super().__init__()
        self.gn1 = nn.GroupNorm(32, in_ch)
        self.conv1 = nn.Conv2d(in_ch, in_ch, 3, 1, 1)
        self.gn2 = nn.GroupNorm(32, in_ch)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, 1, 1)
        for conv in (self.conv1, self.conv2):
            init_conv_relu_(conv)
            if conv.bias is not None:
                init.zeros_(conv.bias)
        self.act = nn.SiLU()
        # optional final GroupNorm
        if use_gn_out:
            self.gn_out = nn.GroupNorm(32, out_ch, affine=False)
        else:
            self.gn_out = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.gn1(x)
        h = self.act(h)
        h = self.conv1(h)
        h = self.gn2(h)
        h = self.act(h)
        h = self.conv2(h)
        if self.gn_out is not None:
            h = self.gn_out(h)
        return h


class ModelD(nn.Module):
    """
    UNet backbone (diffusers) with multi-scale readouts for 1-step distillation.

    Architecture:
      - Hook UNet intermediate features:
        * 8x8 feature from up_blocks[0]
        * 16x16 feature from up_blocks[1]
        * 32x32 final feature (conv_out removed)
      - Project features to latent_ch:
        * proj_8: 8x8 -> 4 * latent_ch (for 8x branch)
        * proj_16: 16x16 -> 2 * latent_ch (for 16x branch)
        * proj_32: 32x32 -> latent_ch (for 32x branch)
      - Readouts:
        * 8x branch: ReadoutUp(4x) -> ReadoutHead (rin) + ReadoutHead x4 (chunk1..4)
        * 16x branch: ReadoutUp(2x) -> ReadoutHead x3 (chunk5..7)
        * 32x branch: ReadoutHead x3 (chunk8..10)
      - All UNet parameters are trainable (full retraining).

    Output:
      - Channel-concatenated [rin0, chunk1..chunk10] with shape [B, (1+num_chunks)*latent_ch, 32, 32]
    """
    def __init__(self, latent_ch: int, num_chunks: int = 10, T: int = 1000):
        super().__init__()
        assert num_chunks == 10, "Current implementation is fixed to 10 chunks"
        self.T = T

        # UNet backbone
        self.unet = _load_diffusers_unet(model_id='google/ddpm-cifar10-32', remove_conv_in=False, remove_conv_out=True)
        for p in self.unet.parameters():
            p.requires_grad = True

        # get UNet channel dimensions from config (google/ddpm-cifar10-32)
        unet_ch_8 = 256   # up_blocks[0] output channels
        unet_ch_16 = 256  # up_blocks[1] output channels
        unet_ch_32 = 128  # final UNet feature channels when conv_out is removed

        # feature hooks for 8x, 16x (f32 uses UNet final output directly)
        self._feat = {"f8": None, "f16": None}
        self._register_feature_hooks()

        # project UNet features to latent_ch
        self.proj_8 = nn.Conv2d(unet_ch_8, latent_ch * 4, 3, 1, 1)
        self.proj_16 = nn.Conv2d(unet_ch_16, latent_ch * 2, 3, 1, 1)
        self.proj_32 = nn.Conv2d(unet_ch_32, latent_ch, 3, 1, 1)
        for conv in (self.proj_8, self.proj_16, self.proj_32):
            init_conv_relu_(conv)
            if conv.bias is not None:
                init.zeros_(conv.bias)

        # readout upsampling per scale
        self.up_8 = ReadoutUp(latent_ch * 4, latent_ch, upsample_factor=4)
        self.up_16 = ReadoutUp(latent_ch * 2, latent_ch, upsample_factor=2)

        # readout heads per scale
        self.head_8_rin = ReadoutHead(latent_ch, latent_ch, use_gn_out=True)
        self.head_8 = nn.ModuleList([ReadoutHead(latent_ch, latent_ch) for _ in range(4)])
        self.head_16 = nn.ModuleList([ReadoutHead(latent_ch, latent_ch) for _ in range(3)])
        self.head_32 = nn.ModuleList([ReadoutHead(latent_ch, latent_ch) for _ in range(3)])

    def _register_feature_hooks(self) -> None:
        target_f8 = None
        target_f16 = None
        for name, module in self.unet.named_modules():
            if name.endswith("up_blocks.0"):
                target_f8 = module
            elif name.endswith("up_blocks.1"):
                target_f16 = module
        assert (target_f8 is not None) and (target_f16 is not None), "UNet intermediate blocks not found"

        def make_hook(key: str):
            def hook(_m, _i, out):
                y = out
                if isinstance(y, (tuple, list)):
                    y = y[0]
                self._feat[key] = y
            return hook

        self._h8 = target_f8.register_forward_hook(make_hook("f8"))
        self._h16 = target_f16.register_forward_hook(make_hook("f16"))

    def forward(self, x_T: torch.Tensor) -> torch.Tensor:
        B = x_T.shape[0]
        t = x_T.new_ones([B, ], dtype=torch.long) * (self.T - 1)
        self._feat = {"f8": None, "f16": None}
        # run UNet forward to capture intermediate features via hooks
        feat = self.unet(sample=x_T, timestep=t.long())
        f8_raw = self._feat["f8"]
        f16_raw = self._feat["f16"]
        f32_raw = feat.sample if hasattr(feat, 'sample') else feat
        # project 8x, 16x, and 32x features to latent_ch
        f8 = self.proj_8(f8_raw)
        f16 = self.proj_16(f16_raw)
        f32 = self.proj_32(f32_raw)
        assert (f8 is not None) and (f16 is not None) and (f32 is not None), "Intermediate features not captured"

        outputs = []
        f8 = self.up_8(f8)
        rin = self.head_8_rin(f8)
        outputs.append(rin)
        for i in range(4):
            outputs.append(self.head_8[i](rin))
        f16 = self.up_16(f16)
        for i in range(3):
            outputs.append(self.head_16[i](f16))
        for i in range(3):
            outputs.append(self.head_32[i](f32))
        return torch.cat(outputs, dim=1)


