import torch
from torch import nn
from Diffusion.ModelR import (
    init_conv_relu_,
    _set_requires_grad,
    _freeze_all_params,
    _load_diffusers_unet,
    _set_trainable_time_embedding,
    FiLMResBlock,
    Adapter,
)


class UNet(nn.Module):
    """
    Diffusers UNet (for unit test of Rin in RNet).
    """
    def __init__(self, T, device: torch.device | None = None):
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
        # output adapter
        self.out_adapter = Adapter(ch_in=feat_ch, ch_mid=feat_ch, ch_out=3, T_embed=self.T)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        feat = self.unet(sample=x, timestep=t.long())
        out = self.out_adapter(feat.sample, t)
        return out
