import torch
import torch.nn as nn
import torch.nn.functional as F
from spectral_normalization import SpectralNorm


class ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, spec_norm=False, LR=0.01, stride=1):
        super(ConvBlock, self).__init__()

        if spec_norm:
            self.main = nn.Sequential(
                SpectralNorm(nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False)),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )
        else:
            self.main = nn.Sequential(
                nn.Conv2d(dim_in, dim_out, kernel_size=3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(dim_out, affine=True, track_running_stats=True),
                nn.LeakyReLU(LR, inplace=False),
            )

    def forward(self, x):
        return self.main(x)

