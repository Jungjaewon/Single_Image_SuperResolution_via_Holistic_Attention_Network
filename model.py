import torch.nn as nn
import torch
import math
from block import ConvBlock


class HR_Discriminator128(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True, LR=0.02):
        super(HR_Discriminator128, self).__init__()
        self.main = list()
        self.main.append(ConvBlock(3, 16, spec_norm, stride=2, LR=LR)) # 128 -> 64
        self.main.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR))  # 64 -> 32
        self.main.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR))  # 32 -> 16
        self.main.append(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

class HR_Discriminator64(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True, LR=0.02):
        super(HR_Discriminator64, self).__init__()
        self.main = list()
        self.main.append(ConvBlock(3, 16, spec_norm, stride=2, LR=LR)) # 64 -> 32
        self.main.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR))  # 32 -> 16
        self.main.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR))  # 16 -> 8
        self.main.append(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)

class LR_Discriminator32(nn.Module):
    """Discriminator network with PatchGAN.
    W = (W - F + 2P) /S + 1"""

    def __init__(self, spec_norm=True, LR=0.02):
        super(LR_Discriminator32, self).__init__()
        self.main = list()
        self.main.append(ConvBlock(3, 16, spec_norm, stride=2, LR=LR)) # 32 -> 16
        self.main.append(ConvBlock(16, 32, spec_norm, stride=2, LR=LR)) # 16 -> 8
        self.main.append(ConvBlock(32, 64, spec_norm, stride=2, LR=LR)) # 8 -> 4
        self.main.append(nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1))
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


class DSN64(nn.Module):

    def __init__(self, spec_norm=False, LR=0.02, inner_channel=32):
        super(DSN64, self).__init__()
        self.main = list()

        self.first_conv = nn.Conv2d(3, inner_channel, kernel_size=3, stride=1, padding=1)
        self.down = ConvBlock(inner_channel, inner_channel, spec_norm, stride=2, LR=LR)
        self.main.append(self.first_conv)
        self.main.append(self.down)
        for _ in range(5):
            self.main.append(ConvBlock(inner_channel, inner_channel, spec_norm, stride=1, LR=LR))
        self.tanh = nn.Tanh()
        self.last_conv = nn.Conv2d(inner_channel, 3, kernel_size=3, stride=1, padding=1)
        self.main.append(self.last_conv)
        self.main.append(self.tanh)
        self.main = nn.Sequential(*self.main)
    def forward(self, x):
        return self.main(x)


class DSN128(nn.Module):

    def __init__(self, spec_norm=False, LR=0.02, inner_channel=64):
        super(DSN128, self).__init__()
        self.main = list()

        self.first_conv = nn.Conv2d(3, inner_channel, kernel_size=3, stride=1, padding=1)
        self.down1 = ConvBlock(inner_channel, inner_channel, spec_norm, stride=2, LR=LR)
        self.main.append(self.first_conv)
        self.main.append(self.down1)
        for _ in range(5):
            self.main.append(ConvBlock(inner_channel, inner_channel, spec_norm, stride=1, LR=LR))

        self.down2 = ConvBlock(inner_channel, inner_channel, spec_norm, stride=2, LR=LR)
        self.main.append(self.down2)

        for _ in range(5):
            self.main.append(ConvBlock(inner_channel, inner_channel, spec_norm, stride=1, LR=LR))

        self.tanh = nn.Tanh()
        self.last_conv = nn.Conv2d(inner_channel, 3, kernel_size=3, stride=1, padding=1)
        self.main.append(self.last_conv)
        self.main.append(self.tanh)
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x)


class SRN(nn.Module):

    def __init__(self, spec_norm=False, LR=0.02, inner_channel=64, up_scale=2):
        super(SRN, self).__init__()
        assert up_scale in [2, 4]
        self.up = nn.Upsample(scale_factor=up_scale, mode='bicubic')
        self.main = list()
        self.main.append(nn.Conv2d(3, inner_channel, kernel_size=3, stride=1, padding=1))
        for _ in range(10):
            self.main.append(ConvBlock(inner_channel, inner_channel, spec_norm, stride=1, LR=LR))
        self.main.append(nn.Conv2d(inner_channel, inner_channel * up_scale * up_scale, kernel_size=3, stride=1, padding=1))
        self.main.append(nn.PixelShuffle(up_scale))
        self.main.append(nn.Tanh())
        self.main = nn.Sequential(*self.main)

    def forward(self, x):
        return self.main(x) + self.up(x)