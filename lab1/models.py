import math

import torch
import torch.nn as nn
import torch.nn.functional as F

# class ResidualBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn1 = nn.BatchNorm2d(channels)
#         self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
#         self.bn2 = nn.BatchNorm2d(channels)
    
#     def forward(self, x):
#         residual = x
#         out = F.relu(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += residual
#         return F.relu(out)

# class BaseNet(nn.Module):
#     def __init__(self, num_channels=[3,16,32,64]):
#         super().__init__()
#         self.entry = nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1)
#         self.layers = nn.ModuleList([
#             nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=3, padding=1)
#             for i in range(1, len(num_channels)-1)
#         ])
#         self.bns = nn.ModuleList([
#             nn.BatchNorm2d(num_channels[i+1]) for i in range(1, len(num_channels)-1)
#         ])
#         # jeden blok rezydualny
#         self.resblock = ResidualBlock(num_channels[-1])
    
#     def forward_features(self, x):
#         x = F.relu(self.entry(x))
#         for conv, bn in zip(self.layers, self.bns):
#             x = F.relu(bn(conv(x)))
#         x = self.resblock(x)
#         return x

# class UpsampleBlock(nn.Module):
#     def __init__(self, channels):
#         super().__init__()
#         self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
#         self.shuffle = nn.PixelShuffle(2)
#         self.relu = nn.ReLU(inplace=True)
    
#     def forward(self, x):
#         x = self.conv(x)
#         x = self.shuffle(x)
#         x = self.relu(x)
#         return x

# class ResizeNet(BaseNet):
#     def __init__(self, upscale_factor=8, num_channels=[3,16,32,64], num_res_blocks=4):
#         #upscale_factor is 8 when we want to go from 32x32 to 256x256, 4 for 64x64 to 256x256 etc.
#         super().__init__(num_channels)
#         self.upscale_factor = upscale_factor
#         num_upsample = int(math.log2(upscale_factor))
#         self.upsample_blocks = nn.Sequential(*[UpsampleBlock(num_channels[-1]) for _ in range(num_upsample)])
#         self.out_conv = nn.Conv2d(num_channels[-1], 3, kernel_size=3, padding=1)
#         self.activation = nn.Tanh()
    
#     def forward(self, x):
#         x_feat = self.forward_features(x)
#         x_up = self.upsample_blocks(x_feat)
#         out = self.out_conv(x_up)
#         residual = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
#         out += residual
#         return self.activation(out)
    
# class DenoiseNet(BaseNet):
#     def __init__(self, num_channels=[3,16,32,64]):
#         super().__init__(num_channels)
#         self.out_conv = nn.Conv2d(num_channels[-1], 3, kernel_size=3, padding=1)
#         self.activation = nn.Tanh()
    
#     def forward(self, x):
#         x_feat = self.forward_features(x)
#         out = self.out_conv(x_feat)
#         return self.activation(out)

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return out


class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super(UpsampleBlock, self).__init__()
        self.conv = nn.Conv2d(channels, channels * (2 ** 2), kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.conv(x)
        out = self.shuffle(out)
        return out


class UpscaleNet(nn.Module):
    def __init__(self, num_residual_blocks=8, channels=64, upscale_factor=8):
        super(UpscaleNet, self).__init__()
        self.upscale_factor = upscale_factor
        
        self.entry_conv = nn.Conv2d(3, channels, kernel_size=9, padding=4)
        
        self.residual_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_residual_blocks)]
        )

        self.upsample_blocks = nn.Sequential(
            *[UpsampleBlock(channels) for _ in range(int(math.log2(upscale_factor)))]
        )
        
        self.exit_conv = nn.Conv2d(channels, 3, kernel_size=9, padding=4)
        self.activation = nn.Tanh()

    def forward(self, x):
        out = F.relu(self.entry_conv(x))
        out = self.residual_blocks(out)
        out = self.upsample_blocks(out)
        out = self.exit_conv(out)
        residual = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out += residual
        return self.activation(out)