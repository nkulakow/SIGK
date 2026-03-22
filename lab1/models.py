import math
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.leaky(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.leaky(out)

class BaseNet(nn.Module):
    def __init__(self, num_channels=[3,16,32,64], num_residual=12):
        super().__init__()
        self.entry = nn.Conv2d(num_channels[0], num_channels[1], kernel_size=3, padding=1)
        self.layers = nn.ModuleList([
            nn.Conv2d(num_channels[i], num_channels[i+1], kernel_size=3, padding=1)
            for i in range(1, len(num_channels)-1)
        ])
        self.bns = nn.ModuleList([
            nn.BatchNorm2d(num_channels[i+1]) for i in range(1, len(num_channels)-1)
        ])
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(num_channels[-1]) for _ in range(num_residual)]
        )
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
    
    def forward_features(self, x):
        x = self.leaky(self.entry(x))
        for conv, bn in zip(self.layers, self.bns):
            x = self.leaky(bn(conv(x)))
        x = self.res_blocks(x)
        return x

class UpsampleBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels * 4, kernel_size=3, padding=1)
        self.shuffle = nn.PixelShuffle(2)
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.shuffle(x)
        x = self.leaky(x)
        return x

class ResizeNet(BaseNet):
    def __init__(self, upscale_factor=8, num_channels=[3,16,32,64]):
        #upscale_factor is 8 when we want to go from 32x32 to 256x256, 4 for 64x64 to 256x256 etc.
        super().__init__(num_channels)
        self.upscale_factor = upscale_factor
        num_upsample = int(math.log2(upscale_factor))
        self.upsample_blocks = nn.Sequential(*[UpsampleBlock(num_channels[-1]) for _ in range(num_upsample)])
        self.out_conv = nn.Conv2d(num_channels[-1], 3, kernel_size=3, padding=1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x_feat = self.forward_features(x)
        x_up = self.upsample_blocks(x_feat)
        out = self.out_conv(x_up)
        residual = F.interpolate(x, scale_factor=self.upscale_factor, mode='bicubic', align_corners=False)
        out += residual
        return self.activation(out)
    
class DenoiseNet(BaseNet):
    def __init__(self, num_channels=[3,16,32,64]):
        super().__init__(num_channels)
        self.out_conv = nn.Conv2d(num_channels[-1], 3, kernel_size=3, padding=1)
        self.activation = nn.Tanh()
    
    def forward(self, x):
        x_feat = self.forward_features(x)
        out = self.out_conv(x_feat)
        return self.activation(out)
