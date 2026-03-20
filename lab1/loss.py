import lpips
import torch
import torch.nn as nn

class PixelLPIPSLoss(nn.Module):
    def __init__(self, device, pixel_weight=0.5, lpips_weight=0.5):
        super().__init__()
        self.pixel_weight = pixel_weight
        self.lpips_weight = lpips_weight
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)

    def forward(self, output, target):
        pixel_loss = torch.mean(torch.abs(output - target))
        lpips_loss = self.lpips_fn(output, target).mean()
        return self.pixel_weight * pixel_loss + self.lpips_weight * lpips_loss
