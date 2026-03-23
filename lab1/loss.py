import lpips
import torch.nn as nn
from torchmetrics.image import StructuralSimilarityIndexMeasure

class MseSsimLipsLoss(nn.Module):
    def __init__(self, device, mse_weight=0.75, ssim_weight=0.20, lpips_weight=0.05):
        super().__init__()
        self.mse_loss = nn.MSELoss()
        self.mse_weight = mse_weight
        self.ssim_weight = ssim_weight
        self.lpips_weight = lpips_weight
        self.lpips_fn = lpips.LPIPS(net='alex').to(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)

    def forward(self, output, target):
        mse_loss = self.mse_loss(output, target)
        lpips_loss = self.lpips_fn(output, target).mean()
        ssim_value = self.ssim(output, target)
        return self.mse_weight * mse_loss + self.lpips_weight * lpips_loss + self.ssim_weight * (1 - ssim_value)

