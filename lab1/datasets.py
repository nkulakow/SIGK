from torch.utils.data import Dataset
from torchvision import transforms
import cv2
import glob
import os
import torch

class ResizeDataset(Dataset):
    def __init__(self, root_dir, img_size=32):
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png'))
        self.transform = transforms.ToTensor()
        self.img_size = img_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        low_res = cv2.resize(
            image,
            (self.img_size, self.img_size),
            interpolation=cv2.INTER_CUBIC
        )

        low_res = self.transform(low_res) * 2 - 1
        high_res = self.transform(image) * 2 - 1

        return low_res, high_res

class DenoiseDataset(Dataset):
    def __init__(self, root_dir, sigma=0.01):
        self.image_paths = glob.glob(os.path.join(root_dir, '*.png'))
        self.transform = transforms.ToTensor()
        self.sigma = sigma

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = cv2.imread(self.image_paths[idx])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        clean = self.transform(image)  # [0,1]
        noise = torch.randn_like(clean) * self.sigma
        noisy = clean + noise
        noisy = torch.clamp(noisy, 0, 1)

        return noisy, clean
