from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr
import math
from skimage.restoration import denoise_bilateral
from torchvision import transforms

def calculate_snr(clean, output):
    noise = clean - output
    signal_var = torch.var(clean)
    noise_var = torch.var(noise)
    if noise_var == 0:
        return float('inf')
    return 10 * math.log10(signal_var / noise_var)

def bilateral_image(image):
    # image is tensor [1,3,H,W] in [0,1]
    img_np = image.squeeze(0).cpu().permute(1,2,0).numpy()
    # denoise_bilateral expects [H,W,C] in [0,1]
    denoised = denoise_bilateral(img_np, sigma_color=0.05, sigma_spatial=15, channel_axis=-1)
    return transforms.ToTensor()(denoised)

def calculate_snr(clean, output):
    noise = clean - output
    signal_var = torch.var(clean)
    noise_var = torch.var(noise)
    if noise_var == 0:
        return float('inf')
    return 10 * math.log10(signal_var / noise_var)

def evaluate_model(model, dataset, reference_func, reference_name, device):
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)  # changed to 1.0 since [0,1]
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    model.eval()
    
    total_psnr_model = 0
    total_ssim_model = 0
    total_lpips_model = 0
    total_sne_model = 0
    
    total_psnr_ref = 0
    total_ssim_ref = 0
    total_lpips_ref = 0
    total_sne_ref = 0
    
    count = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            noisy, clean = dataset[i]
            
            noisy = noisy.unsqueeze(0).to(device)
            clean = clean.unsqueeze(0).to(device)

            output = model(noisy)

            reference_tensor = reference_func(noisy).unsqueeze(0).to(device)

            total_psnr_model += psnr(output, clean, data_range=1.0).item()
            total_ssim_model += ssim_fn(output, clean).item()
            total_lpips_model += lpips_fn(output, clean).mean().item()
            total_sne_model += calculate_snr(clean, output)

            total_psnr_ref += psnr(reference_tensor, clean, data_range=1.0).item()
            total_ssim_ref += ssim_fn(reference_tensor, clean).item()
            total_lpips_ref += lpips_fn(reference_tensor, clean).mean().item()
            total_sne_ref += calculate_snr(clean, reference_tensor)

            count += 1

    results = {
        "model": {
            "psnr": total_psnr_model / count,
            "ssim": total_ssim_model / count,
            "lpips": total_lpips_model / count,
            "sne": total_sne_model / count
        },
        reference_name: {
            "psnr": total_psnr_ref / count,
            "ssim": total_ssim_ref / count,
            "lpips": total_lpips_ref / count,
            "sne": total_sne_ref / count
        }
    }

    return results

def print_results_table(results, title, reference_name):
    print("\n" + "="*70)
    print(f"{title}")
    print("="*70)
    print(f"{'Method':<25} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10} {'SNE':>10}")
    print("-"*70)

    print(f"{reference_name + ' denoising':<25} "
          f"{results[reference_name]['psnr']:>10.2f} "
          f"{results[reference_name]['ssim']:>10.4f} "
          f"{results[reference_name]['lpips']:>10.4f} "
          f"{results[reference_name]['sne']:>10.2f}")

    print(f"{'Model':<25} "
          f"{results['model']['psnr']:>10.2f} "
          f"{results['model']['ssim']:>10.4f} "
          f"{results['model']['lpips']:>10.4f} "
          f"{results['model']['sne']:>10.2f}")

    print("="*70)