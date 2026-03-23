from torchmetrics.image import StructuralSimilarityIndexMeasure
import lpips
import torch
from torchmetrics.functional import peak_signal_noise_ratio as psnr


def evaluate_model(model, dataset, reference_func, reference_name, device):
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=2.0).to(device)
    lpips_fn = lpips.LPIPS(net='alex').to(device)
    
    model.eval()
    
    total_psnr_model = 0
    total_ssim_model = 0
    total_lpips_model = 0
    
    total_psnr_ref = 0
    total_ssim_ref = 0
    total_lpips_ref = 0
    
    count = 0

    with torch.no_grad():
        for i in range(len(dataset)):
            lr, hr = dataset[i]
            
            lr = lr.unsqueeze(0).to(device)
            hr = hr.unsqueeze(0).to(device)

            output = model(lr)

            reference_tensor = reference_func(lr).unsqueeze(0).to(device)

            total_psnr_model += psnr(output, hr, data_range=2.0).item()
            total_ssim_model += ssim_fn(output, hr).item()
            total_lpips_model += lpips_fn(output, hr).mean().item()

            total_psnr_ref += psnr(reference_tensor, hr, data_range=2.0).item()
            total_ssim_ref += ssim_fn(reference_tensor, hr).item()
            total_lpips_ref += lpips_fn(reference_tensor, hr).mean().item()

            count += 1

    results = {
        "model": {
            "psnr": total_psnr_model / count,
            "ssim": total_ssim_model / count,
            "lpips": total_lpips_model / count
        },
        reference_name: {
            "psnr": total_psnr_ref / count,
            "ssim": total_ssim_ref / count,
            "lpips": total_lpips_ref / count
        }
    }

    return results

def print_results_table(results, title, reference_name):
    print("\n" + "="*60)
    print(f"{title}")
    print("="*60)
    print(f"{'Method':<25} {'PSNR':>10} {'SSIM':>10} {'LPIPS':>10}")
    print("-"*60)

    print(f"{reference_name + ' interpolation':<25} "
          f"{results[reference_name]['psnr']:>10.2f} "
          f"{results[reference_name]['ssim']:>10.4f} "
          f"{results[reference_name]['lpips']:>10.4f}")

    print(f"{'Model':<25} "
          f"{results['model']['psnr']:>10.2f} "
          f"{results['model']['ssim']:>10.4f} "
          f"{results['model']['lpips']:>10.4f}")

    print("="*60)