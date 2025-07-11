#!/usr/bin/env python3
"""
Скрипт для тестирования обученной SwinIR температурной Super-Resolution модели
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import cv2
import matplotlib

matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from models.network_swinir import SwinIR
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
import torch.nn.functional as F




def parse_args():
    parser = argparse.ArgumentParser(description='Test Temperature SwinIR Super-Resolution Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model checkpoint')
    parser.add_argument('--input_npz', type=str, required=True,
                        help='Input NPZ file for testing')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save comparison plots')
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Super-resolution scale factor')
    parser.add_argument('--patch_size', type=int, default=160,
                        help='Patch size used in training')
    parser.add_argument('--window_size', type=int, default=8,
                        help='Window size for SwinIR')
    return parser.parse_args()


def define_model(args):
    """Определение модели SwinIR для температурных данных"""
    model = SwinIR(
        upscale=args.scale_factor,
        in_chans=1,  # Одноканальные данные
        img_size=args.patch_size,
        window_size=args.window_size,
        img_range=1.,
        depths=[8, 8, 8, 8, 8, 8, 8, 8],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='3conv'
    )
    return model


def test_model(model, test_data, device):
    """Тестирование модели на одном образце"""
    model.eval()

    with torch.no_grad():
        # Подготовка данных
        lr_tensor = test_data['lq'].unsqueeze(0).to(device)
        hr_tensor = test_data['gt'].unsqueeze(0).to(device)

        # Паддинг для window_size
        _, _, h_old, w_old = lr_tensor.size()
        window_size = 8
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [2])], 2)[:, :, :h_old + h_pad, :]
        lr_tensor = torch.cat([lr_tensor, torch.flip(lr_tensor, [3])], 3)[:, :, :, :w_old + w_pad]

        # Прогон через модель
        sr_tensor = model(lr_tensor)

        # Обрезка паддинга
        sr_tensor = sr_tensor[..., :h_old * test_data['scale_factor'], :w_old * test_data['scale_factor']]
        sr_tensor = torch.clamp(sr_tensor, 0, 1)

    # Конвертация в numpy
    lr_img = lr_tensor[0, 0, :h_old, :w_old].cpu().numpy()
    hr_img = hr_tensor[0, 0].cpu().numpy()
    sr_img = sr_tensor[0, 0].cpu().numpy()

    # Восстановление оригинальных значений температуры
    if 'metadata' in test_data:
        meta = test_data['metadata']
        if 'original_min' in meta and 'original_max' in meta:
            # Денормализация
            hr_img = hr_img * (meta['original_max'] - meta['original_min']) + meta['original_min']
            sr_img = sr_img * (meta['original_max'] - meta['original_min']) + meta['original_min']
            lr_img = lr_img * (meta['original_max'] - meta['original_min']) + meta['original_min']

    results = {
        'lr': lr_img,
        'hr': hr_img,
        'sr': sr_img,
        'metadata': test_data.get('metadata', {})
    }

    # Вычисление метрик
    # Конвертируем в uint8 для PSNR/SSIM
    hr_uint8 = ((hr_img - hr_img.min()) / (hr_img.max() - hr_img.min()) * 255).astype(np.uint8)
    sr_uint8 = ((sr_img - sr_img.min()) / (sr_img.max() - sr_img.min()) * 255).astype(np.uint8)

    psnr = calculate_psnr(sr_uint8, hr_uint8, crop_border=0)
    ssim = calculate_ssim(sr_uint8, hr_uint8, crop_border=0)

    # Метрики в физических единицах
    mse = np.mean((sr_img - hr_img) ** 2)
    mae = np.mean(np.abs(sr_img - hr_img))
    max_error = np.max(np.abs(sr_img - hr_img))

    results['metrics'] = {
        'psnr': psnr,
        'ssim': ssim,
        'mse': mse,
        'mae': mae,
        'temperature_error_mean': mae,
        'temperature_error_max': max_error
    }

    return results


def save_comparison_plot(results, save_path, idx):
    """Сохранение сравнительного изображения"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Низкое разрешение
    im1 = axes[0, 0].imshow(results['lr'], cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Low Resolution ({results["lr"].shape[0]}×{results["lr"].shape[1]})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # Высокое разрешение (Ground Truth)
    im2 = axes[0, 1].imshow(results['hr'], cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'High Resolution GT ({results["hr"].shape[0]}×{results["hr"].shape[1]})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # Super Resolution результат
    im3 = axes[0, 2].imshow(results['sr'], cmap='viridis', aspect='auto')
    axes[0, 2].set_title(f'Super Resolution ({results["sr"].shape[0]}×{results["sr"].shape[1]})')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Разница между HR и SR
    diff = np.abs(results['hr'] - results['sr'])
    im4 = axes[1, 0].imshow(diff, cmap='hot', aspect='auto')
    axes[1, 0].set_title('Absolute Difference (HR - SR)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # Увеличенная область для детального сравнения
    h, w = results['hr'].shape
    crop_size = min(h // 4, w // 4, 64)
    start_h, start_w = h // 2 - crop_size // 2, w // 2 - crop_size // 2

    hr_crop = results['hr'][start_h:start_h + crop_size, start_w:start_w + crop_size]
    sr_crop = results['sr'][start_h:start_h + crop_size, start_w:start_w + crop_size]

    im5 = axes[1, 1].imshow(hr_crop, cmap='viridis', aspect='auto')
    axes[1, 1].set_title('HR Crop (Center)')
    axes[1, 1].axis('off')
    plt.colorbar(im5, ax=axes[1, 1], fraction=0.046)

    im6 = axes[1, 2].imshow(sr_crop, cmap='viridis', aspect='auto')
    axes[1, 2].set_title('SR Crop (Center)')
    axes[1, 2].axis('off')
    plt.colorbar(im6, ax=axes[1, 2], fraction=0.046)

    # Добавляем метрики
    metrics_text = f"PSNR: {results['metrics']['psnr']:.2f} dB\n"
    metrics_text += f"SSIM: {results['metrics']['ssim']:.4f}\n"
    metrics_text += f"MSE: {results['metrics']['mse']:.4f}\n"
    metrics_text += f"Mean Temp Error: {results['metrics']['temperature_error_mean']:.2f} K\n"
    metrics_text += f"Max Temp Error: {results['metrics']['temperature_error_max']:.2f} K"

    fig.text(0.02, 0.02, metrics_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'comparison_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Создаем выходную директорию
    os.makedirs(args.output_dir, exist_ok=True)

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Загружаем модель
    print(f"Loading model from {args.model_path}")
    model = define_model(args).to(device)

    # Загружаем checkpoint
    checkpoint = torch.load(args.model_path, map_location=device,  weights_only=False)

    # Debug: print checkpoint keys
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Загружаем веса модели
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'], strict=True)
        elif 'params' in checkpoint:
            model.load_state_dict(checkpoint['params'], strict=True)
        elif 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'], strict=True)
        else:
            # Пробуем загрузить напрямую
            model.load_state_dict(checkpoint, strict=True)
    else:
        # Это просто state dict
        model.load_state_dict(checkpoint, strict=True)

    print("Model loaded successfully!")
    model.eval()

    # Загружаем тестовые данные
    print(f"Loading test data from {args.input_npz}")


    # Загружаем данные напрямую
    data = np.load(args.input_npz, allow_pickle=True)
    print(f"Available keys in NPZ: {list(data.keys())}")

    # Обрабатываем данные в зависимости от формата
    if 'temperature' in data and 'metadata' in data:
        # Формат с отдельными полями
        temperature = data['temperature'].astype(np.float32)
        metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']
        swaths = [{
            'temperature': temperature,
            'metadata': metadata
        }]
    elif 'swaths' in data:
        swaths = data['swaths']
    elif 'swath_array' in data:
        swaths = data['swath_array']
    else:
        raise ValueError("Cannot find temperature data in the NPZ file")

    # Ограничиваем количество тестовых образцов
    num_samples = min(args.num_samples, len(swaths))
    print(f"Testing on {num_samples} samples")

    # Тестирование
    all_metrics = []

    for i in tqdm(range(num_samples), desc="Testing"):
        swath = swaths[i]
        temp = swath['temperature'].astype(np.float32)
        meta = swath.get('metadata', {})

        # Убеждаемся, что размеры кратны scale_factor
        h, w = temp.shape
        h = h - h % args.scale_factor
        w = w - w % args.scale_factor
        temp = temp[:h, :w]

        temp_min, temp_max = np.min(temp), np.max(temp)
        if temp_max > temp_min:
            temp_norm = (temp - temp_min) / (temp_max - temp_min)
        else:
            temp_norm = np.zeros_like(temp)

        # Создаем LR версию простым даунсэмплингом для тестирования
        lr = cv2.resize(temp_norm, (w // args.scale_factor, h // args.scale_factor),
                        interpolation=cv2.INTER_AREA)
        hr = temp_norm

        # Подготовка данных для модели
        test_data = {
            'lq': torch.from_numpy(lr).unsqueeze(0).float(),
            'gt': torch.from_numpy(hr).unsqueeze(0).float(),
            'scale_factor': args.scale_factor,
            'metadata': {
                'original_min': temp_min,
                'original_max': temp_max,
                'orbit_type': meta.get('orbit_type', 'unknown')
            }
        }

        # Тестирование
        results = test_model(model, test_data, device)
        all_metrics.append(results['metrics'])

        # Сохранение результатов
        if args.save_comparison:
            save_comparison_plot(results, args.output_dir, i)

        # Сохранение numpy массивов
        np_save_path = os.path.join(args.output_dir, f'result_{i:04d}.npz')
        np.savez(np_save_path,
                 lr=results['lr'],
                 hr=results['hr'],
                 sr=results['sr'],
                 metrics=results['metrics'],
                 metadata=results['metadata'])

    # Вычисление средних метрик
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])

    print("\n=== Average Metrics ===")
    print(f"PSNR: {avg_metrics['psnr']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f}")
    print(f"MSE: {avg_metrics['mse']:.4f}")
    print(f"MAE: {avg_metrics['mae']:.4f}")
    print(f"Mean Temperature Error: {avg_metrics['temperature_error_mean']:.2f} K")
    print(f"Max Temperature Error: {avg_metrics['temperature_error_max']:.2f} K")

    # Сохранение метрик
    metrics_path = os.path.join(args.output_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== Test Results ===\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test data: {args.input_npz}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Scale factor: {args.scale_factor}\n\n")
        f.write("=== Average Metrics ===\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()