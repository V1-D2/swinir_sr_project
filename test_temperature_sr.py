#!/usr/bin/env python3
"""
Скрипт для тестирования обученной температурной Super-Resolution модели
"""

import argparse
import os
import numpy as np
import torch
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

from data_preprocessing import TemperatureDataPreprocessor
from basicsr.utils import tensor2img, imwrite  # ← Changed this line
from hybrid_model import TemperatureSRModel     # ← Changed this line
from config_temperature import *


def parse_args():
    parser = argparse.ArgumentParser(description='Test Temperature Super-Resolution Model')
    parser.add_argument('--model_path', type=str, required=True,
                        help='Path to trained model')
    parser.add_argument('--input_npz', type=str, required=True,
                        help='Input NPZ file for testing')
    parser.add_argument('--output_dir', type=str, default='./test_results',
                        help='Output directory for results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to test')
    parser.add_argument('--save_comparison', action='store_true',
                        help='Save comparison plots')
    parser.add_argument('--stats_path', type=str, default=None,
                        help='Path to preprocessor statistics')
    return parser.parse_args()


def test_model(model, test_data, save_path=None):
    """Тестирование модели на одном образце"""
    model.net_g.eval()

    with torch.no_grad():
        # Подготовка данных
        lr_tensor = test_data['lq'].unsqueeze(0).cuda()
        hr_tensor = test_data['gt'].unsqueeze(0).cuda()

        # Прогон через модель
        sr_tensor = model.net_g(lr_tensor)
        sr_tensor = torch.clamp(sr_tensor, 0, 1)

    # Конвертация в numpy
    lr_img = tensor2img([lr_tensor])
    hr_img = tensor2img([hr_tensor])
    sr_img = tensor2img([sr_tensor])



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
    mse = np.mean((sr_img - hr_img) ** 2)
    psnr = 20 * np.log10(np.max(hr_img) / np.sqrt(mse)) if mse > 0 else float('inf')

    results['metrics'] = {
        'mse': mse,
        'psnr': psnr,
        'temperature_error_mean': np.mean(np.abs(sr_img - hr_img)),
        'temperature_error_max': np.max(np.abs(sr_img - hr_img))
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
    crop_size = min(h // 4, w // 4)
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

    # Создаем конфигурацию для модели
    opt = {
        'name': name,
        'model_type': model_type,
        'scale': scale,
        'num_gpu': 1,
        'network_g': network_g,
        'network_d': network_d,
        'path': path,
        'train': train,
        'is_train': False,
        'dist': False
    }

    # Загружаем модель
    print(f"Loading model from {args.model_path}")
    model = TemperatureSRModel(opt)

    # Load the checkpoint directly
    checkpoint = torch.load(args.model_path, map_location='cuda')

    # Debug: print checkpoint keys
    if isinstance(checkpoint, dict):
        print(f"Checkpoint keys: {list(checkpoint.keys())}")

    # Check if it's a full checkpoint or just the state dict
    if isinstance(checkpoint, dict) and 'params' in checkpoint:
        # It's a full checkpoint from training
        model.net_g.load_state_dict(checkpoint['params'], strict=True)
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        # Alternative checkpoint format
        model.net_g.load_state_dict(checkpoint['state_dict'], strict=True)
    else:
        # It's just the state dict
        model.net_g.load_state_dict(checkpoint, strict=True)

    print("Model loaded successfully!")

    model.net_g.eval()

    # Создаем препроцессор
    preprocessor = TemperatureDataPreprocessor()

    # Загружаем статистику препроцессора если есть
    if args.stats_path and os.path.exists(args.stats_path):
        stats = np.load(args.stats_path)
        preprocessor.stats = dict(stats)
        print(f"Loaded preprocessor statistics from {args.stats_path}")

    # Загружаем тестовые данные
    print(f"Loading test data from {args.input_npz}")
    #data = np.load(args.input_npz, allow_pickle=True)
    #swaths = data['swath_array']

    print(f"Loading test data from {args.input_npz}")
    data = np.load(args.input_npz, allow_pickle=True)

    # Проверяем какие ключи есть в файле
    print(f"Available keys in NPZ: {list(data.keys())}")

    # Создаем структуру данных как в обучающем датасете
    temperature = data['temperature'].astype(np.float32)
    metadata = data['metadata'].item() if hasattr(data['metadata'], 'item') else data['metadata']

    # Создаем список из одного элемента для совместимости
    swaths = [{
        'temperature': temperature,
        'metadata': metadata
    }]

    # Ограничиваем количество тестовых образцов
    num_samples = min(args.num_samples, len(swaths))
    print(f"Testing on {num_samples} samples")

    # Тестирование
    all_metrics = []

    for i in tqdm(range(num_samples), desc="Testing"):
        swath = swaths[i]
        temp = swath['temperature']
        meta = swath['metadata']

        # Предобработка
        temp = preprocessor.crop_or_pad(temp)
        temp_min, temp_max = np.min(temp), np.max(temp)
        temp_norm = preprocessor.normalize_temperature(temp)

        # Создаем пару LR-HR
        lr, hr = preprocessor.create_lr_hr_pair(temp_norm, scale_factor=4)

        # Подготовка данных для модели
        test_data = {
            'lq': torch.from_numpy(lr).unsqueeze(0).float(),
            'gt': torch.from_numpy(hr).unsqueeze(0).float(),
            'metadata': {
                'original_min': temp_min,
                'original_max': temp_max,
                'orbit_type': meta.get('orbit_type', 'unknown')
            }
        }

        # Тестирование
        results = test_model(model, test_data)
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
    print(f"MSE: {avg_metrics['mse']:.4f}")
    print(f"Mean Temperature Error: {avg_metrics['temperature_error_mean']:.2f} K")
    print(f"Max Temperature Error: {avg_metrics['temperature_error_max']:.2f} K")

    # Сохранение метрик
    metrics_path = os.path.join(args.output_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== Test Results ===\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test data: {args.input_npz}\n")
        f.write(f"Number of samples: {num_samples}\n\n")
        f.write("=== Average Metrics ===\n")
        for key, value in avg_metrics.items():
            f.write(f"{key}: {value:.4f}\n")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()