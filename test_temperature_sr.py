#!/usr/bin/env python3
"""
Script for testing trained SwinIR temperature Super-Resolution model
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
from collections import OrderedDict

from models.network_swinir import SwinIR
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim


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
    parser.add_argument('--tile_size', type=int, default=256,
                        help='Tile size for processing large images')
    parser.add_argument('--tile_overlap', type=int, default=32,
                        help='Overlap between tiles')
    return parser.parse_args()


def load_checkpoint_flexible(model, checkpoint_path, device):
    """Load checkpoint with flexible state dict handling"""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Extract state dict from various checkpoint formats
    if isinstance(checkpoint, dict):
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'params' in checkpoint:
            state_dict = checkpoint['params']
        elif 'model' in checkpoint:
            state_dict = checkpoint['model']
        else:
            state_dict = checkpoint
    else:
        state_dict = checkpoint

    # Get current model state dict
    model_state_dict = model.state_dict()

    # Create new state dict with matched keys
    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        if k in model_state_dict:
            if model_state_dict[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                print(f"Shape mismatch for {k}: checkpoint {v.shape} vs model {model_state_dict[k].shape}")
                # Skip mismatched attention masks - they will be recalculated
                if 'attn_mask' not in k:
                    print(f"Warning: Unable to load {k} due to shape mismatch")
        else:
            print(f"Key {k} not found in model")

    # Load the filtered state dict
    model.load_state_dict(new_state_dict, strict=False)

    print(f"Loaded {len(new_state_dict)}/{len(state_dict)} parameters from checkpoint")

    # Return additional info if available
    info = {}
    if isinstance(checkpoint, dict):
        info['epoch'] = checkpoint.get('epoch', 0)
        info['best_psnr'] = checkpoint.get('best_psnr', 0)
        info['args'] = checkpoint.get('args', None)

    return info


def tile_process(model, img, scale, tile_size, tile_overlap, window_size):
    """Process image in tiles to handle large images"""
    batch, channel, height, width = img.shape
    output_height = height * scale
    output_width = width * scale

    # Calculate output tile size
    output_tile_size = tile_size * scale
    output_overlap = tile_overlap * scale

    # Initialize output
    output = torch.zeros((batch, channel, output_height, output_width), device=img.device)

    # Process each tile
    tiles_x = int(np.ceil((width - tile_overlap) / (tile_size - tile_overlap)))
    tiles_y = int(np.ceil((height - tile_overlap) / (tile_size - tile_overlap)))

    for y in range(tiles_y):
        for x in range(tiles_x):
            # Calculate tile boundaries
            ofs_x = x * (tile_size - tile_overlap)
            ofs_y = y * (tile_size - tile_overlap)

            # Ensure we don't go out of bounds
            input_start_x = ofs_x
            input_end_x = min(ofs_x + tile_size, width)
            input_start_y = ofs_y
            input_end_y = min(ofs_y + tile_size, height)

            # Adjust tile size if at boundary
            tile_width = input_end_x - input_start_x
            tile_height = input_end_y - input_start_y

            # Extract tile
            input_tile = img[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

            # Pad tile to window_size if needed
            _, _, h, w = input_tile.size()
            h_pad = (h // window_size + 1) * window_size - h if h % window_size != 0 else 0
            w_pad = (w // window_size + 1) * window_size - w if w % window_size != 0 else 0

            if h_pad > 0 or w_pad > 0:
                input_tile = torch.nn.functional.pad(input_tile, (0, w_pad, 0, h_pad), mode='reflect')

            # Process tile
            with torch.no_grad():
                output_tile = model(input_tile)

            # Remove padding from output
            if h_pad > 0 or w_pad > 0:
                output_tile = output_tile[:, :, :tile_height * scale, :tile_width * scale]

            # Calculate output position
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # Place tile in output with blending for overlaps
            if x > 0:  # Blend left edge
                blend_start_x = output_start_x
                blend_end_x = output_start_x + output_overlap
                alpha = torch.linspace(0, 1, blend_end_x - blend_start_x, device=img.device).view(1, 1, 1, -1)

                output[:, :, output_start_y:output_end_y, blend_start_x:blend_end_x] = (
                        output[:, :, output_start_y:output_end_y, blend_start_x:blend_end_x] * (1 - alpha) +
                        output_tile[:, :, :, :output_overlap] * alpha
                )
                output[:, :, output_start_y:output_end_y, blend_end_x:output_end_x] = \
                    output_tile[:, :, :, output_end_x - blend_end_x]
            else:
                output[:, :, output_start_y:output_end_y, output_start_x:output_end_x] = output_tile

    return output


def test_single_image(model, lr_img, device, args):
    """Test model on a single image"""
    model.eval()

    # Convert to tensor
    lr_tensor = torch.from_numpy(lr_img).unsqueeze(0).unsqueeze(0).float().to(device)

    # Process
    if lr_img.shape[0] > args.tile_size or lr_img.shape[1] > args.tile_size:
        # Use tiling for large images
        sr_tensor = tile_process(model, lr_tensor, args.scale_factor,
                                 args.tile_size, args.tile_overlap, args.window_size)
    else:
        # Process entire image at once
        _, _, h, w = lr_tensor.size()
        h_pad = (h // args.window_size + 1) * args.window_size - h if h % args.window_size != 0 else 0
        w_pad = (w // args.window_size + 1) * args.window_size - w if w % args.window_size != 0 else 0

        if h_pad > 0 or w_pad > 0:
            lr_tensor = torch.nn.functional.pad(lr_tensor, (0, w_pad, 0, h_pad), mode='reflect')

        with torch.no_grad():
            sr_tensor = model(lr_tensor)

        if h_pad > 0 or w_pad > 0:
            sr_tensor = sr_tensor[:, :, :h * args.scale_factor, :w * args.scale_factor]

    # Convert back to numpy
    sr_img = sr_tensor[0, 0].cpu().clamp(0, 1).numpy()

    return sr_img


def save_comparison_plot(results, save_path, idx):
    """Save comparison visualization"""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # Low resolution
    im1 = axes[0, 0].imshow(results['lr'], cmap='viridis', aspect='auto')
    axes[0, 0].set_title(f'Low Resolution ({results["lr"].shape[0]}×{results["lr"].shape[1]})')
    axes[0, 0].axis('off')
    plt.colorbar(im1, ax=axes[0, 0], fraction=0.046)

    # High resolution (Ground Truth)
    im2 = axes[0, 1].imshow(results['hr'], cmap='viridis', aspect='auto')
    axes[0, 1].set_title(f'High Resolution GT ({results["hr"].shape[0]}×{results["hr"].shape[1]})')
    axes[0, 1].axis('off')
    plt.colorbar(im2, ax=axes[0, 1], fraction=0.046)

    # Super Resolution result
    im3 = axes[0, 2].imshow(results['sr'], cmap='viridis', aspect='auto')
    axes[0, 2].set_title(f'Super Resolution ({results["sr"].shape[0]}×{results["sr"].shape[1]})')
    axes[0, 2].axis('off')
    plt.colorbar(im3, ax=axes[0, 2], fraction=0.046)

    # Difference
    diff = np.abs(results['hr'] - results['sr'])
    im4 = axes[1, 0].imshow(diff, cmap='hot', aspect='auto')
    axes[1, 0].set_title('Absolute Difference (HR - SR)')
    axes[1, 0].axis('off')
    plt.colorbar(im4, ax=axes[1, 0], fraction=0.046)

    # Zoomed regions
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

    # Add metrics
    metrics_text = f"PSNR: {results['metrics']['psnr']:.2f} dB\n"
    metrics_text += f"SSIM: {results['metrics']['ssim']:.4f}\n"
    metrics_text += f"Mean Temp Error: {results['metrics']['mae']:.3f} K\n"
    metrics_text += f"Max Temp Error: {results['metrics']['max_error']:.3f} K"

    fig.text(0.02, 0.02, metrics_text, fontsize=10,
             bbox=dict(boxstyle="round,pad=0.5", facecolor="white", alpha=0.8))

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'comparison_{idx:04d}.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    args = parse_args()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Create model with the same architecture as training
    print("Creating model...")
    model = SwinIR(
        upscale=args.scale_factor,
        in_chans=1,
        img_size=args.patch_size,
        window_size=args.window_size,
        img_range=1.,
        depths=[8, 8, 8, 8, 8, 8, 8, 8],
        embed_dim=240,
        num_heads=[8, 8, 8, 8, 8, 8, 8, 8],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='3conv'
    ).to(device)

    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint_info = load_checkpoint_flexible(model, args.model_path, device)
    print("Model loaded successfully!")

    if checkpoint_info.get('epoch'):
        print(f"Loaded model from epoch {checkpoint_info['epoch']}")
    if checkpoint_info.get('best_psnr'):
        print(f"Model's best PSNR during training: {checkpoint_info['best_psnr']:.2f}")

    model.eval()

    # Load test data
    print(f"Loading test data from {args.input_npz}")
    data = np.load(args.input_npz, allow_pickle=True)

    # Handle different data formats
    if 'temperature' in data and 'metadata' in data:
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

    # Limit number of test samples
    num_samples = min(args.num_samples, len(swaths))
    print(f"Testing on {num_samples} samples")

    # Test loop
    all_metrics = []

    for i in tqdm(range(num_samples), desc="Testing"):
        swath = swaths[i]
        temp_hr = swath['temperature'].astype(np.float32)
        meta = swath.get('metadata', {})

        # Remove NaN values
        if np.any(np.isnan(temp_hr)):
            temp_hr = np.nan_to_num(temp_hr, nan=np.nanmean(temp_hr))

        # Ensure size is divisible by scale factor
        h, w = temp_hr.shape
        h_new = h - h % args.scale_factor
        w_new = w - w % args.scale_factor
        temp_hr = temp_hr[:h_new, :w_new]

        # Store original range
        temp_min, temp_max = np.min(temp_hr), np.max(temp_hr)

        # Normalize to [0, 1]
        if temp_max > temp_min:
            temp_hr_norm = (temp_hr - temp_min) / (temp_max - temp_min)
        else:
            temp_hr_norm = np.zeros_like(temp_hr)

        # Create LR version
        h_lr, w_lr = h_new // args.scale_factor, w_new // args.scale_factor
        temp_lr_norm = cv2.resize(temp_hr_norm, (w_lr, h_lr), interpolation=cv2.INTER_AREA)

        # Super-resolve
        temp_sr_norm = test_single_image(model, temp_lr_norm, device, args)

        # Denormalize
        temp_lr = temp_lr_norm * (temp_max - temp_min) + temp_min
        temp_sr = temp_sr_norm * (temp_max - temp_min) + temp_min

        # Calculate metrics
        # For PSNR/SSIM, convert to uint8
        hr_uint8 = ((temp_hr_norm * 255).clip(0, 255)).astype(np.uint8)
        sr_uint8 = ((temp_sr_norm * 255).clip(0, 255)).astype(np.uint8)

        psnr = calculate_psnr(sr_uint8, hr_uint8, crop_border=0)
        ssim = calculate_ssim(sr_uint8, hr_uint8, crop_border=0)

        # Temperature-specific metrics
        mae = np.mean(np.abs(temp_sr - temp_hr))
        max_error = np.max(np.abs(temp_sr - temp_hr))
        rmse = np.sqrt(np.mean((temp_sr - temp_hr) ** 2))

        metrics = {
            'psnr': psnr,
            'ssim': ssim,
            'mae': mae,
            'max_error': max_error,
            'rmse': rmse
        }

        all_metrics.append(metrics)

        # Save results
        results = {
            'lr': temp_lr,
            'hr': temp_hr,
            'sr': temp_sr,
            'metrics': metrics,
            'metadata': meta
        }

        if args.save_comparison:
            save_comparison_plot(results, args.output_dir, i)

        # Save numpy arrays
        np_save_path = os.path.join(args.output_dir, f'result_{i:04d}.npz')
        np.savez(np_save_path,
                 lr=temp_lr,
                 hr=temp_hr,
                 sr=temp_sr,
                 lr_norm=temp_lr_norm,
                 hr_norm=temp_hr_norm,
                 sr_norm=temp_sr_norm,
                 metrics=metrics,
                 metadata=meta)

    # Calculate average metrics
    avg_metrics = {}
    for key in all_metrics[0].keys():
        avg_metrics[key] = np.mean([m[key] for m in all_metrics])
        avg_metrics[f'{key}_std'] = np.std([m[key] for m in all_metrics])

    print("\n=== Average Metrics ===")
    print(f"PSNR: {avg_metrics['psnr']:.2f} ± {avg_metrics['psnr_std']:.2f} dB")
    print(f"SSIM: {avg_metrics['ssim']:.4f} ± {avg_metrics['ssim_std']:.4f}")
    print(f"MAE: {avg_metrics['mae']:.3f} ± {avg_metrics['mae_std']:.3f} K")
    print(f"RMSE: {avg_metrics['rmse']:.3f} ± {avg_metrics['rmse_std']:.3f} K")
    print(f"Max Error: {avg_metrics['max_error']:.3f} ± {avg_metrics['max_error_std']:.3f} K")

    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'test_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("=== Test Results ===\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Test data: {args.input_npz}\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Scale factor: {args.scale_factor}\n")
        f.write(f"Tile size: {args.tile_size}\n")
        f.write(f"Window size: {args.window_size}\n\n")
        f.write("=== Average Metrics ===\n")
        for key in ['psnr', 'ssim', 'mae', 'rmse', 'max_error']:
            f.write(f"{key.upper()}: {avg_metrics[key]:.4f} ± {avg_metrics[f'{key}_std']:.4f}\n")

    print(f"\nResults saved to {args.output_dir}")


if __name__ == '__main__':
    main()