# train_temperature_sr.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
import argparse
import time
from datetime import datetime
import json
from tqdm import tqdm
import cv2

# Импорты из проекта
from models.network_swinir import SwinIR
from data.data_loader import create_train_val_dataloaders
from utils.util_calculate_psnr_ssim import calculate_psnr, calculate_ssim
from utils.logger import Logger
from utils.common import AverageMeter, save_checkpoint, load_checkpoint
from utils.temperature_loss import TemperatureAwareLoss, CharbonnierLoss, PhysicsConsistencyLoss, TemperaturePerceptualLoss


def define_model(args):
    """Определение модели SwinIR для температурных данных"""
    if args.patch_height is not None and args.patch_width is not None:
        img_size = (args.patch_height, args.patch_width)
    else:
        img_size = args.patch_size

    model = SwinIR(
        upscale=args.scale_factor,
        in_chans=1,  # Одноканальные данные
        img_size=img_size,
        window_size=8,
        img_range=1.,
        depths=[6, 6, 6, 6],
        embed_dim=180,
        num_heads=[6, 6, 6, 6],
        mlp_ratio=2,
        upsampler='nearest+conv',
        resi_connection='3conv'
    )
    return model


def train_one_epoch(model, train_loader, criteria, optimizer, epoch, logger, device):
    pixel_criterion, perceptual_criterion, perceptual_weight = criteria
    """Обучение одной эпохи"""
    model.train()

    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    # Прогресс бар
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

    for i, batch in enumerate(pbar):
        # Загружаем данные
        lq = batch['lq'].to(device)  # Low quality (LR)
        gt = batch['gt'].to(device)  # Ground truth (HR)

        # Forward pass
        sr = model(lq)

        # Calculate losses
        pixel_loss, pixel_loss_dict = pixel_criterion(sr, gt)
        perceptual_loss = perceptual_criterion(sr, gt)

        # Combine losses
        total_loss = pixel_loss + perceptual_weight * perceptual_loss

        # Create comprehensive loss dict
        loss_dict = pixel_loss_dict.copy()
        loss_dict['perceptual_loss'] = perceptual_loss.item()
        loss_dict['total_loss'] = total_loss.item()

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        #torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        # Вычисляем метрики
        with torch.no_grad():
            # Конвертируем в numpy для метрик
            sr_np = sr.detach().cpu().numpy()
            gt_np = gt.detach().cpu().numpy()

            psnr_val = 0
            ssim_val = 0

            # Вычисляем PSNR и SSIM для каждого изображения в батче
            for j in range(sr_np.shape[0]):
                sr_clamped = np.clip(sr_np[j, 0], 0, 1)
                gt_clamped = np.clip(gt_np[j, 0], 0, 1)

                sr_img = (sr_clamped * 255).astype(np.uint8)
                gt_img = (gt_clamped * 255).astype(np.uint8)

                psnr_val += calculate_psnr(sr_img, gt_img, crop_border=0)
                ssim_val += calculate_ssim(sr_img, gt_img, crop_border=0)

            psnr_val /= sr_np.shape[0]
            ssim_val /= sr_np.shape[0]

        # Обновляем метрики
        losses.update(loss_dict['total_loss'], lq.size(0))
        psnrs.update(psnr_val, lq.size(0))
        ssims.update(ssim_val, lq.size(0))

        # Обновляем прогресс бар
        pbar.set_postfix({
            'Loss': f'{losses.avg:.4f}',
            'PSNR': f'{psnrs.avg:.2f}',
            'SSIM': f'{ssims.avg:.4f}'
        })

        # Логирование каждые 100 итераций
        if i % 100 == 0:
            logger.log_training(epoch, i, losses.avg, psnrs.avg, ssims.avg)

    return losses.avg, psnrs.avg, ssims.avg


def validate(model, val_loader, criteria, epoch, logger, device):
    pixel_criterion, perceptual_criterion, perceptual_weight = criteria
    """Валидация модели"""
    model.eval()

    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Validation {epoch}')

        for i, batch in enumerate(pbar):
            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)

            # Forward pass
            sr = model(lq)

            # Calculate losses (same as training)
            pixel_loss, pixel_loss_dict = pixel_criterion(sr, gt)
            perceptual_loss = perceptual_criterion(sr, gt)
            total_loss = pixel_loss + perceptual_weight * perceptual_loss

            loss_dict = pixel_loss_dict.copy()
            loss_dict['perceptual_loss'] = perceptual_loss.item()
            loss_dict['total_loss'] = total_loss.item()

            # Метрики
            sr_np = sr.cpu().numpy()
            gt_np = gt.cpu().numpy()

            for j in range(sr_np.shape[0]):
                sr_clamped = np.clip(sr_np[j, 0], 0, 1)
                gt_clamped = np.clip(gt_np[j, 0], 0, 1)

                sr_img = (sr_clamped * 255).astype(np.uint8)
                gt_img = (gt_clamped * 255).astype(np.uint8)

                psnr_val = calculate_psnr(sr_img, gt_img, crop_border=0)
                ssim_val = calculate_ssim(sr_img, gt_img, crop_border=0)

                losses.update(total_loss.item(), 1)
                psnrs.update(psnr_val, 1)
                ssims.update(ssim_val, 1)

            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'PSNR': f'{psnrs.avg:.2f}',
                'SSIM': f'{ssims.avg:.4f}'
            })

    logger.log_validation(epoch, losses.avg, psnrs.avg, ssims.avg)

    return losses.avg, psnrs.avg, ssims.avg

'''


def validate(model, val_loader, criteria, epoch, logger, device):
    pixel_criterion, perceptual_criterion, perceptual_weight = criteria
    """Validation on full images"""
    model.eval()

    losses = AverageMeter()
    psnrs = AverageMeter()
    ssims = AverageMeter()

    # Get the dataset from the dataloader
    val_dataset = val_loader.dataset

    # Process only a subset of full images for validation
    num_val_images = min(10, len(val_dataset.swaths))

    with torch.no_grad():
        pbar = tqdm(range(num_val_images), desc=f'Validation {epoch}')

        for idx in pbar:
            # Get full image split into patches
            patches, positions, full_size, metadata = val_dataset.get_full_image_for_validation(idx)
            h_full, w_full = full_size

            # Create full resolution output image
            sr_full = np.zeros((h_full, w_full), dtype=np.float32)
            weight_map = np.zeros((h_full, w_full), dtype=np.float32)

            # Process each patch
            for patch, (y, x) in zip(patches, positions):
                # CREATE LOW-RESOLUTION VERSION OF THE PATCH
                patch_lr = cv2.resize(patch, (patch.shape[1] // 4, patch.shape[0] // 4), interpolation=cv2.INTER_AREA)

                # Convert to tensor and add batch/channel dimensions
                patch_tensor = torch.from_numpy(patch_lr).unsqueeze(0).unsqueeze(0).float().to(device)

                # Process through model (this will upscale back to original size)
                sr_patch = model(patch_tensor)
                sr_patch = sr_patch[0, 0].cpu().numpy()

                # Add to full image with blending for overlaps
                sr_full[y:y + 512, x:x + 128] += sr_patch
                weight_map[y:y + 512, x:x + 128] += 1.0

            # Normalize by weight map to handle overlaps
            sr_full = sr_full / np.maximum(weight_map, 1.0)
            sr_full = np.clip(sr_full, 0, 1)

            # Use the original full image as ground truth
            swath = val_dataset.swaths[idx]
            temp = swath['temperature'].astype(np.float32)
            temp = cv2.resize(temp, (208, 2000), interpolation=cv2.INTER_LINEAR)
            temp_min, temp_max = np.min(temp), np.max(temp)
            temp_norm = (temp - temp_min) / (temp_max - temp_min) if temp_max > temp_min else np.zeros_like(temp)

            # Convert to tensors for loss calculation
            sr_tensor = torch.from_numpy(sr_full).unsqueeze(0).unsqueeze(0).float().to(device)
            gt_tensor = torch.from_numpy(temp_norm).unsqueeze(0).unsqueeze(0).float().to(device)

            # Calculate losses on full images
            pixel_loss, pixel_loss_dict = pixel_criterion(sr_tensor, gt_tensor)
            perceptual_loss = perceptual_criterion(sr_tensor, gt_tensor)
            total_loss = pixel_loss + perceptual_weight * perceptual_loss

            # Calculate metrics on full images
            sr_img = (sr_full * 255).astype(np.uint8)
            gt_img = (temp_norm * 255).astype(np.uint8)

            psnr_val = calculate_psnr(sr_img, gt_img, crop_border=0)
            ssim_val = calculate_ssim(sr_img, gt_img, crop_border=0)

            losses.update(total_loss.item(), 1)
            psnrs.update(psnr_val, 1)
            ssims.update(ssim_val, 1)

            pbar.set_postfix({
                'Loss': f'{losses.avg:.4f}',
                'PSNR': f'{psnrs.avg:.2f}',
                'SSIM': f'{ssims.avg:.4f}'
            })

    logger.log_validation(epoch, losses.avg, psnrs.avg, ssims.avg)
    return losses.avg, psnrs.avg, ssims.avg
'''

def main(args):
    """Основная функция обучения"""
    # Создаем директории
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'logs'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'samples'), exist_ok=True)

    # Инициализируем логгер
    logger = Logger(os.path.join(args.output_dir, 'logs'))

    # Устройство
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Создаем датасеты
    # Get all NPZ files in the data directory
    # Get all NPZ files in the data directory
    import glob
    all_npz_files = glob.glob(os.path.join(args.data_dir, '*.npz'))
    if not all_npz_files:
        raise FileNotFoundError(f"No NPZ files found in {args.data_dir}")

    # Limit number of files if specified
    if args.max_files is not None and args.max_files < len(all_npz_files):
        all_npz_files = all_npz_files[:args.max_files]
        print(f"Limited to first {args.max_files} files")

    print(f"Using {len(all_npz_files)} NPZ files: {[os.path.basename(f) for f in all_npz_files]}")

    # Use most files for training, last one for validation
    train_files = all_npz_files[:-1] if len(all_npz_files) > 1 else all_npz_files
    val_file = all_npz_files[-1]

    print("Creating data loaders...")
    train_loader, val_loader = create_train_val_dataloaders(
        train_files, val_file,
        batch_size=args.batch_size,
        scale_factor=args.scale_factor,
        patch_size=args.patch_size,
        patch_height=args.patch_height,
        patch_width=args.patch_width
    )

    # Создаем модель
    print("Creating model...")
    model = define_model(args).to(device)

    # For 2 GPU
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
        print(f"Using {torch.cuda.device_count()} GPUs")

    # Подсчет параметров
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Loss function и optimizer
    # Loss functions
    pixel_criterion = PhysicsConsistencyLoss(gradient_weight=0.1, smoothness_weight=0.05)
    perceptual_criterion = TemperaturePerceptualLoss(feature_weights=[0.1, 0.2, 1.0, 1.0]).to(device)
    perceptual_weight = 0.1
    optimizer = optim.Adam(model.parameters(), lr=args.lr, betas=(0.9, 0.99))

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-7
    )

    # Загружаем checkpoint если есть
    start_epoch = 0
    best_psnr = 0

    if args.resume:
        checkpoint_path = os.path.join(args.output_dir, 'checkpoints', 'latest.pth')
        if os.path.exists(checkpoint_path):
            start_epoch, best_psnr = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
            print(f"Resumed from epoch {start_epoch} with best PSNR {best_psnr:.2f}")

    # Сохраняем конфигурацию
    config = vars(args)
    config['model_params'] = total_params
    with open(os.path.join(args.output_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=4)

    # Основной цикл обучения
    print("\nStarting training...")
    for epoch in range(start_epoch, args.num_epochs):
        print(f"\n{'=' * 50}")
        print(f"Epoch {epoch + 1}/{args.num_epochs}")
        print(f"Learning rate: {optimizer.param_groups[0]['lr']:.2e}")
        print(f"{'=' * 50}")

        # Обучение
        train_loss, train_psnr, train_ssim = train_one_epoch(
            model, train_loader, (pixel_criterion, perceptual_criterion, perceptual_weight), optimizer, epoch + 1,
            logger, device
        )

        # Валидация
        val_loss, val_psnr, val_ssim = validate(
            model, val_loader, (pixel_criterion, perceptual_criterion, perceptual_weight), epoch + 1, logger, device
        )

        # Обновляем learning rate
        scheduler.step()

        # Сохраняем checkpoint
        is_best = val_psnr > best_psnr
        if is_best:
            best_psnr = val_psnr

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'best_psnr': best_psnr,
            'args': args
        }, is_best, args.output_dir)

        # Сохраняем примеры каждые 5 эпох
        if (epoch + 1) % 5 == 0:
            save_sample_images(model, val_loader, epoch + 1, args.output_dir, device)

        # Выводим итоги эпохи
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train - Loss: {train_loss:.4f}, PSNR: {train_psnr:.2f}, SSIM: {train_ssim:.4f}")
        print(f"  Val   - Loss: {val_loss:.4f}, PSNR: {val_psnr:.2f}, SSIM: {val_ssim:.4f}")
        print(f"  Best PSNR: {best_psnr:.2f}")

    print("\nTraining completed!")
    print(f"Best validation PSNR: {best_psnr:.2f}")


def save_sample_images(model, val_loader, epoch, output_dir, device, num_samples=4):
    """Сохранение примеров результатов"""
    model.eval()
    samples_dir = os.path.join(output_dir, 'samples', f'epoch_{epoch}')
    os.makedirs(samples_dir, exist_ok=True)

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            if i >= num_samples:
                break

            lq = batch['lq'].to(device)
            gt = batch['gt'].to(device)
            sr = model(lq)

            # Конвертируем в numpy и сохраняем
            lq_np = (lq[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            gt_np = (gt[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            sr_np = (sr[0, 0].cpu().numpy() * 255).clip(0, 255).astype(np.uint8)

            # Создаем композитное изображение
            h_lr, w_lr = lq_np.shape
            h_hr, w_hr = gt_np.shape

            # Апсэмплим LR для визуализации
            lq_up = cv2.resize(lq_np, (w_hr, h_hr), interpolation=cv2.INTER_NEAREST)

            # Соединяем изображения горизонтально
            composite = np.hstack([lq_up, sr_np, gt_np])

            # Сохраняем
            cv2.imwrite(os.path.join(samples_dir, f'sample_{i + 1}.png'), composite)

            # Также сохраняем отдельные изображения
            cv2.imwrite(os.path.join(samples_dir, f'sample_{i + 1}_lr.png'), lq_np)
            cv2.imwrite(os.path.join(samples_dir, f'sample_{i + 1}_sr.png'), sr_np)
            cv2.imwrite(os.path.join(samples_dir, f'sample_{i + 1}_hr.png'), gt_np)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Temperature SR Training')

    # Paths
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to data directory with NPZ files')
    parser.add_argument('--output_dir', type=str, default='./experiments/temperature_sr',
                        help='Path to save outputs')

    parser.add_argument('--max_files', type=int, default=None,
                        help='Maximum number of NPZ files to process (default: all files)')

    # Model parameters
    parser.add_argument('--scale_factor', type=int, default=4,
                        help='Super-resolution scale factor')
    parser.add_argument('--patch_size', type=int, default=128,
                        help='Training patch size')
    parser.add_argument('--patch_height', type=int, default=512,
                        help='Training patch height (defaults to patch_size if not set)')
    parser.add_argument('--patch_width', type=int, default=128,
                        help='Training patch width (defaults to patch_size if not set)')

    # Training parameters
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Number of epochs')
    parser.add_argument('--lr', type=float, default=2e-4,
                        help='Learning rate')
    parser.add_argument('--resume', action='store_true',
                        help='Resume from checkpoint')
    parser.add_argument('--debug', action='store_true',
                        help='Debug mode')
    parser.add_argument('--launcher', type=str, default='none',
                        choices=['none', 'pytorch'],
                        help='Launcher type')
    parser.add_argument('--local_rank', type=int, default=0,
                        help='Local rank for distributed training')

    args = parser.parse_args()
    main(args)