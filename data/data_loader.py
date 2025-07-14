# data/data_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
import cv2
from typing import Dict, Tuple, List, Optional
import gc
import sys

sys.path.append('..')
from models.degradation_bsrgan import TemperatureDegradation


class TemperatureDataset(Dataset):
    """Dataset для температурных данных с BSRGAN деградацией"""

    def __init__(self, npz_file: str, scale_factor: int = 4,
                 patch_size: int = 128, max_samples: Optional[int] = None,
                 phase: str = 'train', patch_height: int = 800, patch_width: int = 200):
        self.npz_file = npz_file
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.phase = phase
        self.max_samples = max_samples

        if patch_height is not None and patch_width is not None:
            self.patch_height = patch_height
            self.patch_width = patch_width
        else:
            self.patch_height = patch_size
            self.patch_width = patch_size


        if patch_size < 32:
            raise ValueError(f"Patch size {patch_size} is too small. Minimum is 32.")
        if patch_size % scale_factor != 0:
            raise ValueError(f"Patch size {patch_size} must be divisible by scale factor {scale_factor}")

        # Инициализируем деградацию BSRGAN
        self.degradation = TemperatureDegradation(scale_factor=scale_factor)

        # Загружаем данные
        print(f"Loading {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)
        if 'swaths' in data:
            self.swaths = data['swaths']
        elif 'swath_array' in data:
            self.swaths = data['swath_array']
        else:
            raise KeyError(f"Neither 'swaths' nor 'swath_array' found in {npz_file}")

        # Подготавливаем список температурных массивов
        self.temperatures = []
        self.metadata = []

        n_samples = len(self.swaths) if max_samples is None else min(len(self.swaths), max_samples)

        print(f"Preprocessing {n_samples} samples...")
        for i in range(n_samples):
            swath = self.swaths[i]
            temp = swath['temperature'].astype(np.float32)

            # Удаляем NaN
            mask = np.isnan(temp)
            if mask.any():
                mean_val = np.nanmean(temp)
                temp[mask] = mean_val

            # Нормализация в [0, 1]
            temp_min, temp_max = np.min(temp), np.max(temp)
            if temp_max > temp_min:
                temp_norm = (temp - temp_min) / (temp_max - temp_min)
            else:
                temp_norm = np.zeros_like(temp)

            self.temperatures.append(temp_norm)
            self.metadata.append({
                'original_min': temp_min,
                'original_max': temp_max,
                'orbit_type': swath['metadata'].get('orbit_type', 'unknown')
            })

            if (i + 1) % 1000 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples")

        data.close()
        gc.collect()

    def __len__(self):
        return len(self.temperatures)

    def random_crop(self, img: np.ndarray, patch_size: int) -> np.ndarray:
        """Случайный кроп патча из изображения"""
        h, w = img.shape
        if h < patch_size or w < patch_size:
            # Паддинг если изображение меньше patch_size
            pad_h = max(0, patch_size - h)
            pad_w = max(0, patch_size - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape

        top = np.random.randint(0, h - patch_size + 1)
        left = np.random.randint(0, w - patch_size + 1)

        return img[top:top + patch_size, left:left + patch_size]

    def random_crop_rect(self, img: np.ndarray, patch_height: int, patch_width: int) -> np.ndarray:
        """Random crop of rectangular patch from image"""
        h, w = img.shape
        if h < patch_height or w < patch_width:
            # Padding logic - adjust for both dimensions
            pad_h = max(0, patch_height - h)
            pad_w = max(0, patch_width - w)
            img = np.pad(img, ((0, pad_h), (0, pad_w)), mode='reflect')
            h, w = img.shape

        top = np.random.randint(0, h - patch_height + 1)
        left = np.random.randint(0, w - patch_width + 1)

        return img[top:top + patch_height, left:left + patch_width]

    def __getitem__(self, idx):
        # Получаем температурный массив
        temp_hr = self.temperatures[idx]
        meta = self.metadata[idx]

        if self.phase == 'train':
            # Для обучения - случайный кроп
            temp_hr_patch = self.random_crop_rect(temp_hr, self.patch_height, self.patch_width)

            # Применяем деградацию (lq_patchsize should be patch_size for HR, it gets divided internally)
            temp_lr_patch, temp_hr_patch = self.degradation.degradation_bsrgan_rect(
                temp_hr_patch,
                lq_patchsize_h=self.patch_height // self.scale_factor,
                lq_patchsize_w=self.patch_width // self.scale_factor
            )
        else:
            # Для валидации - центральный кроп или весь массив
            h, w = temp_hr.shape
            if h > self.patch_size and w > self.patch_size:
                top = (h - self.patch_size) // 2
                left = (w - self.patch_size) // 2
                temp_hr_patch = temp_hr[top:top + self.patch_size, left:left + self.patch_size]
            else:
                temp_hr_patch = temp_hr

            # Применяем фиксированную деградацию для валидации
            h, w = temp_hr_patch.shape
            h = h - h % self.scale_factor
            w = w - w % self.scale_factor
            temp_hr_patch = temp_hr_patch[:h, :w]

            # Простой даунсэмплинг для валидации
            temp_lr_patch = cv2.resize(temp_hr_patch,
                                       (w // self.scale_factor, h // self.scale_factor),
                                       interpolation=cv2.INTER_AREA)

        # Конвертируем в тензоры и добавляем канальное измерение
        lr_tensor = torch.from_numpy(temp_lr_patch).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(temp_hr_patch).unsqueeze(0).float()

        return {
            'lq': lr_tensor,  # low quality (low resolution)
            'gt': hr_tensor,  # ground truth (high resolution)
            'lq_path': f'{self.npz_file}_{idx}',
            'gt_path': f'{self.npz_file}_{idx}'
        }


class MultiFileDataLoader:
    """Загрузчик для работы с несколькими NPZ файлами"""

    def __init__(self, npz_files: List[str], batch_size: int = 4,
                 scale_factor: int = 4, patch_size: int = 128,
                 samples_per_file: Optional[int] = None, phase: str = 'train',
                 patch_height: int = None, patch_width: int = None):
        self.npz_files = npz_files
        self.batch_size = batch_size
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.samples_per_file = samples_per_file
        self.phase = phase
        self.current_file_idx = 0

        self.patch_height = patch_height
        self.patch_width = patch_width

    def get_combined_dataloader(self) -> DataLoader:
        """Создает единый DataLoader для всех файлов"""
        all_datasets = []

        for npz_file in self.npz_files:
            dataset = TemperatureDataset(
                npz_file,
                scale_factor=self.scale_factor,
                patch_size=self.patch_size,
                max_samples=self.samples_per_file,
                phase=self.phase,
                patch_height=self.patch_height,
                patch_width=self.patch_width
            )
            all_datasets.append(dataset)

        # Объединяем все датасеты
        combined_dataset = torch.utils.data.ConcatDataset(all_datasets)

        return DataLoader(
            combined_dataset,
            batch_size=self.batch_size,
            shuffle=(self.phase == 'train'),
            num_workers=4,
            pin_memory=True,
            drop_last=(self.phase == 'train')
        )


def create_train_val_dataloaders(train_files: List[str], val_file: str,
                                 batch_size: int = 4, scale_factor: int = 4,
                                 patch_size: int = 128, patch_height: int = None,
                                 patch_width: int = None) -> Tuple[DataLoader, DataLoader]:
    """Создание train и validation датлоадеров"""

    # Training dataloader
    train_loader = MultiFileDataLoader(
        train_files,
        batch_size=batch_size,
        scale_factor=scale_factor,
        patch_size=patch_size,
        phase='train',
        patch_height=patch_height,
        patch_width=patch_width,
        samples_per_file=10000
    ).get_combined_dataloader()

    # Validation dataloader
    val_dataset = TemperatureDataset(
        val_file,
        scale_factor=scale_factor,
        patch_size=patch_size,
        max_samples=100,  # Используем только 100 примеров для валидации
        phase='val',
        patch_height=patch_height,
        patch_width=patch_width
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )

    return train_loader, val_loader