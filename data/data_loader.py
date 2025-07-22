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

'''
SLOW
class TemperatureDataset(Dataset):
    def __init__(self, npz_file: str, scale_factor: int = 4,
                 patch_size: int = 128, max_samples: Optional[int] = None,
                 phase: str = 'train', patch_height: int = 512, patch_width: int = 128):
        self.npz_file = npz_file
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.phase = phase
        self.max_samples = max_samples
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Initialize degradation
        self.degradation = TemperatureDegradation(scale_factor=scale_factor)

        # ONLY store indices, not actual data
        print(f"Loading metadata from {npz_file}...")
        data = np.load(npz_file, allow_pickle=True)

        if 'swaths' in data:
            swaths = data['swaths']
        elif 'swath_array' in data:
            swaths = data['swath_array']
        else:
            raise KeyError(f"Neither 'swaths' nor 'swath_array' found in {npz_file}")

        # Calculate total number of patches WITHOUT loading data
        self.patch_indices = []  # [(swath_idx, patch_y, patch_x), ...]

        n_samples = len(swaths) if max_samples is None else min(len(swaths), max_samples)

        for i in range(n_samples):
            # Calculate how many patches this swath will generate
            patch_height, patch_width = 512, 128
            overlap = 32
            stride_h, stride_w = patch_height - overlap, patch_width - overlap

            # Assume standard size after resize
            img_h, img_w = 2000, 208

            for y in range(0, img_h - patch_height + 1, stride_h):
                for x in range(0, img_w - patch_width + 1, stride_w):
                    self.patch_indices.append((i, y, x))

        data.close()
        print(f"Dataset ready with {len(self.patch_indices)} patches")

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        swath_idx, patch_y, patch_x = self.patch_indices[idx]

        # Load data on-demand
        data = np.load(self.npz_file, allow_pickle=True)
        if 'swaths' in data:
            swaths = data['swaths']
        elif 'swath_array' in data:
            swaths = data['swath_array']

        swath = swaths[swath_idx]
        temp = swath['temperature'].astype(np.float32)

        # Process the temperature data
        mask = np.isnan(temp)
        if mask.any():
            mean_val = np.nanmean(temp)
            temp[mask] = mean_val

        # Resize
        temp = cv2.resize(temp, (208, 2000), interpolation=cv2.INTER_LINEAR)

        # Normalize
        temp_min, temp_max = np.min(temp), np.max(temp)
        if temp_max > temp_min:
            temp_norm = (temp - temp_min) / (temp_max - temp_min)
        else:
            temp_norm = np.zeros_like(temp)

        # Extract the specific patch
        patch_height, patch_width = 512, 128
        patch = temp_norm[patch_y:patch_y + patch_height, patch_x:patch_x + patch_width]

        # Apply degradation
        if self.phase == 'train':
            temp_lr_patch, temp_hr_patch = self.degradation.degradation_bsrgan_rect(
                patch,
                lq_patchsize_h=patch_height // self.scale_factor,
                lq_patchsize_w=patch_width // self.scale_factor
            )
        else:
            h, w = patch.shape
            temp_lr_patch = cv2.resize(patch,
                                       (w // self.scale_factor, h // self.scale_factor),
                                       interpolation=cv2.INTER_AREA)
            temp_hr_patch = patch

        # Convert to tensors
        lr_tensor = torch.from_numpy(temp_lr_patch).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(temp_hr_patch).unsqueeze(0).float()

        data.close()  # Important: close the file

        return {
            'lq': lr_tensor,
            'gt': hr_tensor,
            'lq_path': f'{self.npz_file}_{idx}',
            'gt_path': f'{self.npz_file}_{idx}'
        }

----------------------
FAST
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
        # Подготавливаем список температурных массивов
        self.temperature_patches = []
        self.metadata_list = []

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

            # Resize to 2000x208
            h, w = temp.shape
            if h > 2000 and w > 208:
                temp = cv2.resize(temp, (208, 2000), interpolation=cv2.INTER_LINEAR)

            # Нормализация в [0, 1]
            temp_min, temp_max = np.min(temp), np.max(temp)
            if temp_max > temp_min:
                temp_norm = (temp - temp_min) / (temp_max - temp_min)
            else:
                temp_norm = np.zeros_like(temp)

            # Split into patches (e.g., 512x128 with small overlap)
            patch_width = 128
            patch_height = 512
            overlap = 32  # Small overlap between patches

            # Calculate number of patches needed
            h, w = temp_norm.shape
            stride_w = patch_width - overlap
            stride_h = patch_height - overlap

            for y in range(0, h - patch_height + 1, stride_h):
                for x in range(0, w - patch_width + 1, stride_w):
                    patch = temp_norm[y:y + patch_height, x:x + patch_width]
                    self.temperature_patches.append(patch)
                    self.metadata_list.append({
                        'original_min': temp_min,
                        'original_max': temp_max,
                        'orbit_type': swath['metadata'].get('orbit_type', 'unknown'),
                        'patch_idx': len(self.temperature_patches) - 1,
                        'source_idx': i
                    })

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{n_samples} samples, total patches: {len(self.temperature_patches)}")

        data.close()
        gc.collect()

    def get_full_image_for_validation(self, idx):
        """Get full image split into patches for validation"""
        swath = self.swaths[idx]
        temp = swath['temperature'].astype(np.float32)

        # Resize to standard size
        temp = cv2.resize(temp, (208, 2000), interpolation=cv2.INTER_LINEAR)

        # Normalize
        temp_min, temp_max = np.min(temp), np.max(temp)
        temp_norm = (temp - temp_min) / (temp_max - temp_min) if temp_max > temp_min else np.zeros_like(temp)

        # Split into patches
        patches = []
        positions = []
        h, w = temp_norm.shape

        for y in range(0, h, 512):  # Step by patch height (512)
            for x in range(0, w, 128):  # Step by patch width (128)
                if y + 512 <= h and x + 128 <= w:
                    patch = temp_norm[y:y + 512, x:x + 128]  # 512x128 patch
                    patches.append(patch)
                    positions.append((y, x))

        return patches, positions, (h, w), {'original_min': temp_min, 'original_max': temp_max}

    def __len__(self):
        return len(self.temperature_patches)

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
        # Получаем патч
        temp_hr_patch = self.temperature_patches[idx]
        meta = self.metadata_list[idx]

        if self.phase == 'train':
            # Применяем деградацию
            '' Comment BSRGAN
            temp_lr_patch, temp_hr_patch = self.degradation.degradation_bsrgan_rect(
                temp_hr_patch,
                lq_patchsize_h=temp_hr_patch.shape[0] // self.scale_factor,
                lq_patchsize_w=temp_hr_patch.shape[1] // self.scale_factor
            )
             Comment BSRGAN''
            # Replace the BSRGAN degradation with simple downsampling
            h, w = temp_hr_patch.shape
            temp_lr_patch = cv2.resize(temp_hr_patch,
                                       (w // self.scale_factor, h // self.scale_factor),
                                       interpolation=cv2.INTER_AREA)


        else:
            # Простой даунсэмплинг для валидации
            h, w = temp_hr_patch.shape
            temp_lr_patch = cv2.resize(temp_hr_patch,
                                       (w // self.scale_factor, h // self.scale_factor),
                                       interpolation=cv2.INTER_AREA)

        # Конвертируем в тензоры и добавляем канальное измерение
        lr_tensor = torch.from_numpy(temp_lr_patch).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(temp_hr_patch).unsqueeze(0).float()

        return {
            'lq': lr_tensor,
            'gt': hr_tensor,
            'lq_path': f'{self.npz_file}_{idx}',
            'gt_path': f'{self.npz_file}_{idx}'
        }
'''


class TemperatureDataset(Dataset):
    def __init__(self, npz_file: str, scale_factor: int = 4,
                 patch_size: int = 128, max_samples: Optional[int] = None,
                 phase: str = 'train', patch_height: int = 512, patch_width: int = 128):
        self.npz_file = npz_file
        self.scale_factor = scale_factor
        self.patch_size = patch_size
        self.phase = phase
        self.max_samples = max_samples
        self.patch_height = patch_height
        self.patch_width = patch_width

        # Initialize degradation
        self.degradation = TemperatureDegradation(scale_factor=scale_factor)

        # Don't open file here, just get metadata
        self._npz_file = None  # Will be opened lazily

        # Quick open just to get structure
        with np.load(npz_file, allow_pickle=True) as data:
            if 'swaths' in data:
                self.swaths_key = 'swaths'
            elif 'swath_array' in data:
                self.swaths_key = 'swath_array'
            else:
                raise KeyError(f"Neither 'swaths' nor 'swath_array' found in {npz_file}")

            swaths = data[self.swaths_key]
            n_samples = len(swaths) if max_samples is None else min(len(swaths), max_samples)

        # Calculate patch indices (same as before)
        self.patch_indices = []
        patch_height, patch_width = 512, 128
        overlap = 32
        stride_h, stride_w = patch_height - overlap, patch_width - overlap
        img_h, img_w = 2000, 208

        for i in range(n_samples):
            for y in range(0, img_h - patch_height + 1, stride_h):
                for x in range(0, img_w - patch_width + 1, stride_w):
                    self.patch_indices.append((i, y, x))

        print(f"Dataset ready with {len(self.patch_indices)} patches")

    def _ensure_file_open(self):
        """Open file if not already open (per-worker)"""
        if self._npz_file is None:
            self._npz_file = np.load(self.npz_file, allow_pickle=True, mmap_mode='r')

    def __len__(self):
        return len(self.patch_indices)

    def __getitem__(self, idx):
        # Ensure file is open in this worker
        self._ensure_file_open()

        swath_idx, patch_y, patch_x = self.patch_indices[idx]

        # Access swath directly from memory-mapped file
        swaths = self._npz_file[self.swaths_key]
        swath = swaths[swath_idx]
        temp = swath['temperature'].astype(np.float32)

        # Process the temperature data
        mask = np.isnan(temp)
        if mask.any():
            mean_val = np.nanmean(temp)
            temp[mask] = mean_val

        # Resize
        temp = cv2.resize(temp, (208, 2000), interpolation=cv2.INTER_LINEAR)

        # Normalize
        temp_min, temp_max = np.min(temp), np.max(temp)
        if temp_max > temp_min:
            temp_norm = (temp - temp_min) / (temp_max - temp_min)
        else:
            temp_norm = np.zeros_like(temp)

        # Extract patch
        patch_height, patch_width = 512, 128
        patch = temp_norm[patch_y:patch_y + patch_height, patch_x:patch_x + patch_width]

        # Apply degradation
        if self.phase == 'train':
            temp_lr_patch, temp_hr_patch = self.degradation.degradation_bsrgan_rect(
                patch,
                lq_patchsize_h=patch_height // self.scale_factor,
                lq_patchsize_w=patch_width // self.scale_factor
            )
        else:
            h, w = patch.shape
            temp_lr_patch = cv2.resize(patch,
                                       (w // self.scale_factor, h // self.scale_factor),
                                       interpolation=cv2.INTER_AREA)
            temp_hr_patch = patch

        # Convert to tensors
        lr_tensor = torch.from_numpy(temp_lr_patch).unsqueeze(0).float()
        hr_tensor = torch.from_numpy(temp_hr_patch).unsqueeze(0).float()

        return {
            'lq': lr_tensor,
            'gt': hr_tensor,
            'lq_path': f'{self.npz_file}_{idx}',
            'gt_path': f'{self.npz_file}_{idx}'
        }

    def __del__(self):
        if hasattr(self, '_npz_file'):
            self._npz_file.close()

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
        samples_per_file=1000
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