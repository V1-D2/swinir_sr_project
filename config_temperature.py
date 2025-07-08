# Конфигурация для обучения температурной Super-Resolution модели

# Общие параметры
name = 'TemperatureSR_SwinIR_x4'
model_type = 'TemperatureSRModel'
scale = 4
num_gpu = 1  # Количество GPU

# Параметры данных
datasets = {
    'train': {
        'name': 'TemperatureTrainDataset',
        'dataroot_gt': None,  # Будет задан в train script
        'npz_files': [],  # Будет задан в train script
        'preprocessor_args': {
            'target_height': 2000,
            'target_width': 220
        },
        'scale_factor': 4,
        'batch_size': 16,
        'samples_per_file': 10000,  # Ограничение для управления памятью
        'num_worker': 4,
        'pin_memory': True,
        'persistent_workers': True
    },
    'val': {
        'name': 'TemperatureValDataset',
        'dataroot_gt': None,
        'npz_file': None,  # Будет задан в train script
        'n_samples': 10,
        'scale_factor': 8
    }
}

# Параметры сети
network_g = {
    'type': 'SwinIR',
    'upscale': 4,
    'in_chans': 1,  # Температурные данные - 1 канал
    'img_size': 64,
    'window_size': 8,
    'img_range': 1.,
    'depths': [6, 6, 6, 6, 6, 6, 6, 6],
    'embed_dim': 180,
    'num_heads': [8, 8, 8, 8, 8, 8, 8, 8],
    'mlp_ratio': 2,
    'upsampler': 'pixelshuffle',
    'resi_connection': '3conv'
}

network_d = None

# Путь к файлам
path = {
    'pretrain_network_g': None,
    'strict_load_g': True,
    'resume_state': None,
    'root': './',
    'experiments_root': './experiments',
    'models': './experiments/models',
    'training_states': './experiments/training_states',
    'log': './experiments/log',
    'visualization': './experiments/visualization'
}

# Параметры обучения
train = {
    'ema_decay': 0.999,
    'optim_g': {
        'type': 'Adam',
        'lr': 2e-4,
        'weight_decay': 0,
        'betas': [0.9, 0.99]
    },
    'scheduler': {
        'type': 'CosineAnnealingLR',
        'T_max': 70000,
        'eta_min': 1e-6
    },
    # Loss функции
    'pixel_opt': {
        'type': 'PhysicsConsistencyLoss',
        'loss_weight': 1.0,
        'gradient_weight': 0.01,
        'smoothness_weight': 0.05,
        'reduction': 'mean'
    },
    'perceptual_opt': {
        'type': 'TemperaturePerceptualLoss',
        'loss_weight': 0.1,
        'feature_weights': [0.1, 0.1, 1.0, 1.0]
    },
    # Частота сохранения
    'manual_seed': 10,
    'use_grad_clip': True,
    'grad_clip_norm': 0.5,
    'use_ema': True                 # Exponential Moving Averag
}

# Параметры валидации
val = {
    'val_freq': 2000,
    'save_img': True,
    'metrics': {
        'psnr': {
            'type': 'calculate_psnr',
            'crop_border': 0,
            'test_y_channel': False
        },
        'ssim': {
            'type': 'calculate_ssim',
            'crop_border': 0,
            'test_y_channel': False
        }
    }
}

# Логирование
logger = {
    'print_freq': 1000,
    'save_checkpoint_freq': 100000,
    'use_tb_logger': True,
    'wandb': {
        'project': 'temperature-sr',
        'resume_id': None
    }
}

# Распределенное обучение
dist_params = {
    'backend': 'nccl',
    'port': 29500
}

# Специфичные для температурных данных параметры
temperature_specific = {
    'preserve_relative_values': True,
    'temperature_range': [80, 400],  # Кельвины
    'physical_constraints': {
        'enforce_smoothness': True,
        'preserve_gradients': True,
        'max_gradient': 10.0  # Максимальный градиент температуры
    }
}

# Инкрементальное обучение
incremental_training = {
    'enabled': True,
    'epochs_per_file': 1,
    'learning_rate_decay_per_file': 1.0,
    'checkpoint_per_file': False,
    'shuffle_files': True
}

# Дополнительные параметры
others = {
    'use_amp': False,  # Automatic Mixed Precision
    'num_threads': 8,
    'seed': 10
}