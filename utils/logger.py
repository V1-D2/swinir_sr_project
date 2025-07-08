# utils/logger.py
import os
import sys
import datetime
from torch.utils.tensorboard import SummaryWriter


class Logger:
    """Логгер для отслеживания процесса обучения"""

    def __init__(self, log_dir):
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir)

        # Текстовый лог
        self.log_file = os.path.join(log_dir, 'training.log')

        # Начальное сообщение
        self.log_message(f"Training started at {datetime.datetime.now()}")
        self.log_message("=" * 50)

    def log_message(self, message):
        """Записать сообщение в лог"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.datetime.now()}: {message}\n")
        print(message)

    def log_training(self, epoch, iteration, loss, psnr, ssim):
        """Логирование метрик обучения"""
        global_step = epoch * 10000 + iteration

        self.writer.add_scalar('Train/Loss', loss, global_step)
        self.writer.add_scalar('Train/PSNR', psnr, global_step)
        self.writer.add_scalar('Train/SSIM', ssim, global_step)

    def log_validation(self, epoch, loss, psnr, ssim):
        """Логирование метрик валидации"""
        self.writer.add_scalar('Val/Loss', loss, epoch)
        self.writer.add_scalar('Val/PSNR', psnr, epoch)
        self.writer.add_scalar('Val/SSIM', ssim, epoch)

        self.log_message(f"Epoch {epoch} - Val Loss: {loss:.4f}, PSNR: {psnr:.2f}, SSIM: {ssim:.4f}")

    def log_images(self, tag, images, epoch):
        """Логирование изображений"""
        self.writer.add_images(tag, images, epoch)

    def close(self):
        """Закрыть writer"""
        self.writer.close()