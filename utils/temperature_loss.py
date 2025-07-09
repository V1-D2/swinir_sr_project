# utils/temperature_loss.py
import torch
import torch.nn as nn
import torch.nn.functional as F


class TemperatureAwareLoss(nn.Module):
    """Combined loss function for temperature super-resolution"""

    def __init__(self, alpha=1.0, beta=0.1, gamma=0.05):
        """
        Args:
            alpha: Weight for L1 loss
            beta: Weight for gradient loss
            gamma: Weight for relative error loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def forward(self, pred, target):
        """
        Args:
            pred: Predicted temperature map
            target: Ground truth temperature map
        """
        # L1 loss
        l1_loss = F.l1_loss(pred, target)

        # Gradient loss (preserve temperature gradients)
        grad_loss = self._gradient_loss(pred, target)

        # Relative error loss (penalize errors more in lower temperature regions)
        rel_loss = self._relative_error_loss(pred, target)

        # Combined loss
        total_loss = self.alpha * l1_loss + self.beta * grad_loss + self.gamma * rel_loss

        return total_loss, {
            'l1_loss': l1_loss.item(),
            'grad_loss': grad_loss.item(),
            'rel_loss': rel_loss.item(),
            'total_loss': total_loss.item()
        }

    def _gradient_loss(self, pred, target):
        """Compute gradient loss to preserve temperature gradients"""
        # Sobel filters
        sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                               dtype=torch.float32).unsqueeze(0).unsqueeze(0)

        sobel_x = sobel_x.to(pred.device)
        sobel_y = sobel_y.to(pred.device)

        # Compute gradients
        pred_grad_x = F.conv2d(pred, sobel_x, padding=1)
        pred_grad_y = F.conv2d(pred, sobel_y, padding=1)
        target_grad_x = F.conv2d(target, sobel_x, padding=1)
        target_grad_y = F.conv2d(target, sobel_y, padding=1)

        # Gradient magnitude
        pred_grad = torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2 + 1e-8)
        target_grad = torch.sqrt(target_grad_x ** 2 + target_grad_y ** 2 + 1e-8)

        return F.l1_loss(pred_grad, target_grad)

    def _relative_error_loss(self, pred, target):
        """Compute relative error loss"""
        # Add small epsilon to avoid division by zero
        epsilon = 1e-3
        relative_error = torch.abs(pred - target) / (target + epsilon)
        return relative_error.mean()


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 with better gradients near zero)"""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.epsilon ** 2))