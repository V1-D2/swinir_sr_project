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


# ADD these classes to utils/temperature_loss.py (after existing TemperatureAwareLoss class)

class TemperaturePerceptualLoss(nn.Module):
    """Perceptual loss adapted for temperature data"""

    def __init__(self, feature_weights=None):
        super().__init__()
        if feature_weights is None:
            self.feature_weights = [1.0, 1.0, 1.0, 1.0]
        else:
            self.feature_weights = feature_weights

        # Simple feature extractor for temperature data
        self.feature_extractor = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(1, 32, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, 64, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(128, 128, 3, 2, 1),
                nn.ReLU(inplace=True)
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, 3, 1, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(256, 256, 3, 2, 1),
                nn.ReLU(inplace=True)
            )
        ])

        # Freeze weights for stability
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x, y):
        loss = 0
        feat_x = x
        feat_y = y

        for i, layer in enumerate(self.feature_extractor):
            feat_x = layer(feat_x)
            feat_y = layer(feat_y)
            loss += self.feature_weights[i] * F.l1_loss(feat_x, feat_y)

        return loss


class PhysicsConsistencyLoss(nn.Module):
    """Loss for maintaining physical consistency of temperature data"""

    def __init__(self, gradient_weight=0.1, smoothness_weight=0.05):
        super().__init__()
        self.gradient_weight = gradient_weight
        self.smoothness_weight = smoothness_weight

    def forward(self, pred, target):
        # Main L1 loss
        main_loss = F.l1_loss(pred, target)

        # Gradient loss - preserve temperature gradients
        pred_dx = pred[:, :, :, 1:] - pred[:, :, :, :-1]
        pred_dy = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        target_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
        target_dy = target[:, :, 1:, :] - target[:, :, :-1, :]

        gradient_loss = F.l1_loss(pred_dx, target_dx) + F.l1_loss(pred_dy, target_dy)

        # Smoothness loss - avoid artifacts
        smooth_x = pred[:, :, :, 2:] - 2 * pred[:, :, :, 1:-1] + pred[:, :, :, :-2]
        smooth_y = pred[:, :, 2:, :] - 2 * pred[:, :, 1:-1, :] + pred[:, :, :-2, :]
        smoothness_loss = torch.mean(torch.abs(smooth_x)) + torch.mean(torch.abs(smooth_y))

        total_loss = main_loss + self.gradient_weight * gradient_loss + self.smoothness_weight * smoothness_loss

        return total_loss, {
            'main': main_loss,
            'gradient': gradient_loss,
            'smoothness': smoothness_loss
        }


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1 with better gradients near zero)"""

    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon

    def forward(self, pred, target):
        return torch.mean(torch.sqrt((pred - target) ** 2 + self.epsilon ** 2))