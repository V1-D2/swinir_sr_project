# models/degradation_bsrgan.py
"""
Temperature-specific degradation model inspired by BSRGAN
Adapted for single-channel temperature data with physically plausible degradations
"""
import numpy as np
import cv2
import random
from scipy import ndimage
import scipy.stats as ss
from scipy.interpolate import interp2d
from typing import Tuple, Optional


def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, HxW
        sf: scale factor
    Return:
        cropped image
    '''
    h, w = img.shape[:2]
    im = np.copy(img)
    return im[:h - h % sf, :w - w % sf]


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """Generate an anisotropic Gaussian kernel"""
    v = np.dot(np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]), np.array([1., 0.]))
    V = np.array([[v[0], v[1]], [v[1], -v[0]]])
    D = np.array([[l1, 0], [0, l2]])
    Sigma = np.dot(np.dot(V, D), np.linalg.inv(V))
    k = gm_blur_kernel(mean=[0, 0], cov=Sigma, size=ksize)
    return k


def gm_blur_kernel(mean, cov, size=15):
    center = size / 2.0 + 0.5
    k = np.zeros([size, size])
    for y in range(size):
        for x in range(size):
            cy = y - center + 1
            cx = x - center + 1
            k[y, x] = ss.multivariate_normal.pdf([cx, cy], mean=mean, cov=cov)
    k = k / np.sum(k)
    return k


def fspecial_gaussian(hsize, sigma):
    hsize = [hsize, hsize]
    siz = [(hsize[0] - 1.0) / 2.0, (hsize[1] - 1.0) / 2.0]
    std = sigma
    [x, y] = np.meshgrid(np.arange(-siz[1], siz[1] + 1), np.arange(-siz[0], siz[0] + 1))
    arg = -(x * x + y * y) / (2 * std * std)
    h = np.exp(arg)
    h[h < np.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def shift_pixel(x, sf, upper_left=True):
    """Shift pixel for super-resolution with different scale factors"""
    h, w = x.shape[:2]
    shift = (sf - 1) * 0.5
    xv, yv = np.arange(0, w, 1.0), np.arange(0, h, 1.0)
    if upper_left:
        x1 = xv + shift
        y1 = yv + shift
    else:
        x1 = xv - shift
        y1 = yv - shift

    x1 = np.clip(x1, 0, w - 1)
    y1 = np.clip(y1, 0, h - 1)

    # For 2D array
    x = interp2d(xv, yv, x)(x1, y1)
    return x


def add_temperature_blur(img, sf=4):
    """Add physically plausible blur for temperature data"""
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf

    # Use more conservative blur for temperature data
    blur_prob = random.random()

    if blur_prob < 0.3:
        # Isotropic Gaussian blur (thermal diffusion-like)
        k = fspecial_gaussian(2 * random.randint(2, 8) + 3, wd * random.uniform(0.5, 1.0))
    elif blur_prob < 0.6:
        # Mild anisotropic blur (directional heat transfer)
        l1 = wd2 * random.uniform(0.5, 1.0)
        l2 = wd2 * random.uniform(0.5, 1.0)
        k = anisotropic_Gaussian(ksize=2 * random.randint(2, 6) + 3,
                                 theta=random.random() * np.pi, l1=l1, l2=l2)
    else:
        # No blur
        return img

    # Apply convolution
    img_blurred = ndimage.filters.convolve(img, k, mode='mirror')
    return img_blurred


def add_temperature_noise(img, noise_level1=1, noise_level2=15):
    """Add sensor-like noise appropriate for temperature measurements"""
    # Lower noise levels for temperature data
    noise_level = random.randint(noise_level1, noise_level2)

    # Add Gaussian noise (sensor noise)
    noise = np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)

    # Add slight spatially correlated noise (atmospheric effects)
    if random.random() < 0.3:
        # Create correlated noise by blurring white noise
        corr_noise = np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
        kernel = fspecial_gaussian(5, 1.0)
        corr_noise = ndimage.filters.convolve(corr_noise, kernel, mode='mirror')
        noise = noise * 0.7 + corr_noise * 0.3

    img_noisy = img + noise
    return np.clip(img_noisy, 0.0, 1.0)


def add_temperature_quantization(img, bits=10):
    """Add quantization noise (ADC effects in temperature sensors)"""
    if random.random() < 0.3:  # 30% chance
        levels = 2 ** bits
        img_quantized = np.round(img * levels) / levels
        return img_quantized
    return img


def add_temperature_artifacts(img):
    """Add measurement artifacts specific to temperature imaging"""
    if random.random() < 0.2:  # 20% chance
        # Simulate calibration drift (slow spatial variation)
        h, w = img.shape
        drift = np.random.normal(0, 0.02, (5, 5))
        drift = cv2.resize(drift, (w, h), interpolation=cv2.INTER_CUBIC)
        img = img + drift

    if random.random() < 0.1:  # 10% chance
        # Simulate bad pixels (dead/hot pixels)
        num_bad_pixels = random.randint(1, 10)
        h, w = img.shape
        for _ in range(num_bad_pixels):
            x, y = random.randint(0, w - 1), random.randint(0, h - 1)
            if random.random() < 0.5:
                img[y, x] = 0  # Dead pixel
            else:
                img[y, x] = 1  # Hot pixel

    return np.clip(img, 0.0, 1.0)


def add_shifted_downsampling(img, sf):
    """Add pixel shift effect during downsampling"""
    if random.random() < 0.5:
        # Apply sub-pixel shift before downsampling
        k = fspecial_gaussian(25, random.uniform(0.1, 0.6 * sf))
        k_shifted = shift_pixel(k, sf)
        k_shifted = k_shifted / k_shifted.sum()
        img = ndimage.filters.convolve(img, k_shifted, mode='mirror')
    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    """Random crop for training patches"""
    h, w = lq.shape
    rnd_h = random.randint(0, h - lq_patchsize)
    rnd_w = random.randint(0, w - lq_patchsize)

    lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf, rnd_w_H:rnd_w_H + lq_patchsize * sf]

    return lq, hq


def degradation_bsrgan_temperature(img, sf=4, lq_patchsize=72):
    """
    Temperature-specific degradation model inspired by BSRGAN
    WITHOUT any resizing - maintains strict size relationships

    Args:
        img: HxW temperature array, normalized to [0, 1]
        sf: scale factor
        lq_patchsize: size of low-quality patches

    Returns:
        lq: low-quality patch (lq_patchsize x lq_patchsize)
        hq: high-quality patch (lq_patchsize*sf x lq_patchsize*sf)
    """
    # Probabilities for different degradations
    noise_prob = 0.8
    artifact_prob = 0.3
    shift_prob = 0.5

    h1, w1 = img.shape
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf]
    h, w = img.shape

    if h < lq_patchsize * sf or w < lq_patchsize * sf:
        raise ValueError(f'Image size ({h1}x{w1}) is too small for patch size {lq_patchsize} with scale factor {sf}!')

    hq = img.copy()

    # Apply degradations in random order (NO RESIZING)
    degradations = []

    # Always include at least one blur
    degradations.append('blur1')

    # Randomly add other degradations
    if random.random() < 0.5:
        degradations.append('blur2')
    if random.random() < noise_prob:
        degradations.append('noise')
    if random.random() < artifact_prob:
        degradations.append('artifacts')
    if random.random() < shift_prob:
        degradations.append('shift')

    # Shuffle degradations
    random.shuffle(degradations)

    # Apply degradations
    for degradation in degradations:
        if degradation == 'blur1':
            img = add_temperature_blur(img, sf=sf)
        elif degradation == 'blur2':
            img = add_temperature_blur(img, sf=sf)
        elif degradation == 'noise':
            img = add_temperature_noise(img)
        elif degradation == 'artifacts':
            img = add_temperature_artifacts(img)
        elif degradation == 'shift':
            img = add_shifted_downsampling(img, sf)

    # Add quantization
    img = add_temperature_quantization(img)

    # Final downsampling with EXACT scale factor (no intermediate resizing)
    h_lq, w_lq = h // sf, w // sf

    # Choose downsampling method
    if random.random() < 0.5:
        # Standard area interpolation
        img = cv2.resize(img, (w_lq, h_lq), interpolation=cv2.INTER_AREA)
    else:
        # Nearest neighbor downsampling (simulates direct subsampling)
        img = img[::sf, ::sf]

    # Ensure output is properly clipped
    img = np.clip(img, 0.0, 1.0)

    # Random crop to get exact patch sizes
    img, hq = random_crop(img, hq, sf, lq_patchsize)

    return img, hq


class TemperatureDegradation:
    """Wrapper class for temperature-specific degradation"""

    def __init__(self, scale_factor=4):
        self.scale_factor = scale_factor

    def degradation_bsrgan(self, img, lq_patchsize=128):
        """Apply temperature-specific degradation"""
        return degradation_bsrgan_temperature(img, sf=self.scale_factor, lq_patchsize=lq_patchsize // self.scale_factor)