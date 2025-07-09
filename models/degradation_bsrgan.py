# models/degradation_bsrgan.py
"""
Адаптированная версия BSRGAN degradation из utils_blindsr.py
для одноканальных температурных данных
"""
import numpy as np
import torch
import cv2
import random
from scipy import ndimage
import scipy
import scipy.stats as ss
from scipy.linalg import orth
from scipy.interpolate import interp2d
from typing import Tuple, Optional


def modcrop_np(img, sf):
    '''
    Args:
        img: numpy image, HxW or HxWxC
        sf: scale factor
    Return:
        cropped image
    '''
    h, w = img.shape[:2]
    im = np.copy(img)
    return im[:h - h % sf, :w - w % sf, ...]


def anisotropic_Gaussian(ksize=15, theta=np.pi, l1=6, l2=6):
    """ generate an anisotropic Gaussian kernel
    Args:
        ksize : e.g., 15, kernel size
        theta : [0,  pi], rotation angle range
        l1    : [0.1,50], scaling of eigenvalues
        l2    : [0.1,l1], scaling of eigenvalues
        If l1 = l2, will get an isotropic Gaussian kernel.
    Returns:
        k     : kernel
    """
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
    h[h < scipy.finfo(float).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h = h / sumh
    return h


def fspecial(filter_type, *args, **kwargs):
    '''
    python code from:
    https://github.com/ronaldosena/imagens-medicas-2/blob/40171a6c259edec7827a6693a93955de2bd39e76/Aulas/aula_2_-_uniform_filter/matlab_fspecial.py
    '''
    if filter_type == 'gaussian':
        return fspecial_gaussian(*args, **kwargs)


def shift_pixel(x, sf, upper_left=True):
    """shift pixel for super-resolution with different scale factors
    Args:
        x: HxW
        sf: scale factor
        upper_left: shift direction
    """
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

    # Для 2D массива
    x = interp2d(xv, yv, x)(x1, y1)
    return x


def add_blur(img, sf=4):
    """Добавление размытия для одноканального изображения"""
    wd2 = 4.0 + sf
    wd = 2.0 + 0.2 * sf

    if random.random() < 0.5:
        l1 = wd2 * random.random()
        l2 = wd2 * random.random()
        k = anisotropic_Gaussian(ksize=2 * random.randint(2, 11) + 3, theta=random.random() * np.pi, l1=l1, l2=l2)
    else:
        k = fspecial('gaussian', 2 * random.randint(2, 11) + 3, wd * random.random())

    # Для одноканального изображения
    if img.ndim == 2:
        img = ndimage.filters.convolve(img, k, mode='mirror')
    else:
        img = ndimage.filters.convolve(img, np.expand_dims(k, axis=2), mode='mirror')

    return img


def add_resize(img, sf=4):
    """Изменение размера для одноканального изображения"""
    rnum = np.random.rand()
    if rnum > 0.8:  # up
        sf1 = random.uniform(1, 2)
    elif rnum < 0.7:  # down
        sf1 = random.uniform(0.5 / sf, 1)
    else:
        sf1 = 1.0

    # Обрабатываем одноканальное изображение
    if img.ndim == 2:
        img = cv2.resize(img, (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])),
                         interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
    else:
        img = cv2.resize(img, (int(sf1 * img.shape[1]), int(sf1 * img.shape[0])),
                         interpolation=random.choice([1, 2, 3]))

    img = np.clip(img, 0.0, 1.0)
    return img


def add_Gaussian_noise(img, noise_level1=2, noise_level2=25):
    """Добавление Гауссова шума для одноканального изображения"""
    noise_level = random.randint(noise_level1, noise_level2)

    # Для одноканального изображения всегда добавляем простой шум
    if img.ndim == 2:
        img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
    else:
        # Для многоканального оставляем оригинальную логику
        rnum = np.random.rand()
        if rnum > 0.6:  # add color Gaussian noise
            img += np.random.normal(0, noise_level / 255.0, img.shape).astype(np.float32)
        elif rnum < 0.4:  # add grayscale Gaussian noise
            img += np.random.normal(0, noise_level / 255.0, (*img.shape[:2], 1)).astype(np.float32)
        else:  # add  noise
            L = noise_level2 / 255.
            D = np.diag(np.random.rand(3))
            U = orth(np.random.rand(3, 3))
            conv = np.dot(np.dot(np.transpose(U), D), U)
            img += np.random.multivariate_normal([0, 0, 0], np.abs(L ** 2 * conv), img.shape[:2]).astype(np.float32)

    img = np.clip(img, 0.0, 1.0)
    return img


def add_JPEG_noise(img):
    """Добавление JPEG артефактов для одноканального изображения"""
    quality_factor = random.randint(30, 95)

    if img.ndim == 2:
        # Для одноканального изображения
        img_uint8 = (img * 255).astype(np.uint8)
        result, encimg = cv2.imencode('.jpg', img_uint8, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img = cv2.imdecode(encimg, cv2.IMREAD_GRAYSCALE)
        img = img.astype(np.float32) / 255.0
    else:
        # Оригинальная логика для RGB
        img = cv2.cvtColor((img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        result, encimg = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img = cv2.imdecode(encimg, 1)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    return img


def random_crop(lq, hq, sf=4, lq_patchsize=64):
    """Случайная обрезка патчей"""
    h, w = lq.shape[:2]
    rnd_h = random.randint(0, h - lq_patchsize)
    rnd_w = random.randint(0, w - lq_patchsize)

    if lq.ndim == 2:
        lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize]
    else:
        lq = lq[rnd_h:rnd_h + lq_patchsize, rnd_w:rnd_w + lq_patchsize, :]

    rnd_h_H, rnd_w_H = int(rnd_h * sf), int(rnd_w * sf)
    if hq.ndim == 2:
        hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf, rnd_w_H:rnd_w_H + lq_patchsize * sf]
    else:
        hq = hq[rnd_h_H:rnd_h_H + lq_patchsize * sf, rnd_w_H:rnd_w_H + lq_patchsize * sf, :]

    return lq, hq


def degradation_bsrgan(img, sf=4, lq_patchsize=72, isp_model=None):
    """
    This is the degradation model of BSRGAN from the paper
    "Designing a Practical Degradation Model for Deep Blind Image Super-Resolution"
    Адаптировано для одноканальных изображений
    ----------
    img: HxW, [0, 1], its size should be large than (lq_patchsizexsf)x(lq_patchsizexsf)
    sf: scale factor
    isp_model: camera ISP model

    Returns
    -------
    img: low-quality patch, size: lq_patchsizeXlq_patchsize, range: [0, 1]
    hq: corresponding high-quality patch, size: (lq_patchsizexsf)X(lq_patchsizexsf), range: [0, 1]
    """
    isp_prob, jpeg_prob, scale2_prob = 0.25, 0.9, 0.25
    sf_ori = sf

    h1, w1 = img.shape[:2]
    img = img.copy()[:h1 - h1 % sf, :w1 - w1 % sf, ...]  # mod crop
    h, w = img.shape[:2]

    if h < lq_patchsize * sf or w < lq_patchsize * sf:
        raise ValueError(f'img size ({h1}X{w1}) is too small!')

    hq = img.copy()

    if sf == 4 and random.random() < scale2_prob:  # downsample1
        if np.random.rand() < 0.5:
            img = cv2.resize(img, (int(1 / 2 * img.shape[1]), int(1 / 2 * img.shape[0])),
                             interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
        else:
            # Для одноканальных данных используем cv2.resize вместо util.imresize_np
            img = cv2.resize(img, (int(img.shape[1] / 2), int(img.shape[0] / 2)), interpolation=cv2.INTER_CUBIC)
        img = np.clip(img, 0.0, 1.0)
        sf = 2

    shuffle_order = random.sample(range(7), 7)
    idx1, idx2 = shuffle_order.index(2), shuffle_order.index(3)
    if idx1 > idx2:  # keep downsample3 last
        shuffle_order[idx1], shuffle_order[idx2] = shuffle_order[idx2], shuffle_order[idx1]

    for i in shuffle_order:
        if i == 0:
            img = add_blur(img, sf=sf)
        elif i == 1:
            img = add_blur(img, sf=sf)
        elif i == 2:
            a, b = img.shape[1], img.shape[0]
            # downsample2
            if random.random() < 0.75:
                sf1 = random.uniform(1, 2 * sf)
                img = cv2.resize(img, (int(1 / sf1 * img.shape[1]), int(1 / sf1 * img.shape[0])),
                                 interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
            else:
                k = fspecial('gaussian', 25, random.uniform(0.1, 0.6 * sf))
                k_shifted = shift_pixel(k, sf)
                k_shifted = k_shifted / k_shifted.sum()  # blur with shifted kernel
                # Для одноканального изображения
                if img.ndim == 2:
                    img = ndimage.filters.convolve(img, k_shifted, mode='mirror')
                else:
                    img = ndimage.filters.convolve(img, np.expand_dims(k_shifted, axis=2), mode='mirror')
                img = img[0::sf, 0::sf, ...]  # nearest downsampling
            img = np.clip(img, 0.0, 1.0)
        elif i == 3:
            # downsample3
            img = cv2.resize(img, (int(1 / sf * a), int(1 / sf * b)),
                             interpolation=random.choice([cv2.INTER_LINEAR, cv2.INTER_CUBIC, cv2.INTER_AREA]))
            img = np.clip(img, 0.0, 1.0)
        elif i == 4:
            # add Gaussian noise
            img = add_Gaussian_noise(img, noise_level1=2, noise_level2=25)
        elif i == 5:
            # add JPEG noise
            if random.random() < jpeg_prob:
                img = add_JPEG_noise(img)
        elif i == 6:
            # add processed camera sensor noise
            if random.random() < isp_prob and isp_model is not None:
                with torch.no_grad():
                    img, hq = isp_model.forward(img.copy(), hq)

    # add final JPEG compression noise
    img = add_JPEG_noise(img)

    # random crop
    img, hq = random_crop(img, hq, sf_ori, lq_patchsize)

    return img, hq


class BSRGANDegradation:
    """Обертка для совместимости с data loader"""

    def __init__(self, scale_factor=4, use_sharp=False):
        self.scale_factor = scale_factor
        self.use_sharp = use_sharp

    def degradation_bsrgan(self, img, lq_patchsize=128):
        """Вызов функции деградации"""
        return degradation_bsrgan(img, sf=self.scale_factor, lq_patchsize=lq_patchsize)