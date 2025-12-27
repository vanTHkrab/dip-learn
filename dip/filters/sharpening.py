"""
Sharpening Filters

ฟิลเตอร์สำหรับ sharpening
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Literal

from ..transforms.color import to_grayscale


def unsharp_mask(
    image: np.ndarray,
    kernel_size: Tuple[int, int] = (5, 5),
    sigma: float = 1.0,
    amount: float = 1.5
) -> np.ndarray:
    """
    Unsharp masking - sharpening technique
    
    Args:
        image: Input image
        kernel_size: Gaussian blur kernel size
        sigma: Gaussian sigma
        amount: Sharpening strength
        
    Returns:
        Sharpened image
    """
    blurred = cv.GaussianBlur(image, kernel_size, sigmaX=sigma)
    sharpened = cv.addWeighted(image, 1 + amount, blurred, -amount, 0)
    return cv.normalize(sharpened, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def laplacian_sharpen(image: np.ndarray, strength: float = 1.0) -> np.ndarray:
    """
    Sharpening using Laplacian operator
    
    Args:
        image: Input grayscale image
        strength: Sharpening strength
        
    Returns:
        Sharpened image
    """
    gray = to_grayscale(image)
    laplacian = cv.Laplacian(gray, cv.CV_64F)
    sharpened = gray - strength * laplacian
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def kernel_sharpen(
    image: np.ndarray,
    strength: Literal["light", "normal", "strong"] = "normal"
) -> np.ndarray:
    """
    Sharpening using convolution kernel
    
    Args:
        image: Input image
        strength: "light", "normal", or "strong"
        
    Returns:
        Sharpened image
    """
    kernels = {
        "light": np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32),
        "normal": np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]], dtype=np.float32),
        "strong": np.array([[-1, -1, -1], [-1, 10, -1], [-1, -1, -1]], dtype=np.float32) / 2
    }
    kernel = kernels.get(strength, kernels["normal"])
    return cv.filter2D(image, -1, kernel)


def high_boost_filter(
    image: np.ndarray,
    amplification: float = 1.5,
    blur_ksize: int = 5
) -> np.ndarray:
    """
    High-boost filtering for sharpening
    
    Args:
        image: Input image
        amplification: Amplification factor (A > 1)
        blur_ksize: Blur kernel size
        
    Returns:
        Sharpened image
    """
    gray = to_grayscale(image)
    blurred = cv.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)
    mask = gray.astype(np.float32) - blurred.astype(np.float32)
    sharpened = gray.astype(np.float32) + amplification * mask
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def frequency_high_boost(
    image: np.ndarray,
    alpha: float = 1.4,
    cutoff_ratio: float = 0.1
) -> np.ndarray:
    """
    Frequency domain high-boost filtering
    
    Args:
        image: Input grayscale image
        alpha: Boost factor
        cutoff_ratio: Cutoff frequency ratio
        
    Returns:
        Sharpened image
    """
    gray = to_grayscale(image)
    image_float = gray.astype(np.float32)
    dft = np.fft.fftshift(np.fft.fft2(image_float))
    rows, cols = gray.shape
    crow, ccol = rows // 2, cols // 2
    y, x = np.ogrid[:rows, :cols]
    distance = np.sqrt((y - crow) ** 2 + (x - ccol) ** 2)
    cutoff = cutoff_ratio * min(rows, cols)

    mask = np.ones_like(dft, dtype=np.float32)
    mask[distance <= cutoff] = 1 - alpha

    boosted = np.fft.ifft2(np.fft.ifftshift(dft * mask))
    boosted = np.abs(boosted)
    return cv.normalize(boosted, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def custom_sharpen(
    image: np.ndarray,
    kernel: np.ndarray
) -> np.ndarray:
    """
    Apply custom sharpening kernel
    
    Args:
        image: Input image
        kernel: Custom convolution kernel
        
    Returns:
        Sharpened image
    """
    return cv.filter2D(image, -1, kernel)


__all__ = [
    'unsharp_mask',
    'laplacian_sharpen',
    'kernel_sharpen',
    'high_boost_filter',
    'frequency_high_boost',
    'custom_sharpen',
]
