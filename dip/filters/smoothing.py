"""
Smoothing / Blur Filters

ฟิลเตอร์สำหรับ blur และ smoothing
"""

import cv2 as cv
import numpy as np
from typing import Tuple


def gaussian_blur(
    image: np.ndarray,
    ksize: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Gaussian blur
    
    Args:
        image: Input image
        ksize: Kernel size (must be odd)
        sigma: Gaussian kernel standard deviation (0 = auto)
        
    Returns:
        Blurred image
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv.GaussianBlur(image, (ksize, ksize), sigma)


def median_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Median blur - good for salt and pepper noise
    
    Args:
        image: Input image
        ksize: Kernel size (must be odd)
        
    Returns:
        Blurred image
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv.medianBlur(image, ksize)


def bilateral_filter(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Bilateral filter - smoothing ที่รักษา edge
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in the color space
        sigma_space: Filter sigma in the coordinate space
        
    Returns:
        Filtered image
    """
    return cv.bilateralFilter(image, d, sigma_color, sigma_space)


def box_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Box (mean) blur
    
    Args:
        image: Input image
        ksize: Kernel size
        
    Returns:
        Blurred image
    """
    return cv.blur(image, (ksize, ksize))


def stack_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Stack blur - fast approximation of Gaussian blur
    
    Args:
        image: Input image
        ksize: Kernel size (must be odd)
        
    Returns:
        Blurred image
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv.stackBlur(image, (ksize, ksize))


def blur_2d(
    image: np.ndarray,
    kernel: np.ndarray
) -> np.ndarray:
    """
    Apply custom blur kernel
    
    Args:
        image: Input image
        kernel: 2D convolution kernel
        
    Returns:
        Filtered image
    """
    return cv.filter2D(image, -1, kernel)


def average_blur(image: np.ndarray, ksize: int = 5) -> np.ndarray:
    """
    Average blur (same as box_blur)
    
    Args:
        image: Input image
        ksize: Kernel size
        
    Returns:
        Blurred image
    """
    return box_blur(image, ksize)


__all__ = [
    'gaussian_blur',
    'median_blur',
    'bilateral_filter',
    'box_blur',
    'stack_blur',
    'blur_2d',
    'average_blur',
]
