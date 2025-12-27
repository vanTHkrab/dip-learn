"""
Brightness and Contrast Adjustment

การปรับความสว่างและ contrast ของภาพ
"""

import cv2 as cv
import numpy as np
from typing import Tuple

from ..transforms.color import to_grayscale


def adjust_brightness(image: np.ndarray, value: int) -> np.ndarray:
    """
    ปรับความสว่าง
    
    Args:
        image: Input image
        value: Brightness value (-255 to 255)
        
    Returns:
        Adjusted image
    """
    if value >= 0:
        lim = 255 - value
        image = np.where(image > lim, 255, image + value)
    else:
        lim = -value
        image = np.where(image < lim, 0, image + value)
    return image.astype(np.uint8)


def adjust_contrast(image: np.ndarray, factor: float) -> np.ndarray:
    """
    ปรับ contrast
    
    Args:
        image: Input image
        factor: Contrast factor (< 1 = decrease, > 1 = increase)
        
    Returns:
        Adjusted image
    """
    mean = np.mean(image)
    adjusted = (image.astype(np.float32) - mean) * factor + mean
    return np.clip(adjusted, 0, 255).astype(np.uint8)


def adjust_brightness_contrast(
    image: np.ndarray,
    brightness: int = 0,
    contrast: float = 1.0
) -> np.ndarray:
    """
    ปรับทั้ง brightness และ contrast
    
    Args:
        image: Input image
        brightness: Brightness adjustment (-255 to 255)
        contrast: Contrast factor (0.5 to 2.0 recommended)
        
    Returns:
        Adjusted image
    """
    return cv.convertScaleAbs(image, alpha=contrast, beta=brightness)


def gamma_correction(image: np.ndarray, gamma: float = 1.0) -> np.ndarray:
    """
    Gamma correction
    
    Args:
        image: Input image
        gamma: Gamma value (< 1 = brighter, > 1 = darker)
        
    Returns:
        Gamma corrected image
    """
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255
                     for i in np.arange(0, 256)]).astype(np.uint8)
    return cv.LUT(image, table)


def auto_brightness_contrast(
    image: np.ndarray,
    clip_hist_percent: float = 1.0
) -> np.ndarray:
    """
    ปรับ brightness และ contrast อัตโนมัติ
    
    Args:
        image: Input grayscale image
        clip_hist_percent: Histogram clipping percentage
        
    Returns:
        Auto-adjusted image
    """
    gray = to_grayscale(image)
    
    # Calculate histogram
    hist = cv.calcHist([gray], [0], None, [256], [0, 256])
    hist_size = len(hist)
    
    # Calculate cumulative distribution
    accumulator = [float(hist[0])]
    for i in range(1, hist_size):
        accumulator.append(accumulator[i - 1] + float(hist[i]))
    
    # Locate points to clip
    maximum = accumulator[-1]
    clip_hist_percent *= (maximum / 100.0)
    clip_hist_percent /= 2.0
    
    # Locate left cut
    minimum_gray = 0
    while accumulator[minimum_gray] < clip_hist_percent:
        minimum_gray += 1
    
    # Locate right cut
    maximum_gray = hist_size - 1
    while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
        maximum_gray -= 1
    
    # Calculate alpha and beta values
    if maximum_gray == minimum_gray:
        return gray
        
    alpha = 255 / (maximum_gray - minimum_gray)
    beta = -minimum_gray * alpha
    
    return cv.convertScaleAbs(gray, alpha=alpha, beta=beta)


def sigmoid_correction(image: np.ndarray, cutoff: float = 0.5, gain: float = 10) -> np.ndarray:
    """
    Sigmoid contrast correction
    
    Args:
        image: Input image
        cutoff: Cutoff point (0-1)
        gain: Gain factor
        
    Returns:
        Corrected image
    """
    gray = to_grayscale(image)
    normalized = gray.astype(np.float32) / 255.0
    
    # Apply sigmoid
    result = 1 / (1 + np.exp(gain * (cutoff - normalized)))
    
    return (result * 255).astype(np.uint8)


def log_transform(image: np.ndarray, c: float = 1.0) -> np.ndarray:
    """
    Logarithmic transform
    
    Args:
        image: Input image
        c: Scaling constant
        
    Returns:
        Transformed image
    """
    gray = to_grayscale(image).astype(np.float32)
    result = c * np.log1p(gray)
    return cv.normalize(result, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def power_law_transform(image: np.ndarray, gamma: float, c: float = 1.0) -> np.ndarray:
    """
    Power-law (gamma) transform
    
    Args:
        image: Input image
        gamma: Gamma value
        c: Scaling constant
        
    Returns:
        Transformed image
    """
    gray = to_grayscale(image).astype(np.float32) / 255.0
    result = c * np.power(gray, gamma)
    return (result * 255).clip(0, 255).astype(np.uint8)


__all__ = [
    'adjust_brightness',
    'adjust_contrast',
    'adjust_brightness_contrast',
    'gamma_correction',
    'auto_brightness_contrast',
    'sigmoid_correction',
    'log_transform',
    'power_law_transform',
]
