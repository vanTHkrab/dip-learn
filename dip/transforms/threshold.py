"""
Thresholding Operations

การทำ threshold ทั้งแบบ global และ adaptive
"""

import cv2 as cv
import numpy as np
from typing import Tuple

from .color import to_grayscale


def binary_threshold(
    image: np.ndarray,
    threshold: int = 127,
    max_value: int = 255,
    inverse: bool = False
) -> np.ndarray:
    """
    Binary threshold แบบกำหนด threshold เอง
    
    Args:
        image: Grayscale image
        threshold: Threshold value (0-255)
        max_value: Maximum value for thresholding
        inverse: If True, use inverse threshold
        
    Returns:
        Binary image
    """
    gray = to_grayscale(image)
    thresh_type = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    _, binary = cv.threshold(gray, threshold, max_value, thresh_type)
    return binary


def otsu_threshold(
    image: np.ndarray,
    max_value: int = 255,
    inverse: bool = False
) -> np.ndarray:
    """
    Otsu's automatic threshold
    
    Args:
        image: Grayscale image
        max_value: Maximum value for thresholding
        inverse: If True, use inverse threshold
        
    Returns:
        Binary image
    """
    gray = to_grayscale(image)
    thresh_type = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    thresh_type += cv.THRESH_OTSU
    _, binary = cv.threshold(gray, 0, max_value, thresh_type)
    return binary


def adaptive_threshold_mean(
    image: np.ndarray,
    max_value: int = 255,
    block_size: int = 11,
    c: int = 2,
    inverse: bool = False
) -> np.ndarray:
    """
    Adaptive threshold using mean method
    
    Args:
        image: Grayscale image
        max_value: Maximum value
        block_size: Size of neighborhood area (must be odd)
        c: Constant subtracted from mean
        inverse: If True, use inverse threshold
        
    Returns:
        Binary image
    """
    gray = to_grayscale(image)
    thresh_type = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    return cv.adaptiveThreshold(
        gray, max_value, cv.ADAPTIVE_THRESH_MEAN_C, thresh_type, block_size, c
    )


def adaptive_threshold_gaussian(
    image: np.ndarray,
    max_value: int = 255,
    block_size: int = 11,
    c: int = 2,
    inverse: bool = False
) -> np.ndarray:
    """
    Adaptive threshold using Gaussian weighted method
    
    Args:
        image: Grayscale image
        max_value: Maximum value
        block_size: Size of neighborhood area (must be odd)
        c: Constant subtracted from weighted mean
        inverse: If True, use inverse threshold
        
    Returns:
        Binary image
    """
    gray = to_grayscale(image)
    thresh_type = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    return cv.adaptiveThreshold(
        gray, max_value, cv.ADAPTIVE_THRESH_GAUSSIAN_C, thresh_type, block_size, c
    )


def auto_threshold(image: np.ndarray) -> np.ndarray:
    """
    เลือก threshold method ที่เหมาะสมอัตโนมัติ
    
    ใช้ adaptive สำหรับภาพที่มี lighting ไม่สม่ำเสมอ
    Use Otsu for uniform images.
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    gray = to_grayscale(image)
    
    # Calculate local standard deviation
    blur = cv.GaussianBlur(gray, (5, 5), 0)
    local_std = cv.absdiff(gray, blur)
    std_value = np.std(local_std)
    
    # Calculate intensity range
    intensity_range = np.max(gray) - np.min(gray)
    
    # Use adaptive threshold for low contrast or uneven lighting
    if intensity_range < 100 or std_value > 30:
        return cv.adaptiveThreshold(
            gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2
        )
    else:
        _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
        return binary


def triangle_threshold(
    image: np.ndarray,
    max_value: int = 255,
    inverse: bool = False
) -> np.ndarray:
    """
    Triangle threshold method
    
    Args:
        image: Grayscale image
        max_value: Maximum value
        inverse: If True, use inverse threshold
        
    Returns:
        Binary image
    """
    gray = to_grayscale(image)
    thresh_type = cv.THRESH_BINARY_INV if inverse else cv.THRESH_BINARY
    thresh_type += cv.THRESH_TRIANGLE
    _, binary = cv.threshold(gray, 0, max_value, thresh_type)
    return binary


__all__ = [
    'binary_threshold',
    'otsu_threshold',
    'adaptive_threshold_mean',
    'adaptive_threshold_gaussian',
    'auto_threshold',
    'triangle_threshold',
]
