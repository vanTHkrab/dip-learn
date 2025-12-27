"""
Color Space Conversion Operations

การแปลง color space ต่างๆ เช่น grayscale, BGR, RGB, HSV
"""

import cv2 as cv
import numpy as np
from typing import Tuple


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    แปลงภาพเป็น grayscale
    
    Args:
        image: Input image (BGR, BGRA, or already grayscale)
        
    Returns:
        Grayscale image
    """
    if len(image.shape) == 2:
        return image.copy()
    if len(image.shape) == 3:
        if image.shape[2] == 4:  # BGRA
            return cv.cvtColor(image, cv.COLOR_BGRA2GRAY)
        return cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    return image.copy()


def to_bgr(image: np.ndarray) -> np.ndarray:
    """
    แปลงภาพเป็น BGR
    
    Args:
        image: Input image (grayscale or BGR)
        
    Returns:
        BGR image
    """
    if len(image.shape) == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return image.copy()


def to_rgb(image: np.ndarray) -> np.ndarray:
    """
    แปลงภาพจาก BGR เป็น RGB
    
    Args:
        image: Input image (BGR or grayscale)
        
    Returns:
        RGB image
    """
    if len(image.shape) == 2:
        return cv.cvtColor(image, cv.COLOR_GRAY2RGB)
    return cv.cvtColor(image, cv.COLOR_BGR2RGB)


def to_hsv(image: np.ndarray) -> np.ndarray:
    """
    แปลงภาพเป็น HSV color space
    
    Args:
        image: Input BGR image
        
    Returns:
        HSV image
    """
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return cv.cvtColor(image, cv.COLOR_BGR2HSV)


def to_lab(image: np.ndarray) -> np.ndarray:
    """
    แปลงภาพเป็น LAB color space
    
    Args:
        image: Input BGR image
        
    Returns:
        LAB image
    """
    if len(image.shape) == 2:
        image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)
    return cv.cvtColor(image, cv.COLOR_BGR2LAB)


def split_channels(image: np.ndarray) -> Tuple[np.ndarray, ...]:
    """
    แยก channels ของภาพ
    
    Args:
        image: Input multi-channel image
        
    Returns:
        Tuple of channel arrays
    """
    return cv.split(image)


def merge_channels(channels: Tuple[np.ndarray, ...]) -> np.ndarray:
    """
    รวม channels เข้าด้วยกัน
    
    Args:
        channels: Tuple of channel arrays
        
    Returns:
        Merged image
    """
    return cv.merge(channels)


__all__ = [
    'to_grayscale',
    'to_bgr',
    'to_rgb',
    'to_hsv',
    'to_lab',
    'split_channels',
    'merge_channels',
]
