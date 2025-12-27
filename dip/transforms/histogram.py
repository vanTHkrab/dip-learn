"""
Histogram Operations

การทำ histogram operations เช่น equalization, CLAHE, stretching
"""

import cv2 as cv
import numpy as np
from typing import Tuple

from .color import to_grayscale


def histogram_equalization(image: np.ndarray) -> np.ndarray:
    """
    Global histogram equalization
    
    Args:
        image: Grayscale image
        
    Returns:
        Equalized image
    """
    gray = to_grayscale(image)
    return cv.equalizeHist(gray)


def clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid_size: Tuple[int, int] = (8, 8)
) -> np.ndarray:
    """
    Contrast Limited Adaptive Histogram Equalization (CLAHE)
    
    Args:
        image: Grayscale image
        clip_limit: Threshold for contrast limiting
        tile_grid_size: Size of grid for histogram equalization
        
    Returns:
        CLAHE enhanced image
    """
    gray = to_grayscale(image)
    clahe_obj = cv.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe_obj.apply(gray)


def contrast_stretch(
    image: np.ndarray,
    percentile_low: float = 2,
    percentile_high: float = 98
) -> np.ndarray:
    """
    Contrast stretching using percentile clipping
    
    Args:
        image: Input image
        percentile_low: Lower percentile for clipping
        percentile_high: Upper percentile for clipping
        
    Returns:
        Contrast-stretched image
    """
    gray = to_grayscale(image)
    p_low = np.percentile(gray, percentile_low)
    p_high = np.percentile(gray, percentile_high)
    
    if p_high <= p_low:
        return gray
    
    stretched = np.clip(gray, p_low, p_high)
    stretched = ((stretched - p_low) / (p_high - p_low) * 255).astype(np.uint8)
    return stretched


def normalize(image: np.ndarray, alpha: int = 0, beta: int = 255) -> np.ndarray:
    """
    Normalize image intensity to range [alpha, beta]
    
    Args:
        image: Input image
        alpha: Lower bound
        beta: Upper bound
        
    Returns:
        Normalized image
    """
    return cv.normalize(image, None, alpha, beta, cv.NORM_MINMAX)


def histogram_matching(
    source: np.ndarray,
    reference: np.ndarray
) -> np.ndarray:
    """
    Match histogram of source image to reference image
    
    Args:
        source: Source image
        reference: Reference image
        
    Returns:
        Image with matched histogram
    """
    src_gray = to_grayscale(source)
    ref_gray = to_grayscale(reference)
    
    # Compute histograms
    src_hist, _ = np.histogram(src_gray.flatten(), bins=256, range=(0, 256))
    ref_hist, _ = np.histogram(ref_gray.flatten(), bins=256, range=(0, 256))
    
    # Compute CDFs
    src_cdf = np.cumsum(src_hist).astype(np.float64)
    ref_cdf = np.cumsum(ref_hist).astype(np.float64)
    
    # Normalize CDFs
    src_cdf = src_cdf / src_cdf[-1]
    ref_cdf = ref_cdf / ref_cdf[-1]
    
    # Create lookup table
    lookup = np.zeros(256, dtype=np.uint8)
    for i in range(256):
        j = 255
        while j > 0 and ref_cdf[j] > src_cdf[i]:
            j -= 1
        lookup[i] = j
    
    return cv.LUT(src_gray, lookup)


def calculate_histogram(image: np.ndarray, bins: int = 256) -> np.ndarray:
    """
    Calculate histogram of image
    
    Args:
        image: Input grayscale image
        bins: Number of bins
        
    Returns:
        Histogram array
    """
    gray = to_grayscale(image)
    hist = cv.calcHist([gray], [0], None, [bins], [0, 256])
    return hist.flatten()


def normalize_lighting(image: np.ndarray) -> np.ndarray:
    """
    Normalize uneven lighting
    
    Args:
        image: Input grayscale image
        
    Returns:
        Image with normalized lighting
    """
    gray = to_grayscale(image)
    
    # Create background estimation using large blur
    blur_size = min(51, (gray.shape[0] // 2) * 2 + 1, (gray.shape[1] // 2) * 2 + 1)
    blur_size = max(3, blur_size)
    blur = cv.GaussianBlur(gray, (blur_size, blur_size), 0)
    
    # Subtract background
    normalized = cv.subtract(gray, blur)
    
    # Normalize to full range
    return cv.normalize(normalized, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


__all__ = [
    'histogram_equalization',
    'clahe',
    'contrast_stretch',
    'normalize',
    'histogram_matching',
    'calculate_histogram',
    'normalize_lighting',
]
