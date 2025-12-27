"""
Image Metrics Utilities

เครื่องมือสำหรับวัด metrics ของภาพ
"""

import cv2 as cv
import numpy as np
from typing import Tuple

from ..transforms.color import to_grayscale


def variance_of_laplacian(image: np.ndarray) -> float:
    """
    Calculate variance of Laplacian (sharpness measure)
    Higher value = sharper image
    
    Args:
        image: Input grayscale image
        
    Returns:
        Variance value
    """
    gray = to_grayscale(image)
    return float(cv.Laplacian(gray, cv.CV_64F).var())


def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
    """
    Check if image is blurry
    
    Args:
        image: Input image
        threshold: Variance threshold (lower = blurry)
        
    Returns:
        True if image is blurry
    """
    return variance_of_laplacian(image) < threshold


def calculate_snr(image: np.ndarray) -> float:
    """
    Calculate Signal-to-Noise Ratio
    
    Args:
        image: Input grayscale image
        
    Returns:
        SNR value in dB
    """
    gray = to_grayscale(image).astype(np.float64)
    mean = np.mean(gray)
    std = np.std(gray)
    
    if std == 0:
        return float('inf')
    
    return 20 * np.log10(mean / std)


def calculate_contrast(image: np.ndarray) -> float:
    """
    Calculate Michelson contrast
    
    Args:
        image: Input grayscale image
        
    Returns:
        Contrast value (0-1)
    """
    gray = to_grayscale(image).astype(np.float64)
    max_val = np.max(gray)
    min_val = np.min(gray)
    
    if max_val + min_val == 0:
        return 0.0
    
    return (max_val - min_val) / (max_val + min_val)


def calculate_rms_contrast(image: np.ndarray) -> float:
    """
    Calculate RMS (Root Mean Square) contrast
    
    Args:
        image: Input grayscale image
        
    Returns:
        RMS contrast value
    """
    gray = to_grayscale(image).astype(np.float64)
    normalized = gray / 255.0
    mean = np.mean(normalized)
    
    return np.sqrt(np.mean((normalized - mean) ** 2))


def calculate_entropy(image: np.ndarray) -> float:
    """
    Calculate image entropy (measure of information content)
    
    Args:
        image: Input grayscale image
        
    Returns:
        Entropy value
    """
    gray = to_grayscale(image)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist = hist / hist.sum()  # Normalize
    
    # Remove zeros to avoid log(0)
    hist = hist[hist > 0]
    
    return -np.sum(hist * np.log2(hist))


def calculate_mean_brightness(image: np.ndarray) -> float:
    """
    Calculate mean brightness
    
    Args:
        image: Input image
        
    Returns:
        Mean brightness (0-255)
    """
    gray = to_grayscale(image)
    return float(np.mean(gray))


def calculate_std_brightness(image: np.ndarray) -> float:
    """
    Calculate standard deviation of brightness
    
    Args:
        image: Input image
        
    Returns:
        Standard deviation
    """
    gray = to_grayscale(image)
    return float(np.std(gray))


def calculate_histogram_spread(image: np.ndarray) -> float:
    """
    Calculate histogram spread (dynamic range utilization)
    
    Args:
        image: Input grayscale image
        
    Returns:
        Spread value (0-1)
    """
    gray = to_grayscale(image)
    hist = cv.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    
    # Find range where histogram has values
    non_zero = np.where(hist > 0)[0]
    if len(non_zero) == 0:
        return 0.0
    
    return (non_zero[-1] - non_zero[0]) / 255.0


def calculate_psnr(
    original: np.ndarray,
    processed: np.ndarray,
    max_pixel: float = 255.0
) -> float:
    """
    Calculate Peak Signal-to-Noise Ratio between two images
    
    Args:
        original: Original image
        processed: Processed image
        max_pixel: Maximum pixel value
        
    Returns:
        PSNR value in dB
    """
    gray_orig = to_grayscale(original).astype(np.float64)
    gray_proc = to_grayscale(processed).astype(np.float64)
    
    if gray_orig.shape != gray_proc.shape:
        raise ValueError("Images must have the same shape")
    
    mse = np.mean((gray_orig - gray_proc) ** 2)
    
    if mse == 0:
        return float('inf')
    
    return 20 * np.log10(max_pixel / np.sqrt(mse))


def calculate_ssim(
    original: np.ndarray,
    processed: np.ndarray,
    k1: float = 0.01,
    k2: float = 0.03,
    L: float = 255.0
) -> float:
    """
    Calculate Structural Similarity Index (simplified version)
    
    Args:
        original: Original image
        processed: Processed image
        k1, k2: Constants
        L: Dynamic range
        
    Returns:
        SSIM value (0-1)
    """
    gray_orig = to_grayscale(original).astype(np.float64)
    gray_proc = to_grayscale(processed).astype(np.float64)
    
    if gray_orig.shape != gray_proc.shape:
        raise ValueError("Images must have the same shape")
    
    c1 = (k1 * L) ** 2
    c2 = (k2 * L) ** 2
    
    mu_x = np.mean(gray_orig)
    mu_y = np.mean(gray_proc)
    
    sigma_x = np.var(gray_orig)
    sigma_y = np.var(gray_proc)
    sigma_xy = np.mean((gray_orig - mu_x) * (gray_proc - mu_y))
    
    numerator = (2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)
    denominator = (mu_x ** 2 + mu_y ** 2 + c1) * (sigma_x + sigma_y + c2)
    
    return numerator / denominator


def get_image_stats(image: np.ndarray) -> dict:
    """
    Get comprehensive image statistics
    
    Args:
        image: Input image
        
    Returns:
        Dictionary of statistics
    """
    gray = to_grayscale(image)
    
    return {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'mean_brightness': calculate_mean_brightness(gray),
        'std_brightness': calculate_std_brightness(gray),
        'min_value': int(np.min(gray)),
        'max_value': int(np.max(gray)),
        'contrast': calculate_contrast(gray),
        'rms_contrast': calculate_rms_contrast(gray),
        'entropy': calculate_entropy(gray),
        'sharpness': variance_of_laplacian(gray),
        'is_blurry': is_blurry(gray),
        'histogram_spread': calculate_histogram_spread(gray),
    }


__all__ = [
    'variance_of_laplacian',
    'is_blurry',
    'calculate_snr',
    'calculate_contrast',
    'calculate_rms_contrast',
    'calculate_entropy',
    'calculate_mean_brightness',
    'calculate_std_brightness',
    'calculate_histogram_spread',
    'calculate_psnr',
    'calculate_ssim',
    'get_image_stats',
]
