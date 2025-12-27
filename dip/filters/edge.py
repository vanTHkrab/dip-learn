"""
Edge Detection Filters

ฟิลเตอร์สำหรับ edge detection
"""

import cv2 as cv
import numpy as np
from typing import Tuple

from ..transforms.color import to_grayscale


def canny_edge(
    image: np.ndarray,
    threshold1: int = 50,
    threshold2: int = 150,
    aperture_size: int = 3
) -> np.ndarray:
    """
    Canny edge detection
    
    Args:
        image: Input grayscale image
        threshold1: First threshold for hysteresis
        threshold2: Second threshold for hysteresis
        aperture_size: Aperture size for Sobel operator
        
    Returns:
        Edge image
    """
    gray = to_grayscale(image)
    return cv.Canny(gray, threshold1, threshold2, apertureSize=aperture_size)


def sobel_edge(
    image: np.ndarray,
    dx: int = 1,
    dy: int = 1,
    ksize: int = 3
) -> np.ndarray:
    """
    Sobel edge detection
    
    Args:
        image: Input grayscale image
        dx: Order of derivative x
        dy: Order of derivative y
        ksize: Kernel size
        
    Returns:
        Edge magnitude image
    """
    gray = to_grayscale(image)
    sobel_x = cv.Sobel(gray, cv.CV_64F, dx, 0, ksize=ksize)
    sobel_y = cv.Sobel(gray, cv.CV_64F, 0, dy, ksize=ksize)
    magnitude = cv.magnitude(sobel_x, sobel_y)
    return cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def sobel_x(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Sobel edge detection in X direction
    
    Args:
        image: Input grayscale image
        ksize: Kernel size
        
    Returns:
        Horizontal edge image
    """
    gray = to_grayscale(image)
    sobel = cv.Sobel(gray, cv.CV_64F, 1, 0, ksize=ksize)
    return cv.convertScaleAbs(sobel)


def sobel_y(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Sobel edge detection in Y direction
    
    Args:
        image: Input grayscale image
        ksize: Kernel size
        
    Returns:
        Vertical edge image
    """
    gray = to_grayscale(image)
    sobel = cv.Sobel(gray, cv.CV_64F, 0, 1, ksize=ksize)
    return cv.convertScaleAbs(sobel)


def laplacian_edge(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian edge detection
    
    Args:
        image: Input grayscale image
        ksize: Kernel size
        
    Returns:
        Edge image
    """
    gray = to_grayscale(image)
    laplacian = cv.Laplacian(gray, cv.CV_16S, ksize=ksize)
    return cv.convertScaleAbs(laplacian)


def scharr_edge(image: np.ndarray) -> np.ndarray:
    """
    Scharr edge detection
    
    Args:
        image: Input grayscale image
        
    Returns:
        Edge magnitude image
    """
    gray = to_grayscale(image)
    scharr_x = cv.Scharr(gray, cv.CV_64F, 1, 0)
    scharr_y = cv.Scharr(gray, cv.CV_64F, 0, 1)
    magnitude = cv.magnitude(scharr_x, scharr_y)
    return cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def prewitt_edge(image: np.ndarray) -> np.ndarray:
    """
    Prewitt edge detection
    
    Args:
        image: Input grayscale image
        
    Returns:
        Edge magnitude image
    """
    gray = to_grayscale(image).astype(np.float64)
    
    kernel_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
    kernel_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)
    
    edge_x = cv.filter2D(gray, -1, kernel_x)
    edge_y = cv.filter2D(gray, -1, kernel_y)
    
    magnitude = np.sqrt(edge_x**2 + edge_y**2)
    return cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def roberts_edge(image: np.ndarray) -> np.ndarray:
    """
    Roberts cross edge detection
    
    Args:
        image: Input grayscale image
        
    Returns:
        Edge magnitude image
    """
    gray = to_grayscale(image).astype(np.float64)
    
    kernel_x = np.array([[1, 0], [0, -1]], dtype=np.float64)
    kernel_y = np.array([[0, 1], [-1, 0]], dtype=np.float64)
    
    edge_x = cv.filter2D(gray, -1, kernel_x)
    edge_y = cv.filter2D(gray, -1, kernel_y)
    
    magnitude = np.sqrt(edge_x**2 + edge_y**2)
    return cv.normalize(magnitude, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


def morphological_gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morphological gradient (dilation - erosion) for edge detection
    
    Args:
        image: Input image
        kernel_size: Structuring element size
        
    Returns:
        Gradient image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    return cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)


def auto_canny(image: np.ndarray, sigma: float = 0.33) -> np.ndarray:
    """
    Automatic Canny edge detection using median
    
    Args:
        image: Input grayscale image
        sigma: Percentage of median for threshold calculation
        
    Returns:
        Edge image
    """
    gray = to_grayscale(image)
    median = np.median(gray)
    
    lower = int(max(0, (1.0 - sigma) * median))
    upper = int(min(255, (1.0 + sigma) * median))
    
    return cv.Canny(gray, lower, upper)


__all__ = [
    'canny_edge',
    'sobel_edge',
    'sobel_x',
    'sobel_y',
    'laplacian_edge',
    'scharr_edge',
    'prewitt_edge',
    'roberts_edge',
    'morphological_gradient',
    'auto_canny',
]
