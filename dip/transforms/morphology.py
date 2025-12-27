"""
Morphological Operations

การทำ morphological operations เช่น erosion, dilation, opening, closing
"""

import cv2 as cv
import numpy as np
from typing import Literal
from enum import Enum


class KernelShape(Enum):
    """Kernel shapes for morphological operations."""
    RECT = cv.MORPH_RECT
    ELLIPSE = cv.MORPH_ELLIPSE
    CROSS = cv.MORPH_CROSS


def _get_kernel(
    kernel_size: int,
    kernel_shape: Literal["rect", "ellipse", "cross"] = "rect"
) -> np.ndarray:
    """Get structuring element."""
    shapes = {
        "rect": cv.MORPH_RECT,
        "ellipse": cv.MORPH_ELLIPSE,
        "cross": cv.MORPH_CROSS
    }
    shape = shapes.get(kernel_shape, cv.MORPH_RECT)
    return cv.getStructuringElement(shape, (kernel_size, kernel_size))


def erode(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: Literal["rect", "ellipse", "cross"] = "rect"
) -> np.ndarray:
    """
    Erosion - ลดขนาด foreground
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        iterations: Number of iterations
        kernel_shape: "rect", "ellipse", or "cross"
        
    Returns:
        Eroded image
    """
    kernel = _get_kernel(kernel_size, kernel_shape)
    return cv.erode(image, kernel, iterations=iterations)


def dilate(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1,
    kernel_shape: Literal["rect", "ellipse", "cross"] = "rect"
) -> np.ndarray:
    """
    Dilation - ขยาย foreground
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        iterations: Number of iterations
        kernel_shape: "rect", "ellipse", or "cross"
        
    Returns:
        Dilated image
    """
    kernel = _get_kernel(kernel_size, kernel_shape)
    return cv.dilate(image, kernel, iterations=iterations)


def morph_open(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Morphological opening (erosion -> dilation)
    ใช้กำจัด noise ขนาดเล็ก
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        iterations: Number of iterations
        
    Returns:
        Opened image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    return cv.morphologyEx(image, cv.MORPH_OPEN, kernel, iterations=iterations)


def morph_close(
    image: np.ndarray,
    kernel_size: int = 3,
    iterations: int = 1
) -> np.ndarray:
    """
    Morphological closing (dilation -> erosion)
    ใช้เติมช่องว่างเล็กๆ
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        iterations: Number of iterations
        
    Returns:
        Closed image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    return cv.morphologyEx(image, cv.MORPH_CLOSE, kernel, iterations=iterations)


def morph_gradient(image: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    Morphological gradient (dilation - erosion)
    ใช้หา edge
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        
    Returns:
        Gradient image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    return cv.morphologyEx(image, cv.MORPH_GRADIENT, kernel)


def top_hat(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """
    Top-hat transform (image - opening)
    ใช้แยก bright objects บน dark background
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        
    Returns:
        Top-hat image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    return cv.morphologyEx(image, cv.MORPH_TOPHAT, kernel)


def black_hat(image: np.ndarray, kernel_size: int = 9) -> np.ndarray:
    """
    Black-hat transform (closing - image)
    ใช้แยก dark objects บน bright background
    
    Args:
        image: Input image
        kernel_size: Size of structuring element
        
    Returns:
        Black-hat image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    return cv.morphologyEx(image, cv.MORPH_BLACKHAT, kernel)


def skeletonize(binary: np.ndarray) -> np.ndarray:
    """
    Skeletonization - ลดรูปให้เหลือเส้นกลาง
    
    Args:
        binary: Binary input image
        
    Returns:
        Skeleton image
    """
    binary = binary.copy()
    skeleton = np.zeros(binary.shape, np.uint8)
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    
    while True:
        eroded = cv.erode(binary, kernel)
        temp = cv.dilate(eroded, kernel)
        temp = cv.subtract(binary, temp)
        skeleton = cv.bitwise_or(skeleton, temp)
        binary = eroded.copy()
        if cv.countNonZero(binary) == 0:
            break
            
    return skeleton


def morphological_reconstruction(marker: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Morphological reconstruction
    
    Args:
        marker: Marker image
        mask: Mask image
        
    Returns:
        Reconstructed image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    prev = np.zeros_like(marker)
    current = marker.copy()
    
    while True:
        dilated = cv.dilate(current, kernel)
        current = cv.min(dilated, mask)
        if np.array_equal(current, prev):
            break
        prev = current.copy()
        
    return current


def hit_or_miss(image: np.ndarray, kernel1: np.ndarray, kernel2: np.ndarray) -> np.ndarray:
    """
    Hit-or-miss transform
    
    Args:
        image: Binary input image
        kernel1: First structuring element
        kernel2: Second structuring element
        
    Returns:
        Result image
    """
    return cv.morphologyEx(image, cv.MORPH_HITMISS, kernel1)


__all__ = [
    'erode',
    'dilate',
    'morph_open',
    'morph_close',
    'morph_gradient',
    'top_hat',
    'black_hat',
    'skeletonize',
    'morphological_reconstruction',
    'hit_or_miss',
    'KernelShape',
]
