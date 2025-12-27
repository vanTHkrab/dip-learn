"""
Denoising Operations

การลด noise จากภาพ
"""

import cv2 as cv
import numpy as np
from typing import Tuple

from ..transforms.color import to_grayscale


def non_local_means_denoising(
    image: np.ndarray,
    h: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Non-local means denoising - high quality denoising
    
    Args:
        image: Input grayscale image
        h: Filter strength (higher = more denoising)
        template_window_size: Should be odd
        search_window_size: Should be odd
        
    Returns:
        Denoised image
    """
    gray = to_grayscale(image)
    return cv.fastNlMeansDenoising(
        gray, None, h, template_window_size, search_window_size
    )


def non_local_means_colored(
    image: np.ndarray,
    h: float = 10,
    h_color: float = 10,
    template_window_size: int = 7,
    search_window_size: int = 21
) -> np.ndarray:
    """
    Non-local means denoising for colored images
    
    Args:
        image: Input BGR image
        h: Filter strength for luminance
        h_color: Filter strength for color components
        template_window_size: Should be odd
        search_window_size: Should be odd
        
    Returns:
        Denoised image
    """
    if len(image.shape) == 2:
        return non_local_means_denoising(image, h, template_window_size, search_window_size)
    
    return cv.fastNlMeansDenoisingColored(
        image, None, h, h_color, template_window_size, search_window_size
    )


def remove_salt_pepper_noise(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Remove salt and pepper noise using median filter
    
    Args:
        image: Input image
        ksize: Kernel size (must be odd)
        
    Returns:
        Denoised image
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv.medianBlur(image, ksize)


def denoise_morphological(
    image: np.ndarray,
    kernel_size: int = 2
) -> np.ndarray:
    """
    Denoise using morphological operations (open -> close)
    
    Args:
        image: Binary input image
        kernel_size: Size of structuring element
        
    Returns:
        Cleaned image
    """
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (kernel_size, kernel_size))
    cleaned = cv.morphologyEx(image, cv.MORPH_OPEN, kernel)
    cleaned = cv.morphologyEx(cleaned, cv.MORPH_CLOSE, kernel)
    return cleaned


def denoise_bilateral(
    image: np.ndarray,
    d: int = 9,
    sigma_color: float = 75,
    sigma_space: float = 75
) -> np.ndarray:
    """
    Denoise using bilateral filter (edge-preserving)
    
    Args:
        image: Input image
        d: Diameter of each pixel neighborhood
        sigma_color: Filter sigma in color space
        sigma_space: Filter sigma in coordinate space
        
    Returns:
        Denoised image
    """
    return cv.bilateralFilter(image, d, sigma_color, sigma_space)


def denoise_gaussian(
    image: np.ndarray,
    ksize: int = 5,
    sigma: float = 0
) -> np.ndarray:
    """
    Denoise using Gaussian blur
    
    Args:
        image: Input image
        ksize: Kernel size (must be odd)
        sigma: Standard deviation
        
    Returns:
        Denoised image
    """
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    return cv.GaussianBlur(image, (ksize, ksize), sigma)


def anisotropic_diffusion(
    image: np.ndarray,
    iterations: int = 15,
    kappa: float = 30.0,
    gamma: float = 0.2
) -> np.ndarray:
    """
    Anisotropic diffusion (Perona-Malik) denoising
    
    Args:
        image: Input grayscale image
        iterations: Number of iterations
        kappa: Conduction coefficient
        gamma: Integration constant (0 < gamma <= 0.25)
        
    Returns:
        Denoised image
    """
    gray = to_grayscale(image)
    img = gray.astype(np.float32)
    
    for _ in range(iterations):
        north = np.roll(img, -1, axis=0) - img
        south = np.roll(img, 1, axis=0) - img
        east = np.roll(img, -1, axis=1) - img
        west = np.roll(img, 1, axis=1) - img

        c_n = np.exp(-(north / kappa) ** 2)
        c_s = np.exp(-(south / kappa) ** 2)
        c_e = np.exp(-(east / kappa) ** 2)
        c_w = np.exp(-(west / kappa) ** 2)

        img += gamma * (c_n * north + c_s * south + c_e * east + c_w * west)
    
    return img.clip(0, 255).astype(np.uint8)


def richardson_lucy_deblur(
    image: np.ndarray,
    iterations: int = 15,
    psf_size: int = 5
) -> np.ndarray:
    """
    Richardson-Lucy deconvolution for deblurring
    
    Args:
        image: Input grayscale image
        iterations: Number of iterations
        psf_size: Point spread function size
        
    Returns:
        Deblurred image
    """
    gray = to_grayscale(image)
    
    # Create uniform PSF
    psf = np.ones((psf_size, psf_size), dtype=np.float32)
    psf /= psf.sum()
    psf_mirror = psf[::-1, ::-1]
    
    EPS = 1e-6
    img = gray.astype(np.float32) / 255.0 + EPS
    estimate = np.full_like(img, 0.5)

    for _ in range(iterations):
        conv_estimate = cv.filter2D(estimate, -1, psf, borderType=cv.BORDER_REPLICATE)
        relative_blur = img / (conv_estimate + EPS)
        estimate = estimate * cv.filter2D(relative_blur, -1, psf_mirror, borderType=cv.BORDER_REPLICATE)
        estimate = estimate.clip(0.0, 1.0)

    return cv.normalize(estimate, None, 0, 255, cv.NORM_MINMAX).astype(np.uint8)


__all__ = [
    'non_local_means_denoising',
    'non_local_means_colored',
    'remove_salt_pepper_noise',
    'denoise_morphological',
    'denoise_bilateral',
    'denoise_gaussian',
    'anisotropic_diffusion',
    'richardson_lucy_deblur',
]
