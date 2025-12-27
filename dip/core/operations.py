"""
Image Operations Class

รวม operations ทั้งหมดไว้ใน static class เดียว
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Optional, Union, Literal

from .. import transforms
from .. import filters
from .. import enhancement


class ImageOps:
    """
    Static class containing all image processing operations.
    
    แต่ละ method รับ numpy array และ return numpy array
    สามารถใช้งานได้ทั้งแบบเรียก function โดยตรง หรือผ่าน Pipeline
    """
    
    # =====================================================================
    # COLOR SPACE CONVERSION
    # =====================================================================
    
    to_grayscale = staticmethod(transforms.to_grayscale)
    to_bgr = staticmethod(transforms.to_bgr)
    to_rgb = staticmethod(transforms.to_rgb)
    to_hsv = staticmethod(transforms.to_hsv)
    to_lab = staticmethod(transforms.to_lab)
    
    # =====================================================================
    # THRESHOLDING
    # =====================================================================
    
    binary_threshold = staticmethod(transforms.binary_threshold)
    otsu_threshold = staticmethod(transforms.otsu_threshold)
    adaptive_threshold_mean = staticmethod(transforms.adaptive_threshold_mean)
    adaptive_threshold_gaussian = staticmethod(transforms.adaptive_threshold_gaussian)
    auto_threshold = staticmethod(transforms.auto_threshold)
    triangle_threshold = staticmethod(transforms.triangle_threshold)
    
    # =====================================================================
    # HISTOGRAM OPERATIONS
    # =====================================================================
    
    histogram_equalization = staticmethod(transforms.histogram_equalization)
    clahe = staticmethod(transforms.clahe)
    contrast_stretch = staticmethod(transforms.contrast_stretch)
    normalize = staticmethod(transforms.normalize)
    histogram_matching = staticmethod(transforms.histogram_matching)
    normalize_lighting = staticmethod(transforms.normalize_lighting)
    
    # =====================================================================
    # MORPHOLOGICAL OPERATIONS
    # =====================================================================
    
    erode = staticmethod(transforms.erode)
    dilate = staticmethod(transforms.dilate)
    morph_open = staticmethod(transforms.morph_open)
    morph_close = staticmethod(transforms.morph_close)
    morph_gradient = staticmethod(transforms.morph_gradient)
    top_hat = staticmethod(transforms.top_hat)
    black_hat = staticmethod(transforms.black_hat)
    skeletonize = staticmethod(transforms.skeletonize)
    
    # =====================================================================
    # GEOMETRIC TRANSFORMATIONS
    # =====================================================================
    
    resize = staticmethod(transforms.resize)
    resize_keep_aspect = staticmethod(transforms.resize_keep_aspect)
    rotate = staticmethod(transforms.rotate)
    rotate_bound = staticmethod(transforms.rotate_bound)
    flip = staticmethod(transforms.flip)
    crop = staticmethod(transforms.crop)
    crop_center = staticmethod(transforms.crop_center)
    pad = staticmethod(transforms.pad)
    pad_to_size = staticmethod(transforms.pad_to_size)
    translate = staticmethod(transforms.translate)
    
    # =====================================================================
    # SMOOTHING / BLURRING
    # =====================================================================
    
    gaussian_blur = staticmethod(filters.gaussian_blur)
    median_blur = staticmethod(filters.median_blur)
    bilateral_filter = staticmethod(filters.bilateral_filter)
    box_blur = staticmethod(filters.box_blur)
    
    # =====================================================================
    # SHARPENING
    # =====================================================================
    
    unsharp_mask = staticmethod(filters.unsharp_mask)
    laplacian_sharpen = staticmethod(filters.laplacian_sharpen)
    kernel_sharpen = staticmethod(filters.kernel_sharpen)
    high_boost_filter = staticmethod(filters.high_boost_filter)
    
    # =====================================================================
    # EDGE DETECTION
    # =====================================================================
    
    canny_edge = staticmethod(filters.canny_edge)
    sobel_edge = staticmethod(filters.sobel_edge)
    laplacian_edge = staticmethod(filters.laplacian_edge)
    scharr_edge = staticmethod(filters.scharr_edge)
    prewitt_edge = staticmethod(filters.prewitt_edge)
    morphological_gradient = staticmethod(filters.morphological_gradient)
    auto_canny = staticmethod(filters.auto_canny)
    
    # =====================================================================
    # BRIGHTNESS & CONTRAST ADJUSTMENT
    # =====================================================================
    
    adjust_brightness = staticmethod(enhancement.adjust_brightness)
    adjust_contrast = staticmethod(enhancement.adjust_contrast)
    adjust_brightness_contrast = staticmethod(enhancement.adjust_brightness_contrast)
    gamma_correction = staticmethod(enhancement.gamma_correction)
    auto_brightness_contrast = staticmethod(enhancement.auto_brightness_contrast)
    
    # =====================================================================
    # DENOISING
    # =====================================================================
    
    non_local_means_denoising = staticmethod(enhancement.non_local_means_denoising)
    remove_salt_pepper_noise = staticmethod(enhancement.remove_salt_pepper_noise)
    denoise_morphological = staticmethod(enhancement.denoise_morphological)
    denoise_bilateral = staticmethod(enhancement.denoise_bilateral)
    anisotropic_diffusion = staticmethod(enhancement.anisotropic_diffusion)
    
    # =====================================================================
    # UTILITY FUNCTIONS
    # =====================================================================
    
    @staticmethod
    def invert(image: np.ndarray) -> np.ndarray:
        """Invert image (bitwise NOT)"""
        return cv.bitwise_not(image)
    
    @staticmethod
    def variance_of_laplacian(image: np.ndarray) -> float:
        """Calculate variance of Laplacian (sharpness measure)"""
        gray = transforms.to_grayscale(image)
        return float(cv.Laplacian(gray, cv.CV_64F).var())
    
    @staticmethod
    def is_blurry(image: np.ndarray, threshold: float = 100.0) -> bool:
        """Check if image is blurry"""
        return ImageOps.variance_of_laplacian(image) < threshold


__all__ = ['ImageOps']
