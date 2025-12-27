"""
Scikit-image Filters

Advanced filters using scikit-image library
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Optional, Union

try:
    from skimage import filters, restoration, feature, morphology as sk_morphology
    from skimage import exposure, segmentation, measure, transform
    from skimage.util import img_as_float, img_as_ubyte
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False


def _check_skimage():
    """Check if scikit-image is available."""
    if not SKIMAGE_AVAILABLE:
        raise ImportError(
            "scikit-image is required for this function. "
            "Install it with: pip install scikit-image"
        )


def _to_float(image: np.ndarray) -> np.ndarray:
    """Convert image to float [0, 1]."""
    if image.dtype == np.float64 or image.dtype == np.float32:
        return image
    return img_as_float(image)


def _to_uint8(image: np.ndarray) -> np.ndarray:
    """Convert image to uint8 [0, 255]."""
    if image.dtype == np.uint8:
        return image
    return img_as_ubyte(np.clip(image, 0, 1))


# =============================================================================
# Threshold (scikit-image)
# =============================================================================

def threshold_otsu_sk(image: np.ndarray) -> np.ndarray:
    """
    Otsu threshold using scikit-image.
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    thresh = filters.threshold_otsu(image)
    return (image > thresh).astype(np.uint8) * 255


def threshold_yen(image: np.ndarray) -> np.ndarray:
    """
    Yen threshold - good for images with clear bimodal distribution.
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    thresh = filters.threshold_yen(image)
    return (image > thresh).astype(np.uint8) * 255


def threshold_isodata(image: np.ndarray) -> np.ndarray:
    """
    ISODATA threshold (iterative selection method).
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    thresh = filters.threshold_isodata(image)
    return (image > thresh).astype(np.uint8) * 255


def threshold_li(image: np.ndarray) -> np.ndarray:
    """
    Li threshold - minimum cross entropy method.
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    thresh = filters.threshold_li(image)
    return (image > thresh).astype(np.uint8) * 255


def threshold_minimum(image: np.ndarray) -> np.ndarray:
    """
    Minimum threshold - finds minimum between peaks.
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    thresh = filters.threshold_minimum(image)
    return (image > thresh).astype(np.uint8) * 255


def threshold_mean_sk(image: np.ndarray) -> np.ndarray:
    """
    Mean threshold - uses mean intensity as threshold.
    
    Args:
        image: Grayscale image
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    thresh = filters.threshold_mean(image)
    return (image > thresh).astype(np.uint8) * 255


def threshold_local(
    image: np.ndarray,
    block_size: int = 35,
    method: str = 'gaussian',
    offset: float = 0
) -> np.ndarray:
    """
    Local adaptive threshold using scikit-image.
    
    Args:
        image: Grayscale image
        block_size: Size of neighborhood (must be odd)
        method: 'gaussian', 'mean', 'median'
        offset: Constant subtracted from weighted mean
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    block_size = block_size if block_size % 2 == 1 else block_size + 1
    local_thresh = filters.threshold_local(image, block_size, method=method, offset=offset)
    return (image > local_thresh).astype(np.uint8) * 255


def threshold_sauvola(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.2
) -> np.ndarray:
    """
    Sauvola threshold - good for text/document images.
    
    Args:
        image: Grayscale image
        window_size: Size of window (must be odd)
        k: Parameter controlling threshold value
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    thresh = filters.threshold_sauvola(image, window_size=window_size, k=k)
    return (image > thresh).astype(np.uint8) * 255


def threshold_niblack(
    image: np.ndarray,
    window_size: int = 25,
    k: float = 0.8
) -> np.ndarray:
    """
    Niblack threshold - local threshold for document images.
    
    Args:
        image: Grayscale image
        window_size: Size of window (must be odd)
        k: Parameter controlling threshold value
        
    Returns:
        Binary image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    window_size = window_size if window_size % 2 == 1 else window_size + 1
    thresh = filters.threshold_niblack(image, window_size=window_size, k=k)
    return (image > thresh).astype(np.uint8) * 255


def threshold_multiotsu(
    image: np.ndarray,
    classes: int = 3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Multi-Otsu threshold - segments image into multiple classes.
    
    Args:
        image: Grayscale image
        classes: Number of classes (2-5)
        
    Returns:
        Tuple of (thresholds, segmented_image)
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    thresholds = filters.threshold_multiotsu(image, classes=classes)
    regions = np.digitize(image, bins=thresholds)
    return thresholds, (regions * (255 // (classes - 1))).astype(np.uint8)


# =============================================================================
# Edge Detection (scikit-image)
# =============================================================================

def edge_roberts(image: np.ndarray) -> np.ndarray:
    """
    Roberts cross edge detection.
    
    Args:
        image: Grayscale image
        
    Returns:
        Edge image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    edges = filters.roberts(image)
    return _to_uint8(edges)


def edge_sobel_sk(image: np.ndarray) -> np.ndarray:
    """
    Sobel edge detection using scikit-image.
    
    Args:
        image: Grayscale image
        
    Returns:
        Edge image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    edges = filters.sobel(image)
    return _to_uint8(edges)


def edge_scharr_sk(image: np.ndarray) -> np.ndarray:
    """
    Scharr edge detection using scikit-image.
    
    Args:
        image: Grayscale image
        
    Returns:
        Edge image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    edges = filters.scharr(image)
    return _to_uint8(edges)


def edge_prewitt_sk(image: np.ndarray) -> np.ndarray:
    """
    Prewitt edge detection using scikit-image.
    
    Args:
        image: Grayscale image
        
    Returns:
        Edge image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    edges = filters.prewitt(image)
    return _to_uint8(edges)


def edge_farid(image: np.ndarray) -> np.ndarray:
    """
    Farid edge detection - high accuracy edge filter.
    
    Args:
        image: Grayscale image
        
    Returns:
        Edge image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    edges = filters.farid(image)
    return _to_uint8(edges)


def edge_laplace_sk(image: np.ndarray, ksize: int = 3) -> np.ndarray:
    """
    Laplacian edge detection using scikit-image.
    
    Args:
        image: Grayscale image
        ksize: Kernel size
        
    Returns:
        Edge image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    edges = filters.laplace(image, ksize=ksize)
    return _to_uint8(np.abs(edges))


def canny_sk(
    image: np.ndarray,
    sigma: float = 1.0,
    low_threshold: Optional[float] = None,
    high_threshold: Optional[float] = None
) -> np.ndarray:
    """
    Canny edge detection using scikit-image.
    
    Args:
        image: Grayscale image
        sigma: Gaussian smoothing sigma
        low_threshold: Low threshold (auto if None)
        high_threshold: High threshold (auto if None)
        
    Returns:
        Binary edge image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    edges = feature.canny(
        image, sigma=sigma,
        low_threshold=low_threshold,
        high_threshold=high_threshold
    )
    return (edges * 255).astype(np.uint8)


# =============================================================================
# Denoising (scikit-image)
# =============================================================================

def denoise_tv_chambolle(
    image: np.ndarray,
    weight: float = 0.1,
    max_num_iter: int = 200
) -> np.ndarray:
    """
    Total Variation denoising (Chambolle method).
    Good for removing noise while preserving edges.
    
    Args:
        image: Input image
        weight: Denoising weight (higher = more smoothing)
        max_num_iter: Maximum iterations
        
    Returns:
        Denoised image
    """
    _check_skimage()
    img_float = _to_float(image)
    
    if len(image.shape) == 3:
        denoised = restoration.denoise_tv_chambolle(
            img_float, weight=weight, max_num_iter=max_num_iter, channel_axis=-1
        )
    else:
        denoised = restoration.denoise_tv_chambolle(
            img_float, weight=weight, max_num_iter=max_num_iter
        )
    
    return _to_uint8(denoised)


def denoise_tv_bregman(
    image: np.ndarray,
    weight: float = 5.0,
    max_num_iter: int = 100,
    isotropic: bool = True
) -> np.ndarray:
    """
    Total Variation denoising (Split Bregman method).
    Faster than Chambolle for certain images.
    
    Args:
        image: Input image
        weight: Denoising weight
        max_num_iter: Maximum iterations
        isotropic: Use isotropic TV
        
    Returns:
        Denoised image
    """
    _check_skimage()
    img_float = _to_float(image)
    
    if len(image.shape) == 3:
        denoised = restoration.denoise_tv_bregman(
            img_float, weight=weight, max_num_iter=max_num_iter,
            isotropic=isotropic, channel_axis=-1
        )
    else:
        denoised = restoration.denoise_tv_bregman(
            img_float, weight=weight, max_num_iter=max_num_iter,
            isotropic=isotropic
        )
    
    return _to_uint8(denoised)


def denoise_bilateral_sk(
    image: np.ndarray,
    sigma_color: float = 0.05,
    sigma_spatial: float = 15
) -> np.ndarray:
    """
    Bilateral denoising using scikit-image.
    Edge-preserving smoothing.
    
    Args:
        image: Input image
        sigma_color: Color/intensity sigma
        sigma_spatial: Spatial sigma
        
    Returns:
        Denoised image
    """
    _check_skimage()
    img_float = _to_float(image)
    
    if len(image.shape) == 3:
        denoised = restoration.denoise_bilateral(
            img_float, sigma_color=sigma_color, sigma_spatial=sigma_spatial,
            channel_axis=-1
        )
    else:
        denoised = restoration.denoise_bilateral(
            img_float, sigma_color=sigma_color, sigma_spatial=sigma_spatial
        )
    
    return _to_uint8(denoised)


def denoise_wavelet(
    image: np.ndarray,
    sigma: Optional[float] = None,
    wavelet: str = 'db1',
    mode: str = 'soft'
) -> np.ndarray:
    """
    Wavelet denoising - good for various types of noise.
    
    Args:
        image: Input image
        sigma: Noise standard deviation (auto-estimated if None)
        wavelet: Wavelet type ('db1', 'haar', 'sym2', etc.)
        mode: 'soft' or 'hard' thresholding
        
    Returns:
        Denoised image
    """
    _check_skimage()
    img_float = _to_float(image)
    
    if len(image.shape) == 3:
        denoised = restoration.denoise_wavelet(
            img_float, sigma=sigma, wavelet=wavelet, mode=mode,
            channel_axis=-1, rescale_sigma=True
        )
    else:
        denoised = restoration.denoise_wavelet(
            img_float, sigma=sigma, wavelet=wavelet, mode=mode,
            rescale_sigma=True
        )
    
    return _to_uint8(denoised)


def denoise_nl_means_sk(
    image: np.ndarray,
    patch_size: int = 7,
    patch_distance: int = 11,
    h: float = 0.1
) -> np.ndarray:
    """
    Non-local means denoising using scikit-image.
    
    Args:
        image: Input image
        patch_size: Size of patches
        patch_distance: Search distance
        h: Cut-off distance (higher = more smoothing)
        
    Returns:
        Denoised image
    """
    _check_skimage()
    img_float = _to_float(image)
    
    if len(image.shape) == 3:
        denoised = restoration.denoise_nl_means(
            img_float, patch_size=patch_size, patch_distance=patch_distance,
            h=h, channel_axis=-1
        )
    else:
        denoised = restoration.denoise_nl_means(
            img_float, patch_size=patch_size, patch_distance=patch_distance, h=h
        )
    
    return _to_uint8(denoised)


# =============================================================================
# Morphology (scikit-image)
# =============================================================================

def thin(image: np.ndarray, max_num_iter: Optional[int] = None) -> np.ndarray:
    """
    Thin binary image to 1-pixel wide skeleton.
    
    Args:
        image: Binary image
        max_num_iter: Max iterations (None for until convergence)
        
    Returns:
        Thinned image
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    thinned = sk_morphology.thin(binary, max_num_iter=max_num_iter)
    return (thinned * 255).astype(np.uint8)


def skeletonize_sk(image: np.ndarray, method: str = 'lee') -> np.ndarray:
    """
    Skeletonize binary image.
    
    Args:
        image: Binary image
        method: 'lee' (fast) or 'zhang' (more accurate)
        
    Returns:
        Skeleton image
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    skeleton = sk_morphology.skeletonize(binary, method=method)
    return (skeleton * 255).astype(np.uint8)


def medial_axis_transform(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute medial axis (skeleton) and distance transform.
    
    Args:
        image: Binary image
        
    Returns:
        Tuple of (skeleton, distance_transform)
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    skeleton, distance = sk_morphology.medial_axis(binary, return_distance=True)
    return (skeleton * 255).astype(np.uint8), distance.astype(np.float32)


def remove_small_objects_sk(
    image: np.ndarray,
    min_size: int = 64,
    connectivity: int = 1
) -> np.ndarray:
    """
    Remove small connected components.
    
    Args:
        image: Binary image
        min_size: Minimum size to keep
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity
        
    Returns:
        Cleaned image
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    cleaned = sk_morphology.remove_small_objects(binary, min_size=min_size, connectivity=connectivity)
    return (cleaned * 255).astype(np.uint8)


def remove_small_holes(
    image: np.ndarray,
    area_threshold: int = 64,
    connectivity: int = 1
) -> np.ndarray:
    """
    Remove small holes in binary image.
    
    Args:
        image: Binary image
        area_threshold: Maximum hole size to fill
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity
        
    Returns:
        Filled image
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    filled = sk_morphology.remove_small_holes(binary, area_threshold=area_threshold, connectivity=connectivity)
    return (filled * 255).astype(np.uint8)


def area_opening(image: np.ndarray, area_threshold: int = 64) -> np.ndarray:
    """
    Area opening - removes bright regions smaller than threshold.
    
    Args:
        image: Grayscale image
        area_threshold: Minimum area to keep
        
    Returns:
        Filtered image
    """
    _check_skimage()
    return sk_morphology.area_opening(image, area_threshold=area_threshold)


def area_closing(image: np.ndarray, area_threshold: int = 64) -> np.ndarray:
    """
    Area closing - fills dark regions smaller than threshold.
    
    Args:
        image: Grayscale image
        area_threshold: Maximum area to fill
        
    Returns:
        Filtered image
    """
    _check_skimage()
    return sk_morphology.area_closing(image, area_threshold=area_threshold)


def white_tophat_sk(image: np.ndarray, footprint_size: int = 5) -> np.ndarray:
    """
    White top-hat transform - extracts bright spots.
    
    Args:
        image: Grayscale image
        footprint_size: Size of structuring element
        
    Returns:
        Filtered image
    """
    _check_skimage()
    footprint = sk_morphology.disk(footprint_size)
    return sk_morphology.white_tophat(image, footprint)


def black_tophat_sk(image: np.ndarray, footprint_size: int = 5) -> np.ndarray:
    """
    Black top-hat transform - extracts dark spots.
    
    Args:
        image: Grayscale image
        footprint_size: Size of structuring element
        
    Returns:
        Filtered image
    """
    _check_skimage()
    footprint = sk_morphology.disk(footprint_size)
    return sk_morphology.black_tophat(image, footprint)


# =============================================================================
# Exposure / Enhancement (scikit-image)
# =============================================================================

def equalize_hist_sk(image: np.ndarray) -> np.ndarray:
    """
    Histogram equalization using scikit-image.
    
    Args:
        image: Input image
        
    Returns:
        Equalized image
    """
    _check_skimage()
    return _to_uint8(exposure.equalize_hist(image))


def equalize_adapthist(
    image: np.ndarray,
    kernel_size: Optional[int] = None,
    clip_limit: float = 0.01
) -> np.ndarray:
    """
    Adaptive histogram equalization (CLAHE) using scikit-image.
    
    Args:
        image: Input image
        kernel_size: Tile size (None for default)
        clip_limit: Clipping limit (0-1)
        
    Returns:
        Enhanced image
    """
    _check_skimage()
    img_float = _to_float(image)
    enhanced = exposure.equalize_adapthist(img_float, kernel_size=kernel_size, clip_limit=clip_limit)
    return _to_uint8(enhanced)


def rescale_intensity(
    image: np.ndarray,
    in_range: Union[str, Tuple[float, float]] = 'image',
    out_range: Union[str, Tuple[float, float]] = (0, 255)
) -> np.ndarray:
    """
    Rescale image intensity to a range.
    
    Args:
        image: Input image
        in_range: Input range ('image' for actual, 'dtype' for dtype range, or tuple)
        out_range: Output range
        
    Returns:
        Rescaled image
    """
    _check_skimage()
    rescaled = exposure.rescale_intensity(image, in_range=in_range, out_range=out_range)
    return rescaled.astype(np.uint8) if out_range == (0, 255) else rescaled


def adjust_gamma_sk(image: np.ndarray, gamma: float = 1.0, gain: float = 1.0) -> np.ndarray:
    """
    Gamma correction using scikit-image.
    
    Args:
        image: Input image
        gamma: Gamma value (< 1 brightens, > 1 darkens)
        gain: Constant multiplier
        
    Returns:
        Adjusted image
    """
    _check_skimage()
    adjusted = exposure.adjust_gamma(image, gamma=gamma, gain=gain)
    return adjusted.astype(np.uint8) if image.dtype == np.uint8 else adjusted


def adjust_log_sk(image: np.ndarray, gain: float = 1.0, inv: bool = False) -> np.ndarray:
    """
    Logarithmic correction.
    
    Args:
        image: Input image
        gain: Constant multiplier
        inv: If True, use inverse log
        
    Returns:
        Adjusted image
    """
    _check_skimage()
    adjusted = exposure.adjust_log(image, gain=gain, inv=inv)
    return _to_uint8(adjusted)


def adjust_sigmoid(
    image: np.ndarray,
    cutoff: float = 0.5,
    gain: float = 10,
    inv: bool = False
) -> np.ndarray:
    """
    Sigmoid correction for contrast adjustment.
    
    Args:
        image: Input image
        cutoff: Midpoint of sigmoid
        gain: Slope of sigmoid
        inv: If True, use inverse sigmoid
        
    Returns:
        Adjusted image
    """
    _check_skimage()
    img_float = _to_float(image)
    adjusted = exposure.adjust_sigmoid(img_float, cutoff=cutoff, gain=gain, inv=inv)
    return _to_uint8(adjusted)


# =============================================================================
# Feature Detection (scikit-image)
# =============================================================================

def detect_blob_dog(
    image: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 50,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Detect blobs using Difference of Gaussian.
    
    Args:
        image: Grayscale image
        min_sigma: Minimum sigma for blob detection
        max_sigma: Maximum sigma for blob detection
        threshold: Detection threshold
        
    Returns:
        Array of (y, x, sigma) for each blob
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    img_float = _to_float(image)
    blobs = feature.blob_dog(img_float, min_sigma=min_sigma, max_sigma=max_sigma, threshold=threshold)
    return blobs


def detect_blob_log(
    image: np.ndarray,
    min_sigma: float = 1,
    max_sigma: float = 50,
    num_sigma: int = 10,
    threshold: float = 0.1
) -> np.ndarray:
    """
    Detect blobs using Laplacian of Gaussian.
    
    Args:
        image: Grayscale image
        min_sigma: Minimum sigma
        max_sigma: Maximum sigma
        num_sigma: Number of intermediate sigma values
        threshold: Detection threshold
        
    Returns:
        Array of (y, x, sigma) for each blob
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    img_float = _to_float(image)
    blobs = feature.blob_log(
        img_float, min_sigma=min_sigma, max_sigma=max_sigma,
        num_sigma=num_sigma, threshold=threshold
    )
    return blobs


def detect_corners_harris(
    image: np.ndarray,
    method: str = 'k',
    k: float = 0.05,
    sigma: float = 1.0
) -> np.ndarray:
    """
    Harris corner detection.
    
    Args:
        image: Grayscale image
        method: 'k' or 'eps'
        k: Harris detector parameter
        sigma: Gaussian sigma for smoothing
        
    Returns:
        Corner response image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    return feature.corner_harris(image, method=method, k=k, sigma=sigma)


def detect_corners_shi_tomasi(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Shi-Tomasi corner detection (minimum eigenvalue method).
    
    Args:
        image: Grayscale image
        sigma: Gaussian sigma
        
    Returns:
        Corner response image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    return feature.corner_shi_tomasi(image, sigma=sigma)


# =============================================================================
# Segmentation (scikit-image)
# =============================================================================

def clear_border(image: np.ndarray, buffer_size: int = 0) -> np.ndarray:
    """
    Clear objects touching the border.
    
    Args:
        image: Binary image
        buffer_size: Size of border buffer
        
    Returns:
        Cleaned image
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    cleared = segmentation.clear_border(binary, buffer_size=buffer_size)
    return (cleared * 255).astype(np.uint8)


def find_boundaries(image: np.ndarray, mode: str = 'thick') -> np.ndarray:
    """
    Find boundaries of labeled regions.
    
    Args:
        image: Labeled image
        mode: 'thick', 'inner', 'outer', 'subpixel'
        
    Returns:
        Boundary image
    """
    _check_skimage()
    boundaries = segmentation.find_boundaries(image, mode=mode)
    return (boundaries * 255).astype(np.uint8)


def label_image(image: np.ndarray, connectivity: int = 2) -> Tuple[np.ndarray, int]:
    """
    Label connected components.
    
    Args:
        image: Binary image
        connectivity: 1 for 4-connectivity, 2 for 8-connectivity
        
    Returns:
        Tuple of (labeled_image, num_labels)
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    labeled = measure.label(binary, connectivity=connectivity)
    num_labels = labeled.max()
    return labeled.astype(np.int32), num_labels


# =============================================================================
# Restoration (scikit-image)
# =============================================================================

def wiener_filter(
    image: np.ndarray,
    psf: np.ndarray,
    balance: float = 0.1
) -> np.ndarray:
    """
    Wiener deconvolution filter.
    
    Args:
        image: Blurred image
        psf: Point spread function
        balance: Regularization parameter
        
    Returns:
        Restored image
    """
    _check_skimage()
    img_float = _to_float(image)
    restored = restoration.wiener(img_float, psf, balance)
    return _to_uint8(np.clip(restored, 0, 1))


def unsupervised_wiener(
    image: np.ndarray,
    psf: np.ndarray,
    reg: Optional[np.ndarray] = None
) -> np.ndarray:
    """
    Unsupervised Wiener filter.
    
    Args:
        image: Blurred image
        psf: Point spread function
        reg: Regularization operator
        
    Returns:
        Restored image
    """
    _check_skimage()
    img_float = _to_float(image)
    restored, _ = restoration.unsupervised_wiener(img_float, psf, reg=reg)
    return _to_uint8(np.clip(restored, 0, 1))


def richardson_lucy_sk(
    image: np.ndarray,
    psf: np.ndarray,
    num_iter: int = 50
) -> np.ndarray:
    """
    Richardson-Lucy deconvolution.
    
    Args:
        image: Blurred image
        psf: Point spread function
        num_iter: Number of iterations
        
    Returns:
        Restored image
    """
    _check_skimage()
    img_float = _to_float(image)
    restored = restoration.richardson_lucy(img_float, psf, num_iter=num_iter)
    return _to_uint8(np.clip(restored, 0, 1))


def inpaint_biharmonic(
    image: np.ndarray,
    mask: np.ndarray
) -> np.ndarray:
    """
    Inpaint image using biharmonic equations.
    
    Args:
        image: Input image
        mask: Binary mask (True where to inpaint)
        
    Returns:
        Inpainted image
    """
    _check_skimage()
    img_float = _to_float(image)
    mask_bool = mask > 127 if mask.max() > 1 else mask > 0.5
    
    if len(image.shape) == 3:
        inpainted = restoration.inpaint_biharmonic(img_float, mask_bool, channel_axis=-1)
    else:
        inpainted = restoration.inpaint_biharmonic(img_float, mask_bool)
    
    return _to_uint8(inpainted)


# =============================================================================
# Transform (scikit-image)
# =============================================================================

def hough_line_transform(image: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Hough line transform.
    
    Args:
        image: Binary edge image
        
    Returns:
        Tuple of (hspace, angles, distances)
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    return transform.hough_line(binary)


def hough_line_peaks(
    hspace: np.ndarray,
    angles: np.ndarray,
    distances: np.ndarray,
    num_peaks: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract peaks from Hough transform.
    
    Args:
        hspace: Hough accumulator
        angles: Array of angles
        distances: Array of distances
        num_peaks: Number of peaks to find
        
    Returns:
        Tuple of (accum, angles, distances) for peaks
    """
    _check_skimage()
    return transform.hough_line_peaks(hspace, angles, distances, num_peaks=num_peaks)


def hough_circle_transform(
    image: np.ndarray,
    radii: np.ndarray
) -> np.ndarray:
    """
    Hough circle transform.
    
    Args:
        image: Binary edge image
        radii: Array of radii to search
        
    Returns:
        Hough accumulator array
    """
    _check_skimage()
    binary = image > 127 if image.max() > 1 else image > 0.5
    return transform.hough_circle(binary, radii)


def hough_circle_peaks(
    hspaces: np.ndarray,
    radii: np.ndarray,
    total_num_peaks: int = 10
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract peaks from circle Hough transform.
    
    Args:
        hspaces: Hough spaces for each radius
        radii: Array of radii
        total_num_peaks: Total number of peaks
        
    Returns:
        Tuple of (accums, cx, cy, radii)
    """
    _check_skimage()
    return transform.hough_circle_peaks(hspaces, radii, total_num_peaks=total_num_peaks)


# =============================================================================
# Filters (scikit-image)
# =============================================================================

def unsharp_mask_sk(
    image: np.ndarray,
    radius: float = 1.0,
    amount: float = 1.0
) -> np.ndarray:
    """
    Unsharp mask using scikit-image.
    
    Args:
        image: Input image
        radius: Blur radius
        amount: Sharpening amount
        
    Returns:
        Sharpened image
    """
    _check_skimage()
    if len(image.shape) == 3:
        sharpened = filters.unsharp_mask(image, radius=radius, amount=amount, channel_axis=-1)
    else:
        sharpened = filters.unsharp_mask(image, radius=radius, amount=amount)
    
    return _to_uint8(sharpened)


def gaussian_sk(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """
    Gaussian filter using scikit-image.
    
    Args:
        image: Input image
        sigma: Standard deviation
        
    Returns:
        Filtered image
    """
    _check_skimage()
    if len(image.shape) == 3:
        filtered = filters.gaussian(image, sigma=sigma, channel_axis=-1)
    else:
        filtered = filters.gaussian(image, sigma=sigma)
    
    return _to_uint8(filtered)


def median_sk(image: np.ndarray, footprint_size: int = 3) -> np.ndarray:
    """
    Median filter using scikit-image.
    
    Args:
        image: Input image (grayscale)
        footprint_size: Size of disk footprint
        
    Returns:
        Filtered image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    footprint = sk_morphology.disk(footprint_size)
    return filters.median(image, footprint)


def rank_filter(
    image: np.ndarray,
    footprint_size: int = 3,
    rank: int = -1
) -> np.ndarray:
    """
    Rank filter (generalized median).
    
    Args:
        image: Grayscale image
        footprint_size: Size of footprint
        rank: Rank value (-1 for median)
        
    Returns:
        Filtered image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    footprint = sk_morphology.disk(footprint_size)
    if rank == -1:
        rank = footprint.sum() // 2
    
    return filters.rank.mean(image, footprint)


def frangi_filter(
    image: np.ndarray,
    sigmas: Tuple[float, ...] = (1, 2, 3),
    black_ridges: bool = True
) -> np.ndarray:
    """
    Frangi vesselness filter - enhances tubular structures.
    
    Args:
        image: Grayscale image
        sigmas: Scales to analyze
        black_ridges: If True, detect dark ridges
        
    Returns:
        Filtered image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    img_float = _to_float(image)
    result = filters.frangi(img_float, sigmas=sigmas, black_ridges=black_ridges)
    return _to_uint8(result / result.max() if result.max() > 0 else result)


def meijering_filter(
    image: np.ndarray,
    sigmas: Tuple[float, ...] = (1,),
    black_ridges: bool = True
) -> np.ndarray:
    """
    Meijering neuriteness filter - enhances neurite structures.
    
    Args:
        image: Grayscale image
        sigmas: Scales to analyze
        black_ridges: If True, detect dark ridges
        
    Returns:
        Filtered image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    img_float = _to_float(image)
    result = filters.meijering(img_float, sigmas=sigmas, black_ridges=black_ridges)
    return _to_uint8(result / result.max() if result.max() > 0 else result)


def sato_filter(
    image: np.ndarray,
    sigmas: Tuple[float, ...] = (1, 2),
    black_ridges: bool = True
) -> np.ndarray:
    """
    Sato tubeness filter - enhances tubular structures.
    
    Args:
        image: Grayscale image
        sigmas: Scales to analyze
        black_ridges: If True, detect dark tubes
        
    Returns:
        Filtered image
    """
    _check_skimage()
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    img_float = _to_float(image)
    result = filters.sato(img_float, sigmas=sigmas, black_ridges=black_ridges)
    return _to_uint8(result / result.max() if result.max() > 0 else result)


# =============================================================================
# Exports
# =============================================================================

__all__ = [
    # Availability check
    'SKIMAGE_AVAILABLE',
    
    # Threshold
    'threshold_otsu_sk',
    'threshold_yen',
    'threshold_isodata',
    'threshold_li',
    'threshold_minimum',
    'threshold_mean_sk',
    'threshold_local',
    'threshold_sauvola',
    'threshold_niblack',
    'threshold_multiotsu',
    
    # Edge
    'edge_roberts',
    'edge_sobel_sk',
    'edge_scharr_sk',
    'edge_prewitt_sk',
    'edge_farid',
    'edge_laplace_sk',
    'canny_sk',
    
    # Denoise
    'denoise_tv_chambolle',
    'denoise_tv_bregman',
    'denoise_bilateral_sk',
    'denoise_wavelet',
    'denoise_nl_means_sk',
    
    # Morphology
    'thin',
    'skeletonize_sk',
    'medial_axis_transform',
    'remove_small_objects_sk',
    'remove_small_holes',
    'area_opening',
    'area_closing',
    'white_tophat_sk',
    'black_tophat_sk',
    
    # Exposure
    'equalize_hist_sk',
    'equalize_adapthist',
    'rescale_intensity',
    'adjust_gamma_sk',
    'adjust_log_sk',
    'adjust_sigmoid',
    
    # Feature detection
    'detect_blob_dog',
    'detect_blob_log',
    'detect_corners_harris',
    'detect_corners_shi_tomasi',
    
    # Segmentation
    'clear_border',
    'find_boundaries',
    'label_image',
    
    # Restoration
    'wiener_filter',
    'unsupervised_wiener',
    'richardson_lucy_sk',
    'inpaint_biharmonic',
    
    # Transform
    'hough_line_transform',
    'hough_line_peaks',
    'hough_circle_transform',
    'hough_circle_peaks',
    
    # Filters
    'unsharp_mask_sk',
    'gaussian_sk',
    'median_sk',
    'rank_filter',
    'frangi_filter',
    'meijering_filter',
    'sato_filter',
]
