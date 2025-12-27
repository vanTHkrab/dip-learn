"""
Filters Module

รวม filter operations ทั้งหมด
"""

from .smoothing import (
    gaussian_blur,
    median_blur,
    bilateral_filter,
    box_blur,
    stack_blur,
    blur_2d,
    average_blur,
)

from .sharpening import (
    unsharp_mask,
    laplacian_sharpen,
    kernel_sharpen,
    high_boost_filter,
    frequency_high_boost,
    custom_sharpen,
)

from .edge import (
    canny_edge,
    sobel_edge,
    sobel_x,
    sobel_y,
    laplacian_edge,
    scharr_edge,
    prewitt_edge,
    roberts_edge,
    morphological_gradient,
    auto_canny,
)

# Scikit-image filters (optional import)
try:
    from .skimage_filters import (
        # Threshold
        threshold_otsu_sk,
        threshold_yen,
        threshold_isodata,
        threshold_li,
        threshold_minimum,
        threshold_mean_sk,
        threshold_local,
        threshold_sauvola,
        threshold_niblack,
        threshold_multiotsu,
        # Edge
        edge_roberts,
        edge_sobel_sk,
        edge_scharr_sk,
        edge_prewitt_sk,
        edge_farid,
        edge_laplace_sk,
        canny_sk,
        # Denoise
        denoise_tv_chambolle,
        denoise_tv_bregman,
        denoise_bilateral_sk,
        denoise_wavelet,
        denoise_nl_means_sk,
        # Morphology
        thin,
        skeletonize_sk,
        medial_axis_transform,
        remove_small_objects_sk,
        remove_small_holes,
        area_opening,
        area_closing,
        white_tophat_sk,
        black_tophat_sk,
        # Exposure
        equalize_hist_sk,
        equalize_adapthist,
        rescale_intensity,
        adjust_gamma_sk,
        adjust_log_sk,
        adjust_sigmoid,
        # Feature detection
        detect_blob_dog,
        detect_blob_log,
        detect_corners_harris,
        detect_corners_shi_tomasi,
        # Segmentation
        clear_border,
        find_boundaries,
        label_image,
        # Restoration
        wiener_filter,
        unsupervised_wiener,
        richardson_lucy_sk,
        inpaint_biharmonic,
        # Transform
        hough_line_transform,
        hough_line_peaks,
        hough_circle_transform,
        hough_circle_peaks,
        # Filters
        unsharp_mask_sk,
        gaussian_sk,
        median_sk,
        rank_filter,
        frangi_filter,
        meijering_filter,
        sato_filter,
        SKIMAGE_AVAILABLE,
    )
    _SKIMAGE_IMPORTS = [
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
        'SKIMAGE_AVAILABLE',
    ]
except ImportError:
    SKIMAGE_AVAILABLE = False
    _SKIMAGE_IMPORTS = []


__all__ = [
    # Smoothing
    'gaussian_blur',
    'median_blur',
    'bilateral_filter',
    'box_blur',
    'stack_blur',
    'blur_2d',
    'average_blur',
    # Sharpening
    'unsharp_mask',
    'laplacian_sharpen',
    'kernel_sharpen',
    'high_boost_filter',
    'frequency_high_boost',
    'custom_sharpen',
    # Edge
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
] + _SKIMAGE_IMPORTS
