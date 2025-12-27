"""
Transforms Module

รวม transformation operations ทั้งหมด
"""

from .color import (
    to_grayscale,
    to_bgr,
    to_rgb,
    to_hsv,
    to_lab,
    split_channels,
    merge_channels,
)

from .threshold import (
    binary_threshold,
    otsu_threshold,
    adaptive_threshold_mean,
    adaptive_threshold_gaussian,
    auto_threshold,
    triangle_threshold,
)

from .morphology import (
    erode,
    dilate,
    morph_open,
    morph_close,
    morph_gradient,
    top_hat,
    black_hat,
    skeletonize,
    morphological_reconstruction,
    KernelShape,
)

from .geometric import (
    resize,
    resize_keep_aspect,
    rotate,
    rotate_bound,
    flip,
    crop,
    crop_center,
    pad,
    pad_to_size,
    translate,
    perspective_transform,
)

from .histogram import (
    histogram_equalization,
    clahe,
    contrast_stretch,
    normalize,
    histogram_matching,
    calculate_histogram,
    normalize_lighting,
)


__all__ = [
    # Color
    'to_grayscale',
    'to_bgr',
    'to_rgb',
    'to_hsv',
    'to_lab',
    'split_channels',
    'merge_channels',
    # Threshold
    'binary_threshold',
    'otsu_threshold',
    'adaptive_threshold_mean',
    'adaptive_threshold_gaussian',
    'auto_threshold',
    'triangle_threshold',
    # Morphology
    'erode',
    'dilate',
    'morph_open',
    'morph_close',
    'morph_gradient',
    'top_hat',
    'black_hat',
    'skeletonize',
    'morphological_reconstruction',
    'KernelShape',
    # Geometric
    'resize',
    'resize_keep_aspect',
    'rotate',
    'rotate_bound',
    'flip',
    'crop',
    'crop_center',
    'pad',
    'pad_to_size',
    'translate',
    'perspective_transform',
    # Histogram
    'histogram_equalization',
    'clahe',
    'contrast_stretch',
    'normalize',
    'histogram_matching',
    'calculate_histogram',
    'normalize_lighting',
]
