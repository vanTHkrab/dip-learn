"""
Images Process Library
======================

A comprehensive image processing library built with OpenCV and NumPy.
Organized into modular subpackages for easy use and maintenance.

Subpackages:
- core: Core operations and pipeline system
- transforms: Image transformations (color, threshold, morphology, geometric, histogram)
- filters: Image filters (smoothing, sharpening, edge detection)
- enhancement: Image enhancement (brightness, contrast, denoising)
- presets: Pre-built processing pipelines
- utils: Utility functions (I/O, visualization, metrics)
- cli: Command-line interface

Usage:
------
# Quick operations
from images_process import ImageOps, ImagePipeline
img = ImageOps.to_grayscale(img)
img = ImageOps.gaussian_blur(img, 5)

# Using pipeline
pipeline = ImagePipeline()
pipeline.add_step('grayscale', ImageOps.to_grayscale)
pipeline.add_step('blur', ImageOps.gaussian_blur, ksize=5)
result = pipeline.run(img)

# Using presets
from images_process import PresetPipelines
result = PresetPipelines.ocr_basic().run(img)

# Direct function access
from images_process import to_grayscale, gaussian_blur, canny_edge
"""

from .__version__ import __version__, __author__

# =============================================================================
# Core Components
# =============================================================================
from .core import (
    ImageOps,
    ImagePipeline,
    PipelineStep,
    PipelineResult,
    quick_process,
    apply_sequence,
)

# =============================================================================
# Annotation
# =============================================================================
from .annotated import (
    # Basic Shapes
    draw_rectangle,
    draw_circle,
    draw_line,
    draw_ellipse,
    draw_polygon,
    draw_filled_polygon,
    draw_polylines,
    draw_arrow,
    
    # Text Operations
    draw_text,
    draw_text_with_background,
    get_text_size,
    draw_multiline_text,
    
    # Annotation Utilities
    draw_bounding_box,
    draw_bounding_boxes,
    draw_contours,
    draw_keypoints,
    draw_grid,
    draw_crosshair,
    draw_marker,
    
    # Overlay Operations
    overlay_image,
    add_alpha_channel,
    create_mask_overlay,
)

# =============================================================================
# Transforms - Color
# =============================================================================
from .transforms.color import (
    to_grayscale,
    to_bgr,
    to_rgb,
    to_hsv,
    to_lab,
    split_channels,
    merge_channels,
)

# =============================================================================
# Transforms - Threshold
# =============================================================================
from .transforms.threshold import (
    binary_threshold,
    otsu_threshold,
    adaptive_threshold_mean,
    adaptive_threshold_gaussian,
    auto_threshold,
    triangle_threshold,
)

# =============================================================================
# Transforms - Morphology
# =============================================================================
from .transforms.morphology import (
    erode,
    dilate,
    morph_open,
    morph_close,
    morph_gradient,
    top_hat,
    black_hat,
    skeletonize,
)

# =============================================================================
# Transforms - Geometric
# =============================================================================
from .transforms.geometric import (
    resize,
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

# =============================================================================
# Transforms - Histogram
# =============================================================================
from .transforms.histogram import (
    histogram_equalization,
    clahe,
    contrast_stretch,
    normalize,
    histogram_matching,
    normalize_lighting,
)

# =============================================================================
# Filters - Smoothing
# =============================================================================
from .filters.smoothing import (
    gaussian_blur,
    median_blur,
    bilateral_filter,
    box_blur,
    stack_blur,
    blur_2d,
)

# =============================================================================
# Filters - Sharpening
# =============================================================================
from .filters.sharpening import (
    unsharp_mask,
    laplacian_sharpen,
    kernel_sharpen,
    high_boost_filter,
    frequency_high_boost,
)

# =============================================================================
# Filters - Edge Detection
# =============================================================================
from .filters.edge import (
    canny_edge,
    sobel_edge,
    sobel_x,
    sobel_y,
    laplacian_edge,
    scharr_edge,
    prewitt_edge,
    roberts_edge,
    auto_canny,
)

# =============================================================================
# Filters - Scikit-image (Optional)
# =============================================================================
try:
    from .filters.skimage_filters import (
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
except ImportError:
    SKIMAGE_AVAILABLE = False

# =============================================================================
# Enhancement - Brightness/Contrast
# =============================================================================
from .enhancement.brightness import (
    adjust_brightness,
    adjust_contrast,
    gamma_correction,
    auto_brightness_contrast,
    sigmoid_correction,
    log_transform,
    power_law_transform,
)

# =============================================================================
# Enhancement - Denoising
# =============================================================================
from .enhancement.denoise import (
    non_local_means_denoising,
    remove_salt_pepper_noise,
    denoise_morphological,
    denoise_bilateral,
    denoise_gaussian,
    anisotropic_diffusion,
    richardson_lucy_deblur,
)

# =============================================================================
# Augmentation - Data Augmentation for ML/DL
# =============================================================================
try:
    from .augmentation import (
        # Geometric Transformations
        random_flip,
        random_rotation,
        random_scale,
        random_translate,
        random_shear,
        random_crop,
        random_perspective,
        random_elastic_transform,
        
        # Color Augmentations
        random_brightness,
        random_contrast,
        random_saturation,
        random_hue,
        random_color_jitter,
        random_gamma,
        random_channel_shuffle,
        to_grayscale_augment,
        
        # Noise Augmentations
        random_gaussian_noise,
        random_salt_pepper_noise,
        random_speckle_noise,
        random_poisson_noise,
        
        # Blur Augmentations
        random_gaussian_blur,
        random_motion_blur,
        random_median_blur,
        
        # Cutout / Erasing
        random_cutout,
        random_grid_mask,
        
        # Mixup and CutMix
        mixup,
        cutmix,
        
        # Pipeline
        AugmentationPipeline,
        get_default_augmentation_pipeline,
        
        # Utility Functions
        augment_batch,
        create_augmented_dataset,
    )
    AUGMENTATION_AVAILABLE = True
except ImportError:
    AUGMENTATION_AVAILABLE = False

# =============================================================================
# Presets
# =============================================================================
from .presets import PresetPipelines

# =============================================================================
# Utils - I/O
# =============================================================================
from .utils.io import (
    load_image,
    save_image,
    load_images_from_dir,
    image_from_bytes,
    image_to_bytes,
)

# =============================================================================
# Utils - Visualization
# =============================================================================
from .utils.visualization import (
    visualize_pipeline_result,
    compare_images,
    show_image,
    show_histogram,
    create_comparison_grid,
)

# =============================================================================
# Utils - Metrics
# =============================================================================
from .utils.metrics import (
    variance_of_laplacian,
    is_blurry,
    calculate_snr,
    calculate_contrast,
    calculate_entropy,
    calculate_psnr,
    calculate_ssim,
    get_image_stats,
)

# =============================================================================
# Web Interface (Optional - requires streamlit)
# =============================================================================
try:
    from .web import launch_app, run_server
    WEB_AVAILABLE = True
except ImportError:
    WEB_AVAILABLE = False
    launch_app = None
    run_server = None

# =============================================================================
# All Public Symbols
# =============================================================================
__all__ = [
    # Version info
    '__version__',
    '__author__',
    
    # Core
    'ImageOps',
    'ImagePipeline',
    'PipelineStep',
    'PipelineResult',
    'quick_process',
    'apply_sequence',
    
    # Annotation - Basic Shapes
    'draw_rectangle',
    'draw_circle',
    'draw_line',
    'draw_ellipse',
    'draw_polygon',
    'draw_filled_polygon',
    'draw_polylines',
    'draw_arrow',
    
    # Annotation - Text
    'draw_text',
    'draw_text_with_background',
    'get_text_size',
    'draw_multiline_text',
    
    # Annotation - Utilities
    'draw_bounding_box',
    'draw_bounding_boxes',
    'draw_contours',
    'draw_keypoints',
    'draw_grid',
    'draw_crosshair',
    'draw_marker',
    
    # Annotation - Overlay
    'overlay_image',
    'add_alpha_channel',
    'create_mask_overlay',
    
    # Transforms - Color
    'to_grayscale',
    'to_bgr',
    'to_rgb',
    'to_hsv',
    'to_lab',
    'split_channels',
    'merge_channels',
    
    # Transforms - Threshold
    'binary_threshold',
    'otsu_threshold',
    'adaptive_threshold_mean',
    'adaptive_threshold_gaussian',
    'auto_threshold',
    'triangle_threshold',
    
    # Transforms - Morphology
    'erode',
    'dilate',
    'morph_open',
    'morph_close',
    'morph_gradient',
    'top_hat',
    'black_hat',
    'skeletonize',
    
    # Transforms - Geometric
    'resize',
    'rotate',
    'rotate_bound',
    'flip',
    'crop',
    'crop_center',
    'pad',
    'pad_to_size',
    'translate',
    'perspective_transform',
    
    # Transforms - Histogram
    'histogram_equalization',
    'clahe',
    'contrast_stretch',
    'normalize',
    'histogram_matching',
    'normalize_lighting',
    
    # Filters - Smoothing
    'gaussian_blur',
    'median_blur',
    'bilateral_filter',
    'box_blur',
    'stack_blur',
    'blur_2d',
    
    # Filters - Sharpening
    'unsharp_mask',
    'laplacian_sharpen',
    'kernel_sharpen',
    'high_boost_filter',
    'frequency_high_boost',
    
    # Filters - Edge
    'canny_edge',
    'sobel_edge',
    'sobel_x',
    'sobel_y',
    'laplacian_edge',
    'scharr_edge',
    'prewitt_edge',
    'roberts_edge',
    'auto_canny',
    
    # Enhancement - Brightness
    'adjust_brightness',
    'adjust_contrast',
    'gamma_correction',
    'auto_brightness_contrast',
    'sigmoid_correction',
    'log_transform',
    'power_law_transform',
    
    # Enhancement - Denoise
    'non_local_means_denoising',
    'remove_salt_pepper_noise',
    'denoise_morphological',
    'denoise_bilateral',
    'denoise_gaussian',
    'anisotropic_diffusion',
    'richardson_lucy_deblur',
    
    # Presets
    'PresetPipelines',
    
    # Utils - I/O
    'load_image',
    'save_image',
    'load_images_from_dir',
    'image_from_bytes',
    'image_to_bytes',
    
    # Utils - Visualization
    'visualize_pipeline_result',
    'compare_images',
    'show_image',
    'show_histogram',
    'create_comparison_grid',
    
    # Utils - Metrics
    'variance_of_laplacian',
    'is_blurry',
    'calculate_snr',
    'calculate_contrast',
    'calculate_entropy',
    'calculate_psnr',
    'calculate_ssim',
    'get_image_stats',
]

# Add augmentation functions to __all__ if available
_AUGMENTATION_ALL = [
    # Geometric Transformations
    'random_flip',
    'random_rotation',
    'random_scale',
    'random_translate',
    'random_shear',
    'random_crop',
    'random_perspective',
    'random_elastic_transform',
    
    # Color Augmentations
    'random_brightness',
    'random_contrast',
    'random_saturation',
    'random_hue',
    'random_color_jitter',
    'random_gamma',
    'random_channel_shuffle',
    'to_grayscale_augment',
    
    # Noise Augmentations
    'random_gaussian_noise',
    'random_salt_pepper_noise',
    'random_speckle_noise',
    'random_poisson_noise',
    
    # Blur Augmentations
    'random_gaussian_blur',
    'random_motion_blur',
    'random_median_blur',
    
    # Cutout / Erasing
    'random_cutout',
    'random_grid_mask',
    
    # Mixup and CutMix
    'mixup',
    'cutmix',
    
    # Pipeline
    'AugmentationPipeline',
    'get_default_augmentation_pipeline',
    
    # Utility Functions
    'augment_batch',
    'create_augmented_dataset',
    
    # Flag
    'AUGMENTATION_AVAILABLE',
]

try:
    if AUGMENTATION_AVAILABLE:
        __all__.extend(_AUGMENTATION_ALL)
except NameError:
    pass

# Add scikit-image functions to __all__ if available
_SKIMAGE_ALL = [
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

try:
    if SKIMAGE_AVAILABLE:
        __all__.extend(_SKIMAGE_ALL)
except NameError:
    pass


# =============================================================================
# Convenience Functions for Quick Access
# =============================================================================
def process_for_ocr(image, preset='basic'):
    """
    Quick function to process image for OCR.
    
    Args:
        image: Input image (numpy array or file path)
        preset: 'basic' or 'advanced'
        
    Returns:
        Processed image ready for OCR
    """
    if isinstance(image, str):
        image = load_image(image)
    
    if preset == 'basic':
        pipeline = PresetPipelines.ocr_basic()
    else:
        pipeline = PresetPipelines.ocr_advanced()
    
    return pipeline.run(image).output


def process_seven_segment(image):
    """
    Quick function to process seven segment display images.
    
    Args:
        image: Input image (numpy array or file path)
        
    Returns:
        Processed image optimized for seven segment recognition
    """
    if isinstance(image, str):
        image = load_image(image)
    
    return PresetPipelines.seven_segment().run(image).output


def batch_process(images, pipeline):
    """
    Process multiple images with the same pipeline.
    
    Args:
        images: List of images (numpy arrays or file paths)
        pipeline: ImagePipeline instance or preset name
        
    Returns:
        List of processed images
    """
    if isinstance(pipeline, str):
        pipeline_map = {
            'ocr_basic': PresetPipelines.ocr_basic,
            'ocr_advanced': PresetPipelines.ocr_advanced,
            'seven_segment': PresetPipelines.seven_segment,
            'denoise': PresetPipelines.denoise_light,
            'enhance': PresetPipelines.enhance_contrast,
            'document': PresetPipelines.document_scan,
        }
        pipeline = pipeline_map.get(pipeline, PresetPipelines.ocr_basic)()
    
    results = []
    for img in images:
        if isinstance(img, str):
            img = load_image(img)
        results.append(pipeline.run(img).output)
    
    return results


# Add convenience functions to __all__
__all__.extend([
    'process_for_ocr',
    'process_seven_segment',
    'batch_process',
])

# Add web interface to __all__ if available
if WEB_AVAILABLE:
    __all__.extend([
        'launch_app',
        'run_server',
        'WEB_AVAILABLE',
    ])
