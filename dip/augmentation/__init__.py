"""
Augmentation Module

Data augmentation operations for machine learning and deep learning.
Provides various image augmentation techniques including:
- Geometric transformations (flip, rotate, scale, crop, etc.)
- Color augmentations (brightness, contrast, saturation, hue)
- Noise injection (gaussian, salt-pepper, speckle)
- Blur effects (gaussian, motion, median)
- Cutout and erasing techniques
- Mixup and CutMix augmentations
- Pipeline for composing multiple augmentations
"""

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
except ImportError:
    pass

__all__ = [
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
]
