"""
Image Augmentation Module

Data augmentation operations for machine learning and deep learning.
Includes geometric transforms, color adjustments, noise injection, and more.
"""

import cv2 as cv
import numpy as np
from typing import Tuple, Optional, List, Union, Callable
import random


# =============================================================================
# Type Aliases
# =============================================================================
Image = np.ndarray
BBox = Tuple[int, int, int, int]  # (x, y, w, h)


# =============================================================================
# Random Geometric Transformations
# =============================================================================

def random_flip(
    image: Image,
    horizontal: bool = True,
    vertical: bool = False,
    p: float = 0.5
) -> Image:
    """
    Randomly flip the image horizontally and/or vertically.
    
    Args:
        image: Input image
        horizontal: Allow horizontal flip
        vertical: Allow vertical flip
        p: Probability of flipping
        
    Returns:
        Flipped image (or original if not flipped)
    """
    result = image.copy()
    
    if horizontal and random.random() < p:
        result = cv.flip(result, 1)
    
    if vertical and random.random() < p:
        result = cv.flip(result, 0)
    
    return result


def random_rotation(
    image: Image,
    angle_range: Tuple[float, float] = (-30, 30),
    border_mode: int = cv.BORDER_REFLECT_101,
    fill_value: int = 0
) -> Image:
    """
    Randomly rotate the image within a specified angle range.
    
    Args:
        image: Input image
        angle_range: (min_angle, max_angle) in degrees
        border_mode: Border mode for out-of-bounds pixels
        fill_value: Fill value for constant border mode
        
    Returns:
        Rotated image
    """
    angle = random.uniform(angle_range[0], angle_range[1])
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    return cv.warpAffine(
        image, M, (w, h),
        borderMode=border_mode,
        borderValue=fill_value
    )


def random_scale(
    image: Image,
    scale_range: Tuple[float, float] = (0.8, 1.2),
    keep_size: bool = True
) -> Image:
    """
    Randomly scale the image.
    
    Args:
        image: Input image
        scale_range: (min_scale, max_scale)
        keep_size: If True, resize back to original size
        
    Returns:
        Scaled image
    """
    scale = random.uniform(scale_range[0], scale_range[1])
    h, w = image.shape[:2]
    new_h, new_w = int(h * scale), int(w * scale)
    
    result = cv.resize(image, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    
    if keep_size:
        if scale > 1.0:
            # Crop center
            start_y = (new_h - h) // 2
            start_x = (new_w - w) // 2
            result = result[start_y:start_y + h, start_x:start_x + w]
        else:
            # Pad
            pad_h = (h - new_h) // 2
            pad_w = (w - new_w) // 2
            result = cv.copyMakeBorder(
                result,
                pad_h, h - new_h - pad_h,
                pad_w, w - new_w - pad_w,
                cv.BORDER_REFLECT_101
            )
    
    return result


def random_translate(
    image: Image,
    translate_range: Tuple[float, float] = (-0.1, 0.1),
    border_mode: int = cv.BORDER_REFLECT_101
) -> Image:
    """
    Randomly translate the image.
    
    Args:
        image: Input image
        translate_range: (min_ratio, max_ratio) as fraction of image size
        border_mode: Border mode for out-of-bounds pixels
        
    Returns:
        Translated image
    """
    h, w = image.shape[:2]
    tx = random.uniform(translate_range[0], translate_range[1]) * w
    ty = random.uniform(translate_range[0], translate_range[1]) * h
    
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv.warpAffine(image, M, (w, h), borderMode=border_mode)


def random_shear(
    image: Image,
    shear_range: Tuple[float, float] = (-0.2, 0.2),
    border_mode: int = cv.BORDER_REFLECT_101
) -> Image:
    """
    Randomly shear the image.
    
    Args:
        image: Input image
        shear_range: (min_shear, max_shear)
        border_mode: Border mode for out-of-bounds pixels
        
    Returns:
        Sheared image
    """
    h, w = image.shape[:2]
    shear_x = random.uniform(shear_range[0], shear_range[1])
    shear_y = random.uniform(shear_range[0], shear_range[1])
    
    M = np.float32([
        [1, shear_x, 0],
        [shear_y, 1, 0]
    ])
    
    return cv.warpAffine(image, M, (w, h), borderMode=border_mode)


def random_crop(
    image: Image,
    crop_size: Optional[Tuple[int, int]] = None,
    crop_ratio: float = 0.8
) -> Image:
    """
    Randomly crop a portion of the image.
    
    Args:
        image: Input image
        crop_size: (height, width) of crop. If None, use crop_ratio
        crop_ratio: Ratio of original size to crop
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    
    if crop_size is None:
        crop_h = int(h * crop_ratio)
        crop_w = int(w * crop_ratio)
    else:
        crop_h, crop_w = crop_size
    
    max_y = h - crop_h
    max_x = w - crop_w
    
    start_y = random.randint(0, max(0, max_y))
    start_x = random.randint(0, max(0, max_x))
    
    return image[start_y:start_y + crop_h, start_x:start_x + crop_w]


def random_perspective(
    image: Image,
    distortion_scale: float = 0.1
) -> Image:
    """
    Apply random perspective transformation.
    
    Args:
        image: Input image
        distortion_scale: Amount of distortion (0-1)
        
    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    
    # Original corners
    pts1 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])
    
    # Random displacement
    d = distortion_scale * min(h, w)
    pts2 = pts1 + np.float32([
        [random.uniform(-d, d), random.uniform(-d, d)],
        [random.uniform(-d, d), random.uniform(-d, d)],
        [random.uniform(-d, d), random.uniform(-d, d)],
        [random.uniform(-d, d), random.uniform(-d, d)]
    ])
    
    M = cv.getPerspectiveTransform(pts1, pts2)
    return cv.warpPerspective(image, M, (w, h), borderMode=cv.BORDER_REFLECT_101)


def random_elastic_transform(
    image: Image,
    alpha: float = 50,
    sigma: float = 5,
    alpha_affine: float = 5
) -> Image:
    """
    Apply elastic deformation to the image.
    
    Args:
        image: Input image
        alpha: Scaling factor for displacement
        sigma: Sigma for Gaussian filter
        alpha_affine: Affine transform strength
        
    Returns:
        Elastically deformed image
    """
    h, w = image.shape[:2]
    
    # Random affine
    center = np.float32([w, h]) / 2
    square_size = min(w, h) // 3
    
    pts1 = np.float32([
        center + np.array([-square_size, -square_size]),
        center + np.array([square_size, -square_size]),
        center + np.array([-square_size, square_size])
    ])
    pts2 = pts1 + np.random.uniform(-alpha_affine, alpha_affine, pts1.shape).astype(np.float32)
    M = cv.getAffineTransform(pts1, pts2)
    
    result = cv.warpAffine(image, M, (w, h), borderMode=cv.BORDER_REFLECT_101)
    
    # Random elastic deformation
    dx = cv.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    dy = cv.GaussianBlur((np.random.rand(h, w) * 2 - 1).astype(np.float32), (0, 0), sigma) * alpha
    
    x, y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (x + dx).astype(np.float32)
    map_y = (y + dy).astype(np.float32)
    
    return cv.remap(result, map_x, map_y, cv.INTER_LINEAR, borderMode=cv.BORDER_REFLECT_101)


# =============================================================================
# Color Augmentations
# =============================================================================

def random_brightness(
    image: Image,
    brightness_range: Tuple[float, float] = (-0.3, 0.3)
) -> Image:
    """
    Randomly adjust brightness.
    
    Args:
        image: Input image
        brightness_range: (min_delta, max_delta) as fraction
        
    Returns:
        Brightness-adjusted image
    """
    delta = random.uniform(brightness_range[0], brightness_range[1]) * 255
    return np.clip(image.astype(np.float32) + delta, 0, 255).astype(np.uint8)


def random_contrast(
    image: Image,
    contrast_range: Tuple[float, float] = (0.7, 1.3)
) -> Image:
    """
    Randomly adjust contrast.
    
    Args:
        image: Input image
        contrast_range: (min_factor, max_factor)
        
    Returns:
        Contrast-adjusted image
    """
    factor = random.uniform(contrast_range[0], contrast_range[1])
    mean = np.mean(image, axis=(0, 1), keepdims=True)
    return np.clip((image - mean) * factor + mean, 0, 255).astype(np.uint8)


def random_saturation(
    image: Image,
    saturation_range: Tuple[float, float] = (0.7, 1.3)
) -> Image:
    """
    Randomly adjust saturation.
    
    Args:
        image: Input BGR image
        saturation_range: (min_factor, max_factor)
        
    Returns:
        Saturation-adjusted image
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    
    factor = random.uniform(saturation_range[0], saturation_range[1])
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * factor, 0, 255)
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)


def random_hue(
    image: Image,
    hue_range: Tuple[int, int] = (-10, 10)
) -> Image:
    """
    Randomly adjust hue.
    
    Args:
        image: Input BGR image
        hue_range: (min_delta, max_delta) in degrees
        
    Returns:
        Hue-adjusted image
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    
    delta = random.randint(hue_range[0], hue_range[1])
    hsv = cv.cvtColor(image, cv.COLOR_BGR2HSV).astype(np.int16)
    hsv[:, :, 0] = (hsv[:, :, 0] + delta) % 180
    return cv.cvtColor(hsv.astype(np.uint8), cv.COLOR_HSV2BGR)


def random_color_jitter(
    image: Image,
    brightness: float = 0.2,
    contrast: float = 0.2,
    saturation: float = 0.2,
    hue: float = 0.1
) -> Image:
    """
    Apply random color jittering (brightness, contrast, saturation, hue).
    
    Args:
        image: Input image
        brightness: Max brightness adjustment
        contrast: Max contrast adjustment
        saturation: Max saturation adjustment
        hue: Max hue adjustment
        
    Returns:
        Color-jittered image
    """
    result = image.copy()
    
    # Random order of transformations
    transforms = []
    if brightness > 0:
        transforms.append(lambda img: random_brightness(img, (-brightness, brightness)))
    if contrast > 0:
        transforms.append(lambda img: random_contrast(img, (1 - contrast, 1 + contrast)))
    if saturation > 0:
        transforms.append(lambda img: random_saturation(img, (1 - saturation, 1 + saturation)))
    if hue > 0:
        transforms.append(lambda img: random_hue(img, (int(-hue * 180), int(hue * 180))))
    
    random.shuffle(transforms)
    
    for transform in transforms:
        result = transform(result)
    
    return result


def random_gamma(
    image: Image,
    gamma_range: Tuple[float, float] = (0.7, 1.5)
) -> Image:
    """
    Randomly adjust gamma.
    
    Args:
        image: Input image
        gamma_range: (min_gamma, max_gamma)
        
    Returns:
        Gamma-adjusted image
    """
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    inv_gamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
    return cv.LUT(image, table)


def random_channel_shuffle(image: Image, p: float = 0.5) -> Image:
    """
    Randomly shuffle color channels.
    
    Args:
        image: Input BGR image
        p: Probability of shuffling
        
    Returns:
        Channel-shuffled image
    """
    if len(image.shape) != 3 or image.shape[2] != 3:
        return image
    
    if random.random() < p:
        channels = [0, 1, 2]
        random.shuffle(channels)
        return image[:, :, channels]
    
    return image.copy()


def to_grayscale_augment(image: Image, p: float = 0.5) -> Image:
    """
    Randomly convert image to grayscale.
    
    Args:
        image: Input image
        p: Probability of conversion
        
    Returns:
        Grayscale image (as 3-channel) or original
    """
    if random.random() < p:
        if len(image.shape) == 3:
            gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
            return cv.cvtColor(gray, cv.COLOR_GRAY2BGR)
    return image.copy()


# =============================================================================
# Noise Augmentations
# =============================================================================

def random_gaussian_noise(
    image: Image,
    mean: float = 0,
    std_range: Tuple[float, float] = (5, 25)
) -> Image:
    """
    Add random Gaussian noise.
    
    Args:
        image: Input image
        mean: Noise mean
        std_range: (min_std, max_std) for noise standard deviation
        
    Returns:
        Noisy image
    """
    std = random.uniform(std_range[0], std_range[1])
    noise = np.random.normal(mean, std, image.shape)
    return np.clip(image.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def random_salt_pepper_noise(
    image: Image,
    amount_range: Tuple[float, float] = (0.001, 0.01),
    salt_vs_pepper: float = 0.5
) -> Image:
    """
    Add random salt and pepper noise.
    
    Args:
        image: Input image
        amount_range: (min_amount, max_amount) as fraction of pixels
        salt_vs_pepper: Ratio of salt to pepper
        
    Returns:
        Noisy image
    """
    result = image.copy()
    amount = random.uniform(amount_range[0], amount_range[1])
    
    # Salt
    num_salt = int(np.ceil(amount * image.size * salt_vs_pepper))
    coords = [np.random.randint(0, i - 1, num_salt) for i in image.shape[:2]]
    if len(image.shape) == 3:
        result[coords[0], coords[1], :] = 255
    else:
        result[coords[0], coords[1]] = 255
    
    # Pepper
    num_pepper = int(np.ceil(amount * image.size * (1 - salt_vs_pepper)))
    coords = [np.random.randint(0, i - 1, num_pepper) for i in image.shape[:2]]
    if len(image.shape) == 3:
        result[coords[0], coords[1], :] = 0
    else:
        result[coords[0], coords[1]] = 0
    
    return result


def random_speckle_noise(
    image: Image,
    intensity_range: Tuple[float, float] = (0.1, 0.3)
) -> Image:
    """
    Add random speckle (multiplicative) noise.
    
    Args:
        image: Input image
        intensity_range: (min_intensity, max_intensity)
        
    Returns:
        Noisy image
    """
    intensity = random.uniform(intensity_range[0], intensity_range[1])
    noise = np.random.randn(*image.shape) * intensity
    return np.clip(image.astype(np.float32) * (1 + noise), 0, 255).astype(np.uint8)


def random_poisson_noise(image: Image, scale: float = 1.0) -> Image:
    """
    Add Poisson noise.
    
    Args:
        image: Input image
        scale: Scale factor for noise
        
    Returns:
        Noisy image
    """
    vals = len(np.unique(image))
    vals = 2 ** np.ceil(np.log2(vals))
    noisy = np.random.poisson(image.astype(np.float32) * scale / 255.0 * vals) / vals * 255.0 / scale
    return np.clip(noisy, 0, 255).astype(np.uint8)


# =============================================================================
# Blur Augmentations
# =============================================================================

def random_gaussian_blur(
    image: Image,
    kernel_range: Tuple[int, int] = (3, 7),
    sigma_range: Tuple[float, float] = (0.1, 2.0)
) -> Image:
    """
    Apply random Gaussian blur.
    
    Args:
        image: Input image
        kernel_range: (min_kernel, max_kernel) - must be odd
        sigma_range: (min_sigma, max_sigma)
        
    Returns:
        Blurred image
    """
    ksize = random.randint(kernel_range[0] // 2, kernel_range[1] // 2) * 2 + 1
    sigma = random.uniform(sigma_range[0], sigma_range[1])
    return cv.GaussianBlur(image, (ksize, ksize), sigma)


def random_motion_blur(
    image: Image,
    kernel_range: Tuple[int, int] = (3, 15),
    angle_range: Tuple[float, float] = (0, 360)
) -> Image:
    """
    Apply random motion blur.
    
    Args:
        image: Input image
        kernel_range: (min_kernel, max_kernel)
        angle_range: (min_angle, max_angle) in degrees
        
    Returns:
        Motion-blurred image
    """
    ksize = random.randint(kernel_range[0], kernel_range[1])
    angle = random.uniform(angle_range[0], angle_range[1])
    
    # Create motion blur kernel
    kernel = np.zeros((ksize, ksize))
    kernel[ksize // 2, :] = np.ones(ksize)
    
    # Rotate kernel
    center = (ksize // 2, ksize // 2)
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    kernel = cv.warpAffine(kernel, M, (ksize, ksize))
    kernel = kernel / kernel.sum()
    
    return cv.filter2D(image, -1, kernel)


def random_median_blur(
    image: Image,
    kernel_range: Tuple[int, int] = (3, 7)
) -> Image:
    """
    Apply random median blur.
    
    Args:
        image: Input image
        kernel_range: (min_kernel, max_kernel) - must be odd
        
    Returns:
        Blurred image
    """
    ksize = random.randint(kernel_range[0] // 2, kernel_range[1] // 2) * 2 + 1
    return cv.medianBlur(image, ksize)


# =============================================================================
# Cutout / Erasing Augmentations
# =============================================================================

def random_cutout(
    image: Image,
    num_holes: int = 1,
    hole_size_range: Tuple[float, float] = (0.1, 0.3),
    fill_value: Union[int, Tuple[int, int, int]] = 0
) -> Image:
    """
    Randomly cut out rectangular regions from the image.
    
    Args:
        image: Input image
        num_holes: Number of holes to cut
        hole_size_range: (min_ratio, max_ratio) of image size
        fill_value: Value to fill the holes
        
    Returns:
        Image with cutout regions
    """
    result = image.copy()
    h, w = image.shape[:2]
    
    for _ in range(num_holes):
        hole_h = int(random.uniform(hole_size_range[0], hole_size_range[1]) * h)
        hole_w = int(random.uniform(hole_size_range[0], hole_size_range[1]) * w)
        
        y = random.randint(0, h - hole_h)
        x = random.randint(0, w - hole_w)
        
        if len(image.shape) == 3:
            result[y:y + hole_h, x:x + hole_w, :] = fill_value
        else:
            result[y:y + hole_h, x:x + hole_w] = fill_value
    
    return result


def random_grid_mask(
    image: Image,
    grid_size: Tuple[int, int] = (4, 4),
    mask_ratio: float = 0.5,
    fill_value: Union[int, Tuple[int, int, int]] = 0
) -> Image:
    """
    Apply random grid mask to the image.
    
    Args:
        image: Input image
        grid_size: (rows, cols) grid divisions
        mask_ratio: Ratio of cells to mask
        fill_value: Value to fill masked cells
        
    Returns:
        Grid-masked image
    """
    result = image.copy()
    h, w = image.shape[:2]
    rows, cols = grid_size
    
    cell_h = h // rows
    cell_w = w // cols
    
    for i in range(rows):
        for j in range(cols):
            if random.random() < mask_ratio:
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                
                if len(image.shape) == 3:
                    result[y1:y2, x1:x2, :] = fill_value
                else:
                    result[y1:y2, x1:x2] = fill_value
    
    return result


# =============================================================================
# Mixup and CutMix
# =============================================================================

def mixup(
    image1: Image,
    image2: Image,
    alpha: float = 0.2
) -> Tuple[Image, float]:
    """
    Apply mixup augmentation (blend two images).
    
    Args:
        image1: First image
        image2: Second image (should be same size)
        alpha: Beta distribution parameter
        
    Returns:
        (mixed_image, lambda) where lambda is the mix ratio
    """
    lam = np.random.beta(alpha, alpha)
    
    # Resize if necessary
    if image1.shape != image2.shape:
        image2 = cv.resize(image2, (image1.shape[1], image1.shape[0]))
    
    mixed = (lam * image1 + (1 - lam) * image2).astype(np.uint8)
    return mixed, lam


def cutmix(
    image1: Image,
    image2: Image,
    alpha: float = 1.0
) -> Tuple[Image, float]:
    """
    Apply CutMix augmentation (paste patch from one image to another).
    
    Args:
        image1: First image (base)
        image2: Second image (patch source)
        alpha: Beta distribution parameter
        
    Returns:
        (mixed_image, lambda) where lambda is the area ratio
    """
    h, w = image1.shape[:2]
    lam = np.random.beta(alpha, alpha)
    
    # Calculate patch size
    cut_ratio = np.sqrt(1 - lam)
    cut_h = int(h * cut_ratio)
    cut_w = int(w * cut_ratio)
    
    # Random position
    cx = random.randint(0, w)
    cy = random.randint(0, h)
    
    x1 = np.clip(cx - cut_w // 2, 0, w)
    x2 = np.clip(cx + cut_w // 2, 0, w)
    y1 = np.clip(cy - cut_h // 2, 0, h)
    y2 = np.clip(cy + cut_h // 2, 0, h)
    
    # Resize image2 if necessary
    if image1.shape != image2.shape:
        image2 = cv.resize(image2, (w, h))
    
    result = image1.copy()
    result[y1:y2, x1:x2] = image2[y1:y2, x1:x2]
    
    # Adjust lambda based on actual area
    lam = 1 - (x2 - x1) * (y2 - y1) / (w * h)
    
    return result, lam


# =============================================================================
# Compound Augmentation Pipeline
# =============================================================================

class AugmentationPipeline:
    """
    Pipeline for applying multiple augmentations sequentially or randomly.
    """
    
    def __init__(self, transforms: Optional[List[Callable]] = None):
        """
        Initialize augmentation pipeline.
        
        Args:
            transforms: List of augmentation functions
        """
        self.transforms = transforms or []
    
    def add(self, transform: Callable, p: float = 1.0):
        """
        Add a transform to the pipeline.
        
        Args:
            transform: Augmentation function
            p: Probability of applying this transform
        """
        self.transforms.append((transform, p))
        return self
    
    def __call__(self, image: Image) -> Image:
        """
        Apply all transforms to the image.
        
        Args:
            image: Input image
            
        Returns:
            Augmented image
        """
        result = image.copy()
        
        for transform, p in self.transforms:
            if random.random() < p:
                result = transform(result)
        
        return result
    
    def apply_batch(self, images: List[Image]) -> List[Image]:
        """
        Apply pipeline to a batch of images.
        
        Args:
            images: List of images
            
        Returns:
            List of augmented images
        """
        return [self(img) for img in images]


def get_default_augmentation_pipeline(
    strong: bool = False
) -> AugmentationPipeline:
    """
    Get a default augmentation pipeline.
    
    Args:
        strong: If True, use stronger augmentations
        
    Returns:
        AugmentationPipeline instance
    """
    pipeline = AugmentationPipeline()
    
    if strong:
        pipeline.add(lambda img: random_flip(img, horizontal=True), p=0.5)
        pipeline.add(lambda img: random_rotation(img, (-30, 30)), p=0.5)
        pipeline.add(lambda img: random_scale(img, (0.8, 1.2)), p=0.3)
        pipeline.add(lambda img: random_color_jitter(img, 0.3, 0.3, 0.3, 0.1), p=0.8)
        pipeline.add(lambda img: random_gaussian_blur(img, (3, 7)), p=0.3)
        pipeline.add(lambda img: random_gaussian_noise(img), p=0.3)
        pipeline.add(lambda img: random_cutout(img, num_holes=2), p=0.3)
    else:
        pipeline.add(lambda img: random_flip(img, horizontal=True), p=0.5)
        pipeline.add(lambda img: random_rotation(img, (-15, 15)), p=0.3)
        pipeline.add(lambda img: random_color_jitter(img, 0.2, 0.2, 0.2, 0.05), p=0.5)
        pipeline.add(lambda img: random_gaussian_blur(img, (3, 5)), p=0.2)
    
    return pipeline


# =============================================================================
# Utility Functions
# =============================================================================

def augment_batch(
    images: List[Image],
    augment_fn: Callable,
    num_augments: int = 1
) -> List[Image]:
    """
    Apply augmentation to a batch of images.
    
    Args:
        images: List of input images
        augment_fn: Augmentation function
        num_augments: Number of augmented versions per image
        
    Returns:
        List of augmented images
    """
    results = []
    for img in images:
        for _ in range(num_augments):
            results.append(augment_fn(img))
    return results


def create_augmented_dataset(
    images: List[Image],
    labels: List[int],
    pipeline: AugmentationPipeline,
    augment_factor: int = 5,
    include_original: bool = True
) -> Tuple[List[Image], List[int]]:
    """
    Create an augmented dataset from original images.
    
    Args:
        images: List of original images
        labels: List of corresponding labels
        pipeline: Augmentation pipeline to use
        augment_factor: Number of augmented versions per image
        include_original: Whether to include original images
        
    Returns:
        (augmented_images, augmented_labels)
    """
    aug_images = []
    aug_labels = []
    
    for img, label in zip(images, labels):
        if include_original:
            aug_images.append(img)
            aug_labels.append(label)
        
        for _ in range(augment_factor):
            aug_images.append(pipeline(img))
            aug_labels.append(label)
    
    return aug_images, aug_labels
