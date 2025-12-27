"""
Test suite for augmentation module.
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from augmentation import (
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


class TestGeometricTransformations:
    """Test geometric transformation augmentations."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample color image for testing."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def grayscale_image(self):
        """Create a sample grayscale image for testing."""
        return np.random.randint(0, 256, (100, 100), dtype=np.uint8)
    
    def test_random_flip_horizontal(self, sample_image):
        """Test horizontal flip."""
        result = random_flip(sample_image, horizontal=True, vertical=False, p=1.0)
        assert result.shape == sample_image.shape
        assert result.dtype == sample_image.dtype
    
    def test_random_flip_vertical(self, sample_image):
        """Test vertical flip."""
        result = random_flip(sample_image, horizontal=False, vertical=True, p=1.0)
        assert result.shape == sample_image.shape
    
    def test_random_flip_no_change(self, sample_image):
        """Test flip with p=0 returns original."""
        result = random_flip(sample_image, horizontal=True, vertical=True, p=0.0)
        np.testing.assert_array_equal(result, sample_image)
    
    def test_random_rotation(self, sample_image):
        """Test random rotation."""
        result = random_rotation(sample_image, angle_range=(-30, 30))
        assert result.shape == sample_image.shape
    
    def test_random_scale(self, sample_image):
        """Test random scale."""
        result = random_scale(sample_image, scale_range=(0.8, 1.2), keep_size=True)
        assert result.shape == sample_image.shape
    
    def test_random_translate(self, sample_image):
        """Test random translation."""
        result = random_translate(sample_image, max_dx=0.1, max_dy=0.1)
        assert result.shape == sample_image.shape
    
    def test_random_shear(self, sample_image):
        """Test random shear."""
        result = random_shear(sample_image, shear_range=(-0.2, 0.2))
        assert result.shape == sample_image.shape
    
    def test_random_crop(self, sample_image):
        """Test random crop."""
        result = random_crop(sample_image, crop_size=(80, 80))
        assert result.shape == (80, 80, 3)
    
    def test_random_perspective(self, sample_image):
        """Test random perspective transform."""
        result = random_perspective(sample_image, distortion_scale=0.3)
        assert result.shape == sample_image.shape
    
    def test_random_elastic_transform(self, sample_image):
        """Test elastic transform."""
        result = random_elastic_transform(sample_image, alpha=50, sigma=5)
        assert result.shape == sample_image.shape


class TestColorAugmentations:
    """Test color augmentation functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample color image for testing."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_random_brightness(self, sample_image):
        """Test random brightness adjustment."""
        result = random_brightness(sample_image, brightness_range=(-0.3, 0.3))
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_random_contrast(self, sample_image):
        """Test random contrast adjustment."""
        result = random_contrast(sample_image, contrast_range=(0.7, 1.3))
        assert result.shape == sample_image.shape
    
    def test_random_saturation(self, sample_image):
        """Test random saturation adjustment."""
        result = random_saturation(sample_image, saturation_range=(0.5, 1.5))
        assert result.shape == sample_image.shape
    
    def test_random_hue(self, sample_image):
        """Test random hue adjustment."""
        result = random_hue(sample_image, hue_range=(-0.1, 0.1))
        assert result.shape == sample_image.shape
    
    def test_random_color_jitter(self, sample_image):
        """Test random color jitter."""
        result = random_color_jitter(
            sample_image,
            brightness=0.3,
            contrast=0.3,
            saturation=0.3,
            hue=0.1
        )
        assert result.shape == sample_image.shape
    
    def test_random_gamma(self, sample_image):
        """Test random gamma correction."""
        result = random_gamma(sample_image, gamma_range=(0.8, 1.2))
        assert result.shape == sample_image.shape
    
    def test_random_channel_shuffle(self, sample_image):
        """Test random channel shuffle."""
        result = random_channel_shuffle(sample_image, p=1.0)
        assert result.shape == sample_image.shape
    
    def test_to_grayscale_augment(self, sample_image):
        """Test random grayscale conversion."""
        result = to_grayscale_augment(sample_image, p=1.0)
        assert result.shape == sample_image.shape


class TestNoiseAugmentations:
    """Test noise augmentation functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample color image for testing."""
        return np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    
    def test_random_gaussian_noise(self, sample_image):
        """Test Gaussian noise injection."""
        result = random_gaussian_noise(sample_image, mean=0, std_range=(10, 30))
        assert result.shape == sample_image.shape
    
    def test_random_salt_pepper_noise(self, sample_image):
        """Test salt and pepper noise injection."""
        result = random_salt_pepper_noise(sample_image, amount_range=(0.01, 0.05))
        assert result.shape == sample_image.shape
    
    def test_random_speckle_noise(self, sample_image):
        """Test speckle noise injection."""
        result = random_speckle_noise(sample_image, noise_range=(0.1, 0.3))
        assert result.shape == sample_image.shape
    
    def test_random_poisson_noise(self, sample_image):
        """Test Poisson noise injection."""
        result = random_poisson_noise(sample_image, scale=1.0)
        assert result.shape == sample_image.shape


class TestBlurAugmentations:
    """Test blur augmentation functions."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample color image for testing."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_random_gaussian_blur(self, sample_image):
        """Test random Gaussian blur."""
        result = random_gaussian_blur(sample_image, kernel_range=(3, 7))
        assert result.shape == sample_image.shape
    
    def test_random_motion_blur(self, sample_image):
        """Test random motion blur."""
        result = random_motion_blur(sample_image, kernel_range=(5, 15))
        assert result.shape == sample_image.shape
    
    def test_random_median_blur(self, sample_image):
        """Test random median blur."""
        result = random_median_blur(sample_image, kernel_range=(3, 7))
        assert result.shape == sample_image.shape


class TestCutoutAugmentations:
    """Test cutout and erasing augmentations."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample color image for testing."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_random_cutout(self, sample_image):
        """Test random cutout."""
        result = random_cutout(
            sample_image,
            num_holes=3,
            hole_size_range=(10, 20),
            fill_value=0
        )
        assert result.shape == sample_image.shape
    
    def test_random_grid_mask(self, sample_image):
        """Test random grid mask."""
        result = random_grid_mask(sample_image, grid_size=10, ratio=0.5)
        assert result.shape == sample_image.shape


class TestMixupCutMix:
    """Test mixup and cutmix augmentations."""
    
    @pytest.fixture
    def sample_images(self):
        """Create two sample images for mixing."""
        img1 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
        return img1, img2
    
    def test_mixup(self, sample_images):
        """Test mixup augmentation."""
        img1, img2 = sample_images
        result, alpha = mixup(img1, img2, alpha_range=(0.3, 0.7))
        assert result.shape == img1.shape
        assert 0.3 <= alpha <= 0.7
    
    def test_cutmix(self, sample_images):
        """Test cutmix augmentation."""
        img1, img2 = sample_images
        result, (x1, y1, x2, y2) = cutmix(img1, img2, beta=1.0)
        assert result.shape == img1.shape
        assert x1 < x2 and y1 < y2


class TestAugmentationPipeline:
    """Test augmentation pipeline functionality."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample color image for testing."""
        return np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    
    def test_pipeline_creation(self):
        """Test creating an augmentation pipeline."""
        pipeline = AugmentationPipeline()
        assert len(pipeline) == 0
    
    def test_pipeline_add_transform(self):
        """Test adding transforms to pipeline."""
        pipeline = AugmentationPipeline()
        pipeline.add(random_flip, horizontal=True, p=0.5)
        assert len(pipeline) == 1
    
    def test_pipeline_apply(self, sample_image):
        """Test applying pipeline to image."""
        pipeline = AugmentationPipeline()
        pipeline.add(random_flip, horizontal=True, p=0.5)
        pipeline.add(random_rotation, angle_range=(-15, 15))
        
        result = pipeline(sample_image)
        assert result.shape == sample_image.shape
    
    def test_get_default_pipeline(self):
        """Test getting default augmentation pipeline."""
        pipeline = get_default_augmentation_pipeline()
        assert isinstance(pipeline, AugmentationPipeline)
        assert len(pipeline) > 0
    
    def test_default_pipeline_apply(self, sample_image):
        """Test applying default pipeline."""
        pipeline = get_default_augmentation_pipeline(strength='light')
        result = pipeline(sample_image)
        assert result.shape == sample_image.shape


class TestUtilityFunctions:
    """Test utility functions for augmentation."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for batch testing."""
        return [
            np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
            for _ in range(5)
        ]
    
    def test_augment_batch(self, sample_images):
        """Test batch augmentation."""
        pipeline = AugmentationPipeline()
        pipeline.add(random_flip, horizontal=True, p=0.5)
        
        results = augment_batch(sample_images, pipeline)
        assert len(results) == len(sample_images)
        for result in results:
            assert result.shape == sample_images[0].shape
    
    def test_create_augmented_dataset(self, sample_images):
        """Test creating augmented dataset."""
        pipeline = AugmentationPipeline()
        pipeline.add(random_flip, horizontal=True, p=0.5)
        
        augmented = create_augmented_dataset(
            sample_images,
            pipeline,
            augmentations_per_image=2
        )
        assert len(augmented) == len(sample_images) * 2


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_grayscale_image_augmentation(self):
        """Test augmentation on grayscale images."""
        gray_img = np.random.randint(0, 256, (100, 100), dtype=np.uint8)
        result = random_flip(gray_img, horizontal=True, p=1.0)
        assert result.shape == gray_img.shape
    
    def test_small_image(self):
        """Test augmentation on small images."""
        small_img = np.random.randint(0, 256, (10, 10, 3), dtype=np.uint8)
        result = random_brightness(small_img, brightness_range=(-0.3, 0.3))
        assert result.shape == small_img.shape
    
    def test_large_image(self):
        """Test augmentation on large images."""
        large_img = np.random.randint(0, 256, (1000, 1000, 3), dtype=np.uint8)
        result = random_gaussian_blur(large_img, kernel_range=(3, 7))
        assert result.shape == large_img.shape
    
    def test_float_image(self):
        """Test augmentation on float images."""
        float_img = np.random.rand(100, 100, 3).astype(np.float32)
        result = random_flip(float_img, horizontal=True, p=1.0)
        assert result.shape == float_img.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
