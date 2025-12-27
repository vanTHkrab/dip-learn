"""
Tests for Transforms Module

ทดสอบการแปลงภาพต่างๆ: color, threshold, morphology, geometric, histogram
"""

import pytest
import numpy as np
import cv2 as cv

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bgr_image():
    """สร้างภาพ BGR สีขนาด 100x100"""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [100, 150, 200]  # BGR rectangle
    return img


@pytest.fixture
def sample_grayscale():
    """สร้างภาพ grayscale ขนาด 100x100"""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 200
    return img


@pytest.fixture
def binary_image():
    """สร้างภาพ binary ขนาด 100x100"""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[40:60, 40:60] = 255
    return img


# =============================================================================
# Test Color Transforms
# =============================================================================

class TestColorTransforms:
    def test_to_grayscale_from_bgr(self, sample_bgr_image):
        from transforms.color import to_grayscale
        result = to_grayscale(sample_bgr_image)
        assert len(result.shape) == 2
        
    def test_to_grayscale_from_gray(self, sample_grayscale):
        from transforms.color import to_grayscale
        result = to_grayscale(sample_grayscale)
        assert len(result.shape) == 2
        
    def test_to_bgr_from_gray(self, sample_grayscale):
        from transforms.color import to_bgr
        result = to_bgr(sample_grayscale)
        assert result.shape[2] == 3
        
    def test_to_rgb(self, sample_bgr_image):
        from transforms.color import to_rgb
        result = to_rgb(sample_bgr_image)
        # RGB and BGR should have swapped channels
        np.testing.assert_array_equal(result[:, :, 0], sample_bgr_image[:, :, 2])
        
    def test_to_hsv(self, sample_bgr_image):
        from transforms.color import to_hsv
        result = to_hsv(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape
        
    def test_to_lab(self, sample_bgr_image):
        from transforms.color import to_lab
        result = to_lab(sample_bgr_image)
        assert result.shape == sample_bgr_image.shape


# =============================================================================
# Test Threshold Transforms
# =============================================================================

class TestThresholdTransforms:
    def test_binary_threshold(self, sample_grayscale):
        from transforms.threshold import binary_threshold
        result = binary_threshold(sample_grayscale, thresh=100)
        # Should only contain 0 and 255
        unique = np.unique(result)
        assert len(unique) <= 2
        
    def test_otsu_threshold(self, sample_grayscale):
        from transforms.threshold import otsu_threshold
        result = otsu_threshold(sample_grayscale)
        unique = np.unique(result)
        assert len(unique) <= 2
        
    def test_adaptive_threshold_mean(self, sample_grayscale):
        from transforms.threshold import adaptive_threshold_mean
        result = adaptive_threshold_mean(sample_grayscale)
        unique = np.unique(result)
        assert len(unique) <= 2
        
    def test_adaptive_threshold_gaussian(self, sample_grayscale):
        from transforms.threshold import adaptive_threshold_gaussian
        result = adaptive_threshold_gaussian(sample_grayscale)
        unique = np.unique(result)
        assert len(unique) <= 2


# =============================================================================
# Test Morphology Transforms
# =============================================================================

class TestMorphologyTransforms:
    def test_erode(self, binary_image):
        from transforms.morphology import erode
        result = erode(binary_image, iterations=1)
        # Erosion should reduce white area
        assert result.sum() < binary_image.sum()
        
    def test_dilate(self, binary_image):
        from transforms.morphology import dilate
        result = dilate(binary_image, iterations=1)
        # Dilation should increase white area
        assert result.sum() > binary_image.sum()
        
    def test_morph_open(self, binary_image):
        from transforms.morphology import morph_open
        result = morph_open(binary_image)
        assert result.shape == binary_image.shape
        
    def test_morph_close(self, binary_image):
        from transforms.morphology import morph_close
        result = morph_close(binary_image)
        assert result.shape == binary_image.shape
        
    def test_morph_gradient(self, binary_image):
        from transforms.morphology import morph_gradient
        result = morph_gradient(binary_image)
        assert result.shape == binary_image.shape


# =============================================================================
# Test Geometric Transforms
# =============================================================================

class TestGeometricTransforms:
    def test_resize_scale(self, sample_bgr_image):
        from transforms.geometric import resize
        result = resize(sample_bgr_image, scale=0.5)
        assert result.shape[0] == 50
        assert result.shape[1] == 50
        
    def test_resize_dimensions(self, sample_bgr_image):
        from transforms.geometric import resize
        result = resize(sample_bgr_image, width=200, height=150)
        assert result.shape[0] == 150
        assert result.shape[1] == 200
        
    def test_rotate(self, sample_bgr_image):
        from transforms.geometric import rotate
        result = rotate(sample_bgr_image, angle=45)
        assert result.shape[:2] == sample_bgr_image.shape[:2]
        
    def test_rotate_bound(self, sample_bgr_image):
        from transforms.geometric import rotate_bound
        result = rotate_bound(sample_bgr_image, angle=45)
        # Size may change to accommodate rotated image
        assert result is not None
        
    def test_flip_horizontal(self, sample_bgr_image):
        from transforms.geometric import flip
        result = flip(sample_bgr_image, direction='horizontal')
        np.testing.assert_array_equal(result[:, 0], sample_bgr_image[:, -1])
        
    def test_flip_vertical(self, sample_bgr_image):
        from transforms.geometric import flip
        result = flip(sample_bgr_image, direction='vertical')
        np.testing.assert_array_equal(result[0, :], sample_bgr_image[-1, :])
        
    def test_crop(self, sample_bgr_image):
        from transforms.geometric import crop
        result = crop(sample_bgr_image, x=10, y=10, width=50, height=50)
        assert result.shape[:2] == (50, 50)
        
    def test_crop_center(self, sample_bgr_image):
        from transforms.geometric import crop_center
        result = crop_center(sample_bgr_image, width=50, height=50)
        assert result.shape[:2] == (50, 50)


# =============================================================================
# Test Histogram Transforms
# =============================================================================

class TestHistogramTransforms:
    def test_histogram_equalization(self, sample_grayscale):
        from transforms.histogram import histogram_equalization
        result = histogram_equalization(sample_grayscale)
        assert result.shape == sample_grayscale.shape
        
    def test_clahe(self, sample_grayscale):
        from transforms.histogram import clahe
        result = clahe(sample_grayscale)
        assert result.shape == sample_grayscale.shape
        
    def test_contrast_stretch(self, sample_grayscale):
        from transforms.histogram import contrast_stretch
        result = contrast_stretch(sample_grayscale)
        assert result.min() == 0
        assert result.max() == 255
        
    def test_normalize(self, sample_grayscale):
        from transforms.histogram import normalize
        result = normalize(sample_grayscale, new_min=0, new_max=1)
        assert result.min() >= 0
        assert result.max() <= 1
