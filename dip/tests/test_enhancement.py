"""
Tests for Enhancement Module

ทดสอบ image enhancement: brightness, contrast, denoise
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
def dark_image():
    """สร้างภาพมืดขนาด 100x100"""
    return np.full((100, 100), 50, dtype=np.uint8)


@pytest.fixture
def low_contrast_image():
    """สร้างภาพที่มี contrast ต่ำ"""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 150
    img[:30, :] = 100
    img[70:, :] = 100
    return img


@pytest.fixture
def noisy_image():
    """สร้างภาพที่มี noise"""
    np.random.seed(42)
    img = np.full((100, 100), 128, dtype=np.uint8)
    noise = np.random.normal(0, 25, (100, 100)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


@pytest.fixture
def noisy_bgr():
    """สร้างภาพ BGR ที่มี noise"""
    np.random.seed(42)
    img = np.full((100, 100, 3), 128, dtype=np.uint8)
    noise = np.random.normal(0, 25, (100, 100, 3)).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    return img


# =============================================================================
# Test Brightness Operations
# =============================================================================

class TestBrightnessOperations:
    def test_adjust_brightness_increase(self, dark_image):
        from enhancement.brightness import adjust_brightness
        result = adjust_brightness(dark_image, value=50)
        assert result.mean() > dark_image.mean()
        
    def test_adjust_brightness_decrease(self, dark_image):
        from enhancement.brightness import adjust_brightness
        result = adjust_brightness(dark_image, value=-30)
        assert result.mean() < dark_image.mean()
        
    def test_adjust_contrast_increase(self, low_contrast_image):
        from enhancement.brightness import adjust_contrast
        result = adjust_contrast(low_contrast_image, factor=2.0)
        # Higher contrast should increase standard deviation
        assert result.std() > low_contrast_image.std()
        
    def test_gamma_correction_brighten(self, dark_image):
        from enhancement.brightness import gamma_correction
        result = gamma_correction(dark_image, gamma=0.5)  # < 1 = brighten
        assert result.mean() > dark_image.mean()
        
    def test_gamma_correction_darken(self, dark_image):
        from enhancement.brightness import gamma_correction
        result = gamma_correction(dark_image, gamma=2.0)  # > 1 = darken
        assert result.mean() < dark_image.mean()


# =============================================================================
# Test Denoise Operations
# =============================================================================

class TestDenoiseOperations:
    def test_denoise_bilateral(self, noisy_image):
        from enhancement.denoise import denoise_bilateral
        result = denoise_bilateral(noisy_image)
        assert result.shape == noisy_image.shape
        
    def test_denoise_nlm_grayscale(self, noisy_image):
        from enhancement.denoise import denoise_nlm
        result = denoise_nlm(noisy_image)
        assert result.shape == noisy_image.shape
        
    def test_denoise_nlm_color(self, noisy_bgr):
        from enhancement.denoise import denoise_nlm_color
        result = denoise_nlm_color(noisy_bgr)
        assert result.shape == noisy_bgr.shape
