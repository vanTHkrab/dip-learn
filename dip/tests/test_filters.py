"""
Tests for Filters Module

ทดสอบ filters: smoothing, sharpening, edge detection
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
def sample_image():
    """สร้างภาพ grayscale ที่มี noise"""
    np.random.seed(42)
    img = np.random.randint(50, 200, (100, 100), dtype=np.uint8)
    return img


@pytest.fixture
def sample_bgr():
    """สร้างภาพ BGR สีขนาด 100x100"""
    np.random.seed(42)
    img = np.random.randint(50, 200, (100, 100, 3), dtype=np.uint8)
    return img


@pytest.fixture
def edge_image():
    """สร้างภาพที่มี edge ชัดเจน"""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[:, 50:] = 255
    return img


# =============================================================================
# Test Smoothing Filters
# =============================================================================

class TestSmoothingFilters:
    def test_gaussian_blur(self, sample_image):
        from filters.smoothing import gaussian_blur
        result = gaussian_blur(sample_image, ksize=5)
        assert result.shape == sample_image.shape
        # Blur should reduce pixel differences
        assert result.std() < sample_image.std()
        
    def test_gaussian_blur_even_ksize(self, sample_image):
        from filters.smoothing import gaussian_blur
        # Should convert ksize to odd number
        result = gaussian_blur(sample_image, ksize=4)
        assert result.shape == sample_image.shape
        
    def test_median_blur(self, sample_image):
        from filters.smoothing import median_blur
        result = median_blur(sample_image, ksize=5)
        assert result.shape == sample_image.shape
        
    def test_bilateral_filter(self, sample_bgr):
        from filters.smoothing import bilateral_filter
        result = bilateral_filter(sample_bgr)
        assert result.shape == sample_bgr.shape
        
    def test_box_blur(self, sample_image):
        from filters.smoothing import box_blur
        result = box_blur(sample_image, ksize=5)
        assert result.shape == sample_image.shape


# =============================================================================
# Test Sharpening Filters
# =============================================================================

class TestSharpeningFilters:
    def test_unsharp_mask(self, sample_image):
        from filters.sharpening import unsharp_mask
        result = unsharp_mask(sample_image)
        assert result.shape == sample_image.shape
        
    def test_laplacian_sharpen(self, sample_image):
        from filters.sharpening import laplacian_sharpen
        result = laplacian_sharpen(sample_image)
        assert result.shape == sample_image.shape
        
    def test_kernel_sharpen(self, sample_image):
        from filters.sharpening import kernel_sharpen
        result = kernel_sharpen(sample_image)
        assert result.shape == sample_image.shape


# =============================================================================
# Test Edge Detection Filters
# =============================================================================

class TestEdgeDetection:
    def test_canny_edge(self, edge_image):
        from filters.edge import canny_edge
        result = canny_edge(edge_image, low=50, high=150)
        assert result.shape == edge_image.shape
        # Should detect edge at position x=50
        assert result[:, 49:51].max() == 255
        
    def test_sobel_edge(self, edge_image):
        from filters.edge import sobel_edge
        result = sobel_edge(edge_image)
        assert result.shape == edge_image.shape
        
    def test_sobel_x(self, edge_image):
        from filters.edge import sobel_x
        result = sobel_x(edge_image)
        assert result.shape == edge_image.shape
        
    def test_sobel_y(self, edge_image):
        from filters.edge import sobel_y
        result = sobel_y(edge_image)
        assert result.shape == edge_image.shape
        
    def test_laplacian_edge(self, edge_image):
        from filters.edge import laplacian_edge
        result = laplacian_edge(edge_image)
        assert result.shape == edge_image.shape
        
    def test_auto_canny(self, sample_image):
        from filters.edge import auto_canny
        result = auto_canny(sample_image)
        assert result.shape == sample_image.shape
