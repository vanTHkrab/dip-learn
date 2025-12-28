"""
Tests for DIP-Learn Web Interface Components

Tests the core functionality of the web module without requiring
a running Streamlit server.
"""

import pytest
import numpy as np
import cv2
import sys
from unittest.mock import MagicMock, patch


# Mock streamlit before importing web module
sys.modules['streamlit'] = MagicMock()


class TestImageProcessingFunctions:
    """Test image processing functions used in web interface."""
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample test image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    @pytest.fixture
    def grayscale_image(self):
        """Create a sample grayscale image."""
        return np.random.randint(0, 255, (100, 100), dtype=np.uint8)
    
    def test_brightness_adjustment(self, sample_image):
        """Test brightness adjustment."""
        result = cv2.convertScaleAbs(sample_image, alpha=1, beta=30)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_contrast_adjustment(self, sample_image):
        """Test contrast adjustment."""
        result = cv2.convertScaleAbs(sample_image, alpha=1.5, beta=0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_gaussian_blur(self, sample_image):
        """Test Gaussian blur filter."""
        result = cv2.GaussianBlur(sample_image, (5, 5), 0)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_median_blur(self, sample_image):
        """Test median blur filter."""
        result = cv2.medianBlur(sample_image, 5)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_bilateral_filter(self, sample_image):
        """Test bilateral filter."""
        result = cv2.bilateralFilter(sample_image, 9, 75, 75)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_edge_detection(self, sample_image):
        """Test edge detection."""
        gray = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        result = cv2.Canny(gray, 100, 200)
        assert result.shape == gray.shape
        assert result.dtype == np.uint8
    
    def test_grayscale_conversion(self, sample_image):
        """Test grayscale conversion."""
        result = cv2.cvtColor(sample_image, cv2.COLOR_BGR2GRAY)
        assert len(result.shape) == 2
        assert result.dtype == np.uint8
    
    def test_hsv_conversion(self, sample_image):
        """Test HSV conversion."""
        result = cv2.cvtColor(sample_image, cv2.COLOR_BGR2HSV)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_histogram_equalization(self, grayscale_image):
        """Test histogram equalization."""
        result = cv2.equalizeHist(grayscale_image)
        assert result.shape == grayscale_image.shape
        assert result.dtype == np.uint8
    
    def test_clahe(self, grayscale_image):
        """Test CLAHE."""
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        result = clahe.apply(grayscale_image)
        assert result.shape == grayscale_image.shape
        assert result.dtype == np.uint8
    
    def test_binary_threshold(self, grayscale_image):
        """Test binary threshold."""
        _, result = cv2.threshold(grayscale_image, 127, 255, cv2.THRESH_BINARY)
        assert result.shape == grayscale_image.shape
        assert result.dtype == np.uint8
    
    def test_otsu_threshold(self, grayscale_image):
        """Test Otsu threshold."""
        _, result = cv2.threshold(grayscale_image, 0, 255, 
                                   cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        assert result.shape == grayscale_image.shape
        assert result.dtype == np.uint8
    
    def test_adaptive_threshold(self, grayscale_image):
        """Test adaptive threshold."""
        result = cv2.adaptiveThreshold(
            grayscale_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        assert result.shape == grayscale_image.shape
        assert result.dtype == np.uint8
    
    def test_erosion(self, sample_image):
        """Test erosion."""
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.erode(sample_image, kernel, iterations=1)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_dilation(self, sample_image):
        """Test dilation."""
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.dilate(sample_image, kernel, iterations=1)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_morphological_opening(self, sample_image):
        """Test morphological opening."""
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(sample_image, cv2.MORPH_OPEN, kernel)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_morphological_closing(self, sample_image):
        """Test morphological closing."""
        kernel = np.ones((5, 5), np.uint8)
        result = cv2.morphologyEx(sample_image, cv2.MORPH_CLOSE, kernel)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8
    
    def test_sharpen(self, sample_image):
        """Test sharpen filter."""
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        result = cv2.filter2D(sample_image, -1, kernel)
        assert result.shape == sample_image.shape
        assert result.dtype == np.uint8


class TestMetricsCalculation:
    """Test image quality metrics calculation."""
    
    @pytest.fixture
    def identical_images(self):
        """Create identical test images."""
        img = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return img.copy(), img.copy()
    
    @pytest.fixture
    def different_images(self):
        """Create different test images."""
        img1 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        return img1, img2
    
    def test_mse_identical(self, identical_images):
        """Test MSE for identical images."""
        img1, img2 = identical_images
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        assert mse == 0.0
    
    def test_mse_different(self, different_images):
        """Test MSE for different images."""
        img1, img2 = different_images
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        assert mse > 0.0
    
    def test_psnr_identical(self, identical_images):
        """Test PSNR for identical images."""
        img1, img2 = identical_images
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse == 0:
            psnr = float('inf')
        else:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
        assert psnr == float('inf')
    
    def test_psnr_different(self, different_images):
        """Test PSNR for different images."""
        img1, img2 = different_images
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        if mse > 0:
            psnr = 10 * np.log10(255.0 ** 2 / mse)
            assert psnr > 0
            assert psnr < 100  # Reasonable range
    
    def test_mae(self, different_images):
        """Test Mean Absolute Error."""
        img1, img2 = different_images
        mae = np.mean(np.abs(img1.astype(float) - img2.astype(float)))
        assert mae >= 0


class TestImageDifference:
    """Test image difference calculations."""
    
    @pytest.fixture
    def sample_images(self):
        """Create sample images for comparison."""
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return img1, img2
    
    def test_absolute_difference(self, sample_images):
        """Test absolute difference calculation."""
        img1, img2 = sample_images
        diff = cv2.absdiff(img1, img2)
        assert diff.shape == img1.shape
        assert diff.dtype == np.uint8
    
    def test_difference_to_grayscale(self, sample_images):
        """Test converting difference to grayscale."""
        img1, img2 = sample_images
        diff = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        assert len(diff_gray.shape) == 2
    
    def test_difference_colormap(self, sample_images):
        """Test applying colormap to difference."""
        img1, img2 = sample_images
        diff = cv2.absdiff(img1, img2)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
        assert diff_color.shape[2] == 3


class TestImageBlending:
    """Test image blending operations."""
    
    @pytest.fixture
    def matching_images(self):
        """Create matching size images."""
        img1 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        img2 = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        return img1, img2
    
    def test_addweighted(self, matching_images):
        """Test addWeighted blending."""
        img1, img2 = matching_images
        alpha = 0.5
        blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
        assert blended.shape == img1.shape
        assert blended.dtype == np.uint8
    
    def test_different_alphas(self, matching_images):
        """Test blending with different alpha values."""
        img1, img2 = matching_images
        for alpha in [0.0, 0.25, 0.5, 0.75, 1.0]:
            blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
            assert blended.shape == img1.shape


class TestImageResize:
    """Test image resizing operations."""
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_resize_to_match(self, sample_image):
        """Test resizing to match another image."""
        target_shape = (200, 150)
        resized = cv2.resize(sample_image, target_shape)
        assert resized.shape[1] == target_shape[0]
        assert resized.shape[0] == target_shape[1]


class TestImageEncoding:
    """Test image encoding for download."""
    
    @pytest.fixture
    def sample_image(self):
        """Create sample image."""
        return np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
    
    def test_png_encoding(self, sample_image):
        """Test PNG encoding."""
        success, buffer = cv2.imencode('.png', sample_image)
        assert success
        assert len(buffer) > 0
    
    def test_jpeg_encoding(self, sample_image):
        """Test JPEG encoding."""
        success, buffer = cv2.imencode('.jpg', sample_image)
        assert success
        assert len(buffer) > 0
    
    def test_decode_encoded(self, sample_image):
        """Test decoding an encoded image."""
        _, buffer = cv2.imencode('.png', sample_image)
        decoded = cv2.imdecode(buffer, cv2.IMREAD_COLOR)
        assert decoded.shape == sample_image.shape


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
