"""
Tests for Core Module

Tests for ImageOps and ImagePipeline functionality.
"""

import pytest
import numpy as np

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# =============================================================================
# Fixtures
# =============================================================================

@pytest.fixture
def sample_bgr_image():
    """Create a 100x100 BGR color image."""
    img = np.zeros((100, 100, 3), dtype=np.uint8)
    img[20:80, 20:80] = [100, 150, 200]
    return img


@pytest.fixture
def sample_grayscale():
    """Create a 100x100 grayscale image."""
    img = np.zeros((100, 100), dtype=np.uint8)
    img[30:70, 30:70] = 200
    return img


# =============================================================================
# Test ImageOps
# =============================================================================

class TestImageOps:
    def test_to_grayscale(self, sample_bgr_image):
        from core.operations import ImageOps
        result = ImageOps.to_grayscale(sample_bgr_image)
        assert len(result.shape) == 2
        
    def test_gaussian_blur(self, sample_grayscale):
        from core.operations import ImageOps
        result = ImageOps.gaussian_blur(sample_grayscale, ksize=5)
        assert result.shape == sample_grayscale.shape
        
    def test_otsu_threshold(self, sample_grayscale):
        from core.operations import ImageOps
        result = ImageOps.otsu_threshold(sample_grayscale)
        unique = np.unique(result)
        assert len(unique) <= 2
        
    def test_clahe(self, sample_grayscale):
        from core.operations import ImageOps
        result = ImageOps.clahe(sample_grayscale)
        assert result.shape == sample_grayscale.shape
        
    def test_resize(self, sample_bgr_image):
        from core.operations import ImageOps
        result = ImageOps.resize(sample_bgr_image, scale=0.5)
        assert result.shape[:2] == (50, 50)


# =============================================================================
# Test ImagePipeline
# =============================================================================

class TestImagePipeline:
    def test_pipeline_creation(self):
        from core.pipeline import ImagePipeline
        pipeline = ImagePipeline()
        assert pipeline is not None
        
    def test_pipeline_add_step(self):
        from core.pipeline import ImagePipeline
        from transforms.color import to_grayscale
        
        pipeline = ImagePipeline()
        pipeline.add_step('grayscale', to_grayscale)
        assert len(pipeline.steps) == 1
        
    def test_pipeline_run(self, sample_bgr_image):
        from core.pipeline import ImagePipeline
        from transforms.color import to_grayscale
        from filters.smoothing import gaussian_blur
        
        pipeline = ImagePipeline()
        pipeline.add_step('grayscale', to_grayscale)
        pipeline.add_step('blur', gaussian_blur, ksize=5)
        
        result = pipeline.run(sample_bgr_image)
        assert result.output is not None
        assert len(result.output.shape) == 2
        
    def test_pipeline_history(self, sample_bgr_image):
        from core.pipeline import ImagePipeline
        from transforms.color import to_grayscale
        from filters.smoothing import gaussian_blur
        
        pipeline = ImagePipeline()
        pipeline.add_step('grayscale', to_grayscale)
        pipeline.add_step('blur', gaussian_blur, ksize=5)
        
        result = pipeline.run(sample_bgr_image)
        assert 'grayscale' in result.history
        assert 'blur' in result.history


# =============================================================================
# Test Quick Process Functions
# =============================================================================

class TestQuickProcess:
    def test_quick_process(self, sample_bgr_image):
        from core.pipeline import quick_process
        from transforms.color import to_grayscale
        from filters.smoothing import gaussian_blur
        
        result = quick_process(
            sample_bgr_image,
            [
                (to_grayscale, {}),
                (gaussian_blur, {'ksize': 5}),
            ]
        )
        assert len(result.shape) == 2
        
    def test_apply_sequence(self, sample_bgr_image):
        from core.pipeline import apply_sequence
        from transforms.color import to_grayscale
        
        result = apply_sequence(sample_bgr_image, [to_grayscale])
        assert len(result.shape) == 2
