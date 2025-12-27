"""
Tests for Annotation Module

Tests for drawing and annotation functions on images.
"""

import pytest
import numpy as np
import cv2 as cv

import sys
from pathlib import Path
# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from annotated import (
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
# Fixtures
# =============================================================================

@pytest.fixture
def blank_image():
    """Create a blank 100x100 BGR image."""
    return np.zeros((100, 100, 3), dtype=np.uint8)


@pytest.fixture
def blank_grayscale():
    """Create a blank 100x100 grayscale image."""
    return np.zeros((100, 100), dtype=np.uint8)


@pytest.fixture
def sample_contours():
    """Create sample contours for testing."""
    contour1 = np.array([[[10, 10]], [[10, 50]], [[50, 50]], [[50, 10]]], dtype=np.int32)
    contour2 = np.array([[[60, 60]], [[60, 90]], [[90, 90]], [[90, 60]]], dtype=np.int32)
    return [contour1, contour2]


# =============================================================================
# Test Basic Shapes
# =============================================================================

class TestDrawRectangle:
    def test_draw_rectangle_basic(self, blank_image):
        result = draw_rectangle(blank_image, (10, 10), (50, 50))
        # Check that drawing occurred (not all pixels are 0)
        assert result.max() > 0
        
    def test_draw_rectangle_filled(self, blank_image):
        result = draw_rectangle(blank_image, (10, 10), (50, 50), 
                               color=(0, 255, 0), thickness=-1)
        # Check that green color is present in result
        assert result[:, :, 1].max() == 255
        
    def test_draw_rectangle_copy(self, blank_image):
        original = blank_image.copy()
        result = draw_rectangle(blank_image, (10, 10), (50, 50), copy=True)
        # Check that original image is unchanged
        np.testing.assert_array_equal(blank_image, original)
        
    def test_draw_rectangle_no_copy(self, blank_image):
        result = draw_rectangle(blank_image, (10, 10), (50, 50), copy=False)
        # Check that it's the same image
        assert result is blank_image


class TestDrawCircle:
    def test_draw_circle_basic(self, blank_image):
        result = draw_circle(blank_image, (50, 50), 20)
        assert result.max() > 0
        
    def test_draw_circle_filled(self, blank_image):
        result = draw_circle(blank_image, (50, 50), 20, 
                            color=(255, 0, 0), thickness=-1)
        # Check that blue color is present in result
        assert result[:, :, 0].max() == 255


class TestDrawLine:
    def test_draw_line_basic(self, blank_image):
        result = draw_line(blank_image, (0, 0), (99, 99))
        assert result.max() > 0
        
    def test_draw_line_color(self, blank_image):
        result = draw_line(blank_image, (0, 0), (99, 99), color=(0, 0, 255))
        # Check that red color is present in result
        assert result[:, :, 2].max() == 255


class TestDrawEllipse:
    def test_draw_ellipse_basic(self, blank_image):
        result = draw_ellipse(blank_image, (50, 50), (30, 20))
        assert result.max() > 0
        
    def test_draw_ellipse_rotated(self, blank_image):
        result = draw_ellipse(blank_image, (50, 50), (30, 20), angle=45)
        assert result.max() > 0


class TestDrawPolygon:
    def test_draw_polygon_basic(self, blank_image):
        points = [(10, 10), (50, 10), (30, 50)]
        result = draw_polygon(blank_image, points)
        assert result.max() > 0
        
    def test_draw_filled_polygon(self, blank_image):
        points = [(10, 10), (50, 10), (30, 50)]
        result = draw_filled_polygon(blank_image, points, color=(0, 255, 0))
        assert result[:, :, 1].max() == 255


class TestDrawPolylines:
    def test_draw_polylines_basic(self, blank_image):
        points_list = [[(10, 10), (30, 30), (50, 10)], 
                      [(60, 60), (80, 80), (90, 60)]]
        result = draw_polylines(blank_image, points_list)
        assert result.max() > 0


class TestDrawArrow:
    def test_draw_arrow_basic(self, blank_image):
        result = draw_arrow(blank_image, (10, 50), (90, 50))
        assert result.max() > 0


# =============================================================================
# Test Text Operations
# =============================================================================

class TestDrawText:
    def test_draw_text_basic(self, blank_image):
        result = draw_text(blank_image, "Hello", (10, 50))
        assert result.max() > 0
        
    def test_draw_text_color(self, blank_image):
        result = draw_text(blank_image, "Hello", (10, 50), color=(255, 0, 0))
        assert result[:, :, 0].max() > 0


class TestDrawTextWithBackground:
    def test_draw_text_with_background(self, blank_image):
        result = draw_text_with_background(blank_image, "Test", (10, 10))
        assert result.max() > 0


class TestGetTextSize:
    def test_get_text_size(self):
        size, baseline = get_text_size("Hello")
        assert size[0] > 0  # width
        assert size[1] > 0  # height
        assert baseline >= 0


class TestDrawMultilineText:
    def test_draw_multiline_text(self, blank_image):
        result = draw_multiline_text(blank_image, "Line1\nLine2\nLine3", (10, 10))
        assert result.max() > 0


# =============================================================================
# Test Annotation Utilities
# =============================================================================

class TestDrawBoundingBox:
    def test_draw_bounding_box_basic(self, blank_image):
        result = draw_bounding_box(blank_image, (10, 10, 40, 40))
        assert result.max() > 0
        
    def test_draw_bounding_box_with_label(self, blank_image):
        result = draw_bounding_box(blank_image, (10, 20, 40, 40), label="Test")
        assert result.max() > 0


class TestDrawBoundingBoxes:
    def test_draw_bounding_boxes_basic(self, blank_image):
        bboxes = [(10, 10, 30, 30), (50, 50, 30, 30)]
        result = draw_bounding_boxes(blank_image, bboxes)
        assert result.max() > 0
        
    def test_draw_bounding_boxes_with_labels(self, blank_image):
        bboxes = [(10, 20, 30, 30), (50, 60, 30, 30)]
        labels = ["A", "B"]
        result = draw_bounding_boxes(blank_image, bboxes, labels=labels)
        assert result.max() > 0


class TestDrawContours:
    def test_draw_contours_basic(self, blank_image, sample_contours):
        result = draw_contours(blank_image, sample_contours)
        assert result.max() > 0
        
    def test_draw_contours_single(self, blank_image, sample_contours):
        result = draw_contours(blank_image, sample_contours, contour_idx=0)
        assert result.max() > 0


class TestDrawKeypoints:
    def test_draw_keypoints_basic(self, blank_image):
        keypoints = [(20, 20), (50, 50), (80, 80)]
        result = draw_keypoints(blank_image, keypoints)
        assert result.max() > 0


class TestDrawGrid:
    def test_draw_grid_basic(self, blank_image):
        result = draw_grid(blank_image, grid_size=(5, 5))
        assert result.max() > 0


class TestDrawCrosshair:
    def test_draw_crosshair_center(self, blank_image):
        result = draw_crosshair(blank_image)
        assert result.max() > 0
        
    def test_draw_crosshair_custom(self, blank_image):
        result = draw_crosshair(blank_image, center=(30, 30), size=30)
        assert result.max() > 0


class TestDrawMarker:
    def test_draw_marker_basic(self, blank_image):
        result = draw_marker(blank_image, (50, 50))
        assert result.max() > 0
        
    def test_draw_marker_types(self, blank_image):
        markers = [cv.MARKER_CROSS, cv.MARKER_TILTED_CROSS, 
                  cv.MARKER_STAR, cv.MARKER_DIAMOND]
        for marker in markers:
            result = draw_marker(blank_image.copy(), (50, 50), marker_type=marker)
            assert result.max() > 0


# =============================================================================
# Test Overlay Operations
# =============================================================================

class TestOverlayImage:
    def test_overlay_image_basic(self, blank_image):
        overlay = np.ones((20, 20, 3), dtype=np.uint8) * 255
        result = overlay_image(blank_image, overlay, position=(10, 10))
        assert result[15, 15].max() == 255
        
    def test_overlay_image_alpha(self, blank_image):
        blank_image[:] = 100
        overlay = np.ones((20, 20, 3), dtype=np.uint8) * 200
        result = overlay_image(blank_image, overlay, position=(10, 10), alpha=0.5)
        # Value should be between 100 and 200
        assert 100 < result[15, 15, 0] < 200


class TestAddAlphaChannel:
    def test_add_alpha_to_bgr(self, blank_image):
        result = add_alpha_channel(blank_image)
        assert result.shape[2] == 4
        
    def test_add_alpha_to_grayscale(self, blank_grayscale):
        result = add_alpha_channel(blank_grayscale)
        assert result.shape[2] == 4


class TestCreateMaskOverlay:
    def test_create_mask_overlay(self, blank_image):
        mask = np.zeros((100, 100), dtype=np.uint8)
        mask[30:70, 30:70] = 255
        result = create_mask_overlay(blank_image, mask, color=(0, 255, 0))
        # Check that mask area has green color
        assert result[50, 50, 1] > 0
