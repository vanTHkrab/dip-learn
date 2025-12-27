"""
Annotation Module

โมดูลสำหรับ annotated และ drawing บนภาพ
รวมถึง shapes, text, bounding boxes และ overlays
"""

from .drawing import (
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

__all__ = [
    # Basic Shapes
    'draw_rectangle',
    'draw_circle',
    'draw_line',
    'draw_ellipse',
    'draw_polygon',
    'draw_filled_polygon',
    'draw_polylines',
    'draw_arrow',
    
    # Text Operations
    'draw_text',
    'draw_text_with_background',
    'get_text_size',
    'draw_multiline_text',
    
    # Annotation Utilities
    'draw_bounding_box',
    'draw_bounding_boxes',
    'draw_contours',
    'draw_keypoints',
    'draw_grid',
    'draw_crosshair',
    'draw_marker',
    
    # Overlay Operations
    'overlay_image',
    'add_alpha_channel',
    'create_mask_overlay',
]
