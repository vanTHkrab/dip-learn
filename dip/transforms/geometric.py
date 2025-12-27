"""
Geometric Transformations

การทำ geometric transformations เช่น resize, rotate, flip, crop
"""

import cv2 as cv
import numpy as np
from typing import Optional, Tuple, Union, Literal


def resize(
    image: np.ndarray,
    width: Optional[int] = None,
    height: Optional[int] = None,
    scale: Optional[float] = None,
    interpolation: int = cv.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image
    
    Args:
        image: Input image
        width: Target width (None to auto-calculate)
        height: Target height (None to auto-calculate)
        scale: Scale factor (overrides width/height if provided)
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if scale is not None:
        new_w = int(w * scale)
        new_h = int(h * scale)
    elif width is not None and height is not None:
        new_w, new_h = width, height
    elif width is not None:
        aspect = h / w
        new_w = width
        new_h = int(width * aspect)
    elif height is not None:
        aspect = w / h
        new_h = height
        new_w = int(height * aspect)
    else:
        return image.copy()
    
    return cv.resize(image, (new_w, new_h), interpolation=interpolation)


def resize_keep_aspect(
    image: np.ndarray,
    target_width: Optional[int] = None,
    target_height: Optional[int] = None,
    interpolation: int = cv.INTER_LINEAR
) -> np.ndarray:
    """
    Resize image while keeping aspect ratio
    
    Args:
        image: Input image
        target_width: Maximum width
        target_height: Maximum height
        interpolation: Interpolation method
        
    Returns:
        Resized image
    """
    h, w = image.shape[:2]
    
    if target_width is None and target_height is None:
        return image.copy()
    
    if target_width is not None and target_height is not None:
        scale_w = target_width / w
        scale_h = target_height / h
        scale = min(scale_w, scale_h)
    elif target_width is not None:
        scale = target_width / w
    else:
        scale = target_height / h
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    return cv.resize(image, (new_w, new_h), interpolation=interpolation)


def rotate(
    image: np.ndarray,
    angle: float,
    center: Optional[Tuple[int, int]] = None,
    scale: float = 1.0,
    border_value: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    Rotate image
    
    Args:
        image: Input image
        angle: Rotation angle in degrees (counter-clockwise)
        center: Center of rotation (default: image center)
        scale: Scale factor
        border_value: Border fill value
        
    Returns:
        Rotated image
    """
    h, w = image.shape[:2]
    if center is None:
        center = (w // 2, h // 2)
    
    M = cv.getRotationMatrix2D(center, angle, scale)
    return cv.warpAffine(image, M, (w, h), borderValue=border_value)


def rotate_bound(
    image: np.ndarray,
    angle: float,
    border_value: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    Rotate image without cropping (expand canvas)
    
    Args:
        image: Input image
        angle: Rotation angle in degrees
        border_value: Border fill value
        
    Returns:
        Rotated image with expanded canvas
    """
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    
    M = cv.getRotationMatrix2D(center, angle, 1.0)
    
    # Calculate new image size
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)
    
    # Adjust rotation matrix
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    
    return cv.warpAffine(image, M, (new_w, new_h), borderValue=border_value)


def flip(image: np.ndarray, direction: Literal["horizontal", "vertical", "both"] = "horizontal") -> np.ndarray:
    """
    Flip image
    
    Args:
        image: Input image
        direction: "horizontal", "vertical", or "both"
        
    Returns:
        Flipped image
    """
    codes = {
        "horizontal": 1,
        "vertical": 0,
        "both": -1
    }
    flip_code = codes.get(direction, 1)
    return cv.flip(image, flip_code)


def crop(
    image: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int
) -> np.ndarray:
    """
    Crop image
    
    Args:
        image: Input image
        x: Starting x coordinate
        y: Starting y coordinate
        width: Crop width
        height: Crop height
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x1 = max(0, x)
    y1 = max(0, y)
    x2 = min(w, x + width)
    y2 = min(h, y + height)
    return image[y1:y2, x1:x2].copy()


def crop_center(image: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Crop image from center
    
    Args:
        image: Input image
        width: Crop width
        height: Crop height
        
    Returns:
        Cropped image
    """
    h, w = image.shape[:2]
    x = (w - width) // 2
    y = (h - height) // 2
    return crop(image, x, y, width, height)


def pad(
    image: np.ndarray,
    top: int = 0,
    bottom: int = 0,
    left: int = 0,
    right: int = 0,
    color: Union[int, Tuple[int, int, int]] = 0,
    mode: Literal["constant", "reflect", "replicate"] = "constant"
) -> np.ndarray:
    """
    Add padding to image
    
    Args:
        image: Input image
        top, bottom, left, right: Padding sizes
        color: Padding color (for constant mode)
        mode: Padding mode ("constant", "reflect", "replicate")
        
    Returns:
        Padded image
    """
    border_types = {
        "constant": cv.BORDER_CONSTANT,
        "reflect": cv.BORDER_REFLECT,
        "replicate": cv.BORDER_REPLICATE
    }
    border_type = border_types.get(mode, cv.BORDER_CONSTANT)
    
    return cv.copyMakeBorder(
        image, top, bottom, left, right,
        border_type, value=color
    )


def pad_to_size(
    image: np.ndarray,
    target_width: int,
    target_height: int,
    color: Union[int, Tuple[int, int, int]] = 0,
    position: Literal["center", "top-left", "top-right", "bottom-left", "bottom-right"] = "center"
) -> np.ndarray:
    """
    Pad image to specific size
    
    Args:
        image: Input image
        target_width: Target width
        target_height: Target height
        color: Padding color
        position: Position of original image
        
    Returns:
        Padded image
    """
    h, w = image.shape[:2]
    
    if w >= target_width and h >= target_height:
        return image.copy()
    
    pad_w = max(0, target_width - w)
    pad_h = max(0, target_height - h)
    
    if position == "center":
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
    elif position == "top-left":
        left, top = 0, 0
        right, bottom = pad_w, pad_h
    elif position == "top-right":
        left, top = pad_w, 0
        right, bottom = 0, pad_h
    elif position == "bottom-left":
        left, top = 0, pad_h
        right, bottom = pad_w, 0
    elif position == "bottom-right":
        left, top = pad_w, pad_h
        right, bottom = 0, 0
    else:
        left = pad_w // 2
        right = pad_w - left
        top = pad_h // 2
        bottom = pad_h - top
    
    return pad(image, top, bottom, left, right, color)


def translate(
    image: np.ndarray,
    tx: int,
    ty: int,
    border_value: Union[int, Tuple[int, int, int]] = 0
) -> np.ndarray:
    """
    Translate (shift) image
    
    Args:
        image: Input image
        tx: Translation in x direction
        ty: Translation in y direction
        border_value: Border fill value
        
    Returns:
        Translated image
    """
    h, w = image.shape[:2]
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    return cv.warpAffine(image, M, (w, h), borderValue=border_value)


def perspective_transform(
    image: np.ndarray,
    src_points: np.ndarray,
    dst_points: np.ndarray,
    output_size: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Apply perspective transform
    
    Args:
        image: Input image
        src_points: Source points (4x2 array)
        dst_points: Destination points (4x2 array)
        output_size: Output image size (width, height)
        
    Returns:
        Transformed image
    """
    h, w = image.shape[:2]
    if output_size is None:
        output_size = (w, h)
    
    M = cv.getPerspectiveTransform(
        src_points.astype(np.float32),
        dst_points.astype(np.float32)
    )
    return cv.warpPerspective(image, M, output_size)


__all__ = [
    'resize',
    'resize_keep_aspect',
    'rotate',
    'rotate_bound',
    'flip',
    'crop',
    'crop_center',
    'pad',
    'pad_to_size',
    'translate',
    'perspective_transform',
]
