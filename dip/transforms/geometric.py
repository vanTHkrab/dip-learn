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


def detect_skew_angle(
    image: np.ndarray,
    method: Literal["hough", "projection"] = "hough",
    max_angle: float = 45.0,
    min_line_length: int = 50,
    max_line_gap: int = 10
) -> float:
    """
    Detect skew angle of an image using line detection or projection profile
    
    ตรวจจับมุมเอียงของภาพโดยใช้การหาเส้นตรง (Hough Transform) หรือ Projection Profile
    เหมาะสำหรับภาพเอกสาร, ตาราง, หรือภาพที่มีเส้นตรงชัดเจน
    
    Args:
        image: Input image (grayscale preferred, color will be converted)
        method: Detection method
            - "hough": Hough Line Transform (good for images with lines)
            - "projection": Projection profile analysis (good for text documents)
        max_angle: Maximum angle to consider (degrees), angles beyond this are ignored
        min_line_length: Minimum line length for Hough method
        max_line_gap: Maximum gap between line segments for Hough method
        
    Returns:
        Detected skew angle in degrees (counter-clockwise positive)
        Returns 0.0 if no skew detected or detection fails
        
    Example:
        >>> angle = detect_skew_angle(img, method="hough")
        >>> rotated = rotate_bound(img, -angle)
    """
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    if method == "hough":
        return _detect_skew_hough(gray, max_angle, min_line_length, max_line_gap)
    elif method == "projection":
        return _detect_skew_projection(gray, max_angle)
    else:
        return _detect_skew_hough(gray, max_angle, min_line_length, max_line_gap)


def _detect_skew_hough(
    gray: np.ndarray,
    max_angle: float,
    min_line_length: int,
    max_line_gap: int
) -> float:
    """
    Detect skew angle using Hough Line Transform
    
    Internal function ที่ใช้ Hough Transform ในการตรวจจับเส้นตรง
    แล้วคำนวณมุมเฉลี่ยของเส้นที่ใกล้เคียงกับแนวนอน/แนวตั้ง
    """
    # Apply edge detection
    edges = cv.Canny(gray, 50, 150, apertureSize=3)
    
    # Apply morphological operations to connect nearby edges
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (3, 3))
    edges = cv.dilate(edges, kernel, iterations=1)
    
    # Detect lines using Probabilistic Hough Transform
    lines = cv.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi / 180,
        threshold=100,
        minLineLength=min_line_length,
        maxLineGap=max_line_gap
    )
    
    if lines is None or len(lines) == 0:
        return 0.0
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        
        # Calculate angle of the line
        if x2 - x1 == 0:
            angle = 90.0
        else:
            angle = np.arctan((y2 - y1) / (x2 - x1)) * 180 / np.pi
        
        # Normalize angle to range [-max_angle, max_angle]
        # Lines near horizontal (0°) or near vertical (90°) are considered
        if abs(angle) <= max_angle:
            angles.append(angle)
        elif abs(angle - 90) <= max_angle:
            angles.append(angle - 90)
        elif abs(angle + 90) <= max_angle:
            angles.append(angle + 90)
    
    if not angles:
        return 0.0
    
    # Use median to be robust against outliers
    median_angle = float(np.median(angles))
    
    # Clamp to max_angle
    if abs(median_angle) > max_angle:
        return 0.0
    
    return median_angle


def _detect_skew_projection(gray: np.ndarray, max_angle: float) -> float:
    """
    Detect skew angle using projection profile analysis
    
    Internal function ที่ใช้ Projection Profile 
    วิเคราะห์ความแปรปรวนของ horizontal projection ที่มุมต่างๆ
    มุมที่ทำให้เกิด variance สูงสุดคือมุมที่ภาพตั้งตรง
    เหมาะสำหรับภาพเอกสารที่มีข้อความ
    """
    # Binarize the image
    _, binary = cv.threshold(gray, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU)
    
    # Try different angles and find the one with maximum variance
    best_angle = 0.0
    max_variance = 0.0
    
    # Search in range [-max_angle, max_angle] with 0.5 degree step
    angle_range = np.arange(-max_angle, max_angle + 0.5, 0.5)
    
    for angle in angle_range:
        # Rotate image
        h, w = binary.shape
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv.warpAffine(binary, M, (w, h), flags=cv.INTER_LINEAR, borderValue=0)
        
        # Calculate horizontal projection (sum of white pixels per row)
        projection = np.sum(rotated, axis=1).astype(float)
        
        # Calculate variance of projection
        if np.mean(projection) > 0:
            variance = np.var(projection)
        else:
            variance = 0.0
        
        if variance > max_variance:
            max_variance = variance
            best_angle = angle
    
    return best_angle


def auto_rotate(
    image: np.ndarray,
    method: Literal["hough", "projection", "auto"] = "auto",
    max_angle: float = 45.0,
    expand: bool = True,
    border_value: Union[int, Tuple[int, int, int]] = 255,
    return_angle: bool = False
) -> Union[np.ndarray, Tuple[np.ndarray, float]]:
    """
    Automatically detect and correct image skew/rotation
    
    ตรวจจับมุมเอียงของภาพอัตโนมัติและหมุนแก้ไขให้ตั้งตรง
    เหมาะสำหรับการทำ preprocessing ก่อนนำไปใช้ OCR หรือการวิเคราะห์อื่นๆ
    
    Args:
        image: Input image
        method: Detection method
            - "hough": Use Hough Line Transform (faster, good for images with lines)
            - "projection": Use projection profile (slower, better for text documents)
            - "auto": Try hough first, fallback to projection if angle is 0
        max_angle: Maximum skew angle to consider (degrees)
        expand: If True, expand canvas to fit rotated image (no cropping)
        border_value: Value to fill new pixels after rotation (default white for documents)
        return_angle: If True, also return the detected angle
        
    Returns:
        If return_angle is False: Rotated image
        If return_angle is True: Tuple of (rotated image, detected angle)
        
    Example:
        >>> # Simple auto-rotation
        >>> corrected = auto_rotate(skewed_image)
        
        >>> # Get the detected angle too
        >>> corrected, angle = auto_rotate(skewed_image, return_angle=True)
        >>> print(f"Corrected {angle:.2f} degrees of skew")
        
        >>> # For document images
        >>> corrected = auto_rotate(doc_image, method="projection")
    """
    # Detect skew angle
    if method == "auto":
        # Try hough first
        angle = detect_skew_angle(image, method="hough", max_angle=max_angle)
        
        # If hough returns 0, try projection method
        if abs(angle) < 0.1:
            angle = detect_skew_angle(image, method="projection", max_angle=max_angle)
    else:
        angle = detect_skew_angle(image, method=method, max_angle=max_angle)
    
    # Skip rotation if angle is very small
    if abs(angle) < 0.1:
        if return_angle:
            return image.copy(), 0.0
        return image.copy()
    
    # Rotate image to correct the skew (negative angle to de-skew)
    if expand:
        rotated = rotate_bound(image, -angle, border_value=border_value)
    else:
        rotated = rotate(image, -angle, border_value=border_value)
    
    if return_angle:
        return rotated, angle
    return rotated


def auto_rotate_90(
    image: np.ndarray,
    target_orientation: Literal["landscape", "portrait", "auto"] = "auto"
) -> np.ndarray:
    """
    Auto-rotate image by 90 degree increments based on orientation
    
    หมุนภาพเป็นจำนวน 90 องศา เพื่อให้ได้ orientation ที่ต้องการ
    ไม่ได้แก้ไข skew เล็กๆ แต่เป็นการหมุน 90/180/270 องศา
    
    Args:
        image: Input image
        target_orientation: Target orientation
            - "landscape": Width > Height
            - "portrait": Height > Width
            - "auto": Keep original, just ensure proper orientation
            
    Returns:
        Rotated image (0, 90, 180, or 270 degrees)
        
    Example:
        >>> # Ensure image is landscape
        >>> landscape_img = auto_rotate_90(img, "landscape")
    """
    h, w = image.shape[:2]
    is_landscape = w > h
    
    if target_orientation == "landscape":
        if not is_landscape:
            # Rotate 90 degrees to make landscape
            return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        return image.copy()
    
    elif target_orientation == "portrait":
        if is_landscape:
            # Rotate 90 degrees to make portrait
            return cv.rotate(image, cv.ROTATE_90_CLOCKWISE)
        return image.copy()
    
    else:  # auto - just return as is
        return image.copy()


__all__ = [
    # Resize
    'resize',
    'resize_keep_aspect',
    # Rotate
    'rotate',
    'rotate_bound',
    'auto_rotate',
    'auto_rotate_90',
    'detect_skew_angle',
    # Flip
    'flip',
    # Crop
    'crop',
    'crop_center',
    # Padding
    'pad',
    'pad_to_size',
    # Transform
    'translate',
    'perspective_transform',
]
