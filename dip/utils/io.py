"""
Image I/O Utilities

การ load และ save ภาพ
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Optional, Union, List


def load_image(
    path: Union[str, Path],
    color_mode: str = "color"
) -> Optional[np.ndarray]:
    """
    Load image from file
    
    Args:
        path: Path to image file
        color_mode: "color" (BGR), "gray", or "unchanged"
        
    Returns:
        Image array or None if failed
    """
    modes = {
        "color": cv.IMREAD_COLOR,
        "gray": cv.IMREAD_GRAYSCALE,
        "unchanged": cv.IMREAD_UNCHANGED,
    }
    mode = modes.get(color_mode, cv.IMREAD_COLOR)
    
    image = cv.imread(str(path), mode)
    return image


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    quality: int = 95
) -> bool:
    """
    Save image to file
    
    Args:
        image: Image array
        path: Output path
        quality: JPEG quality (0-100)
        
    Returns:
        True if successful
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    params = []
    if path.suffix.lower() in ['.jpg', '.jpeg']:
        params = [cv.IMWRITE_JPEG_QUALITY, quality]
    elif path.suffix.lower() == '.png':
        params = [cv.IMWRITE_PNG_COMPRESSION, 9 - quality // 10]
    
    return cv.imwrite(str(path), image, params)


def load_images_from_dir(
    directory: Union[str, Path],
    extensions: List[str] = None,
    color_mode: str = "color"
) -> List[np.ndarray]:
    """
    Load all images from directory
    
    Args:
        directory: Directory path
        extensions: List of extensions to include (default: common image extensions)
        color_mode: Color mode for loading
        
    Returns:
        List of images
    """
    if extensions is None:
        extensions = ['.png', '.jpg', '.jpeg', '.bmp', '.tiff']
    
    directory = Path(directory)
    images = []
    
    for ext in extensions:
        for path in directory.glob(f"*{ext}"):
            img = load_image(path, color_mode)
            if img is not None:
                images.append(img)
    
    return images


def image_from_bytes(data: bytes, color_mode: str = "color") -> Optional[np.ndarray]:
    """
    Load image from bytes
    
    Args:
        data: Image bytes
        color_mode: Color mode
        
    Returns:
        Image array or None
    """
    modes = {
        "color": cv.IMREAD_COLOR,
        "gray": cv.IMREAD_GRAYSCALE,
        "unchanged": cv.IMREAD_UNCHANGED,
    }
    mode = modes.get(color_mode, cv.IMREAD_COLOR)
    
    nparr = np.frombuffer(data, np.uint8)
    return cv.imdecode(nparr, mode)


def image_to_bytes(
    image: np.ndarray,
    format: str = ".png",
    quality: int = 95
) -> bytes:
    """
    Convert image to bytes
    
    Args:
        image: Image array
        format: Image format (.png, .jpg, etc.)
        quality: JPEG quality
        
    Returns:
        Image bytes
    """
    params = []
    if format.lower() in ['.jpg', '.jpeg']:
        params = [cv.IMWRITE_JPEG_QUALITY, quality]
    
    _, buffer = cv.imencode(format, image, params)
    return buffer.tobytes()


__all__ = [
    'load_image',
    'save_image',
    'load_images_from_dir',
    'image_from_bytes',
    'image_to_bytes',
]
