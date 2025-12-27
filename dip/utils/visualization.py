"""
Visualization Utilities

เครื่องมือสำหรับแสดงผลภาพ
"""

import cv2 as cv
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import math


def visualize_pipeline_result(
    result,  # PipelineResult
    cols: int = 4,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    แสดงผลลัพธ์ของ pipeline ทุก step
    
    Args:
        result: PipelineResult จาก process_with_history
        cols: จำนวน columns
        figsize: Figure size (width, height)
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization")
        return
    
    n_images = len(result.history)
    rows = math.ceil(n_images / cols)
    
    if figsize is None:
        figsize = (4 * cols, 3 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for ax, (name, img) in zip(axes, result.history.items()):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(name, fontsize=9)
        ax.axis('off')
    
    # Hide empty subplots
    for ax in axes[n_images:]:
        ax.axis('off')
    
    plt.suptitle(f"Pipeline Execution ({result.execution_time_ms:.1f} ms)", fontsize=12)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    
    plt.show()


def compare_images(
    images: Dict[str, np.ndarray],
    cols: int = 3,
    figsize: Optional[Tuple[int, int]] = None,
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    เปรียบเทียบหลายภาพ
    
    Args:
        images: Dict ของ {name: image}
        cols: จำนวน columns
        figsize: Figure size
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization")
        return
    
    n_images = len(images)
    rows = math.ceil(n_images / cols)
    
    if figsize is None:
        figsize = (4 * cols, 3 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    axes = axes.flatten() if n_images > 1 else [axes]
    
    for ax, (name, img) in zip(axes, images.items()):
        ax.imshow(img, cmap='gray' if len(img.shape) == 2 else None)
        ax.set_title(name, fontsize=10)
        ax.axis('off')
    
    for ax in axes[n_images:]:
        ax.axis('off')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    
    plt.show()


def show_image(
    image: np.ndarray,
    title: str = "Image",
    figsize: Tuple[int, int] = (8, 6),
    save_path: Optional[Union[str, Path]] = None
) -> None:
    """
    แสดงภาพเดียว
    
    Args:
        image: Image array
        title: Window title
        figsize: Figure size
        save_path: Path to save figure
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization")
        return
    
    plt.figure(figsize=figsize)
    plt.imshow(image, cmap='gray' if len(image.shape) == 2 else None)
    plt.title(title)
    plt.axis('off')
    
    if save_path:
        plt.savefig(str(save_path), dpi=150, bbox_inches='tight')
    
    plt.show()


def show_histogram(
    image: np.ndarray,
    title: str = "Histogram",
    figsize: Tuple[int, int] = (10, 5)
) -> None:
    """
    แสดง histogram ของภาพ
    
    Args:
        image: Grayscale image
        title: Plot title
        figsize: Figure size
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is required for visualization")
        return
    
    if len(image.shape) == 3:
        image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    
    plt.figure(figsize=figsize)
    plt.hist(image.ravel(), bins=256, range=[0, 256])
    plt.title(title)
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.xlim([0, 256])
    plt.show()


def create_comparison_grid(
    images: List[np.ndarray],
    titles: Optional[List[str]] = None,
    cols: int = 3,
    cell_size: Tuple[int, int] = (200, 200),
    padding: int = 10,
    bg_color: int = 255
) -> np.ndarray:
    """
    สร้าง grid ของภาพสำหรับเปรียบเทียบ (ไม่ใช้ matplotlib)
    
    Args:
        images: List of images
        titles: List of titles
        cols: Number of columns
        cell_size: Size of each cell (width, height)
        padding: Padding between cells
        bg_color: Background color
        
    Returns:
        Grid image
    """
    n = len(images)
    rows = math.ceil(n / cols)
    
    cell_w, cell_h = cell_size
    grid_w = cols * cell_w + (cols + 1) * padding
    grid_h = rows * cell_h + (rows + 1) * padding
    
    grid = np.full((grid_h, grid_w), bg_color, dtype=np.uint8)
    
    for idx, img in enumerate(images):
        row = idx // cols
        col = idx % cols
        
        x = padding + col * (cell_w + padding)
        y = padding + row * (cell_h + padding)
        
        # Resize image to fit cell
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        h, w = img.shape
        scale = min(cell_w / w, cell_h / h)
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        resized = cv.resize(img, (new_w, new_h))
        
        # Center in cell
        offset_x = (cell_w - new_w) // 2
        offset_y = (cell_h - new_h) // 2
        
        grid[y + offset_y:y + offset_y + new_h, x + offset_x:x + offset_x + new_w] = resized
    
    return grid


__all__ = [
    'visualize_pipeline_result',
    'compare_images',
    'show_image',
    'show_histogram',
    'create_comparison_grid',
]
