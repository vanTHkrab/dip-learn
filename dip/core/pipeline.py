"""
Core Pipeline Module

ระบบ Pipeline สำหรับ step-by-step image processing
"""

from __future__ import annotations

import json
import copy
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any, Callable, Union
import numpy as np


@dataclass
class PipelineStep:
    """
    แทน 1 step ใน pipeline
    
    Attributes:
        name: ชื่อ operation (ต้องตรงกับ method ที่ register ไว้)
        params: parameters ที่จะส่งให้ operation
        enabled: ถ้า False จะข้าม step นี้
    """
    name: str
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'name': self.name,
            'params': self.params,
            'enabled': self.enabled
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'PipelineStep':
        """Create from dictionary."""
        return cls(
            name=data['name'],
            params=data.get('params', {}),
            enabled=data.get('enabled', True)
        )


@dataclass
class PipelineResult:
    """
    ผลลัพธ์จากการรัน pipeline
    
    Attributes:
        final_image: ภาพสุดท้าย
        history: Dict ของภาพในแต่ละ step
        steps_executed: รายชื่อ steps ที่ทำงาน
        execution_time_ms: เวลาที่ใช้ (milliseconds)
    """
    final_image: np.ndarray
    history: Dict[str, np.ndarray] = field(default_factory=dict)
    steps_executed: List[str] = field(default_factory=list)
    execution_time_ms: float = 0.0
    
    @property
    def output(self) -> np.ndarray:
        """Alias for final_image."""
        return self.final_image
    
    @property
    def steps(self) -> List[str]:
        """Alias for steps_executed."""
        return self.steps_executed


class ImagePipeline:
    """
    Image processing pipeline ที่สามารถเพิ่ม/ลบ/แก้ไข steps ได้
    
    รองรับ:
    - การเพิ่ม step แบบ method chaining
    - การ save/load configuration
    - การดู intermediate results
    - การ enable/disable steps
    - Custom operations
    """
    
    # Operation registry - will be populated by register_all_operations
    _operations: Dict[str, Callable] = {}
    _initialized: bool = False
    
    @classmethod
    def _ensure_operations_registered(cls):
        """Ensure all operations are registered."""
        if not cls._initialized:
            cls._register_all_operations()
            cls._initialized = True
    
    @classmethod
    def _register_all_operations(cls):
        """Register all operations from submodules."""
        from .. import transforms
        from .. import filters
        from .. import enhancement
        
        # Color operations
        cls._operations.update({
            'grayscale': transforms.to_grayscale,
            'to_grayscale': transforms.to_grayscale,
            'to_bgr': transforms.to_bgr,
            'to_rgb': transforms.to_rgb,
            'to_hsv': transforms.to_hsv,
        })
        
        # Threshold operations
        cls._operations.update({
            'binary_threshold': transforms.binary_threshold,
            'otsu_threshold': transforms.otsu_threshold,
            'otsu': transforms.otsu_threshold,
            'adaptive_threshold_mean': transforms.adaptive_threshold_mean,
            'adaptive_threshold_gaussian': transforms.adaptive_threshold_gaussian,
            'adaptive_mean': transforms.adaptive_threshold_mean,
            'adaptive_gaussian': transforms.adaptive_threshold_gaussian,
            'auto_threshold': transforms.auto_threshold,
        })
        
        # Morphology operations
        cls._operations.update({
            'erode': transforms.erode,
            'dilate': transforms.dilate,
            'morph_open': transforms.morph_open,
            'open': transforms.morph_open,
            'morph_close': transforms.morph_close,
            'close': transforms.morph_close,
            'morph_gradient': transforms.morph_gradient,
            'top_hat': transforms.top_hat,
            'tophat': transforms.top_hat,
            'black_hat': transforms.black_hat,
            'blackhat': transforms.black_hat,
            'skeleton': transforms.skeletonize,
            'skeletonize': transforms.skeletonize,
        })
        
        # Geometric operations
        cls._operations.update({
            'resize': transforms.resize,
            'rotate': transforms.rotate,
            'flip': transforms.flip,
            'crop': transforms.crop,
            'pad': transforms.pad,
        })
        
        # Histogram operations
        cls._operations.update({
            'histogram_equalization': transforms.histogram_equalization,
            'equalize': transforms.histogram_equalization,
            'clahe': transforms.clahe,
            'contrast_stretch': transforms.contrast_stretch,
            'normalize': transforms.normalize,
            'normalize_lighting': transforms.normalize_lighting,
        })
        
        # Smoothing operations
        cls._operations.update({
            'gaussian_blur': filters.gaussian_blur,
            'gaussian': filters.gaussian_blur,
            'blur': filters.gaussian_blur,
            'median_blur': filters.median_blur,
            'median': filters.median_blur,
            'bilateral_filter': filters.bilateral_filter,
            'bilateral': filters.bilateral_filter,
            'box_blur': filters.box_blur,
        })
        
        # Sharpening operations
        cls._operations.update({
            'unsharp_mask': filters.unsharp_mask,
            'sharpen': filters.unsharp_mask,
            'laplacian_sharpen': filters.laplacian_sharpen,
            'kernel_sharpen': filters.kernel_sharpen,
            'high_boost': filters.high_boost_filter,
        })
        
        # Edge operations
        cls._operations.update({
            'canny': filters.canny_edge,
            'canny_edge': filters.canny_edge,
            'sobel': filters.sobel_edge,
            'sobel_edge': filters.sobel_edge,
            'laplacian_edge': filters.laplacian_edge,
            'laplacian': filters.laplacian_edge,
        })
        
        # Brightness operations
        cls._operations.update({
            'adjust_brightness': enhancement.adjust_brightness,
            'brightness': enhancement.adjust_brightness,
            'adjust_contrast': enhancement.adjust_contrast,
            'contrast': enhancement.adjust_contrast,
            'adjust_brightness_contrast': enhancement.adjust_brightness_contrast,
            'gamma': enhancement.gamma_correction,
            'gamma_correction': enhancement.gamma_correction,
            'auto_brightness_contrast': enhancement.auto_brightness_contrast,
            'auto_bc': enhancement.auto_brightness_contrast,
        })
        
        # Denoise operations
        cls._operations.update({
            'non_local_means': enhancement.non_local_means_denoising,
            'nlm': enhancement.non_local_means_denoising,
            'remove_salt_pepper': enhancement.remove_salt_pepper_noise,
            'denoise_morph': enhancement.denoise_morphological,
            'denoise_bilateral': enhancement.denoise_bilateral,
            'anisotropic': enhancement.anisotropic_diffusion,
        })
    
    def __init__(self, steps: Optional[List[Union[Tuple[str, Dict], PipelineStep, str]]] = None):
        """
        Initialize pipeline
        
        Args:
            steps: List of steps as (name, params) tuples, PipelineStep objects, or strings
        """
        self._ensure_operations_registered()
        
        self.steps: List[PipelineStep] = []
        self._custom_ops: Dict[str, Callable] = {}
        
        if steps:
            for step in steps:
                if isinstance(step, PipelineStep):
                    self.steps.append(step)
                elif isinstance(step, tuple):
                    name, params = step
                    self.steps.append(PipelineStep(name=name, params=params))
                elif isinstance(step, str):
                    self.steps.append(PipelineStep(name=step, params={}))
    
    def add_step(
        self,
        name: str,
        enabled: bool = True,
        **params
    ) -> 'ImagePipeline':
        """
        เพิ่ม step ใน pipeline (method chaining)
        
        Args:
            name: ชื่อ operation
            enabled: Enable/disable step
            **params: Parameters สำหรับ operation
            
        Returns:
            self for method chaining
        """
        self.steps.append(PipelineStep(name=name, params=params, enabled=enabled))
        return self
    
    def insert_step(
        self,
        index: int,
        name: str,
        enabled: bool = True,
        **params
    ) -> 'ImagePipeline':
        """แทรก step ที่ตำแหน่งที่กำหนด"""
        self.steps.insert(index, PipelineStep(name=name, params=params, enabled=enabled))
        return self
    
    def remove_step(self, index: int) -> 'ImagePipeline':
        """ลบ step ที่ตำแหน่งที่กำหนด"""
        if 0 <= index < len(self.steps):
            self.steps.pop(index)
        return self
    
    def remove_step_by_name(self, name: str) -> 'ImagePipeline':
        """ลบ step ตามชื่อ (ลบทั้งหมดที่ตรงกัน)"""
        self.steps = [s for s in self.steps if s.name != name]
        return self
    
    def enable_step(self, index: int) -> 'ImagePipeline':
        """Enable step at index."""
        if 0 <= index < len(self.steps):
            self.steps[index].enabled = True
        return self
    
    def disable_step(self, index: int) -> 'ImagePipeline':
        """Disable step at index."""
        if 0 <= index < len(self.steps):
            self.steps[index].enabled = False
        return self
    
    def clear(self) -> 'ImagePipeline':
        """Clear all steps."""
        self.steps.clear()
        return self
    
    def register_operation(self, name: str, func: Callable) -> 'ImagePipeline':
        """Register custom operation"""
        self._custom_ops[name] = func
        return self
    
    def _get_operation(self, name: str) -> Optional[Callable]:
        """Get operation function by name."""
        if name in self._custom_ops:
            return self._custom_ops[name]
        return self._operations.get(name)
    
    def process(self, image: np.ndarray) -> np.ndarray:
        """
        Process image through pipeline
        
        Args:
            image: Input image
            
        Returns:
            Processed image
        """
        result = image.copy()
        
        for step in self.steps:
            if not step.enabled:
                continue
            
            op = self._get_operation(step.name)
            if op is None:
                raise ValueError(f"Unknown operation: {step.name}")
            
            result = op(result, **step.params)
        
        return result
    
    def process_with_history(
        self,
        image: np.ndarray,
        include_input: bool = True
    ) -> PipelineResult:
        """
        Process image และเก็บผลลัพธ์ของแต่ละ step
        
        Args:
            image: Input image
            include_input: Include original image in history
            
        Returns:
            PipelineResult with all intermediate results
        """
        start_time = time.time()
        result = image.copy()
        history = {}
        steps_executed = []
        
        if include_input:
            history['input'] = image.copy()
        
        for i, step in enumerate(self.steps):
            if not step.enabled:
                continue
            
            op = self._get_operation(step.name)
            if op is None:
                raise ValueError(f"Unknown operation: {step.name}")
            
            result = op(result, **step.params)
            
            step_key = f"{i:02d}_{step.name}"
            history[step_key] = result.copy()
            steps_executed.append(step.name)
        
        execution_time = (time.time() - start_time) * 1000
        
        return PipelineResult(
            final_image=result,
            history=history,
            steps_executed=steps_executed,
            execution_time_ms=execution_time
        )
    
    def run(self, image: np.ndarray, include_history: bool = True) -> PipelineResult:
        """
        Run pipeline on image (alias for process_with_history).
        
        Args:
            image: Input image
            include_history: Include intermediate results
            
        Returns:
            PipelineResult with output and metadata
        """
        return self.process_with_history(image, include_input=include_history)
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Allow pipeline to be called like a function."""
        return self.process(image)
    
    def __len__(self) -> int:
        """Return number of steps."""
        return len(self.steps)
    
    def __repr__(self) -> str:
        """String representation."""
        step_strs = []
        for i, step in enumerate(self.steps):
            status = "✓" if step.enabled else "✗"
            step_strs.append(f"  [{status}] {i}: {step.name}({step.params})")
        return "ImagePipeline(\n" + "\n".join(step_strs) + "\n)"
    
    # Configuration Save/Load
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert pipeline to dictionary."""
        return {
            'version': '1.0',
            'steps': [step.to_dict() for step in self.steps]
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ImagePipeline':
        """Create pipeline from dictionary."""
        steps = [PipelineStep.from_dict(s) for s in data.get('steps', [])]
        return cls(steps=steps)
    
    def save_config(self, filepath: Union[str, Path]) -> None:
        """Save pipeline configuration to JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def from_config(cls, filepath: Union[str, Path]) -> 'ImagePipeline':
        """Load pipeline from JSON file."""
        filepath = Path(filepath)
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data)
    
    def copy(self) -> 'ImagePipeline':
        """Create a copy of this pipeline."""
        new_pipeline = ImagePipeline()
        new_pipeline.steps = copy.deepcopy(self.steps)
        new_pipeline._custom_ops = self._custom_ops.copy()
        return new_pipeline


# Convenience functions

def quick_process(
    image: np.ndarray,
    operations: List[Union[str, Tuple[str, Dict]]]
) -> np.ndarray:
    """
    Quick one-liner processing
    
    Args:
        image: Input image
        operations: List of operation names or (name, params) tuples
        
    Returns:
        Processed image
    """
    pipeline = ImagePipeline(operations)
    return pipeline.process(image)


def apply_sequence(image: np.ndarray, *operations: str, **params) -> np.ndarray:
    """
    Apply sequence of operations with shared params
    
    Args:
        image: Input image
        *operations: Operation names
        **params: Shared parameters
        
    Returns:
        Processed image
    """
    pipeline = ImagePipeline()
    for op in operations:
        pipeline.add_step(op, **params)
    return pipeline.process(image)


__all__ = [
    'PipelineStep',
    'PipelineResult',
    'ImagePipeline',
    'quick_process',
    'apply_sequence',
]
