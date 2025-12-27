"""
Preset Pipelines

Pre-configured pipelines สำหรับงานทั่วไป
"""

from ..core.pipeline import ImagePipeline


class PresetPipelines:
    """
    Pre-configured pipelines สำหรับงานทั่วไป
    """
    
    @staticmethod
    def ocr_basic() -> ImagePipeline:
        """Pipeline พื้นฐานสำหรับ OCR"""
        return ImagePipeline([
            ('grayscale', {}),
            ('gaussian_blur', {'ksize': 3}),
            ('clahe', {'clip_limit': 2.0}),
            ('otsu_threshold', {}),
        ])
    
    @staticmethod
    def ocr_advanced() -> ImagePipeline:
        """Pipeline ขั้นสูงสำหรับ OCR"""
        return ImagePipeline([
            ('grayscale', {}),
            ('contrast_stretch', {'percentile_low': 1, 'percentile_high': 99}),
            ('normalize_lighting', {}),
            ('bilateral', {'d': 9, 'sigma_color': 75, 'sigma_space': 75}),
            ('clahe', {'clip_limit': 2.0, 'tile_grid_size': (8, 8)}),
            ('auto_threshold', {}),
            ('morph_open', {'kernel_size': 2}),
            ('morph_close', {'kernel_size': 2}),
        ])
    
    @staticmethod
    def seven_segment() -> ImagePipeline:
        """Pipeline สำหรับ seven-segment display"""
        return ImagePipeline([
            ('grayscale', {}),
            ('normalize_lighting', {}),
            ('clahe', {'clip_limit': 3.0}),
            ('otsu', {'inverse': True}),
            ('dilate', {'kernel_size': 2, 'iterations': 1}),
            ('morph_open', {'kernel_size': 2}),
        ])
    
    @staticmethod
    def denoise_light() -> ImagePipeline:
        """Pipeline สำหรับลด noise แบบ light"""
        return ImagePipeline([
            ('gaussian_blur', {'ksize': 3}),
            ('bilateral', {'d': 5, 'sigma_color': 50, 'sigma_space': 50}),
        ])
    
    @staticmethod
    def denoise_heavy() -> ImagePipeline:
        """Pipeline สำหรับลด noise แบบ heavy"""
        return ImagePipeline([
            ('median', {'ksize': 5}),
            ('gaussian', {'ksize': 5}),
            ('bilateral', {'d': 9, 'sigma_color': 75, 'sigma_space': 75}),
            ('nlm', {'h': 10}),
        ])
    
    @staticmethod
    def enhance_contrast() -> ImagePipeline:
        """Pipeline สำหรับเพิ่ม contrast"""
        return ImagePipeline([
            ('grayscale', {}),
            ('contrast_stretch', {}),
            ('clahe', {'clip_limit': 2.0}),
            ('sharpen', {'amount': 1.0}),
        ])
    
    @staticmethod
    def edge_detection() -> ImagePipeline:
        """Pipeline สำหรับ edge detection"""
        return ImagePipeline([
            ('grayscale', {}),
            ('gaussian', {'ksize': 5}),
            ('canny', {'threshold1': 50, 'threshold2': 150}),
        ])
    
    @staticmethod
    def binarization() -> ImagePipeline:
        """Pipeline สำหรับ binarization"""
        return ImagePipeline([
            ('grayscale', {}),
            ('gaussian', {'ksize': 3}),
            ('adaptive_gaussian', {'block_size': 11, 'c': 2}),
        ])
    
    @staticmethod
    def sharpen() -> ImagePipeline:
        """Pipeline สำหรับ sharpening"""
        return ImagePipeline([
            ('sharpen', {'amount': 1.5}),
            ('high_boost', {'amplification': 1.3}),
        ])
    
    @staticmethod
    def morphological_clean() -> ImagePipeline:
        """Pipeline สำหรับ clean binary image"""
        return ImagePipeline([
            ('morph_open', {'kernel_size': 2}),
            ('morph_close', {'kernel_size': 2}),
            ('erode', {'kernel_size': 2, 'iterations': 1}),
            ('dilate', {'kernel_size': 2, 'iterations': 1}),
        ])
    
    @staticmethod
    def document_scan() -> ImagePipeline:
        """Pipeline สำหรับ document scanning"""
        return ImagePipeline([
            ('grayscale', {}),
            ('normalize_lighting', {}),
            ('clahe', {'clip_limit': 2.0}),
            ('sharpen', {'amount': 1.0}),
            ('adaptive_gaussian', {'block_size': 15, 'c': 4}),
        ])
    
    @staticmethod
    def photo_enhance() -> ImagePipeline:
        """Pipeline สำหรับ enhance photo"""
        return ImagePipeline([
            ('auto_bc', {}),
            ('bilateral', {'d': 9, 'sigma_color': 75, 'sigma_space': 75}),
            ('sharpen', {'amount': 0.5}),
        ])


__all__ = ['PresetPipelines']
