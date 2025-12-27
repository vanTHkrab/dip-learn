
A comprehensive digital image processing library built with OpenCV, NumPy, and scikit-image for Python.

**Version:** 0.0.1-beta  
**Python:** 3.9 - 3.13

## 📦 Installation

### Install via pip (from PyPI)

```bash
pip install dip-learn
```

### Install from GitHub

```bash
# Install latest version from main branch
pip install git+https://github.com/vanTHkrab/dip-learn.git

# Install specific version/tag
pip install git+https://github.com/vanTHkrab/dip-learn.git@v0.0.1b1

# Install specific branch
pip install git+https://github.com/vanTHkrab/dip-learn.git@main

# Install with optional dependencies
pip install "dip-learn[all] @ git+https://github.com/vanTHkrab/dip-learn.git"
```

### Install from source

```bash
# Clone the repository
git clone https://github.com/vanTHkrab/dip-learn.git
cd dip-learn

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install in editable mode (for development)
pip install -e .

# Install with all optional dependencies
pip install -e ".[all]"

# Install with development dependencies
pip install -e ".[dev]"
```

### Dependencies

**Required:**
- `numpy>=1.20.0`
- `opencv-python>=4.5.0`

**Optional (for advanced features):**
```bash
pip install scikit-image  # Advanced thresholding, morphology, denoising
```

## 🚀 Quick Start

```python
from images_process import ImageOps, ImagePipeline, PresetPipelines

# Load image
from images_process import load_image
img = load_image('path/to/image.png')

# Quick operations
from images_process import to_grayscale, gaussian_blur, otsu_threshold
gray = to_grayscale(img)
blurred = gaussian_blur(gray, ksize=5)
binary = otsu_threshold(blurred)

# Using ImageOps (static class)
result = ImageOps.gaussian_blur(img, ksize=5)
result = ImageOps.clahe(result, clip_limit=2.0)

# Using Pipeline
pipeline = ImagePipeline()
pipeline.add_step('grayscale', to_grayscale)
pipeline.add_step('blur', gaussian_blur, ksize=5)
pipeline.add_step('threshold', otsu_threshold)
result = pipeline.run(img)
print(f"Output: {result.output.shape}")

# Using Presets (recommended)
result = PresetPipelines.ocr_basic().run(img)
result = PresetPipelines.seven_segment().run(img)
```

## 📁 Project Structure

```
images_process/
├── __init__.py          # Main exports
├── core/                # Core components
│   ├── operations.py    # ImageOps static class
│   └── pipeline.py      # ImagePipeline, PipelineResult
├── transforms/          # Image transformations
│   ├── color.py         # Color space conversions
│   ├── threshold.py     # Thresholding operations
│   ├── morphology.py    # Morphological operations
│   ├── geometric.py     # Resize, rotate, flip, crop
│   └── histogram.py     # Histogram equalization, CLAHE
├── filters/             # Image filters
│   ├── smoothing.py     # Gaussian, median, bilateral blur
│   ├── sharpening.py    # Unsharp mask, kernel sharpen
│   └── edge.py          # Canny, Sobel, Laplacian edge detection
├── enhancement/         # Image enhancement
│   ├── brightness.py    # Brightness, contrast, gamma
│   └── denoise.py       # Denoising operations
├── presets/             # Pre-built pipelines
│   └── pipelines.py     # PresetPipelines class
├── utils/               # Utility functions
│   ├── io.py            # Load/save images
│   ├── visualization.py # Display, compare images
│   └── metrics.py       # Image quality metrics
└── cli/                 # Command-line interface
    └── main.py          # CLI entry point
```

## 📚 Modules

### Core

#### ImageOps
Static class with all image operations as methods.

```python
from images_process import ImageOps

# All operations available as static methods
gray = ImageOps.to_grayscale(img)
blurred = ImageOps.gaussian_blur(img, ksize=5)
thresh = ImageOps.otsu_threshold(gray)
enhanced = ImageOps.clahe(gray, clip_limit=2.0)
```

#### ImagePipeline
Chainable pipeline for step-by-step processing.

```python
from images_process import ImagePipeline, to_grayscale, gaussian_blur

pipeline = ImagePipeline()
pipeline.add_step('grayscale', to_grayscale)
pipeline.add_step('blur', gaussian_blur, ksize=5)

# Run pipeline
result = pipeline.run(img)
print(f"Output: {result.output.shape}")
print(f"Steps: {result.steps}")
print(f"Time: {result.execution_time_ms:.2f}ms")

# Access intermediate results
for name, intermediate in result.history.items():
    print(f"{name}: {intermediate.shape}")
```

### Transforms

#### Color Transforms
```python
from images_process import (
    to_grayscale,    # Convert to grayscale
    to_bgr,          # Convert to BGR
    to_rgb,          # Convert to RGB
    to_hsv,          # Convert to HSV
    to_lab,          # Convert to LAB
    split_channels,  # Split into channels
    merge_channels,  # Merge channels
)
```

#### Threshold
```python
from images_process import (
    binary_threshold,           # Simple threshold
    otsu_threshold,             # Otsu's automatic threshold
    adaptive_threshold_mean,    # Adaptive mean threshold
    adaptive_threshold_gaussian,# Adaptive Gaussian threshold
    auto_threshold,             # Auto select best method
    triangle_threshold,         # Triangle method
)
```

#### Morphology
```python
from images_process import (
    erode,           # Erosion
    dilate,          # Dilation
    morph_open,      # Opening (erosion -> dilation)
    morph_close,     # Closing (dilation -> erosion)
    morph_gradient,  # Gradient
    top_hat,         # Top hat transform
    black_hat,       # Black hat transform
    skeletonize,     # Skeletonization
)
```

#### Geometric
```python
from images_process import (
    resize,              # Resize with width/height/scale
    rotate,              # Rotate by angle
    rotate_bound,        # Rotate without cropping
    flip,                # Flip horizontal/vertical
    crop,                # Crop region
    crop_center,         # Crop from center
    pad,                 # Add padding
    pad_to_size,         # Pad to specific size
    translate,           # Translate/shift
    perspective_transform,# Perspective warp
)
```

#### Histogram
```python
from images_process import (
    histogram_equalization,  # Global histogram equalization
    clahe,                   # CLAHE (adaptive)
    contrast_stretch,        # Contrast stretching
    normalize,               # Normalize to range
    histogram_matching,      # Match histogram
    normalize_lighting,      # Normalize lighting
)
```

### Filters

#### Smoothing
```python
from images_process import (
    gaussian_blur,     # Gaussian blur
    median_blur,       # Median blur
    bilateral_filter,  # Bilateral filter
    box_blur,          # Box/average blur
    stack_blur,        # Stack blur (fast)
    blur_2d,           # Custom kernel blur
)
```

#### Sharpening
```python
from images_process import (
    unsharp_mask,       # Unsharp masking
    laplacian_sharpen,  # Laplacian sharpening
    kernel_sharpen,     # Kernel-based sharpening
    high_boost_filter,  # High-boost filter
    frequency_high_boost,# Frequency domain
)
```

#### Edge Detection
```python
from images_process import (
    canny_edge,      # Canny edge detector
    sobel_edge,      # Sobel edge detector
    sobel_x,         # Sobel X gradient
    sobel_y,         # Sobel Y gradient
    laplacian_edge,  # Laplacian edge detector
    scharr_edge,     # Scharr edge detector
    prewitt_edge,    # Prewitt edge detector
    roberts_edge,    # Roberts edge detector
    auto_canny,      # Auto-tuned Canny
)
```

### Annotation

Drawing and annotation utilities for visualizing results and labeling images.

#### Basic Shapes
```python
from images_process import (
    draw_rectangle,      # Draw rectangle
    draw_circle,         # Draw circle
    draw_line,           # Draw line
    draw_ellipse,        # Draw ellipse
    draw_polygon,        # Draw polygon
    draw_filled_polygon, # Draw filled polygon
    draw_polylines,      # Draw multiple polylines
    draw_arrow,          # Draw arrow
)

# Draw shapes on image
result = draw_rectangle(img, (10, 10), (100, 100), color=(0, 255, 0), thickness=2)
result = draw_circle(img, center=(50, 50), radius=30, color=(255, 0, 0))
result = draw_line(img, (0, 0), (100, 100), color=(0, 0, 255), thickness=2)
result = draw_polygon(img, [(10, 10), (50, 10), (30, 50)], color=(0, 255, 255))
```

#### Text Operations
```python
from images_process import (
    draw_text,                  # Draw text
    draw_text_with_background,  # Draw text with background
    get_text_size,              # Calculate text size
    draw_multiline_text,        # Draw multiline text
)

# Add text to image
result = draw_text(img, "Hello", (10, 50), font_scale=1.0, color=(255, 255, 255))
result = draw_text_with_background(img, "Label", (10, 10), 
                                    text_color=(255, 255, 255), 
                                    bg_color=(0, 0, 0))
result = draw_multiline_text(img, "Line 1\nLine 2\nLine 3", (10, 10))

# Get text dimensions
size, baseline = get_text_size("Sample Text", font_scale=1.0)
```

#### Annotation Utilities
```python
from images_process import (
    draw_bounding_box,   # Draw bounding box with label
    draw_bounding_boxes, # Draw multiple bounding boxes
    draw_contours,       # Draw contours
    draw_keypoints,      # Draw keypoints
    draw_grid,           # Draw grid overlay
    draw_crosshair,      # Draw crosshair
    draw_marker,         # Draw marker
)

# Draw bounding boxes for object detection results
result = draw_bounding_box(img, (x, y, w, h), label="Object", color=(0, 255, 0))
result = draw_bounding_boxes(img, 
                              bboxes=[(10, 10, 50, 50), (60, 60, 40, 40)],
                              labels=["Cat", "Dog"],
                              colors=[(0, 255, 0), (255, 0, 0)])

# Draw keypoints for pose estimation
keypoints = [(100, 100), (150, 120), (200, 100)]
result = draw_keypoints(img, keypoints, color=(0, 255, 0), radius=5)

# Draw grid overlay
result = draw_grid(img, grid_size=(10, 10), color=(128, 128, 128))

# Draw crosshair at center
result = draw_crosshair(img, center=(50, 50), size=20, color=(0, 255, 0))
```

#### Overlay Operations
```python
from images_process import (
    overlay_image,       # Overlay one image on another
    add_alpha_channel,   # Add alpha channel to image
    create_mask_overlay, # Create colored mask overlay
)

# Overlay logo or watermark
result = overlay_image(background, overlay, position=(10, 10), alpha=0.5)

# Create mask visualization
result = create_mask_overlay(img, mask, color=(0, 255, 0), alpha=0.5)
```

### Enhancement

#### Brightness & Contrast
```python
from images_process import (
    adjust_brightness,        # Adjust brightness
    adjust_contrast,          # Adjust contrast
    gamma_correction,         # Gamma correction
    auto_brightness_contrast, # Auto adjust
    sigmoid_correction,       # Sigmoid correction
    log_transform,            # Log transform
    power_law_transform,      # Power law transform
)
```

#### Denoising
```python
from images_process import (
    non_local_means_denoising,  # NLM denoising
    remove_salt_pepper_noise,   # Salt & pepper removal
    denoise_morphological,      # Morphological denoising
    denoise_bilateral,          # Bilateral denoising
    denoise_gaussian,           # Gaussian denoising
    anisotropic_diffusion,      # Anisotropic diffusion
    richardson_lucy_deblur,     # Deblurring
)
```

### Augmentation

Data augmentation utilities for machine learning and deep learning training.

#### Geometric Transformations
```python
from dip import (
    random_flip,             # Random horizontal/vertical flip
    random_rotation,         # Random rotation within angle range
    random_scale,            # Random scale/zoom
    random_translate,        # Random translation/shift
    random_shear,            # Random shear transformation
    random_crop,             # Random crop to size
    random_perspective,      # Random perspective transform
    random_elastic_transform,# Elastic deformation
)

# Flip with probability
augmented = random_flip(img, horizontal=True, vertical=False, p=0.5)

# Rotate randomly between -30 and 30 degrees
augmented = random_rotation(img, angle_range=(-30, 30))

# Random scale between 0.8x and 1.2x
augmented = random_scale(img, scale_range=(0.8, 1.2), keep_size=True)

# Random crop to 80x80
augmented = random_crop(img, crop_size=(80, 80))
```

#### Color Augmentations
```python
from dip import (
    random_brightness,       # Random brightness adjustment
    random_contrast,         # Random contrast adjustment
    random_saturation,       # Random saturation change
    random_hue,              # Random hue shift
    random_color_jitter,     # Combined color augmentation
    random_gamma,            # Random gamma correction
    random_channel_shuffle,  # Shuffle color channels
    to_grayscale_augment,    # Random grayscale conversion
)

# Random brightness (-30% to +30%)
augmented = random_brightness(img, brightness_range=(-0.3, 0.3))

# Combined color jitter
augmented = random_color_jitter(img, brightness=0.3, contrast=0.3, 
                                 saturation=0.3, hue=0.1)
```

#### Noise Augmentations
```python
from dip import (
    random_gaussian_noise,   # Add Gaussian noise
    random_salt_pepper_noise,# Add salt and pepper noise
    random_speckle_noise,    # Add speckle noise
    random_poisson_noise,    # Add Poisson noise
)

# Add Gaussian noise
augmented = random_gaussian_noise(img, mean=0, std_range=(10, 30))

# Add salt and pepper noise
augmented = random_salt_pepper_noise(img, amount_range=(0.01, 0.05))
```

#### Blur Augmentations
```python
from dip import (
    random_gaussian_blur,    # Random Gaussian blur
    random_motion_blur,      # Random motion blur
    random_median_blur,      # Random median blur
)

# Random blur
augmented = random_gaussian_blur(img, kernel_range=(3, 7))
augmented = random_motion_blur(img, kernel_range=(5, 15))
```

#### Cutout and Erasing
```python
from dip import (
    random_cutout,           # Random rectangular cutout
    random_grid_mask,        # Grid-based masking
)

# Random cutout (like dropout but for images)
augmented = random_cutout(img, num_holes=3, hole_size_range=(10, 30))

# Grid masking
augmented = random_grid_mask(img, grid_size=10, ratio=0.5)
```

#### Mixup and CutMix
```python
from dip import mixup, cutmix

# Mixup - linear combination of two images
mixed, alpha = mixup(img1, img2, alpha_range=(0.3, 0.7))

# CutMix - paste patch from one image to another
mixed, bbox = cutmix(img1, img2, beta=1.0)
```

#### Augmentation Pipeline
```python
from dip import AugmentationPipeline, get_default_augmentation_pipeline

# Create custom pipeline
pipeline = AugmentationPipeline()
pipeline.add(random_flip, horizontal=True, p=0.5)
pipeline.add(random_rotation, angle_range=(-15, 15))
pipeline.add(random_color_jitter, brightness=0.2, contrast=0.2)
pipeline.add(random_gaussian_blur, kernel_range=(3, 5))

# Apply to image
augmented = pipeline(img)

# Use default pipelines
light_pipeline = get_default_augmentation_pipeline(strength='light')
medium_pipeline = get_default_augmentation_pipeline(strength='medium')
strong_pipeline = get_default_augmentation_pipeline(strength='strong')

augmented = medium_pipeline(img)
```

#### Batch Augmentation
```python
from dip import augment_batch, create_augmented_dataset

# Augment a batch of images
augmented_batch = augment_batch(images, pipeline)

# Create augmented dataset (multiple augmentations per image)
augmented_dataset = create_augmented_dataset(
    images, 
    pipeline, 
    augmentations_per_image=5
)
```

### Presets

Pre-built pipelines for common tasks:

```python
from images_process import PresetPipelines

# OCR preprocessing
ocr_result = PresetPipelines.ocr_basic().run(img)
ocr_result = PresetPipelines.ocr_advanced().run(img)

# Seven segment display
seg_result = PresetPipelines.seven_segment().run(img)

# Denoising
denoised = PresetPipelines.denoise_light().run(img)
denoised = PresetPipelines.denoise_heavy().run(img)

# Enhancement
enhanced = PresetPipelines.enhance_contrast().run(img)
sharpened = PresetPipelines.sharpen().run(img)

# Edge detection
edges = PresetPipelines.edge_detection().run(img)

# Binarization
binary = PresetPipelines.binarization().run(img)

# Document scanning
doc = PresetPipelines.document_scan().run(img)

# Photo enhancement
photo = PresetPipelines.photo_enhance().run(img)

# Morphological cleaning
cleaned = PresetPipelines.morphological_clean().run(img)
```

### Utils

#### I/O
```python
from images_process import (
    load_image,           # Load image from file
    save_image,           # Save image to file
    load_images_from_dir, # Load all images from directory
    image_from_bytes,     # Load from bytes
    image_to_bytes,       # Convert to bytes
)

# Load
img = load_image('image.png')
img = load_image('image.png', grayscale=True)

# Save
save_image(img, 'output.png')
save_image(img, 'output.jpg', quality=95)

# Batch load
images = load_images_from_dir('images/', pattern='*.png')
```

#### Visualization
```python
from images_process import (
    visualize_pipeline_result,  # Show pipeline results
    compare_images,             # Side-by-side comparison
    show_image,                 # Display single image
    show_histogram,             # Show histogram
    create_comparison_grid,     # Create image grid
)

# Compare before/after
compare_images(original, processed, titles=['Original', 'Processed'])

# Visualize pipeline
visualize_pipeline_result(result)
```

#### Metrics
```python
from images_process import (
    variance_of_laplacian,  # Blur detection metric
    is_blurry,              # Check if image is blurry
    calculate_snr,          # Signal-to-noise ratio
    calculate_contrast,     # Contrast metric
    calculate_entropy,      # Entropy metric
    calculate_psnr,         # Peak SNR (compare two images)
    calculate_ssim,         # SSIM (compare two images)
    get_image_stats,        # Get all statistics
)

# Check blur
blur_score = variance_of_laplacian(img)
if is_blurry(img, threshold=100):
    print("Image is blurry!")

# Get stats
stats = get_image_stats(img)
print(f"Contrast: {stats['contrast']:.2f}")
print(f"Entropy: {stats['entropy']:.2f}")
```

## 🔧 Convenience Functions

```python
from images_process import process_for_ocr, process_seven_segment, batch_process

# Quick OCR preprocessing
processed = process_for_ocr('image.png', preset='basic')
processed = process_for_ocr('image.png', preset='advanced')

# Seven segment processing
processed = process_seven_segment('display.png')

# Batch processing
images = ['img1.png', 'img2.png', 'img3.png']
results = batch_process(images, pipeline='ocr_basic')
```

## 💻 CLI Usage

```bash
# Process single image
python -m images_process.cli process input.png -o output.png -p ocr_basic

# Get image info
python -m images_process.cli info image.png

# Compare two images
python -m images_process.cli compare original.png processed.png
```

## 📋 Available Presets

| Preset | Description | Use Case |
|--------|-------------|----------|
| `ocr_basic` | Basic OCR preprocessing | Simple OCR tasks |
| `ocr_advanced` | Advanced OCR with denoising | Complex OCR |
| `seven_segment` | Seven segment display | BP monitor displays |
| `denoise_light` | Light denoising | Low noise images |
| `denoise_heavy` | Heavy denoising | High noise images |
| `enhance_contrast` | Contrast enhancement | Low contrast images |
| `sharpen` | Image sharpening | Blurry images |
| `edge_detection` | Edge detection | Feature extraction |
| `binarization` | Convert to binary | Document processing |
| `document_scan` | Document scanning | Scanned documents |
| `photo_enhance` | Photo enhancement | General photos |
| `morphological_clean` | Morphological cleaning | Binary cleanup |

## 🎯 Examples

### OCR Preprocessing
```python
from images_process import load_image, PresetPipelines, save_image

img = load_image('bp_display.jpg')
result = PresetPipelines.ocr_advanced().run(img)
save_image(result.output, 'preprocessed.png')
```

### Custom Pipeline
```python
from images_process import (
    ImagePipeline, to_grayscale, clahe, 
    gaussian_blur, otsu_threshold, morph_close
)

pipeline = ImagePipeline()
pipeline.add_step('grayscale', to_grayscale)
pipeline.add_step('clahe', clahe, clip_limit=2.0, tile_size=(8, 8))
pipeline.add_step('blur', gaussian_blur, ksize=3)
pipeline.add_step('threshold', otsu_threshold)
pipeline.add_step('clean', morph_close, ksize=3)

result = pipeline.run(img)
```

### Batch Processing with Progress
```python
from images_process import load_images_from_dir, PresetPipelines, save_image
import os

images = load_images_from_dir('input/')
pipeline = PresetPipelines.ocr_basic()

for name, img in images.items():
    result = pipeline.run(img)
    save_image(result.output, f'output/{name}')
    print(f"Processed: {name}")
```

## � Scikit-image Functions

When scikit-image is installed, additional advanced functions become available:

### Advanced Thresholding
```python
from images_process import (
    threshold_sauvola,    # Best for document/text images
    threshold_niblack,    # Local threshold for documents
    threshold_yen,        # Good for bimodal images
    threshold_li,         # Minimum cross entropy
    threshold_isodata,    # Iterative selection
    threshold_multiotsu,  # Multi-class segmentation
)

# Sauvola - excellent for text with uneven lighting
binary = threshold_sauvola(img, window_size=25)

# Multi-Otsu for multi-level segmentation
thresholds, regions = threshold_multiotsu(img, classes=3)
```

### Advanced Denoising
```python
from images_process import (
    denoise_tv_chambolle,  # Total Variation (edge-preserving)
    denoise_tv_bregman,    # Faster TV denoising
    denoise_wavelet,       # Wavelet-based denoising
    denoise_bilateral_sk,  # scikit-image bilateral
    denoise_nl_means_sk,   # Non-local means
)

# TV denoising - excellent edge preservation
denoised = denoise_tv_chambolle(img, weight=0.1)

# Wavelet denoising - good for various noise types
denoised = denoise_wavelet(img, wavelet='db1', mode='soft')
```

### Advanced Morphology
```python
from images_process import (
    thin,                    # Thin to 1-pixel skeleton
    skeletonize_sk,          # Skeletonization
    medial_axis_transform,   # Medial axis + distance
    remove_small_objects_sk, # Remove small components
    remove_small_holes,      # Fill small holes
    area_opening,            # Remove small bright regions
    area_closing,            # Fill small dark regions
)

# Clean binary image
cleaned = remove_small_objects_sk(binary, min_size=100)
cleaned = remove_small_holes(cleaned, area_threshold=50)

# Get skeleton
skeleton = skeletonize_sk(binary, method='lee')
```

### Feature Detection
```python
from images_process import (
    detect_blob_dog,         # Blob detection (DoG)
    detect_blob_log,         # Blob detection (LoG)
    detect_corners_harris,   # Harris corners
    detect_corners_shi_tomasi, # Shi-Tomasi corners
)

# Detect blobs
blobs = detect_blob_dog(gray, min_sigma=1, max_sigma=30)

# Detect corners
corners = detect_corners_harris(gray, sigma=1.0)
```

### Image Restoration
```python
from images_process import (
    wiener_filter,          # Wiener deconvolution
    richardson_lucy_sk,     # Richardson-Lucy deblur
    inpaint_biharmonic,     # Image inpainting
)

# Deblur image
psf = np.ones((5, 5)) / 25  # Simple blur PSF
restored = richardson_lucy_sk(blurred_img, psf, num_iter=30)

# Inpaint damaged areas
restored = inpaint_biharmonic(img, damage_mask)
```

### Ridge/Vessel Enhancement
```python
from images_process import (
    frangi_filter,    # Frangi vesselness
    meijering_filter, # Meijering neuriteness
    sato_filter,      # Sato tubeness
)

# Enhance tubular structures (blood vessels, etc.)
vessels = frangi_filter(img, sigmas=(1, 2, 3))
```

### Hough Transform
```python
from images_process import (
    hough_line_transform,
    hough_line_peaks,
    hough_circle_transform,
    hough_circle_peaks,
)

# Detect lines
hspace, angles, distances = hough_line_transform(edges)
accums, angles, dists = hough_line_peaks(hspace, angles, distances)

# Detect circles
radii = np.arange(20, 100, 5)
hspaces = hough_circle_transform(edges, radii)
```

### Check Availability
```python
from images_process import SKIMAGE_AVAILABLE

if SKIMAGE_AVAILABLE:
    from images_process import threshold_sauvola
    # Use advanced features
else:
    from images_process import adaptive_threshold_gaussian
    # Fallback to OpenCV
```

## 🧪 Testing

The library includes comprehensive test suites for all modules.

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test module
pytest dip/tests/test_annotation.py
pytest dip/tests/test_transforms.py
pytest dip/tests/test_filters.py
pytest dip/tests/test_enhancement.py
pytest dip/tests/test_core.py

# Run with coverage report
pytest --cov=dip --cov-report=html

# Run only specific test class
pytest dip/tests/test_annotation.py::TestDrawRectangle

# Run only specific test function
pytest dip/tests/test_annotation.py::TestDrawRectangle::test_draw_rectangle_basic
```

### Test Structure

```
dip-learn/tests/
├── __init__.py
├── test_annotation.py    # Annotation & drawing tests
├── test_core.py          # ImageOps & Pipeline tests
├── test_transforms.py    # Color, threshold, morphology, geometric tests
├── test_filters.py       # Smoothing, sharpening, edge detection tests
└── test_enhancement.py   # Brightness, contrast, denoise tests
```

### Test Coverage

| Module | Test File | Coverage |
|--------|-----------|----------|
| `annotation/` | `test_annotation.py` | Shapes, Text, Bounding Boxes, Overlays |
| `core/` | `test_core.py` | ImageOps, ImagePipeline |
| `transforms/` | `test_transforms.py` | Color, Threshold, Morphology, Geometric, Histogram |
| `filters/` | `test_filters.py` | Smoothing, Sharpening, Edge Detection |
| `enhancement/` | `test_enhancement.py` | Brightness, Contrast, Denoising |

### Writing New Tests

```python
import pytest
import numpy as np
from dip import draw_rectangle, to_grayscale


@pytest.fixture
def sample_image():
    """Create test image"""
    return np.zeros((100, 100, 3), dtype=np.uint8)


class TestMyFeature:
    def test_basic_functionality(self, sample_image):
        result = draw_rectangle(sample_image, (10, 10), (50, 50))
        assert result.max() > 0

    def test_with_parameters(self, sample_image):
        result = draw_rectangle(sample_image, (10, 10), (50, 50),
                                color=(255, 0, 0), thickness=-1)
        assert result[:, :, 0].max() == 255
```

## 📦 Building as Package

### Development Installation

```bash
# Clone repository
git clone https://github.com/vanTHkrab/dip-learn.git
cd dip

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

### Building the Package

```bash
# Install build tools
pip install build twine

# Build the package
python -m build

# This creates:
# dist/
#   ├── dip-0.1.0-py3-none-any.whl
#   └── dip-0.1.0.tar.gz
```

### Publishing to PyPI

```bash
# Upload to TestPyPI (for testing)
python -m twine upload --repository testpypi dist/*

# Upload to PyPI (production)
python -m twine upload dist/*
```

### Installing from Local Build

```bash
# Install from wheel
pip install dist/dip-0.1.0-py3-none-any.whl

# Or install from source distribution
pip install dist/dip-0.1.0.tar.gz
```

## 📄 License

MIT License

## 👥 Authors

BP-Monitor Team
