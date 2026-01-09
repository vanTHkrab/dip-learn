# Changelog

All notable changes to DIP-Learn will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Web interface module (`dip.web`) with Streamlit support
  - `launch_app()` and `run_server()` functions
  - CLI commands: `dipstream` and `dip-web`
  - Quick Process mode for single filter application
  - Pipeline Builder for chaining operations
  - Compare mode for side-by-side image comparison
  - Metrics mode for image quality assessment
- Reusable web components (`dip.web.components`)
  - `image_uploader`, `image_comparison`, `before_after_viewer`
  - `histogram_viewer`, `metrics_display`, `pipeline_builder`
  - `filter_selector`, `parameter_slider`, `download_button`, `gallery_view`

## [0.0.1b1] - 2024-12-28

### Added
- Initial beta release
- **Core Module**
  - `ImageOps` static class for common operations
  - `ImagePipeline` for chaining operations
  - `PipelineStep` and `PipelineResult` data classes
  
- **Transforms**
  - Color: `to_grayscale`, `to_bgr`, `to_rgb`, `to_hsv`, `to_lab`
  - Threshold: `binary_threshold`, `otsu_threshold`, `adaptive_threshold_*`
  - Morphology: `erode`, `dilate`, `morph_open`, `morph_close`, `top_hat`, `black_hat`
  - Geometric: `resize`, `rotate`, `flip`, `crop`, `pad`, `translate`
  - Histogram: `histogram_equalization`, `clahe`, `contrast_stretch`, `normalize`

- **Filters**
  - Smoothing: `gaussian_blur`, `median_blur`, `bilateral_filter`, `box_blur`
  - Sharpening: `unsharp_mask`, `laplacian_sharpen`, `kernel_sharpen`
  - Edge Detection: `canny_edge`, `sobel_edge`, `laplacian_edge`, `prewitt_edge`

- **Enhancement**
  - `adjust_brightness`, `adjust_contrast`, `gamma_correction`
  - `non_local_means_denoising`, `denoise_bilateral`, `anisotropic_diffusion`

- **Annotation Module**
  - Basic shapes: `draw_rectangle`, `draw_circle`, `draw_line`, `draw_ellipse`
  - Text: `draw_text`, `draw_text_with_background`, `draw_multiline_text`
  - Utilities: `draw_bounding_box`, `draw_contours`, `draw_keypoints`, `draw_grid`
  - Overlay: `overlay_image`, `add_alpha_channel`, `create_mask_overlay`

- **Augmentation Module**
  - Geometric: `random_flip`, `random_rotation`, `random_scale`, `random_crop`
  - Color: `random_brightness`, `random_contrast`, `random_saturation`, `random_hue`
  - Noise: `random_gaussian_noise`, `random_salt_pepper_noise`
  - Blur: `random_gaussian_blur`, `random_motion_blur`
  - Cutout: `random_cutout`, `random_grid_mask`
  - Advanced: `mixup`, `cutmix`, `AugmentationPipeline`

- **Presets**
  - `PresetPipelines` for OCR, document scanning, denoising

- **Utils**
  - I/O: `load_image`, `save_image`, `load_images_from_dir`
  - Visualization: `compare_images`, `show_image`, `show_histogram`
  - Metrics: `calculate_psnr`, `calculate_ssim`, `variance_of_laplacian`

- **Scikit-image Integration** (Optional)
  - Advanced thresholding: `threshold_sauvola`, `threshold_niblack`, `threshold_multiotsu`
  - Morphology: `skeletonize_sk`, `medial_axis_transform`
  - Denoising: `denoise_wavelet`, `denoise_tv_chambolle`
  - Feature detection: `detect_blob_dog`, `detect_corners_harris`

- **CLI**
  - `dip-learn` command for command-line processing

### Dependencies
- Required: `numpy>=2.2.6`, `opencv-python>=4.5.0`
- Optional: `scikit-image>=0.26.0`, `streamlit>=1.28.0`, `matplotlib>=3.7.0`

[Unreleased]: https://github.com/vanTHkrab/dip-learn/compare/v0.0.1b1...HEAD
[0.0.1b1]: https://github.com/vanTHkrab/dip-learn/releases/tag/v0.0.1b1
