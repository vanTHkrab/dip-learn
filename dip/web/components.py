"""
Reusable Streamlit Components for DIP-Learn

Provides pre-built UI components for image processing applications.
"""

import streamlit as st
import numpy as np
import cv2
from typing import Optional, List, Dict, Any, Callable, Tuple, Union
from io import BytesIO
import base64


def image_uploader(
    label: str = "Upload Image",
    key: Optional[str] = None,
    accept_multiple: bool = False,
    max_size_mb: float = 10.0,
    allowed_types: List[str] = ["png", "jpg", "jpeg", "bmp", "tiff", "webp"]
) -> Optional[np.ndarray]:
    """
    Upload image component with preview.
    
    Args:
        label: Upload button label
        key: Unique key for the uploader
        accept_multiple: Whether to accept multiple files
        max_size_mb: Maximum file size in MB
        allowed_types: List of allowed file extensions
        
    Returns:
        Uploaded image as numpy array (BGR format) or None
    """
    uploaded_file = st.file_uploader(
        label,
        type=allowed_types,
        accept_multiple_files=accept_multiple,
        key=key
    )
    
    if uploaded_file is not None:
        # Read file bytes
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        
        # Check file size
        if len(file_bytes) > max_size_mb * 1024 * 1024:
            st.error(f"File size exceeds {max_size_mb}MB limit")
            return None
            
        # Decode image
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is None:
            st.error("Failed to decode image")
            return None
            
        return image
        
    return None


def image_comparison(
    image1: np.ndarray,
    image2: np.ndarray,
    label1: str = "Original",
    label2: str = "Processed",
    width: Optional[int] = None,
    mode: str = "side-by-side"
) -> None:
    """
    Display two images for comparison.
    
    Args:
        image1: First image (BGR format)
        image2: Second image (BGR format)
        label1: Label for first image
        label2: Label for second image
        width: Display width (None for auto)
        mode: Display mode - "side-by-side", "overlay", "slider"
    """
    # Convert BGR to RGB for display
    img1_rgb = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
    
    if mode == "side-by-side":
        col1, col2 = st.columns(2)
        with col1:
            st.image(img1_rgb, caption=label1, use_container_width=True)
        with col2:
            st.image(img2_rgb, caption=label2, use_container_width=True)
            
    elif mode == "overlay":
        alpha = st.slider("Blend Alpha", 0.0, 1.0, 0.5, key="overlay_alpha")
        
        # Resize images to match if needed
        if image1.shape != image2.shape:
            img2_resized = cv2.resize(img2_rgb, (img1_rgb.shape[1], img1_rgb.shape[0]))
        else:
            img2_resized = img2_rgb
            
        blended = cv2.addWeighted(img1_rgb, 1 - alpha, img2_resized, alpha, 0)
        st.image(blended, caption=f"{label1} ↔ {label2}", use_container_width=True)
        
    elif mode == "slider":
        _render_slider_comparison(img1_rgb, img2_rgb, label1, label2)


def _render_slider_comparison(
    img1: np.ndarray,
    img2: np.ndarray,
    label1: str,
    label2: str
) -> None:
    """Render slider-based image comparison using HTML/CSS."""
    # Resize images to match
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Convert to base64
    _, buffer1 = cv2.imencode('.png', cv2.cvtColor(img1, cv2.COLOR_RGB2BGR))
    _, buffer2 = cv2.imencode('.png', cv2.cvtColor(img2, cv2.COLOR_RGB2BGR))
    
    img1_b64 = base64.b64encode(buffer1).decode()
    img2_b64 = base64.b64encode(buffer2).decode()
    
    slider_pos = st.slider("Compare Position", 0, 100, 50, key="compare_slider")
    
    # Create combined image
    h, w = img1.shape[:2]
    split_x = int(w * slider_pos / 100)
    
    combined = img2.copy()
    combined[:, :split_x] = img1[:, :split_x]
    
    # Draw divider line
    cv2.line(combined, (split_x, 0), (split_x, h), (255, 255, 255), 2)
    
    st.image(combined, caption=f"{label1} | {label2}", use_container_width=True)


def before_after_viewer(
    original: np.ndarray,
    processed: np.ndarray,
    show_diff: bool = True
) -> None:
    """
    Display before/after comparison with optional difference view.
    
    Args:
        original: Original image
        processed: Processed image
        show_diff: Whether to show difference image
    """
    tabs = ["Side by Side", "Overlay", "Slider"]
    if show_diff:
        tabs.append("Difference")
        
    tab_objects = st.tabs(tabs)
    
    with tab_objects[0]:
        image_comparison(original, processed, mode="side-by-side")
        
    with tab_objects[1]:
        image_comparison(original, processed, mode="overlay")
        
    with tab_objects[2]:
        image_comparison(original, processed, mode="slider")
        
    if show_diff and len(tab_objects) > 3:
        with tab_objects[3]:
            _show_difference(original, processed)


def _show_difference(img1: np.ndarray, img2: np.ndarray) -> None:
    """Show difference between two images."""
    # Ensure same size
    if img1.shape != img2.shape:
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
    
    # Compute absolute difference
    diff = cv2.absdiff(img1, img2)
    
    # Convert to grayscale for visualization
    diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
    
    # Apply colormap
    diff_colored = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
    diff_rgb = cv2.cvtColor(diff_colored, cv2.COLOR_BGR2RGB)
    
    col1, col2 = st.columns(2)
    with col1:
        st.image(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB), 
                 caption="Absolute Difference", use_container_width=True)
    with col2:
        st.image(diff_rgb, caption="Difference Heatmap", use_container_width=True)


def histogram_viewer(
    image: np.ndarray,
    show_rgb: bool = True,
    show_cumulative: bool = False
) -> None:
    """
    Display image histogram.
    
    Args:
        image: Input image
        show_rgb: Show RGB channels separately
        show_cumulative: Show cumulative histogram
    """
    try:
        import matplotlib.pyplot as plt
        
        fig, axes = plt.subplots(1, 2 if show_cumulative else 1, figsize=(12, 4))
        if not show_cumulative:
            axes = [axes]
        
        colors = ('b', 'g', 'r') if len(image.shape) == 3 else ('gray',)
        channel_names = ('Blue', 'Green', 'Red') if len(image.shape) == 3 else ('Gray',)
        
        # Normal histogram
        ax = axes[0]
        for i, (color, name) in enumerate(zip(colors, channel_names)):
            if len(image.shape) == 3:
                hist = cv2.calcHist([image], [i], None, [256], [0, 256])
            else:
                hist = cv2.calcHist([image], [0], None, [256], [0, 256])
            ax.plot(hist, color=color, label=name)
        ax.set_title('Histogram')
        ax.set_xlabel('Pixel Value')
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.set_xlim([0, 256])
        
        # Cumulative histogram
        if show_cumulative:
            ax = axes[1]
            for i, (color, name) in enumerate(zip(colors, channel_names)):
                if len(image.shape) == 3:
                    hist = cv2.calcHist([image], [i], None, [256], [0, 256])
                else:
                    hist = cv2.calcHist([image], [0], None, [256], [0, 256])
                cumhist = np.cumsum(hist)
                ax.plot(cumhist / cumhist[-1], color=color, label=name)
            ax.set_title('Cumulative Histogram')
            ax.set_xlabel('Pixel Value')
            ax.set_ylabel('Cumulative Frequency')
            ax.legend()
            ax.set_xlim([0, 256])
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
        
    except ImportError:
        st.warning("Matplotlib not installed. Cannot display histogram.")


def metrics_display(
    original: np.ndarray,
    processed: np.ndarray,
    metrics: List[str] = ["PSNR", "SSIM", "MSE"]
) -> Dict[str, float]:
    """
    Display image quality metrics.
    
    Args:
        original: Original image
        processed: Processed image
        metrics: List of metrics to compute
        
    Returns:
        Dictionary of computed metrics
    """
    results = {}
    
    # Ensure same size
    if original.shape != processed.shape:
        processed = cv2.resize(processed, (original.shape[1], original.shape[0]))
    
    # Convert to grayscale for metrics
    if len(original.shape) == 3:
        orig_gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        proc_gray = cv2.cvtColor(processed, cv2.COLOR_BGR2GRAY)
    else:
        orig_gray = original
        proc_gray = processed
    
    cols = st.columns(len(metrics))
    
    for i, metric in enumerate(metrics):
        with cols[i]:
            if metric.upper() == "MSE":
                value = np.mean((orig_gray.astype(float) - proc_gray.astype(float)) ** 2)
                st.metric("MSE", f"{value:.2f}")
                results["MSE"] = value
                
            elif metric.upper() == "PSNR":
                mse = np.mean((orig_gray.astype(float) - proc_gray.astype(float)) ** 2)
                if mse == 0:
                    value = float('inf')
                else:
                    value = 10 * np.log10(255.0 ** 2 / mse)
                st.metric("PSNR", f"{value:.2f} dB")
                results["PSNR"] = value
                
            elif metric.upper() == "SSIM":
                try:
                    from skimage.metrics import structural_similarity
                    value = structural_similarity(orig_gray, proc_gray)
                    st.metric("SSIM", f"{value:.4f}")
                    results["SSIM"] = value
                except ImportError:
                    st.metric("SSIM", "N/A")
                    st.caption("Install scikit-image")
                    
    return results


def filter_selector(
    available_filters: Optional[Dict[str, Callable]] = None,
    key: str = "filter_select"
) -> Tuple[Optional[str], Optional[Callable]]:
    """
    Filter selection component.
    
    Args:
        available_filters: Dict of filter names to functions
        key: Unique key for the selectbox
        
    Returns:
        Tuple of (filter_name, filter_function)
    """
    if available_filters is None:
        available_filters = _get_default_filters()
    
    filter_categories = {
        "Enhancement": ["Brightness", "Contrast", "Denoise", "Sharpen"],
        "Filters": ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Edge Detection"],
        "Transforms": ["Grayscale", "HSV", "Histogram Equalization", "CLAHE"],
        "Threshold": ["Binary", "Otsu", "Adaptive"],
        "Morphology": ["Erosion", "Dilation", "Opening", "Closing"]
    }
    
    category = st.selectbox("Category", list(filter_categories.keys()), key=f"{key}_cat")
    filter_name = st.selectbox("Filter", filter_categories[category], key=f"{key}_filter")
    
    if filter_name in available_filters:
        return filter_name, available_filters[filter_name]
    return filter_name, None


def _get_default_filters() -> Dict[str, Callable]:
    """Get default filter functions."""
    filters = {}
    
    # Enhancement
    filters["Brightness"] = lambda img, val=30: cv2.convertScaleAbs(img, alpha=1, beta=val)
    filters["Contrast"] = lambda img, val=1.5: cv2.convertScaleAbs(img, alpha=val, beta=0)
    filters["Sharpen"] = lambda img: cv2.filter2D(img, -1, 
        np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]]))
    
    # Filters
    filters["Gaussian Blur"] = lambda img, k=5: cv2.GaussianBlur(img, (k, k), 0)
    filters["Median Blur"] = lambda img, k=5: cv2.medianBlur(img, k)
    filters["Bilateral Filter"] = lambda img: cv2.bilateralFilter(img, 9, 75, 75)
    filters["Edge Detection"] = lambda img: cv2.Canny(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img, 100, 200)
    
    # Transforms
    filters["Grayscale"] = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    filters["HSV"] = lambda img: cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    filters["Histogram Equalization"] = lambda img: cv2.equalizeHist(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img)
    
    # Threshold
    filters["Binary"] = lambda img, th=127: cv2.threshold(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img,
        th, 255, cv2.THRESH_BINARY)[1]
    filters["Otsu"] = lambda img: cv2.threshold(
        cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img,
        0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    
    # Morphology
    kernel = np.ones((5, 5), np.uint8)
    filters["Erosion"] = lambda img: cv2.erode(img, kernel, iterations=1)
    filters["Dilation"] = lambda img: cv2.dilate(img, kernel, iterations=1)
    filters["Opening"] = lambda img: cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
    filters["Closing"] = lambda img: cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
    
    return filters


def parameter_slider(
    param_name: str,
    min_val: float,
    max_val: float,
    default: float,
    step: Optional[float] = None,
    key: Optional[str] = None,
    help_text: Optional[str] = None
) -> float:
    """
    Create a parameter slider.
    
    Args:
        param_name: Display name for the parameter
        min_val: Minimum value
        max_val: Maximum value
        default: Default value
        step: Step size
        key: Unique key
        help_text: Help text
        
    Returns:
        Selected value
    """
    return st.slider(
        param_name,
        min_value=min_val,
        max_value=max_val,
        value=default,
        step=step,
        key=key,
        help=help_text
    )


def pipeline_builder(
    available_operations: Optional[Dict[str, Callable]] = None
) -> List[Dict[str, Any]]:
    """
    Interactive pipeline builder component.
    
    Args:
        available_operations: Dict of operation names to functions
        
    Returns:
        List of pipeline steps with parameters
    """
    if available_operations is None:
        available_operations = _get_default_filters()
    
    st.subheader("🔧 Pipeline Builder")
    
    # Initialize session state
    if 'pipeline_steps' not in st.session_state:
        st.session_state.pipeline_steps = []
    
    # Add step
    col1, col2 = st.columns([3, 1])
    with col1:
        operation = st.selectbox(
            "Add Operation",
            list(available_operations.keys()),
            key="pipeline_new_op"
        )
    with col2:
        if st.button("➕ Add", key="pipeline_add_btn"):
            st.session_state.pipeline_steps.append({
                'name': operation,
                'params': {},
                'enabled': True
            })
            st.rerun()
    
    # Display current pipeline
    st.markdown("---")
    st.markdown("**Current Pipeline:**")
    
    if not st.session_state.pipeline_steps:
        st.info("No operations added yet. Add operations above.")
    else:
        for i, step in enumerate(st.session_state.pipeline_steps):
            with st.container():
                col1, col2, col3, col4 = st.columns([0.5, 2, 0.5, 0.5])
                
                with col1:
                    st.write(f"**{i+1}.**")
                    
                with col2:
                    st.write(step['name'])
                    step['enabled'] = st.checkbox(
                        "Enabled", 
                        value=step['enabled'], 
                        key=f"step_enabled_{i}"
                    )
                    
                with col3:
                    if i > 0:
                        if st.button("⬆️", key=f"move_up_{i}"):
                            steps = st.session_state.pipeline_steps
                            steps[i], steps[i-1] = steps[i-1], steps[i]
                            st.rerun()
                            
                with col4:
                    if st.button("🗑️", key=f"delete_{i}"):
                        st.session_state.pipeline_steps.pop(i)
                        st.rerun()
                        
                st.markdown("---")
    
    # Clear all
    if st.session_state.pipeline_steps:
        if st.button("🗑️ Clear All", key="clear_pipeline"):
            st.session_state.pipeline_steps = []
            st.rerun()
    
    return st.session_state.pipeline_steps


def execute_pipeline(
    image: np.ndarray,
    pipeline: List[Dict[str, Any]],
    operations: Dict[str, Callable],
    show_intermediate: bool = False
) -> Tuple[np.ndarray, List[np.ndarray]]:
    """
    Execute a pipeline on an image.
    
    Args:
        image: Input image
        pipeline: List of pipeline steps
        operations: Dict of operation names to functions
        show_intermediate: Whether to return intermediate results
        
    Returns:
        Tuple of (final_image, intermediate_images)
    """
    result = image.copy()
    intermediates = [image.copy()]
    
    for step in pipeline:
        if not step.get('enabled', True):
            continue
            
        op_name = step['name']
        if op_name in operations:
            try:
                result = operations[op_name](result)
                if show_intermediate:
                    intermediates.append(result.copy())
            except Exception as e:
                st.error(f"Error in {op_name}: {str(e)}")
                
    return result, intermediates


def download_button(
    image: np.ndarray,
    filename: str = "processed_image.png",
    label: str = "📥 Download Image"
) -> None:
    """
    Create a download button for an image.
    
    Args:
        image: Image to download
        filename: Download filename
        label: Button label
    """
    # Encode image
    _, buffer = cv2.imencode('.png', image)
    
    st.download_button(
        label=label,
        data=buffer.tobytes(),
        file_name=filename,
        mime="image/png"
    )


def gallery_view(
    images: List[np.ndarray],
    labels: Optional[List[str]] = None,
    columns: int = 3
) -> None:
    """
    Display images in a gallery grid.
    
    Args:
        images: List of images
        labels: Optional labels for each image
        columns: Number of columns
    """
    if labels is None:
        labels = [f"Image {i+1}" for i in range(len(images))]
    
    cols = st.columns(columns)
    
    for i, (img, label) in enumerate(zip(images, labels)):
        with cols[i % columns]:
            if len(img.shape) == 3:
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            else:
                img_rgb = img
            st.image(img_rgb, caption=label, use_container_width=True)
