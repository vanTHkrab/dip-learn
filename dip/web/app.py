"""
DIP-Learn Streamlit Web Application

Main application module providing the web interface for image processing.
"""

import streamlit as st
import numpy as np
import cv2
import sys
import os
from typing import Optional, Dict, Any, List
from pathlib import Path


def launch_app(
    title: str = "DIP-Learn Image Processing",
    icon: str = "🖼️",
    layout: str = "wide",
    port: int = 8501,
    host: str = "localhost"
) -> None:
    """
    Launch the Streamlit web application.
    
    Args:
        title: Page title
        icon: Page icon
        layout: Page layout ("centered" or "wide")
        port: Server port
        host: Server host
        
    Example:
        >>> from dip.web import launch_app
        >>> launch_app()
    """
    import subprocess
    
    # Get the path to this module
    app_path = Path(__file__).parent / "app.py"
    
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        str(app_path),
        "--server.port", str(port),
        "--server.address", host,
        "--browser.gatherUsageStats", "false"
    ]
    
    print(f"🚀 Starting DIP-Learn Web Interface at http://{host}:{port}")
    subprocess.run(cmd)


def run_server(port: int = 8501, host: str = "localhost") -> None:
    """
    Alias for launch_app with minimal parameters.
    
    Args:
        port: Server port
        host: Server host
    """
    launch_app(port=port, host=host)


# ============================================================
# Main Streamlit App (runs when executed directly)
# ============================================================

def _setup_page():
    """Setup Streamlit page configuration."""
    st.set_page_config(
        page_title="DIP-Learn Image Processing",
        page_icon="🖼️",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    </style>
    """, unsafe_allow_html=True)


def _sidebar_config() -> Dict[str, Any]:
    """Setup sidebar configuration."""
    st.sidebar.title("⚙️ Settings")
    
    config = {}
    
    # Mode selection
    config['mode'] = st.sidebar.selectbox(
        "Mode",
        ["🔄 Quick Process", "🔧 Pipeline Builder", "📊 Compare", "📐 Metrics"],
        key="mode_select"
    )
    
    st.sidebar.markdown("---")
    
    # Filter settings based on mode
    if config['mode'] == "🔄 Quick Process":
        config['filter_category'] = st.sidebar.selectbox(
            "Category",
            ["Enhancement", "Filters", "Transforms", "Threshold", "Morphology"]
        )
        
        category_filters = {
            "Enhancement": ["Brightness", "Contrast", "Sharpen", "Denoise"],
            "Filters": ["Gaussian Blur", "Median Blur", "Bilateral Filter", "Edge Detection"],
            "Transforms": ["Grayscale", "HSV", "Histogram Equalization", "CLAHE"],
            "Threshold": ["Binary", "Otsu", "Adaptive Mean", "Adaptive Gaussian"],
            "Morphology": ["Erosion", "Dilation", "Opening", "Closing", "Gradient"]
        }
        
        config['filter'] = st.sidebar.selectbox(
            "Filter",
            category_filters[config['filter_category']]
        )
        
        # Filter parameters
        config['params'] = _get_filter_params(config['filter'])
        
    elif config['mode'] == "📊 Compare":
        config['compare_mode'] = st.sidebar.radio(
            "Compare Mode",
            ["Side by Side", "Overlay", "Slider", "Difference"]
        )
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 About")
    st.sidebar.info(
        "**DIP-Learn** is a Python library for Digital Image Processing.\n\n"
        "🔗 [GitHub](https://github.com/vanTHkrab/dip-learn)\n\n"
        "📦 `pip install dip-learn`"
    )
    
    return config


def _get_filter_params(filter_name: str) -> Dict[str, Any]:
    """Get filter-specific parameters from sidebar."""
    params = {}
    
    if filter_name == "Brightness":
        params['value'] = st.sidebar.slider("Brightness", -100, 100, 30)
        
    elif filter_name == "Contrast":
        params['alpha'] = st.sidebar.slider("Contrast", 0.5, 3.0, 1.5, 0.1)
        
    elif filter_name in ["Gaussian Blur", "Median Blur"]:
        params['ksize'] = st.sidebar.slider("Kernel Size", 3, 31, 5, 2)
        
    elif filter_name == "Bilateral Filter":
        params['d'] = st.sidebar.slider("Diameter", 5, 15, 9)
        params['sigma_color'] = st.sidebar.slider("Sigma Color", 10, 200, 75)
        params['sigma_space'] = st.sidebar.slider("Sigma Space", 10, 200, 75)
        
    elif filter_name == "Edge Detection":
        params['threshold1'] = st.sidebar.slider("Threshold 1", 0, 300, 100)
        params['threshold2'] = st.sidebar.slider("Threshold 2", 0, 300, 200)
        
    elif filter_name == "Binary":
        params['threshold'] = st.sidebar.slider("Threshold", 0, 255, 127)
        
    elif filter_name in ["Adaptive Mean", "Adaptive Gaussian"]:
        params['block_size'] = st.sidebar.slider("Block Size", 3, 51, 11, 2)
        params['c'] = st.sidebar.slider("C (constant)", -10, 10, 2)
        
    elif filter_name in ["Erosion", "Dilation", "Opening", "Closing", "Gradient"]:
        params['kernel_size'] = st.sidebar.slider("Kernel Size", 3, 15, 5, 2)
        params['iterations'] = st.sidebar.slider("Iterations", 1, 10, 1)
        
    elif filter_name == "CLAHE":
        params['clip_limit'] = st.sidebar.slider("Clip Limit", 1.0, 10.0, 2.0, 0.5)
        params['tile_size'] = st.sidebar.slider("Tile Size", 4, 16, 8)
        
    elif filter_name == "Denoise":
        params['strength'] = st.sidebar.slider("Strength", 1, 20, 10)
        
    return params


def _apply_filter(image: np.ndarray, filter_name: str, params: Dict[str, Any]) -> np.ndarray:
    """Apply the selected filter to the image."""
    result = image.copy()
    
    # Enhancement
    if filter_name == "Brightness":
        result = cv2.convertScaleAbs(image, alpha=1, beta=params.get('value', 30))
        
    elif filter_name == "Contrast":
        result = cv2.convertScaleAbs(image, alpha=params.get('alpha', 1.5), beta=0)
        
    elif filter_name == "Sharpen":
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        result = cv2.filter2D(image, -1, kernel)
        
    elif filter_name == "Denoise":
        strength = params.get('strength', 10)
        result = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
        
    # Filters
    elif filter_name == "Gaussian Blur":
        k = params.get('ksize', 5)
        if k % 2 == 0:
            k += 1
        result = cv2.GaussianBlur(image, (k, k), 0)
        
    elif filter_name == "Median Blur":
        k = params.get('ksize', 5)
        if k % 2 == 0:
            k += 1
        result = cv2.medianBlur(image, k)
        
    elif filter_name == "Bilateral Filter":
        result = cv2.bilateralFilter(
            image,
            params.get('d', 9),
            params.get('sigma_color', 75),
            params.get('sigma_space', 75)
        )
        
    elif filter_name == "Edge Detection":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        result = cv2.Canny(gray, params.get('threshold1', 100), params.get('threshold2', 200))
        
    # Transforms
    elif filter_name == "Grayscale":
        result = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
    elif filter_name == "HSV":
        result = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
    elif filter_name == "Histogram Equalization":
        if len(image.shape) == 3:
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
            result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            result = cv2.equalizeHist(image)
            
    elif filter_name == "CLAHE":
        clahe = cv2.createCLAHE(
            clipLimit=params.get('clip_limit', 2.0),
            tileGridSize=(params.get('tile_size', 8), params.get('tile_size', 8))
        )
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            result = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        else:
            result = clahe.apply(image)
            
    # Threshold
    elif filter_name == "Binary":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, result = cv2.threshold(gray, params.get('threshold', 127), 255, cv2.THRESH_BINARY)
        
    elif filter_name == "Otsu":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        _, result = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
    elif filter_name == "Adaptive Mean":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        bs = params.get('block_size', 11)
        if bs % 2 == 0:
            bs += 1
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,
            bs, params.get('c', 2)
        )
        
    elif filter_name == "Adaptive Gaussian":
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
        bs = params.get('block_size', 11)
        if bs % 2 == 0:
            bs += 1
        result = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
            bs, params.get('c', 2)
        )
        
    # Morphology
    elif filter_name in ["Erosion", "Dilation", "Opening", "Closing", "Gradient"]:
        k = params.get('kernel_size', 5)
        kernel = np.ones((k, k), np.uint8)
        iters = params.get('iterations', 1)
        
        if filter_name == "Erosion":
            result = cv2.erode(image, kernel, iterations=iters)
        elif filter_name == "Dilation":
            result = cv2.dilate(image, kernel, iterations=iters)
        elif filter_name == "Opening":
            result = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)
        elif filter_name == "Closing":
            result = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
        elif filter_name == "Gradient":
            result = cv2.morphologyEx(image, cv2.MORPH_GRADIENT, kernel)
            
    return result


def _quick_process_mode(config: Dict[str, Any]):
    """Quick process mode - single filter application."""
    st.header("🔄 Quick Process")
    
    # Upload image
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="quick_upload"
    )
    
    if uploaded_file is not None:
        # Read image
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            # Apply filter
            processed = _apply_filter(image, config['filter'], config['params'])
            
            # Display
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Original")
                st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Size: {image.shape[1]}x{image.shape[0]}")
                
            with col2:
                st.subheader(f"Processed ({config['filter']})")
                if len(processed.shape) == 2:
                    st.image(processed, use_container_width=True)
                else:
                    st.image(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Size: {processed.shape[1]}x{processed.shape[0] if len(processed.shape) > 1 else 1}")
            
            # Download button
            _, buffer = cv2.imencode('.png', processed)
            st.download_button(
                label="📥 Download Processed Image",
                data=buffer.tobytes(),
                file_name=f"processed_{config['filter'].lower().replace(' ', '_')}.png",
                mime="image/png"
            )


def _pipeline_builder_mode():
    """Pipeline builder mode - chain multiple operations."""
    st.header("🔧 Pipeline Builder")
    
    # Initialize session state
    if 'pipeline' not in st.session_state:
        st.session_state.pipeline = []
    
    # Available operations
    operations = [
        "Brightness", "Contrast", "Sharpen", "Denoise",
        "Gaussian Blur", "Median Blur", "Bilateral Filter",
        "Grayscale", "Histogram Equalization", "CLAHE",
        "Binary Threshold", "Otsu Threshold",
        "Erosion", "Dilation", "Opening", "Closing"
    ]
    
    # Add operation
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        new_op = st.selectbox("Add Operation", operations, key="new_op")
    with col2:
        if st.button("➕ Add", key="add_op"):
            st.session_state.pipeline.append({'name': new_op, 'enabled': True})
            st.rerun()
    with col3:
        if st.button("🗑️ Clear All", key="clear_all"):
            st.session_state.pipeline = []
            st.rerun()
    
    # Display pipeline
    st.markdown("---")
    st.subheader("Current Pipeline")
    
    if not st.session_state.pipeline:
        st.info("No operations added. Add operations above to build your pipeline.")
    else:
        for i, op in enumerate(st.session_state.pipeline):
            col1, col2, col3 = st.columns([3, 1, 1])
            with col1:
                st.write(f"**{i+1}.** {op['name']}")
            with col2:
                op['enabled'] = st.checkbox("On", value=op['enabled'], key=f"enabled_{i}")
            with col3:
                if st.button("🗑️", key=f"del_{i}"):
                    st.session_state.pipeline.pop(i)
                    st.rerun()
    
    st.markdown("---")
    
    # Upload and process
    uploaded_file = st.file_uploader(
        "Upload an image to process",
        type=["png", "jpg", "jpeg", "bmp", "tiff", "webp"],
        key="pipeline_upload"
    )
    
    if uploaded_file is not None and st.session_state.pipeline:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        
        if image is not None:
            result = image.copy()
            intermediates = [("Original", image.copy())]
            
            # Apply pipeline
            for op in st.session_state.pipeline:
                if op['enabled']:
                    result = _apply_filter(result, op['name'].replace(" Threshold", ""), {})
                    intermediates.append((op['name'], result.copy()))
            
            # Show results
            show_intermediate = st.checkbox("Show intermediate results", value=False)
            
            if show_intermediate:
                cols = st.columns(min(len(intermediates), 4))
                for i, (name, img) in enumerate(intermediates):
                    with cols[i % 4]:
                        if len(img.shape) == 2:
                            st.image(img, caption=name, use_container_width=True)
                        else:
                            st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 
                                    caption=name, use_container_width=True)
            else:
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(image, cv2.COLOR_BGR2RGB), 
                            caption="Original", use_container_width=True)
                with col2:
                    if len(result.shape) == 2:
                        st.image(result, caption="Final Result", use_container_width=True)
                    else:
                        st.image(cv2.cvtColor(result, cv2.COLOR_BGR2RGB), 
                                caption="Final Result", use_container_width=True)
            
            # Download
            _, buffer = cv2.imencode('.png', result)
            st.download_button(
                label="📥 Download Result",
                data=buffer.tobytes(),
                file_name="pipeline_result.png",
                mime="image/png"
            )


def _compare_mode(config: Dict[str, Any]):
    """Compare mode - compare two images."""
    st.header("📊 Compare Images")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader("Image 1", type=["png", "jpg", "jpeg"], key="compare1")
    with col2:
        file2 = st.file_uploader("Image 2", type=["png", "jpg", "jpeg"], key="compare2")
    
    if file1 is not None and file2 is not None:
        bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
        bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
        
        img1 = cv2.imdecode(bytes1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(bytes2, cv2.IMREAD_COLOR)
        
        if img1 is not None and img2 is not None:
            # Resize img2 to match img1 if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
            
            compare_mode = config.get('compare_mode', 'Side by Side')
            
            if compare_mode == "Side by Side":
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), 
                            caption="Image 1", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), 
                            caption="Image 2", use_container_width=True)
                            
            elif compare_mode == "Overlay":
                alpha = st.slider("Blend", 0.0, 1.0, 0.5)
                blended = cv2.addWeighted(img1, 1 - alpha, img2, alpha, 0)
                st.image(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB), 
                        caption="Blended", use_container_width=True)
                        
            elif compare_mode == "Slider":
                pos = st.slider("Position", 0, 100, 50)
                h, w = img1.shape[:2]
                split = int(w * pos / 100)
                combined = img2.copy()
                combined[:, :split] = img1[:, :split]
                cv2.line(combined, (split, 0), (split, h), (255, 255, 255), 2)
                st.image(cv2.cvtColor(combined, cv2.COLOR_BGR2RGB), 
                        use_container_width=True)
                        
            elif compare_mode == "Difference":
                diff = cv2.absdiff(img1, img2)
                diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
                diff_color = cv2.applyColorMap(diff_gray, cv2.COLORMAP_JET)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.image(cv2.cvtColor(diff, cv2.COLOR_BGR2RGB), 
                            caption="Absolute Difference", use_container_width=True)
                with col2:
                    st.image(cv2.cvtColor(diff_color, cv2.COLOR_BGR2RGB), 
                            caption="Difference Heatmap", use_container_width=True)


def _metrics_mode():
    """Metrics mode - compute image quality metrics."""
    st.header("📐 Image Quality Metrics")
    
    col1, col2 = st.columns(2)
    
    with col1:
        file1 = st.file_uploader("Reference Image", type=["png", "jpg", "jpeg"], key="ref")
    with col2:
        file2 = st.file_uploader("Test Image", type=["png", "jpg", "jpeg"], key="test")
    
    if file1 is not None and file2 is not None:
        bytes1 = np.asarray(bytearray(file1.read()), dtype=np.uint8)
        bytes2 = np.asarray(bytearray(file2.read()), dtype=np.uint8)
        
        img1 = cv2.imdecode(bytes1, cv2.IMREAD_COLOR)
        img2 = cv2.imdecode(bytes2, cv2.IMREAD_COLOR)
        
        if img1 is not None and img2 is not None:
            # Resize if needed
            if img1.shape != img2.shape:
                img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]))
                st.warning("Test image was resized to match reference image.")
            
            # Show images
            col1, col2 = st.columns(2)
            with col1:
                st.image(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB), 
                        caption="Reference", use_container_width=True)
            with col2:
                st.image(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB), 
                        caption="Test", use_container_width=True)
            
            st.markdown("---")
            st.subheader("Computed Metrics")
            
            # Convert to grayscale for metrics
            gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
            
            # Compute metrics
            mse = np.mean((gray1.astype(float) - gray2.astype(float)) ** 2)
            psnr = 10 * np.log10(255.0 ** 2 / mse) if mse > 0 else float('inf')
            
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("MSE", f"{mse:.2f}")
                st.caption("Mean Squared Error\n(Lower is better)")
                
            with col2:
                st.metric("PSNR", f"{psnr:.2f} dB")
                st.caption("Peak Signal-to-Noise Ratio\n(Higher is better)")
                
            with col3:
                # SSIM
                try:
                    from skimage.metrics import structural_similarity
                    ssim = structural_similarity(gray1, gray2)
                    st.metric("SSIM", f"{ssim:.4f}")
                    st.caption("Structural Similarity\n(1.0 is perfect)")
                except ImportError:
                    st.metric("SSIM", "N/A")
                    st.caption("Install scikit-image")
                    
            with col4:
                # MAE
                mae = np.mean(np.abs(gray1.astype(float) - gray2.astype(float)))
                st.metric("MAE", f"{mae:.2f}")
                st.caption("Mean Absolute Error\n(Lower is better)")


def main():
    """Main application entry point."""
    _setup_page()
    
    # Header
    st.markdown('<h1 class="main-header">🖼️ DIP-Learn</h1>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Digital Image Processing Made Easy</p>',
        unsafe_allow_html=True
    )
    
    # Get configuration from sidebar
    config = _sidebar_config()
    
    # Route to appropriate mode
    if config['mode'] == "🔄 Quick Process":
        _quick_process_mode(config)
    elif config['mode'] == "🔧 Pipeline Builder":
        _pipeline_builder_mode()
    elif config['mode'] == "📊 Compare":
        _compare_mode(config)
    elif config['mode'] == "📐 Metrics":
        _metrics_mode()


# Run when executed directly by Streamlit
if __name__ == "__main__":
    main()
