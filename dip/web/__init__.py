"""
DIP-Learn Streamlit Web Interface

A web-based interface for image processing using Streamlit.
Provides easy-to-use tools for image comparison, pipeline creation,
and interactive image processing.

Usage:
    from dip.web import launch_app
    launch_app()
    
    # Or from command line:
    # dip-web
    # dipstream
"""

from .app import launch_app, run_server
from .components import (
    image_uploader,
    image_comparison,
    pipeline_builder,
    filter_selector,
    parameter_slider,
    before_after_viewer,
    histogram_viewer,
    metrics_display,
)

__all__ = [
    # Main launcher
    'launch_app',
    'run_server',
    
    # Components
    'image_uploader',
    'image_comparison',
    'pipeline_builder',
    'filter_selector',
    'parameter_slider',
    'before_after_viewer',
    'histogram_viewer',
    'metrics_display',
]
