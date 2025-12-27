#!/usr/bin/env python3
"""
CLI Main Entry Point

Command line interface สำหรับ image processing
"""

import sys
import argparse
from pathlib import Path


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Image Processing CLI for BP Monitor OCR",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single image
  python -m images_process process input.png -o output.png --preset ocr_basic
  
  # Process with custom pipeline
  python -m images_process process input.png -o output.png --steps grayscale blur otsu
  
  # Show image info
  python -m images_process info input.png
  
  # Compare images
  python -m images_process compare img1.png img2.png
"""
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process image')
    process_parser.add_argument('input', type=str, help='Input image path')
    process_parser.add_argument('-o', '--output', type=str, help='Output image path')
    process_parser.add_argument('--preset', type=str, 
                               choices=['ocr_basic', 'ocr_advanced', 'seven_segment',
                                       'denoise_light', 'denoise_heavy', 'enhance_contrast',
                                       'edge_detection', 'binarization', 'sharpen'],
                               help='Preset pipeline to use')
    process_parser.add_argument('--steps', nargs='+', help='Custom processing steps')
    process_parser.add_argument('--show', action='store_true', help='Show result')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show image info')
    info_parser.add_argument('input', type=str, help='Input image path')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare images')
    compare_parser.add_argument('images', nargs='+', help='Images to compare')
    
    args = parser.parse_args()
    
    if args.command == 'process':
        run_process(args)
    elif args.command == 'info':
        run_info(args)
    elif args.command == 'compare':
        run_compare(args)
    else:
        parser.print_help()


def run_process(args):
    """Run process command."""
    from ..utils.io import load_image, save_image
    from ..core.pipeline import ImagePipeline
    from ..presets.pipelines import PresetPipelines
    
    # Load image
    image = load_image(args.input)
    if image is None:
        print(f"Error: Cannot load image from {args.input}")
        sys.exit(1)
    
    print(f"Loaded: {args.input} ({image.shape})")
    
    # Get pipeline
    if args.preset:
        preset_func = getattr(PresetPipelines, args.preset, None)
        if preset_func is None:
            print(f"Error: Unknown preset {args.preset}")
            sys.exit(1)
        pipeline = preset_func()
        print(f"Using preset: {args.preset}")
    elif args.steps:
        pipeline = ImagePipeline([(step, {}) for step in args.steps])
        print(f"Using steps: {args.steps}")
    else:
        pipeline = PresetPipelines.ocr_basic()
        print("Using default preset: ocr_basic")
    
    # Process
    result = pipeline.process_with_history(image)
    print(f"Processed in {result.execution_time_ms:.1f} ms")
    print(f"Steps: {result.steps_executed}")
    
    # Save
    if args.output:
        save_image(result.final_image, args.output)
        print(f"Saved to: {args.output}")
    
    # Show
    if args.show:
        from ..utils.visualization import compare_images
        compare_images({
            'Original': image,
            'Result': result.final_image
        })


def run_info(args):
    """Run info command."""
    from ..utils.io import load_image
    from ..utils.metrics import get_image_stats
    
    image = load_image(args.input)
    if image is None:
        print(f"Error: Cannot load image from {args.input}")
        sys.exit(1)
    
    stats = get_image_stats(image)
    
    print(f"\nImage: {args.input}")
    print("-" * 40)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


def run_compare(args):
    """Run compare command."""
    from ..utils.io import load_image
    from ..utils.visualization import compare_images
    
    images = {}
    for path in args.images:
        img = load_image(path)
        if img is not None:
            name = Path(path).stem
            images[name] = img
    
    if len(images) < 2:
        print("Error: Need at least 2 images to compare")
        sys.exit(1)
    
    compare_images(images)


if __name__ == '__main__':
    main()
