"""
CLI entry point for DIP-Learn Streamlit Web Interface.

This module provides command-line interface to launch the web app.
"""

import argparse
import sys


def main():
    """Main CLI entry point for dipstream command."""
    parser = argparse.ArgumentParser(
        description="Launch DIP-Learn Web Interface",
        prog="dipstream"
    )
    
    parser.add_argument(
        "-p", "--port",
        type=int,
        default=8501,
        help="Server port (default: 8501)"
    )
    
    parser.add_argument(
        "-H", "--host",
        type=str,
        default="localhost",
        help="Server host (default: localhost)"
    )
    
    parser.add_argument(
        "--public",
        action="store_true",
        help="Make server publicly accessible (0.0.0.0)"
    )
    
    parser.add_argument(
        "-v", "--version",
        action="store_true",
        help="Show version and exit"
    )
    
    args = parser.parse_args()
    
    if args.version:
        try:
            from dip import __version__
            print(f"dip-learn version {__version__}")
        except ImportError:
            print("dip-learn (version unknown)")
        return
    
    # Import and launch
    from dip.web import launch_app
    
    host = "0.0.0.0" if args.public else args.host
    
    print(f"""
╔═══════════════════════════════════════════════════════════╗
║                   🖼️  DIP-Learn Web UI                    ║
╠═══════════════════════════════════════════════════════════╣
║  Starting Streamlit server...                             ║
║                                                           ║
║  Local URL: http://{args.host}:{args.port}                         
║                                                           ║
║  Press Ctrl+C to stop                                     ║
╚═══════════════════════════════════════════════════════════╝
    """)
    
    launch_app(port=args.port, host=host)


if __name__ == "__main__":
    main()
