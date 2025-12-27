"""
Version information for dip-learn package.
"""

__version__ = '0.0.1b1'
__author__ = 'BP-Monitor Team'
__description__ = 'A comprehensive digital image processing library for Python'
__license__ = 'MIT'


def get_version():
    """Get the current version string."""
    return __version__


def get_author():
    """Get the author information."""
    return __author__


def get_description():
    """Get the package description."""
    return __description__


if __name__ == "__main__":
    print(f"dip-learn - Digital Image Processing Library")
    print(f"Version: {get_version()}")
    print(f"Author: {get_author()}")
    print(f"Description: {get_description()}")