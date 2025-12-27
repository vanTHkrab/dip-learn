"""
Conftest for pytest

Configuration and fixtures shared across all tests
"""

import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Import the dip package as dip
dip_learn_path = project_root / "dip"
if str(dip_learn_path) not in sys.path:
    sys.path.insert(0, str(dip_learn_path.parent))

# Create dip module that maps to dip folder
import importlib.util
import types

# Check if dip module already exists
if "dip" not in sys.modules:
    # Create a module that acts as an alias
    spec = importlib.util.spec_from_file_location(
        "dip",
        dip_learn_path / "__init__.py",
        submodule_search_locations=[str(dip_learn_path)]
    )
    if spec and spec.loader:
        dip_learn = importlib.util.module_from_spec(spec)
        sys.modules["dip"] = dip_learn
        
        # Also need to add submodules paths
        for submodule in ["annotated", "core", "transforms", "filters",
                          "enhancement", "presets", "utils", "cli"]:
            submodule_path = dip_learn_path / submodule
            if submodule_path.exists():
                sub_spec = importlib.util.spec_from_file_location(
                    f"dip.{submodule}",
                    submodule_path / "__init__.py",
                    submodule_search_locations=[str(submodule_path)]
                )
                if sub_spec and sub_spec.loader:
                    submod = importlib.util.module_from_spec(sub_spec)
                    sys.modules[f"dip.{submodule}"] = submod
