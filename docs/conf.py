from __future__ import annotations

import os
import sys
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, os.fspath(ROOT))

project = "ClusterEval"
author = "Andreas Tiffeau-Mayer"

try:
    release = package_version("clustereval")
except PackageNotFoundError:
    release = "0.1"

version = release

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": False,
}

napoleon_google_docstring = False
napoleon_numpy_docstring = True

html_theme = "alabaster"
