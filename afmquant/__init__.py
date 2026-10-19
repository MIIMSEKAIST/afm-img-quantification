"""afmquant: color-bar-guided quantification of AFM figures.

The validated core of "Recovering quantitative atomic force microscopy data from
published figures: a ground-truth-validated pipeline".
"""
from .engine import quantify_map
from .render import render_to_published_style
from .metrics import evaluate

__all__ = ["quantify_map", "render_to_published_style", "evaluate"]
__version__ = "0.1.0"
