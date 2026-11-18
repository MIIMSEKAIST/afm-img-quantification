"""Publication-style rendering of a ground-truth matrix.

Used to build the validation benchmark: a known numerical matrix is rendered
with a chosen colormap and degraded with JPEG compression to emulate a figure
as it appears in a published PDF, then handed to the quantification engine.

Defaults reproduce the manuscript benchmark exactly: JPEG quality 85 with the
Pillow default chroma subsampling (4:2:0). The ``fmt`` and ``subsampling``
options were added for the chroma-subsampling test requested in review.

Source: extracted (logic-preserving) from scripts/gold_validation.py.
"""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

try:                                   # matplotlib >= 3.5
    from matplotlib import colormaps as _colormaps

    def get_cmap(name):
        return _colormaps[name]
except ImportError:                    # older matplotlib
    import matplotlib.cm as _cm

    def get_cmap(name):
        return _cm.get_cmap(name)


def normalize_to_unit(matrix, vmin, vmax):
    """``(matrix - vmin) / (vmax - vmin)``, exactly; a constant matrix maps to 0.

    No constant is added to the denominator. A fixed guard such as 1e-12 is not small
    for data stored in SI base units (it is 1 pm for a scan in metres) and would make
    the rendering depend on the unit of the data.
    """
    matrix = np.asarray(matrix, dtype=float)
    span = float(vmax) - float(vmin)
    if not span > 0:
        return np.zeros_like(matrix)
    return (matrix - float(vmin)) / span


def colormap_strip(cmap_name, n=256):
    """Ideal (uncompressed) color-bar strip, ordered vmax -> vmin."""
    cmap = get_cmap(cmap_name)
    return (cmap(np.linspace(1, 0, n))[:, :3] * 255).astype(np.uint8)


def encode_decode(rgb, fmt="jpeg", jpeg_quality=85, subsampling=None):
    """Round-trip an RGB array through an image codec.

    fmt : "jpeg" | "png"
    subsampling : None (Pillow default, 4:2:0) | 0 (4:4:4) | 1 (4:2:2) | 2 (4:2:0)
    """
    img = Image.fromarray(np.ascontiguousarray(rgb))
    buf = io.BytesIO()
    if fmt == "png":
        img.save(buf, format="PNG")
    elif fmt == "jpeg":
        kw = {"quality": int(jpeg_quality)}
        if subsampling is not None:
            kw["subsampling"] = int(subsampling)
        img.save(buf, format="JPEG", **kw)
    else:
        raise ValueError(f"fmt must be 'jpeg' or 'png', got {fmt!r}")
    buf.seek(0)
    return np.array(Image.open(buf).convert("RGB"))


def render_to_published_style(gt_matrix, cmap_name, jpeg_quality=85, fmt="jpeg",
                              subsampling=None, strip_fmt=None):
    """Render a ground-truth matrix into a publication-style figure.

    Parameters
    ----------
    gt_matrix : ndarray (H, W), float
        Ground-truth numerical matrix.
    cmap_name : str
        Matplotlib colormap name (e.g. 'viridis', 'jet', 'hot', 'copper').
    jpeg_quality : int, default 85
        JPEG quality used to emulate publication-typical compression.
    fmt : {"jpeg", "png"}, default "jpeg"
        Codec for the rendered map ("png" = lossless control).
    subsampling : int or None, default None
        JPEG chroma subsampling: None = Pillow default (4:2:0), 0 = 4:4:4.
    strip_fmt : str or None
        Codec for the color-bar strip; defaults to ``fmt``.

    Returns
    -------
    map_rgb_compressed : ndarray (H, W, 3), uint8
        The codec-degraded rendered map.
    strip_compressed : ndarray (256, 3), uint8
        The color-bar strip (vmax -> vmin), degraded with the same settings.
    vmin, vmax : float
        Scale endpoints (full data range of the matrix).
    """
    vmin, vmax = float(np.nanmin(gt_matrix)), float(np.nanmax(gt_matrix))
    norm = normalize_to_unit(gt_matrix, vmin, vmax)

    cmap = get_cmap(cmap_name)
    map_rgb = (cmap(norm)[:, :, :3] * 255).astype(np.uint8)
    map_rgb_compressed = encode_decode(map_rgb, fmt, jpeg_quality, subsampling)

    strip_rgb = colormap_strip(cmap_name, 256)
    strip_compressed = encode_decode(strip_rgb.reshape(-1, 1, 3), strip_fmt or fmt,
                                     jpeg_quality, subsampling).reshape(-1, 3)

    return map_rgb_compressed, strip_compressed, vmin, vmax
