"""Publication-style rendering of a ground-truth matrix.

Used to build the validation benchmark: a known numerical matrix is rendered
with a chosen colormap and degraded with JPEG compression to emulate a figure
as it appears in a published PDF, then handed to the quantification engine.

Source: extracted verbatim (logic-preserving) from scripts/gold_validation.py.
"""
from __future__ import annotations

import io

import numpy as np
import matplotlib.cm as cm
from PIL import Image


def render_to_published_style(gt_matrix, cmap_name, jpeg_quality=85):
    """Render a ground-truth matrix into a publication-style figure.

    Parameters
    ----------
    gt_matrix : ndarray (H, W), float
        Ground-truth numerical matrix.
    cmap_name : str
        Matplotlib colormap name (e.g. 'viridis', 'jet', 'hot', 'copper').
    jpeg_quality : int, default 85
        JPEG quality used to emulate publication-typical compression.

    Returns
    -------
    map_rgb_compressed : ndarray (H, W, 3), uint8
        The JPEG-degraded rendered map.
    strip_compressed : ndarray (256, 3), uint8
        The color-bar strip (vmax -> vmin), degraded with the same JPEG quality.
    vmin, vmax : float
        Scale endpoints (full data range of the matrix).
    """
    vmin, vmax = float(np.nanmin(gt_matrix)), float(np.nanmax(gt_matrix))
    norm = (gt_matrix - vmin) / (vmax - vmin + 1e-12)

    cmap = cm.get_cmap(cmap_name)
    rgba = cmap(norm)
    map_rgb = (rgba[:, :, :3] * 255).astype(np.uint8)

    # JPEG compression of the map
    img = Image.fromarray(map_rgb)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=jpeg_quality)
    buf.seek(0)
    map_rgb_compressed = np.array(Image.open(buf))

    # color-bar strip sampled directly from the colormap (top = vmax)
    strip_positions = np.linspace(1, 0, 256)
    strip_rgb = (cmap(strip_positions)[:, :3] * 255).astype(np.uint8)
    strip_img = Image.fromarray(strip_rgb.reshape(-1, 1, 3))
    buf2 = io.BytesIO()
    strip_img.save(buf2, format="JPEG", quality=jpeg_quality)
    buf2.seek(0)
    strip_compressed = np.array(Image.open(buf2)).reshape(-1, 3)

    return map_rgb_compressed, strip_compressed, vmin, vmax
