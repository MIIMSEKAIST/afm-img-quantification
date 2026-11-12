"""Map crop for the colour lookup (revision R1, last change; ``FigurePipeline(crop="panel")``).

On real figures the detector's map box often includes a few rows or columns of page background (a white gutter
between two maps, the margin around a panel label), and colour bars are sometimes glued to a map or drawn inside it.
The validity mask of the colour lookup cannot reject these pixels when they lie on the colormap: page white matches
the white end of afmhot, gold and grey colormaps, and the pixels of a colour bar match its own colormap by
construction. They were therefore read as data (page white as the maximum of the scale), which inflated the upper
percentiles of such maps.

The crop is now taken from the detector's box after two steps, both applied after pairing and reading, so that neither
changes which colour bar a map is paired with or how a colour bar is read:

``refine_box``  shrink-only. From each side, lines of the box are removed while they are margin: at least 98 % page
                background, or at least 85 % page with the rest in at most three runs (a panel label). A line crossed
                by several grid lines or by map content stops the walk, so a map's own white rows (the saturated end
                of a heat map, which the eye cannot tell from the page by colour) are kept. At most ``cap`` of the box
                size is removed per side; the box never grows.
``bar_mask``    the boxes of the colour bars the reader read; their pixels are excluded from the valid pixels of
                every map crop (an inset colour bar is masked, not cut out, so the map keeps its area).

The page colour is the median colour of the image border."""
from __future__ import annotations

import numpy as np


def page_mask(image_rgb, tol=30):
    """True where a pixel has the page colour (within ``tol`` in every RGB channel)."""
    im = np.asarray(image_rgb)
    frame = np.concatenate([im[:2].reshape(-1, 3), im[-2:].reshape(-1, 3), im[:, :2].reshape(-1, 3), im[:, -2:].reshape(-1, 3)])
    return np.abs(im.astype(np.int16) - np.median(frame, axis=0)).max(axis=2) <= tol


def _runs(line):
    """Number of runs of True in a 1-D boolean array."""
    if not line.size:
        return 0
    d = np.diff(line.astype(np.int8))
    return int((d == 1).sum() + (1 if line[0] else 0))


def _margin(line):
    f = float(line.mean()) if line.size else 0.0
    return f >= 0.98 or (f >= 0.85 and _runs(~line) <= 3)


def refine_box(page, box, cap=0.15):
    """The map box with its margin lines removed (see the module docstring). Returns integer (x0, y0, x1, y1)."""
    H, W = page.shape
    x0, y0, x1, y1 = (int(round(v)) for v in box)
    x0, y0, x1, y1 = max(0, x0), max(0, y0), min(W, x1), min(H, y1)
    bw, bh = x1 - x0, y1 - y0
    if bw < 4 or bh < 4:
        return x0, y0, x1, y1
    ya, yb = y0 + int(0.1 * bh), max(y0 + int(0.1 * bh) + 1, y1 - int(0.1 * bh))
    xa, xb = x0 + int(0.1 * bw), max(x0 + int(0.1 * bw) + 1, x1 - int(0.1 * bw))
    n = 0
    while n < cap * bw and x0 < x1 - 1 and _margin(page[ya:yb, x0]):
        x0 += 1; n += 1
    n = 0
    while n < cap * bw and x1 - 1 > x0 and _margin(page[ya:yb, x1 - 1]):
        x1 -= 1; n += 1
    n = 0
    while n < cap * bh and y0 < y1 - 1 and _margin(page[y0, xa:xb]):
        y0 += 1; n += 1
    n = 0
    while n < cap * bh and y1 - 1 > y0 and _margin(page[y1 - 1, xa:xb]):
        y1 -= 1; n += 1
    return x0, y0, x1, y1


def bar_mask(shape, bar_boxes):
    """True inside the given colour-bar boxes."""
    H, W = shape[:2]
    m = np.zeros((H, W), bool)
    for box in bar_boxes:
        x0, y0, x1, y1 = (int(round(v)) for v in box)
        m[max(0, y0):max(0, y1), max(0, x0):max(0, x1)] = True
    return m
