"""Last change of revision R1: map crop trimmed to the panel border with colour-bar pixels excluded (afmquant.crop),
and end labels taken verbatim (afmquant.reader2.verbatim_ends)."""
import numpy as np

from afmquant.crop import bar_mask, page_mask, refine_box
from afmquant.reader2 import verbatim_ends


def _figure(h=200, w=300):
    im = np.full((h, w, 3), 255, np.uint8)                       # white page
    return im


def test_white_gutter_and_label_margin_are_removed():
    im = _figure()
    im[50:150, 60:160] = (180, 60, 20)                            # the map
    im[40:44, 62:70] = 0                                          # a panel label in the margin above it
    page = page_mask(im)
    x0, y0, x1, y1 = refine_box(page, (52, 36, 168, 150))         # box with 8 px gutter left/right, label margin on top
    assert (x0, y0, x1, y1) == (60, 50, 160, 150)


def test_white_data_rows_crossed_by_grid_lines_are_kept():
    im = _figure()
    im[50:150, 60:160] = (180, 60, 20)
    im[140:150, 60:160] = 255                                     # saturated (white) rows of the map itself ...
    im[50:150, 60:160:10] = 0                                     # ... crossed by the map's grid lines
    page = page_mask(im)
    assert refine_box(page, (60, 50, 160, 150))[3] == 150


def test_box_never_grows_and_map_content_is_not_cut():
    im = _figure()
    im[50:150, 60:160] = (180, 60, 20)
    page = page_mask(im)
    assert refine_box(page, (70, 60, 150, 140)) == (70, 60, 150, 140)


def test_inset_colour_bar_is_masked_not_cut():
    im = _figure()
    im[50:150, 60:160] = (180, 60, 20)
    bar = (70, 60, 80, 120)                                       # a colour bar drawn inside the map
    m = bar_mask(im.shape, [bar])
    assert m[60:120, 70:80].all() and m.sum() == 10 * 60
    assert refine_box(page_mask(im), (60, 50, 160, 150)) == (60, 50, 160, 150)


def _fit(start, end, **kw):
    return {"ok": True, "value_at_start": start, "value_at_end": end, "vmin": min(start, end), "vmax": max(start, end),
            "start_is_max": start > end, "offset_scale": 1.0, **kw}


def test_end_labels_replace_the_fitted_ends():
    anchors = [(0.0, 354.88, 0.9, 10.0, "end"), (1.0, 0.0, 0.9, 300.0, "end")]
    f = verbatim_ends(_fit(358.3, -3.47), anchors)
    assert (f["vmin"], f["vmax"]) == (0.0, 354.88) and f["start_is_max"]


def test_unlabelled_end_keeps_the_fitted_value():
    anchors = [(0.1, 500.0, 0.9, 50.0, "col"), (0.5, 300.0, 0.9, 150.0, "col"), (1.0, 0.0, 0.9, 300.0, "col")]
    f = verbatim_ends(_fit(552.99, -2.67), anchors)
    assert (f["vmin"], f["vmax"]) == (0.0, 552.99)


def test_end_label_far_off_the_line_is_not_used():
    anchors = [(0.0, 900.0, 0.9, 10.0, "col"), (0.5, 300.0, 0.9, 150.0, "col")]   # misread end, rejected by the fit
    f = verbatim_ends(_fit(552.99, -2.67), anchors)
    assert (f["vmin"], f["vmax"]) == (-2.67, 552.99)


def test_repaired_or_failed_fits_are_left_alone():
    anchors = [(0.0, 5.0, 0.9, 0.0, "col"), (1.0, -5.0, 0.9, 1.0, "col")]
    assert verbatim_ends(_fit(5.1, -5.2, sign_repaired=True), anchors)["vmax"] == 5.1
    assert verbatim_ends({"ok": False}, anchors) == {"ok": False}
