"""Strip-geometry rules of the label-column reader (afmquant.reader2) on synthetic crops (no OCR needed)."""
import numpy as np

from afmquant.reader2 import recover_light_end


def _bar_crop(frame=True, top_white_rows=5):
    """White page 60x140 with a vertical 8-px bar (x 20..27, y 20..119): black at the bottom, white at the top,
    optionally a 1-px dark frame around it. The top ``top_white_rows`` rows are pure white (as in 'hot')."""
    crop = np.full((140, 60, 3), 255, np.uint8)
    L = 100
    for i in range(L):                      # i = 0 at the top (white) .. 99 at the bottom (black)
        v = 255 if i < top_white_rows else int(round(255 * (1 - (i - top_white_rows) / (L - top_white_rows))))
        crop[20 + i, 20:28] = (v, v, v)
    if frame:
        crop[19, 19:29] = (40, 40, 40)
        crop[120, 19:29] = (40, 40, 40)
        crop[19:121, 19] = (40, 40, 40)
        crop[19:121, 28] = (40, 40, 40)
    return crop


def test_light_end_extends_to_frame_even_when_detector_box_is_tight():
    crop = _bar_crop(frame=True, top_white_rows=5)
    core = (20, 25, 28, 120)                # the white top rows were trimmed as page background
    det_ac = (23, 118)                      # detector box ends inside the trimmed region (tight box)
    bbox, ext_lo, ext_hi = recover_light_end(crop, core, True, det_ac)
    assert ext_lo == 5 and bbox[1] == 20, (bbox, ext_lo)
    assert ext_hi == 0                      # the dark bottom end is not a light end


def test_light_end_without_frame_stays_inside_detector_box():
    crop = _bar_crop(frame=False, top_white_rows=5)
    core = (20, 25, 28, 120)
    bbox, ext_lo, ext_hi = recover_light_end(crop, core, True, (23, 118))
    assert ext_lo == 0 and bbox[1] == 25    # nothing marks the end: the page is as white as the bar
    bbox, ext_lo, ext_hi = recover_light_end(crop, core, True, (14, 125))
    assert 0 < ext_lo <= 11                 # with room in the detector box the end is recovered up to it


def test_frame_detection_ignores_text_like_rows():
    crop = _bar_crop(frame=True, top_white_rows=5)
    crop[16:18, 21:23] = (0, 0, 0)          # a dark glyph fragment above the frame must not matter
    crop[22, 24:26] = (30, 30, 30)          # dark, non-uniform pixels inside the white rows are not a frame line
    core = (20, 25, 28, 120)
    bbox, ext_lo, _ = recover_light_end(crop, core, True, (23, 118))
    assert ext_lo == 5 and bbox[1] == 20, (bbox, ext_lo)   # the frame still bounds the bar
    crop2 = _bar_crop(frame=True, top_white_rows=5)
    crop2[22, 20:28] = (30, 30, 30)         # a uniform dark row inside the white rows looks like a frame: the bar ends there
    bbox, ext_lo, _ = recover_light_end(crop2, core, True, (23, 118))
    assert ext_lo == 2 and bbox[1] == 23, (bbox, ext_lo)


def test_strip_inside_map_panel_is_not_a_colour_bar():
    from afmquant.reader2 import _inside_map
    map_box = (80, 0, 415, 430)
    strip_in_map = (82, 5, 90, 425)         # the map's left edge columns
    det_outside = (0, 0, 80, 430)           # y-axis tick-label column returned as a 'colorbar'
    assert _inside_map(strip_in_map, det_outside, [map_box]) == map_box
    det_inside = (84, 300, 100, 420)        # an inset colour bar: detector box inside the map too
    assert _inside_map(strip_in_map, det_inside, [map_box]) is None
    strip_outside = (430, 5, 445, 425)      # a real colour bar beside the map
    assert _inside_map(strip_outside, (425, 0, 480, 430), [map_box]) is None
