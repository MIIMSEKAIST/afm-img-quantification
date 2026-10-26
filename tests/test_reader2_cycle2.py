"""Cycle-2 rules of the label-column reader and the pairing on synthetic crops (no OCR needed): across refinement
of the gradient region, the row walk with tick tolerance, ink-based signs, short-bar layout evidence, bars laid out
between two maps, and laid-out-first pairing."""
import numpy as np

from afmquant.colorbar import refine_across, choose_candidate, find_gradient_region
from afmquant.reader2 import extend_by_walk, _strokes_before_digits, _apply_ink_signs, _rotated_minus
from afmquant.pairing import layout_evidence, own_bars, pair_maps_to_bars


# ------------------------------------------------------------------------------------------ gradient region
def _map_touching_bar():
    """White page 200x120; a noisy map (x 10..40), a 2-px dark frame, an 8-px grey gradient strip (x 42..50,
    y 20..180) with a 2-px frame on its right."""
    rng = np.random.default_rng(0)
    crop = np.full((200, 120, 3), 255, np.uint8)
    crop[15:185, 10:40] = rng.integers(0, 200, size=(170, 30, 3))
    crop[18:182, 40:42] = 20
    crop[18:182, 50:52] = 20
    for i in range(160):
        v = int(round(255 * (1 - i / 159)))
        crop[20 + i, 42:50] = (v, v, v)
    return crop


def test_refine_across_separates_strip_from_touching_map_and_frame():
    crop = _map_touching_bar()
    box = refine_across(crop, (10, 20, 52, 180), "vertical")
    assert box[0] >= 41 and box[2] <= 51 and box[2] - box[0] >= 6, box
    assert box[1] == 20 and box[3] == 180
    # the finder itself recovers the strip from the merged component (roi = the bar box with its labels)
    found = find_gradient_region(crop, roi=(41, 15, 90, 185), extend=False)
    assert found is not None and found[1] == "vertical"
    (x0, y0, x1, y1), _ = found
    assert 40 <= x0 <= 43 and 49 <= x1 <= 52, found


def test_choose_candidate_prefers_detector_orientation_and_length():
    cands = [((10, 10, 20, 40), "vertical"), ((5, 50, 60, 58), "horizontal"), ((10, 5, 20, 60), "vertical")]
    assert choose_candidate(cands, roi=(8, 0, 24, 70)) == ((10, 5, 20, 60), "vertical")
    assert choose_candidate(cands, roi=(0, 48, 70, 60)) == ((5, 50, 60, 58), "horizontal")
    assert choose_candidate([], roi=(0, 0, 10, 50)) is None


# ------------------------------------------------------------------------------------------ row walk
def _thin_bar_with_ticks():
    """White page; a 6-px strip (x 30..36, y 20..220) from red (top) to black (bottom); black tick marks 1 px tall
    reaching 2 px into the strip every 40 rows; the core found covers only the middle 60 rows."""
    crop = np.full((240, 80, 3), 255, np.uint8)
    for i in range(200):
        t = i / 199.0
        crop[20 + i, 30:36] = (int(round(220 * (1 - t))), 0, 0)
    for y in range(30, 220, 40):
        crop[y, 24:32] = 0                       # tick: from the label side into the strip
    return crop


def test_walk_extends_through_tick_marks_and_stops_at_the_page():
    crop = _thin_bar_with_ticks()
    bg = np.array([255.0, 255.0, 255.0])
    bbox, lo, hi = extend_by_walk(crop, (30, 100, 36, 160), True, (18, 222), bg, light_only=False)
    assert bbox[1] <= 22 and bbox[3] >= 218, bbox              # the whole bar, ticks tolerated (dark ends: light_only off)
    assert bbox[1] >= 19 and bbox[3] <= 222                     # not into the page (bar ends are not page-coloured)
    assert lo > 0 and hi > 0


def test_walk_stops_at_a_colour_jump_and_respects_limits():
    crop = _thin_bar_with_ticks()
    crop[120:124, 30:36] = (0, 0, 255)                           # a blue block across the bar: a jump
    bg = np.array([255.0, 255.0, 255.0])
    bbox, lo, hi = extend_by_walk(crop, (30, 130, 36, 160), True, (18, 222), bg, light_only=False)
    assert bbox[1] >= 123, bbox                                 # upward walk stops at the jump
    bbox2, _, _ = extend_by_walk(crop, (30, 130, 36, 160), True, (18, 222), bg, limits=(125, 190), light_only=False)
    assert bbox2[1] >= 125 and bbox2[3] <= 190


# ------------------------------------------------------------------------------------------ signs from ink
def _label_crop(labels, strip_x1=30, ticks=True, minus=(), gap_tick_minus=2):
    """White page 200x120 with a vertical strip ending at x=strip_x1; labels right of it: digits = a 10x12 ink block
    at x 44..54. ``ticks``: a 1-px stroke x 30..35 at each label's mid-height; ``minus``: labels (by index) with a
    detached minus stroke x 37..41 (gap_tick_minus px after the tick) or, without ticks, x 36..41."""
    crop = np.full((200, 120, 3), 255, np.uint8)
    crop[10:190, 20:strip_x1] = (120, 40, 200)
    out = []
    for k, (y, num, text) in enumerate(labels):
        crop[y:y + 12, 44:54] = 0
        if ticks:
            crop[y + 6, strip_x1:strip_x1 + 5] = 0
        if k in minus:
            x0 = strip_x1 + 5 + gap_tick_minus if ticks else 36
            crop[y + 6, x0:x0 + 5] = 0
        out.append({"side": 1, "num": float(num), "text": text, "bbox": (44.0, float(y), 54.0, float(y + 12)), "conf": 0.9})
    return crop, out


def test_strokes_before_digits_attached_and_detached():
    crop, labs = _label_crop([(20, 5, "5"), (60, -3, "-3")], minus=(1,))
    strokes, dx0 = _strokes_before_digits(crop, labs[0]["bbox"], 31, 12.0)
    assert dx0 == 44 and len(strokes) == 1 and strokes[0][2]              # the tick, attached
    strokes, _ = _strokes_before_digits(crop, labs[1]["bbox"], 31, 12.0)
    kinds = sorted(s[2] for s in strokes)
    assert kinds == [False, True]                                           # tick + detached minus


def test_ink_signs_ticks_are_not_minus_signs():
    """Ticks at every label; the OCR gave a minus to two labels whose boxes start at the strip (they included the
    tick): both become positive; the detached minus of the last label is kept."""
    crop, labs = _label_crop([(20, -5, "-5"), (60, -3, "-3"), (100, 1, "1"), (140, -2, "-2")], minus=(3,))
    labs[0]["bbox"] = (30.0, 20.0, 54.0, 32.0)
    labs[1]["bbox"] = (30.0, 60.0, 54.0, 72.0)
    _apply_ink_signs(crop, labs, (20, 10, 30, 190), True, 12.0)
    assert [c["num"] for c in labs] == [5.0, 3.0, 1.0, -2.0]
    assert labs[0]["text"] == "5" and labs[3]["text"] == "-2"


def test_ink_signs_lost_minus_recovered_when_zero_has_no_stroke():
    """No ticks; the strokes are minus signs (the 0 label has none): the label read without its minus is fixed."""
    crop, labs = _label_crop([(20, 0, "0"), (60, -1, "-1"), (100, 2, "2"), (140, -3, "-3")], ticks=False, minus=(1, 2, 3))
    _apply_ink_signs(crop, labs, (20, 10, 30, 190), True, 12.0)
    assert [c["num"] for c in labs] == [0.0, -1.0, -2.0, -3.0]


def test_ink_signs_all_minus_column_kept():
    """Every label has an equal attached stroke and an OCR minus (a bar of negative values whose minus signs touch
    the strip): nothing is changed."""
    crop, labs = _label_crop([(20, -5, "-5"), (60, -3, "-3"), (100, -1, "-1")], ticks=True)
    _apply_ink_signs(crop, labs, (20, 10, 30, 190), True, 12.0)
    assert [c["num"] for c in labs] == [-5.0, -3.0, -1.0]


def test_ink_signs_left_side_and_end_labels():
    """Labels left of the strip and end labels: a detached stroke next to the digits adds the minus."""
    crop = np.full((200, 120, 3), 255, np.uint8)
    crop[10:190, 80:90] = (120, 40, 200)
    crop[20:32, 50:60] = 0                                      # digits of a label left of the strip
    crop[26, 43:48] = 0                                         # its minus sign
    crop[100:112, 50:60] = 0                                    # a second label without minus
    labs = [{"side": -1, "num": 4.0, "text": "4", "bbox": (50.0, 20.0, 60.0, 32.0), "conf": 0.9},
            {"side": -1, "num": 2.0, "text": "2", "bbox": (50.0, 100.0, 60.0, 112.0), "conf": 0.9}]
    _apply_ink_signs(crop, labs, (80, 10, 90, 190), True, 12.0)
    assert [c["num"] for c in labs] == [-4.0, 2.0]


def test_rotated_minus():
    crop = np.full((200, 80, 3), 255, np.uint8)
    crop[60:100, 40:50] = 0                                     # glyphs of a label written along the bar
    crop[50:56, 44:46] = 0                                      # a vertical stroke above them: the minus sign
    assert _rotated_minus(crop, (40.0, 60.0, 50.0, 100.0))
    crop2 = np.full((200, 80, 3), 255, np.uint8)
    crop2[60:100, 40:50] = 0
    assert not _rotated_minus(crop2, (40.0, 60.0, 50.0, 100.0))


# ------------------------------------------------------------------------------------------ pairing
def test_layout_evidence_short_bar():
    assert layout_evidence((0, 0, 200, 200), (205, 60, 215, 150))          # a short bar centred beside the map
    assert not layout_evidence((0, 0, 200, 200), (205, 180, 215, 260))     # mostly beyond the map's extent
    assert not layout_evidence((0, 0, 200, 200), (205, 90, 215, 110))      # too short to be the map's bar


def _bar(box, strip="s"):
    return {"box": box, "strip": strip, "orientation": "vertical" if (box[3] - box[1]) >= (box[2] - box[0]) else "horizontal"}


def test_own_bars_side_convention_resolves_bars_between_maps():
    """Three maps in a row, each with its bar on its LEFT (the bar of the second map lies 5 px right of the first
    map): the leftmost bar is unambiguous and sets the convention."""
    maps = [(74, 315, 338, 517), (415, 307, 674, 516), (761, 319, 1021, 518)]
    bars = [_bar((0, 297, 40, 495)), _bar((343, 295, 384, 490)), _bar((689, 297, 733, 493))]
    own, owners = own_bars(maps, bars)
    assert own == [[0], [1], [2]], own


def test_own_bars_nearer_map_wins_without_convention():
    maps = [(3, 506, 179, 633), (206, 505, 382, 631), (430, 503, 551, 624)]
    bars = [_bar((170, 500, 196, 640)), _bar((372, 500, 409, 640))]       # each bar overlaps the map on its left
    own, owners = own_bars(maps, bars)
    assert own == [[0], [1], []], own


def test_pairing_laid_out_bar_first_and_exclusive_reach():
    rng = np.random.default_rng(3)
    strip = np.stack([np.linspace(0, 255, 64), np.zeros(64), np.linspace(255, 0, 64)], axis=1).astype(np.uint8)
    img = np.full((120, 1000, 3), 255, np.uint8)
    for x0 in (0, 400, 800):
        img[:100, x0:x0 + 100] = strip[rng.integers(0, 64, size=(100, 100))]
    maps = [(0, 0, 100, 100), (400, 0, 500, 100), (800, 0, 900, 100)]
    bars = [{"box": (110, 0, 125, 100), "strip": strip, "orientation": "vertical"},     # own bar of map 0
            {"box": (560, 0, 575, 100), "strip": strip, "orientation": "vertical"}]     # 60 px right of map 1, far from map 2
    out = pair_maps_to_bars(img, maps, bars, mode="evidence")
    assert out[0]["bar_idx"] == 0 and out[0]["evidence"]["layout"]
    assert out[1]["bar_idx"] == 1 and out[1]["evidence"]["exclusive"]
    assert out[2]["bar_idx"] is None and out[2]["reason"] == "no_exclusive_bar"


def test_pairing_readable_laid_out_bar_beats_unreadable_own_bar():
    rng = np.random.default_rng(4)
    strip = np.stack([np.linspace(0, 255, 64), np.zeros(64), np.linspace(255, 0, 64)], axis=1).astype(np.uint8)
    img = np.full((120, 400, 3), 255, np.uint8)
    img[:100, 100:200] = strip[rng.integers(0, 64, size=(100, 100))]
    maps = [(100, 0, 200, 100)]
    bars = [{"box": (80, 0, 95, 100), "strip": None, "orientation": "vertical"},        # unreadable, left of the map
            {"box": (205, 0, 220, 100), "strip": strip, "orientation": "vertical"}]     # readable, right of the map
    out = pair_maps_to_bars(img, maps, bars, mode="evidence")
    assert out[0]["bar_idx"] == 1
    # a readable own bar that does not match the map's colours holds the map
    other = np.stack([np.zeros(64), np.linspace(0, 255, 64), np.zeros(64)], axis=1).astype(np.uint8)
    out2 = pair_maps_to_bars(img, maps, [{"box": (205, 0, 220, 100), "strip": other, "orientation": "vertical"}], mode="evidence")
    assert out2[0]["bar_idx"] is None and out2[0]["reason"] == "own_bar_unreadable" and out2[0]["evidence"]["own_incompatible"]


def test_stack_evidence_uses_the_maps_column():
    """A 2x2 grid with one tall bar beside the right column: the right maps share it (stack), the left are held."""
    rng = np.random.default_rng(5)
    strip = np.stack([np.linspace(0, 255, 64), np.zeros(64), np.linspace(255, 0, 64)], axis=1).astype(np.uint8)
    img = np.full((620, 500, 3), 255, np.uint8)
    maps = [(85, 1, 407, 203), (443, 1, 725, 211), (84, 195, 408, 601), (444, 201, 723, 599)]
    img = np.full((620, 760, 3), 255, np.uint8)
    for x0, y0, x1, y1 in maps:
        img[y0:y1, x0:x1] = strip[rng.integers(0, 64, size=(y1 - y0, x1 - x0))]
    bars = [{"box": (742, 62, 760, 600), "strip": strip, "orientation": "vertical"}]
    out = pair_maps_to_bars(img, maps, bars, mode="evidence")
    assert out[1]["bar_idx"] == 0 and out[3]["bar_idx"] == 0
    assert out[0]["bar_idx"] is None and out[2]["bar_idx"] is None


# ------------------------------------------------------------------------------------------ label column
def test_column_with_labels_along_the_strip_beats_end_labels_only():
    from afmquant.reader2 import choose_column

    def lab(num, side, off, kind, pos):
        return {"num": num, "side": side, "off": off, "kind": kind, "pos": pos, "text": str(num), "conf": 0.9}

    left = [lab(20, -1, 12.0, "end", 0.0), lab(80, -1, 14.0, "end", 1.0)]          # a neighbouring axis beyond the ends
    right = [lab(4, 1, 15.0, "col", 0.1), lab(2, 1, 15.0, "col", 0.4), lab(0, 1, 15.0, "col", 0.7), lab(-2, 1, 15.0, "col", 1.0)]
    chosen, how = choose_column(left + right, tol=8.0, max_off=40.0)
    assert how == "column" and {c["num"] for c in chosen} == {4, 2, 0, -2}
    # with labels along the strip on both sides the nearer column still wins
    left2 = [lab(20, -1, 12.0, "col", 0.1), lab(80, -1, 14.0, "col", 0.9)]
    chosen, how = choose_column(left2 + right, tol=8.0, max_off=40.0)
    assert {c["num"] for c in chosen} == {20, 80}


# ------------------------------------------------------------------------------------------ core geometry
def test_main_body_drops_a_label_row_but_keeps_a_white_centre():
    from afmquant.reader2 import main_body
    bg = np.array([255.0, 255.0, 255.0])
    crop = np.full((300, 40, 3), 255, np.uint8)
    crop[10:28, 10:20] = 0                                   # a dark label row merged into the region
    for i in range(200):
        v = int(round(255 * i / 199))
        crop[60 + i, 10:20] = (v, 0, 255 - v)                # the strip (y 60..260)
    assert main_body(crop, (10, 10, 20, 260), True, bg) == (10, 60, 20, 260)
    crop2 = np.full((300, 40, 3), 255, np.uint8)
    for i in range(200):                                     # a diverging map with a white centre band of 6 rows
        t = abs(i - 100) / 100.0
        v = int(round(255 * (1 - t)))
        crop2[60 + i, 10:20] = (255, v, v) if i < 100 else (v, v, 255)
    crop2[157:163, 10:20] = 255
    assert main_body(crop2, (10, 60, 20, 260), True, bg) == (10, 60, 20, 260)


def test_restore_beyond_rejects_other_bar_and_flat_block():
    from afmquant.reader2 import _restore_beyond
    crop = np.full((300, 40, 3), 255, np.uint8)
    for i in range(150):
        v = int(round(255 * i / 149))
        crop[100 + i, 10:20] = (v, v, v)                     # the bar (y 100..250)
    crop[40:100, 10:20] = (120, 120, 120)                    # a flat grey block above it
    clipped, walked = (10, 100, 20, 250), (10, 40, 20, 250)
    assert _restore_beyond(crop, clipped, walked, True, [], []) == clipped            # flat block: not restored
    crop2 = crop.copy()
    for i in range(60):
        crop2[40 + i, 10:20] = (0, 0, max(0, 60 - i))        # a gradient continues above
    assert _restore_beyond(crop2, clipped, walked, True, [], []) == walked
    assert _restore_beyond(crop2, clipped, walked, True, [(5, 20, 25, 95)], []) == clipped   # inside another bar's box


def test_choose_candidate_keeps_the_standard_candidate_unless_much_longer():
    std, longer, much = ((10, 20, 20, 120), "vertical"), ((10, 10, 20, 130), "vertical"), ((10, 0, 20, 200), "vertical")
    assert choose_candidate([std, longer], roi=(5, 0, 25, 220)) == std
    assert choose_candidate([std, much], roi=(5, 0, 25, 220)) == much


def test_ink_signs_zero_without_stroke_means_minus_signs():
    """Labels 0, -1, -2 with the minus signs touching the strip and no ticks: the zero label has no stroke, so the
    strokes are minus signs even though they are attached to the strip (the label read without a minus is fixed)."""
    crop = np.full((200, 120, 3), 255, np.uint8)
    crop[10:190, 20:30] = (120, 40, 200)
    labs = []
    for k, (y, num, text) in enumerate([(20, 0, "0"), (60, -1, "-1"), (100, 2, "2")]):
        crop[y:y + 12, 40:50] = 0
        if k > 0:
            crop[y + 6, 31:36] = 0                            # a minus sign 1 px from the strip (attached)
        labs.append({"side": 1, "num": float(num), "text": text, "bbox": (40.0, float(y), 50.0, float(y + 12)), "conf": 0.9})
    _apply_ink_signs(crop, labs, (20, 10, 30, 190), True, 12.0)
    assert [c["num"] for c in labs] == [0.0, -1.0, -2.0]


def test_stack_bar_owned_by_another_map_must_reach_the_map():
    from afmquant.pairing import row_evidence
    maps = [(144, 207, 450, 435), (166, 25, 354, 175)]       # a column: map D below, a diffraction image above
    bars = [_bar((460, 199, 518, 439))]                       # D's own bar, beside D only
    assert row_evidence(0, 0, maps, bars, [0, 0])             # D shares (it is the owner anyway)
    assert not row_evidence(1, 0, maps, bars, [0, 0])         # the image above does not reach the bar


# ------------------------------------------------------------------------------------------ cycle 2c
def test_finder_prefers_a_separate_bar_over_the_gradient_like_edge_of_a_map():
    """A map ends 2 px before the detector box; its last 6 columns are smooth (gradient-like along the bar axis),
    its body is noise. Inside the search margin the map's edge is the largest component and refines to its smooth
    columns, but the bar, a separate component inside the box, is returned; without the bar the map's edge is the
    fallback (a bar merged with a touching map is found the same way)."""
    crop = np.full((220, 180, 3), 255, np.uint8)
    crop[15:195, 5:60] = np.random.default_rng(1).integers(0, 200, (180, 55, 3))   # map body x 5..60
    for i in range(180):                                     # smooth edge x 60..66
        crop[15 + i, 60:66] = (255, int(255 * (1 - i / 179)), 0)
    for i in range(170):                                     # bar x 72..80 with a 1-px frame
        v = int(255 * (1 - i / 169))
        crop[20 + i, 72:80] = (v, v, v)
    crop[19:191, 71] = 0; crop[19:191, 80] = 0; crop[19, 71:81] = 0; crop[190, 71:81] = 0
    for y in (30, 100, 170):                                 # tick labels right of the bar
        crop[y:y + 10, 90:102] = 0
    roi = (68, 15, 160, 195)
    found = find_gradient_region(crop, roi=roi, extend=False)
    assert found is not None and found[1] == "vertical", found
    (x0, y0, x1, y1), _ = found
    assert 70 <= x0 <= 73 and 79 <= x1 <= 82, found
    crop2 = crop.copy()
    crop2[15:195, 68:180] = 255                              # no bar inside the box: the map's edge is the fallback
    found2 = find_gradient_region(crop2, roi=roi, extend=False)
    assert found2 is not None and 58 <= found2[0][0] <= 61 and 65 <= found2[0][2] <= 67, found2


def test_strokes_before_digits_tick_after_a_frame_line_is_attached():
    """A 3-px frame line at the strip edge and, after a 1-px antialiasing gap, a 5-px tick (a 72 dpi figure): the
    tick is attached (it starts right after the frame); a stroke further away is detached."""
    crop = np.full((60, 80, 3), 255, np.uint8)
    crop[5:55, 10:20] = (120, 40, 200)                      # strip x 10..20
    crop[5:55, 20:23] = 0                                    # frame line x 20..22
    crop[20:32, 36:46] = 0                                   # digits, h = 12
    crop[26, 24:29] = 0                                      # tick: 1 px after the frame
    strokes, dx0 = _strokes_before_digits(crop, (36.0, 20.0, 46.0, 32.0), 21, 12.0)
    assert dx0 == 36 and len(strokes) == 1 and strokes[0][2], strokes
    crop[26, 24:29] = 255
    crop[26, 29:34] = 0                                      # a minus sign 6 px after the frame
    strokes, _ = _strokes_before_digits(crop, (36.0, 20.0, 46.0, 32.0), 21, 12.0)
    assert len(strokes) == 1 and not strokes[0][2], strokes
