"""Evidence-based pairing (revision R1): layout, exclusivity, row and stack evidence, and the held status."""
import numpy as np
import pytest

from afmquant.pairing import layout_evidence, row_evidence, pair_maps_to_bars


def _bar(box):
    return {"box": box, "orientation": "vertical" if (box[3] - box[1]) >= (box[2] - box[0]) else "horizontal"}


def test_layout_evidence_adjacent_and_aligned():
    assert layout_evidence((0, 0, 100, 100), (110, 0, 125, 100))          # right beside, same height
    assert not layout_evidence((0, 0, 100, 100), (300, 0, 315, 100))      # far away
    assert not layout_evidence((0, 0, 100, 100), (110, 300, 125, 400))    # not aligned


def test_row_shared_bar_accepted_and_bar_between_refused():
    maps = [(0, 0, 100, 100), (120, 0, 220, 100)]
    bars = [_bar((230, 0, 245, 100))]
    assert row_evidence(0, 0, maps, bars, [0, 0])                       # left map shares the bar at the row end
    bars2 = bars + [_bar((105, 0, 115, 100))]                            # another bar between left map and the shared bar
    assert not row_evidence(0, 0, maps, bars2, [0, 0])


def test_stack_shared_bar_accepted_and_refusals():
    bars = [_bar((498, 381, 538, 537))]                                  # bar beside the top of a column of two maps
    maps = [(306, 534, 497, 680), (295, 834, 492, 979)]
    assert row_evidence(1, 0, maps, bars, [0, 0]) and row_evidence(0, 0, maps, bars, [0, 0])
    maps_far = [(0, 0, 100, 100), (0, 300, 100, 400)]
    assert not row_evidence(1, 0, maps_far, [_bar((600, 0, 615, 100))], [0, 0])       # bar far from the stack
    bars_between = [_bar((110, 0, 125, 100)), _bar((110, 150, 125, 250))]
    assert not row_evidence(1, 0, maps_far, bars_between, [0, 0])                     # another bar in the stack's bar column
    assert not row_evidence(1, 0, [(0, 0, 100, 100), (400, 300, 500, 400)], bars, [0, 0])   # maps not aligned


def test_pipeline_holds_map_without_evidence():
    """Two maps of the same colormap on different rows; one bar laid out beside the upper-right map only.
    Nearest mode pairs both maps with it; evidence mode keeps the right map (layout) and holds the other
    (not aligned with the bar, not exclusive, not in a row or stack with the owner)."""
    rng = np.random.default_rng(0)
    strip = np.stack([np.linspace(0, 255, 64), np.zeros(64), np.linspace(255, 0, 64)], axis=1).astype(np.uint8)
    img = np.full((400, 700, 3), 255, np.uint8)
    img[0:100, 450:550] = strip[rng.integers(0, 64, size=(100, 100))]
    img[250:350, 0:100] = strip[rng.integers(0, 64, size=(100, 100))]
    bars = [{"box": (560, 0, 575, 100), "strip": strip, "orientation": "vertical"}]
    maps = [(0, 250, 100, 350), (450, 0, 550, 100)]
    near = pair_maps_to_bars(img, maps, bars, mode="nearest")
    assert [p["bar_idx"] for p in near] == [0, 0]
    evid = pair_maps_to_bars(img, maps, bars, mode="evidence")
    assert evid[0]["bar_idx"] is None and evid[0]["reason"] == "no_exclusive_bar"
    assert evid[1]["bar_idx"] == 0 and evid[1]["evidence"]["layout"]


def test_shared_bar_in_a_row_is_kept_in_evidence_mode():
    """Two maps side by side sharing one bar at the row end (the Tier 2 pair_shared layout) are both kept."""
    rng = np.random.default_rng(1)
    strip = np.stack([np.linspace(0, 255, 64), np.zeros(64), np.linspace(255, 0, 64)], axis=1).astype(np.uint8)
    img = np.full((100, 700, 3), 255, np.uint8)
    for x0 in (0, 450):
        img[:, x0:x0 + 100] = strip[rng.integers(0, 64, size=(100, 100))]
    bars = [{"box": (560, 0, 575, 100), "strip": strip, "orientation": "vertical"}]
    maps = [(0, 0, 100, 100), (450, 0, 550, 100)]
    evid = pair_maps_to_bars(img, maps, bars, mode="evidence")
    assert [p["bar_idx"] for p in evid] == [0, 0] and evid[0]["evidence"]["row"]
