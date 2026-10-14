"""Unit tests for the afmquant package (run: python -m pytest tests -q).

The tests cover the deterministic parts of the pipeline: label parsing,
scale fitting, strip extraction from a synthetic colour bar, the colour-lookup
engine round trip, map/colour-bar pairing and the technique filter. The OCR
model and the detector are exercised by the benchmark scripts, not here.
"""
import numpy as np
import pytest

from afmquant import ocr, pairing, metrics, colorbar
from afmquant.engine import quantify_map
from afmquant.render import colormap_strip, get_cmap
from afmquant.afm_filter import classify_panel


# ---------------------------------------------------------------- label parsing
@pytest.mark.parametrize("text, value", [
    ("1.5", 1.5), ("−0.5", -0.5), ("1,000", 1000.0), ("0,5", 0.5), ("2e3", 2000.0),
    ("1000", 1000.0), ("-1.5e-7", -1.5e-7), ("1.2×10^3", 1200.0),
])
def test_parse_number(text, value):
    assert ocr.parse_number(text) == pytest.approx(value)


@pytest.mark.parametrize("text", ["nm", "Amplitude (pm)", "", "(a)"])
def test_parse_number_rejects_text(text):
    assert ocr.parse_number(text) is None


@pytest.mark.parametrize("text, factor", [
    ("1e-9", 1e-9), ("le-7", 1e-7), ("x10^-3", 1e-3), ("10^-3", 1e-3), ("1e6", 1e6),
])
def test_parse_offset(text, factor):
    assert ocr.parse_offset(text) == pytest.approx(factor, rel=1e-9)


def test_parse_offset_rejects_plain_numbers():
    assert ocr.parse_offset("1000") is None
    assert ocr.parse_offset("0.5") is None


def test_parse_number_unit():
    assert ocr.parse_number_unit("80 nm") == (80.0, "nm")
    assert ocr.parse_number_unit("+4.5 Hz") == (4.5, "Hz")
    assert ocr.parse_number_unit("80") == (80.0, None)
    assert ocr.parse_number_unit("ESM") == (None, None)


def test_parse_unit():
    assert ocr.parse_unit("Amplitude (pm)") == "pm"
    assert ocr.parse_unit("nm") == "nm"
    assert ocr.parse_unit("(a)") is None
    assert ocr.parse_unit("1e-9") is None


def test_split_merged_decimals():
    assert ocr.split_merged_decimals("0.000.160.320.48") == [0.0, 0.16, 0.32, 0.48]
    assert ocr.split_merged_decimals("0.5") is None
    assert ocr.split_merged_decimals("1.21.11.3") is None       # not monotonic: ambiguous
    assert ocr.split_merged_decimals("0.480.320.16") == [0.48, 0.32, 0.16]   # decreasing is fine


# ---------------------------------------------------------------- scale fitting
def test_fit_scale_linear():
    anchors = [(0.1, 10.0, 0.9), (0.5, 50.0, 0.9), (0.9, 90.0, 0.9)]   # (position, value, confidence)
    fit = ocr.fit_scale(anchors)
    assert fit["ok"] and fit["scale"] == "linear"
    assert fit["vmin"] == pytest.approx(0.0, abs=1e-6)
    assert fit["vmax"] == pytest.approx(100.0, abs=1e-6)
    assert fit["start_is_max"] is False


def test_fit_scale_rejects_one_misread_label():
    anchors = [(0.1, 10.0, 0.9), (0.3, 30.0, 0.9), (0.5, 50.0, 0.9), (0.7, 700.0, 0.6), (0.9, 90.0, 0.9)]
    fit = ocr.fit_scale(anchors)
    assert fit["ok"] and fit["n_inliers"] == 4
    assert fit["vmax"] == pytest.approx(100.0, abs=1e-6)


def test_fit_scale_flags_log_scale():
    anchors = [(0.1, 1.0, 0.9), (0.3, 10.0, 0.9), (0.5, 100.0, 0.9), (0.7, 1000.0, 0.9)]
    fit = ocr.fit_scale(anchors)
    assert not fit["ok"] and fit["scale"] == "log"


def test_fit_scale_needs_two_labels():
    assert not ocr.fit_scale([(0.5, 1.0, 0.9)])["ok"]


# ---------------------------------------------------------------- strip extraction
def _synthetic_bar(cmap="viridis", vertical=True):
    crop = np.full((300, 120, 3), 255, np.uint8)
    strip = colormap_strip(cmap, 260)                      # vmax -> vmin
    if vertical:
        crop[20:280, 30:60] = strip[:, None, :]
        bbox = (30, 20, 60, 280)
    else:
        crop = np.full((120, 300, 3), 255, np.uint8)
        crop[30:60, 20:280] = strip[::-1][None, :, :]       # vmin on the left
        bbox = (20, 30, 280, 60)
    return crop, bbox, strip


def test_extract_strip_vertical():
    crop, bbox, strip = _synthetic_bar()
    ex = colorbar.extract_strip(crop, n=256)
    assert ex["ok"] and ex["orientation"] == "vertical"
    assert all(abs(a - b) <= 2 for a, b in zip(ex["bbox"], bbox))
    assert np.abs(ex["strip"][0].astype(int) - strip[0].astype(int)).max() <= 4
    assert np.abs(ex["strip"][-1].astype(int) - strip[-1].astype(int)).max() <= 4


def test_extract_strip_horizontal():
    crop, bbox, _ = _synthetic_bar(vertical=False)
    ex = colorbar.extract_strip(crop, n=256)
    assert ex["ok"] and ex["orientation"] == "horizontal"
    assert all(abs(a - b) <= 2 for a, b in zip(ex["bbox"], bbox))


# ---------------------------------------------------------------- engine round trip
@pytest.mark.parametrize("cmap", ["copper", "hot", "jet", "viridis"])
def test_engine_round_trip_lossless(cmap):
    rng = np.random.default_rng(0)
    gt = rng.random((48, 48)).astype(np.float32) * 300 + 20          # arbitrary pm values
    vmin, vmax = float(gt.min()), float(gt.max())
    rgba = get_cmap(cmap)((gt - vmin) / (vmax - vmin))
    map_rgb = (rgba[..., :3] * 255).round().astype(np.uint8)
    strip = colormap_strip(cmap, 256)
    vm, valid, qc = quantify_map(map_rgb, strip, vmin, vmax)
    assert valid.mean() > 0.99
    m = metrics.evaluate(gt, vm, valid)
    assert m["nMAE_pct"] < 0.5                                        # 256-level quantisation floor (copper has flat steps)
    assert m["SSIM"] > 0.98


def test_engine_rejects_foreign_colours():
    gt = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    map_rgb = (get_cmap("viridis")(gt)[..., :3] * 255).astype(np.uint8)
    map_rgb[:8, :8] = (255, 255, 255)                                 # white overlay (not in viridis)
    vm, valid, qc = quantify_map(map_rgb, colormap_strip("viridis"), 0.0, 1.0)
    assert valid[:8, :8].sum() == 0 and valid[8:, 8:].all()
    assert qc["valid_pixel_ratio"] == pytest.approx(1 - 64 / 1024, abs=1e-6)


def test_engine_rgb_matching_option():
    gt = np.linspace(0, 1, 32 * 32, dtype=np.float32).reshape(32, 32)
    map_rgb = (get_cmap("hot")(gt)[..., :3] * 255).astype(np.uint8)
    vm, valid, _ = quantify_map(map_rgb, colormap_strip("hot"), 0.0, 1.0, color_space="rgb")
    assert np.abs(vm - gt)[valid].mean() < 0.003


# ---------------------------------------------------------------- metrics
def test_evaluate_identity():
    gt = np.random.default_rng(1).random((40, 40))
    m = metrics.evaluate(gt, gt, np.ones_like(gt, bool))
    assert m["nMAE_pct"] == 0 and m["SSIM"] == pytest.approx(1.0)
    assert m["nMAE_IQR_pct"] == 0 and m["levels_in_IQR"] > 0


# ---------------------------------------------------------------- pairing
def test_pair_maps_to_bars_by_gamut():
    img = np.full((200, 320, 3), 255, np.uint8)
    gt = np.random.default_rng(2).random((100, 100))
    img[10:110, 10:110] = (get_cmap("viridis")(gt)[..., :3] * 255).astype(np.uint8)
    bars = [{"box": (120, 10, 140, 110), "strip": colormap_strip("copper")},     # nearer, wrong gamut
            {"box": (200, 10, 220, 110), "strip": colormap_strip("viridis")}]   # farther, right gamut
    pairs = pairing.pair_maps_to_bars(img, [(10, 10, 110, 110)], bars)
    assert pairs[0]["bar_idx"] == 1 and pairs[0]["compat"] > 0.95


def test_pair_maps_to_bars_no_compatible():
    img = np.full((120, 200, 3), 255, np.uint8)
    img[10:60, 10:60] = (0, 0, 255)
    bars = [{"box": (80, 10, 100, 60), "strip": colormap_strip("copper")}]
    assert pairing.pair_maps_to_bars(img, [(10, 10, 60, 60)], bars)[0]["bar_idx"] is None


def test_box_distance():
    assert pairing.box_distance((0, 0, 10, 10), (5, 5, 20, 20)) == 0
    assert pairing.box_distance((0, 0, 10, 10), (13, 0, 20, 10)) == pytest.approx(3.0)


# ---------------------------------------------------------------- technique filter
@pytest.mark.parametrize("caption, unit, label", [
    ("PFM amplitude image of the film", "pm", "afm"),
    ("Raman map of the flake", "cps", "non_afm_colormap"),
    ("", "nm", "afm"),
    ("", "a.u.", "unknown"),
    ("AFM topography and SEM image", "nm", "afm"),
])
def test_classify_panel(caption, unit, label):
    assert classify_panel(caption, unit)["label"] == label
