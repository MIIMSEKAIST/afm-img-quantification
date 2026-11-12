"""The quality-control record of a quantified panel, on a synthetic figure with the detector and the reader stubbed
(no weights, no OCR): the record carries the number of colour levels inside the interquartile range of the data."""
import numpy as np
import pytest

from afmquant.pipeline import FigurePipeline


def _figure():
    """White page, a 160 x 160 map whose value rises linearly from left to right, and its vertical colour bar."""
    ramp = np.linspace(0.0, 1.0, 256)
    strip = np.stack([255 * ramp, 150 * ramp, 60 * ramp], axis=1).astype(np.uint8)        # dark to orange, never white
    img = np.full((200, 260, 3), 255, np.uint8)
    cols = np.clip(np.round(np.linspace(0, 255, 160)).astype(int), 0, 255)
    img[20:180, 20:180] = strip[cols][None, :, :]
    img[20:180, 200:215] = strip[np.round(np.linspace(255, 0, 160)).astype(int)][:, None, :]
    return img, strip


def test_levels_in_iqr_is_part_of_the_qc_record(monkeypatch):
    img, strip = _figure()
    pipe = FigurePipeline("unused.pt", ocr=object())
    bar = {"box": [200.0, 20.0, 215.0, 180.0], "ok": True, "strip": strip, "vmin": 0.0, "vmax": 10.0, "unit": "nm",
           "n_anchors": 3, "fit": {"ocr_conf_min": 0.9, "residual_rel": 0.0}, "strip_quality": {"tortuosity": 1.0},
           "orientation": "vertical"}
    monkeypatch.setattr(pipe, "detect", lambda image: ([{"xyxy": [20.0, 20.0, 180.0, 180.0], "conf": 0.9, "cls": 0}],
                                                       [{"xyxy": bar["box"], "conf": 0.9, "cls": 1}]))
    monkeypatch.setattr(pipe, "read_bars", lambda image, boxes, maps=None: [bar])
    res = pipe.process(img)
    (panel,) = res["panels"]
    assert panel["status"] == "ok"
    qc = panel["qc"]
    # the values are uniform over the scale, so the central half of the data spans half of the 255 level steps
    assert qc["levels_in_iqr"] == pytest.approx(127.5, abs=3.0)
    assert qc["valid_pixel_ratio"] > 0.95 and qc["unit"] == "nm"
