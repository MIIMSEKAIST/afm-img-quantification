"""Tier 3 helpers: raw-scan library, unit handling and automatic matching of
reconstructed figure panels to the raw scans they were made from.

Used by ``benchmark/tier3_evaluate.py``; see its docstring for the workflow.

Raw candidates
--------------
Every 2-D layer (channel) of every raw file is a candidate, in four processing
variants that cover what the image software usually applied before the figure
was made:

    raw     as stored
    plane   least-squares plane (1st order) removed
    flat0   per-scan-line offset removed      (Igor/AR "Flatten 0", Gwyddion "Align rows: median/mean")
    flat1   per-scan-line linear fit removed  (Igor/AR "Flatten 1")

Arrays are kept in their *display* orientation: Asylum Research .ibw layers are
shown the way Igor/AR displays them (x = first wave dimension to the right, y up,
i.e. ``np.flipud(wData[:, :, k].T)``); scan lines are rows. Matching tries all
eight dihedral orientations relative to that view (``ORIENTATIONS``).

Supported raw formats: Igor binary wave ``.ibw`` (Asylum Research; channel names
from the dimension labels, units from the channel type and the AR note),
Gwyddion simple field ``.gsf`` (any vendor format can be exported to it from
Gwyddion), plain numeric matrices ``.txt/.asc/.csv/.dat`` (Gwyddion ASCII export
header comments are parsed for the unit), ``.npy/.npz`` and single-channel
floating-point ``.tif/.tiff``. Vendor formats (Bruker .spm/.000, Park .tiff,
Nanonis .sxm, JPK, Nanosurf .nid) are listed as unsupported.

Matching (``match_panel``)
--------------------------
Identification score: Pearson r between "detail" maps of the panel and of the
raw region - least-squares plane removed, then winsorised (1st-99th
percentile) - over the panel's valid pixels. A tilt plane is common to many
raw height scans and isolated outliers (raw spikes, scale bars or labels that
map onto the colour bar) would otherwise dominate r.

1. Coarse screening (panel = full scan): panel and every candidate on a common
   coarse grid (48 cells; the candidate stretched to the panel aspect), all
   eight orientations at once (vectorised).
2. Fine scoring of the best (file, orientation) groups (at least 3 files) on
   the coarser of the two native grids with an integer-shift search (+/-4 %).
   For the 3 leading files the field of view is refined (scale 0.95-1.01 and
   position; detector boxes are not pixel-exact) and every channel x variant is
   scored, each with its own shift search (trace and retrace are laterally
   offset by piezo hysteresis); if another channel wins, the field of view is
   re-refined on it. The processing variant of the winning channel is the one
   with the highest plain (not detrended) r; near-ties are resolved by the
   caller by absolute agreement.
3. Crop search (panel = zoomed sub-region), unless the full-scan match is
   convincing (score >= 0.85 or >= 0.25 above the chance level): NCC of a
   20-cell panel template (8 orientations) against unit/magnitude-plausible
   candidates at 10 zooms (90 % ... 20 % of the scan width), refinement of the
   best 16 (file, orientation) groups (two scale/position rounds), then
   all channels/variants of the 5 leading files at their crops.
4. Decision. The chance level ("null") is the best score of an unrelated
   hypothesis (another file, or the same file in another orientation). A crop
   is preferred only if it scores > 0.03 higher than the full-scan match AND
   stands out more from its own chance level (the crop search has more
   freedom, so its chance level is higher). A match is "matched" with score
   >= 0.25 and score - null >= 0.08; the margin to the best other file and to
   the best other channel of the same file are reported (ambiguous below 0.05
   and 0.03).
"""
from __future__ import annotations

import re
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import cv2

# ============================================================== units
SI_PREFIX = {"": 1.0, "f": 1e-15, "p": 1e-12, "n": 1e-9, "u": 1e-6, "µ": 1e-6, "m": 1e-3,
             "k": 1e3, "M": 1e6, "G": 1e9}
# unit symbol -> (canonical base unit, factor to that base); angles are kept in degrees
BASE_UNITS = {"m": ("m", 1.0), "Å": ("m", 1e-10), "V": ("V", 1.0), "A": ("A", 1.0),
              "Hz": ("Hz", 1.0), "deg": ("deg", 1.0), "°": ("deg", 1.0),
              "rad": ("deg", 180.0 / np.pi), "N": ("N", 1.0), "Pa": ("Pa", 1.0), "S": ("S", 1.0)}
# case-insensitive spellings (OCR case errors); ambiguous ones resolved towards AFM-typical magnitudes
_LOWER_ALIASES = {"pm": "pm", "nm": "nm", "um": "µm", "µm": "µm", "mm": "mm", "å": "Å", "a": None,
                  "mv": "mV", "uv": "µV", "µv": "µV", "kv": "kV", "v": "V",
                  "pa": "pA", "na": "nA", "ua": "µA", "µa": "µA", "ma": "mA", "fa": "fA",
                  "hz": "Hz", "khz": "kHz", "mhz": "MHz", "deg": "deg", "°": "°", "rad": "rad", "mrad": "mrad",
                  "nn": "nN", "pn": "pN", "un": "µN", "µn": "µN", "mpa": "MPa", "gpa": "GPa", "kpa": "kPa",
                  "ns": "nS", "us": "µS", "µs": "µS", "ps": "pS"}


def parse_unit(u):
    """Unit text -> (factor, base) with value_SI = value * factor in the base unit.

    'pm' -> (1e-12, 'm'); 'mV' -> (1e-3, 'V'); '°' -> (1, 'deg');
    'mrad' -> (0.0573, 'deg'); 'a.u.', '%', None, unparsable -> (None, None).
    """
    if u is None:
        return None, None
    s = str(u).strip().strip("()[]{} ").replace("μ", "µ").replace("º", "°")
    if s.lower().startswith("deg"):
        s = "deg"
    if not s or s.lower().strip(".") in ("a.u", "au", "arb", "arb. u", "arb.u", "a. u", "%", "counts", "cps"):
        return None, None

    def _split(t):
        for base in sorted(BASE_UNITS, key=len, reverse=True):
            if t.endswith(base):
                pre = t[: -len(base)]
                if pre in SI_PREFIX and not (base in ("Å", "°") and pre):
                    canon, f = BASE_UNITS[base]
                    return SI_PREFIX[pre] * f, canon
        return None, None

    f, b = _split(s)
    if f is None:
        alias = _LOWER_ALIASES.get(s.lower())
        if alias:
            f, b = _split(alias)
    return f, b


def unit_label(factor, base):
    """(1e-12, 'm') -> 'pm' (for axis labels); falls back to the base unit."""
    if factor is None or base is None:
        return ""
    if base == "deg":
        return "deg" if abs(factor - 1) < 1e-9 else ("rad" if abs(factor - 180 / np.pi) < 1e-6 else f"x{factor:g} deg")
    for p, v in SI_PREFIX.items():
        if p != "µ" and abs(v - factor) <= 1e-9 * v:
            return ("µ" if p == "u" else p) + base
    return f"x{factor:g} {base}"


# ======================================================= raw file formats
SUPPORTED_EXT = {".ibw", ".gsf", ".txt", ".asc", ".csv", ".dat", ".npy", ".npz", ".tif", ".tiff"}
KNOWN_UNSUPPORTED = {".spm": "Bruker NanoScope", ".sxm": "Nanonis", ".nid": "Nanosurf", ".jpk": "JPK",
                     ".jpk-force": "JPK", ".mi": "Agilent/Keysight", ".afm": "vendor", ".stp": "WSxM",
                     ".top": "WSxM", ".gwy": "Gwyddion native (export as .gsf)", ".pxp": "Igor packed experiment",
                     ".h5": "HDF5", ".hdf5": "HDF5", ".sdf": "SDF", ".mdt": "NT-MDT", ".xqd": "Park",
                     ".xqdx": "Park", ".tiff_park": "Park"}
_BRUKER_NUM = re.compile(r"^\.\d{3}$")       # Bruker NanoScope numbered files (.000, .001, ...)


@dataclass
class RawLayer:
    """One 2-D channel of a raw file (metadata; data are loaded on demand)."""
    lid: int                 # index in RawLibrary.layers
    file: Path
    rel: str                 # path relative to its raw root (reported)
    channel: str
    index: int               # layer index inside the file
    unit: str | None         # canonical base unit of the stored values (m, deg, Hz, V, A, ...)
    unit_source: str         # 'channel-name' | 'note' | 'header' | 'file' | 'none'
    shape: tuple             # display-orientation shape (rows, cols)
    phys: tuple | None       # (height, width) of the displayed image in metres, if known
    fmt: str
    meta: dict = field(default_factory=dict)


# ---- Igor binary wave (Asylum Research) ------------------------------------
_AR_UNIT_RULES = [
    (re.compile(r"^(nap)?(height|zsensor|z ?sensor|topo\w*|amplitude\d*|amp\d*|mag\d*|deflection|defl|zdrive|"
                r"heightretrace|heighttrace)", re.I), "m"),
    (re.compile(r"^(nap)?phase", re.I), "deg"),
    (re.compile(r"^(nap)?(frequency|freq|drivefreq)", re.I), "Hz"),
    (re.compile(r"^(nap)?(potential|surfacepotential|userin\d*|bias|lateral|drive|voltage|input\w*|"
                r"inphase|quadrature|real|imag)", re.I), "V"),
    (re.compile(r"^(nap)?(current|orca|tuna|cafm)", re.I), "A"),
]


def ar_channel_unit(channel, note_kv=None, header_unit=None):
    """Unit of an Asylum Research channel from its name ('Amplitude1Retrace' -> 'm').

    The .ibw header carries a single data unit for the whole layer stack, so the
    per-channel unit follows the AR channel type; the AR note overrides it for
    user inputs ('UserIn0Unit: V'). Returns (unit, source)."""
    name = re.sub(r"(Re)?[Tt]race$", "", channel or "").strip()
    if note_kv:
        for key in (f"{name}Unit", f"{name.replace('Nap', '')}Unit"):
            if note_kv.get(key):
                f, b = parse_unit(note_kv[key])
                if b:
                    return b, "note"
    # DART SHO results computed by the AR software carry a "DAC" prefix
    # (DACAmplitude, DACPhase, DACFrequency, DACQFactor; shown as AmD/PhD/FrD/QD in AR)
    rule_name = name[3:] if re.match(r"(?i)^dac[a-z]", name) else name
    if re.match(r"(?i)^q(factor)?$", rule_name):
        return None, "channel-name"                      # quality factor: dimensionless
    for rx, unit in _AR_UNIT_RULES:
        if rx.match(rule_name):
            return unit, "channel-name"
    if header_unit:
        f, b = parse_unit(header_unit)
        if b:
            return b, "header"
    return None, "none"


def _ibw_note_kv(note_bytes):
    txt = note_bytes.decode("latin-1", "replace") if isinstance(note_bytes, (bytes, bytearray)) else str(note_bytes)
    kv = {}
    for ln in re.split(r"[\r\n]+", txt):
        k, sep, v = ln.partition(":")
        if sep and k.strip() and k.strip() not in kv:
            kv[k.strip()] = v.strip()
    return kv


def _ibw_layer_labels(raw_bytes, wave, n_layers):
    """Labels of the layer dimension (dim 2). Parsed from the raw bytes because
    igor2 concatenates the 32-byte label fields without stopping at the NUL
    terminator, which leaves stale characters in some names ('Phase1Tracece')."""
    bh = wave.get("bin_header", {})
    sizes = bh.get("dimLabelsSize")
    labels = None
    try:
        if sizes is not None and int(np.sum(sizes)) > 0:
            off = 64 + int(bh["wfmSize"]) + int(bh.get("formulaSize", 0)) + int(bh.get("noteSize", 0)) + \
                int(bh.get("dataEUnitsSize", 0)) + int(np.sum(bh.get("dimEUnitsSize", [0, 0, 0, 0])))
            per_dim = []
            for size in [int(s) for s in sizes]:
                chunk = raw_bytes[off:off + size]
                per_dim.append([chunk[i:i + 32].split(b"\0")[0].decode("latin-1") for i in range(0, size, 32)])
                off += size
            if len(per_dim) > 2 and len(per_dim[2]) > 1:
                labels = per_dim[2][1:]
    except Exception:  # noqa: BLE001 - fall back to igor2's labels
        labels = None
    if labels is None:
        try:
            lab = wave.get("labels", [[], [], [], []])[2]
            labels = [x.decode("latin-1") if isinstance(x, bytes) else str(x) for x in lab][1:]
        except Exception:  # noqa: BLE001
            labels = []
    labels = [s.strip() for s in labels]
    return [labels[k] if k < len(labels) and labels[k] else f"layer{k}" for k in range(n_layers)]


def load_ibw(path):
    """-> list of (channel, display-oriented float array in SI units, unit, unit_source, phys, meta)."""
    import igor2.binarywave as bw
    raw_bytes = Path(path).read_bytes()
    d = bw.load(str(path))
    w = d["wave"]
    a = np.asarray(w["wData"])
    if a.dtype.kind not in "fiu" or a.ndim < 2:
        return []
    if a.ndim == 2:
        a = a[:, :, None]
    if a.ndim == 4:
        if a.shape[3] > 1:
            return []                       # 4-D waves (force-map volumes) are not images
        a = a[:, :, :, 0]
    if min(a.shape[:2]) < 8:
        return []                           # 1-D traces / force curves
    wh = w["wave_header"]
    note_kv = _ibw_note_kv(w.get("note", b""))

    def _units(arr):
        try:
            return b"".join(np.asarray(arr).ravel().tolist()).decode("latin-1").strip("\0 ")
        except Exception:  # noqa: BLE001
            return ""
    header_unit = _units(wh.get("dataUnits", b"")) or (w.get("data_units") or b"").decode("latin-1", "replace")
    sfA = np.asarray(wh.get("sfA", [1, 1, 1, 1]), dtype=float)
    dim_units = wh.get("dimUnits")
    xy_unit = _units(dim_units[0]) if dim_units is not None and len(dim_units) else ""
    phys = None
    if xy_unit in ("m", "") and sfA[0] > 0 and sfA[1] > 0 and sfA[0] < 1e-3:
        phys = (a.shape[1] * sfA[1], a.shape[0] * sfA[0])      # display (height = y, width = x)
    labels = _ibw_layer_labels(raw_bytes, w, a.shape[2])
    out = []
    for k in range(a.shape[2]):
        disp = np.flipud(a[:, :, k].T).astype(np.float64)       # Igor/AR display orientation
        unit, src = ar_channel_unit(labels[k], note_kv, header_unit)
        meta = {"imaging_mode": note_kv.get("ImagingMode"), "scan_size": note_kv.get("ScanSize")}
        try:
            off, rng = float(note_kv.get(f"Display Offset {k}")), float(note_kv.get(f"Display Range {k}"))
            meta.update({"ar_display_lo": off - rng / 2, "ar_display_hi": off + rng / 2})
        except (TypeError, ValueError):
            pass
        for key, name in ((f"ColorMap {k}", "ar_colormap"), (f"FlattenOrder {k}", "ar_flatten_order"),
                          (f"PlanefitOrder {k}", "ar_planefit_order")):
            if note_kv.get(key) is not None:
                meta[name] = note_kv[key]
        out.append((labels[k], disp, unit, src, phys, meta))
    return out


# ---- simple numeric formats -------------------------------------------------
def load_gsf(path):
    """Gwyddion Simple Field: text header, NUL padding to a multiple of 4, float32 LE data."""
    raw = Path(path).read_bytes()
    if not raw.startswith(b"Gwyddion Simple Field 1.0"):
        return []
    end = raw.index(b"\0")
    kv = {}
    for ln in raw[:end].decode("utf-8", "replace").splitlines()[1:]:
        k, sep, v = ln.partition("=")
        if sep:
            kv[k.strip()] = v.strip()
    xres, yres = int(kv["XRes"]), int(kv["YRes"])
    start = (end // 4 + 1) * 4
    data = np.frombuffer(raw[start:start + 4 * xres * yres], dtype="<f4").reshape(yres, xres).astype(np.float64)
    f, b = parse_unit(kv.get("ZUnits"))
    data = data * (f if f else 1.0)
    phys = None
    xf, xb = parse_unit(kv.get("XYUnits", "m"))
    try:
        if xb == "m":
            phys = (float(kv["YReal"]) * xf, float(kv["XReal"]) * xf)
    except (KeyError, ValueError):
        pass
    return [(kv.get("Title", "gsf"), data, b, "file" if b else "none", phys, {})]


def load_text_matrix(path, max_mb=60):
    """Plain numeric matrix (whitespace/comma/semicolon/tab separated); '#'/'%'
    comment lines of a Gwyddion ASCII export give the channel and value unit."""
    if Path(path).stat().st_size > max_mb * 2 ** 20:
        return []
    txt = Path(path).read_text(encoding="utf-8", errors="replace")
    lines = txt.splitlines()
    comments = [ln.lstrip("#% ").strip() for ln in lines if ln.lstrip().startswith(("#", "%"))]
    body = [ln for ln in lines if ln.strip() and not ln.lstrip().startswith(("#", "%"))]
    if len(body) < 8:
        return []
    first = body[0]
    delim = "," if "," in first else (";" if ";" in first else ("\t" if "\t" in first else None))
    try:
        data = np.array([[float(v) for v in (ln.split(delim) if delim else ln.split()) if v.strip()]
                         for ln in body], dtype=np.float64)
    except ValueError:
        return []
    if data.ndim != 2 or min(data.shape) < 8:
        return []
    unit, channel, phys = None, Path(path).stem, None
    for c in comments:
        k, _, v = c.partition(":")
        k = k.strip().lower()
        if k in ("value units", "z units", "zunits", "unit", "units"):
            f, b = parse_unit(v.strip())
            if b:
                data, unit = data * f, b
        elif k == "channel":
            channel = v.strip()
    return [(channel, data, unit, "file" if unit else "none", phys, {})]


def load_numpy(path):
    p = Path(path)
    if p.suffix.lower() == ".npy":
        items = [(p.stem, np.load(p, allow_pickle=False))]
    else:
        with np.load(p, allow_pickle=False) as z:
            items = [(k, z[k]) for k in z.files]
    out = []
    for name, arr in items:
        arr = np.asarray(arr)
        if arr.dtype.kind not in "fiu":
            continue
        if arr.ndim == 2 and min(arr.shape) >= 8:
            out.append((name, arr.astype(np.float64), None, "none", None, {}))
        elif arr.ndim == 3 and min(arr.shape[:2]) >= 8 and arr.shape[2] <= 32:
            out += [(f"{name}[{k}]", arr[:, :, k].astype(np.float64), None, "none", None, {}) for k in range(arr.shape[2])]
    return out


def load_tiff(path):
    from PIL import Image
    im = Image.open(path)
    tags = getattr(im, "tag_v2", {})
    if 50432 in tags or 50434 in tags:
        raise ValueError("Park Systems TIFF (data in private tags) - export it as .gsf from Gwyddion")
    if im.mode not in ("F", "I", "I;16", "I;16B", "I;16L", "I;32"):
        raise ValueError(f"TIFF mode {im.mode} is a picture, not a data matrix")
    arr = np.array(im, dtype=np.float64)
    return [(Path(path).stem, arr, None, "none", None, {})]


LOADERS = {".ibw": load_ibw, ".gsf": load_gsf, ".txt": load_text_matrix, ".asc": load_text_matrix,
           ".csv": load_text_matrix, ".dat": load_text_matrix, ".npy": load_numpy, ".npz": load_numpy,
           ".tif": load_tiff, ".tiff": load_tiff}


# ================================================= processing variants
def _fill(a):
    """NaN/inf -> median (processing and resampling need finite arrays)."""
    m = np.isfinite(a)
    if m.all():
        return a
    out = a.copy()
    out[~m] = np.median(a[m]) if m.any() else 0.0
    return out


def v_raw(a):
    return a


def v_plane(a):
    h, w = a.shape
    yy, xx = np.mgrid[0:h, 0:w]
    m = np.isfinite(a)
    A = np.stack([np.ones(int(m.sum())), xx[m], yy[m]], 1)
    c, *_ = np.linalg.lstsq(A, a[m], rcond=None)
    return a - (c[0] + c[1] * xx + c[2] * yy)


def v_flat0(a):
    return a - np.nanmean(a, axis=1, keepdims=True)


def v_flat1(a):
    h, w = a.shape
    x = np.arange(w) - (w - 1) / 2.0
    if np.isfinite(a).all():
        mean = a.mean(axis=1, keepdims=True)
        slope = ((a - mean) * x).sum(axis=1, keepdims=True) / (x * x).sum()
        return a - mean - slope * x
    out = np.full_like(a, np.nan)
    for i in range(h):
        m = np.isfinite(a[i])
        if m.sum() >= 2:
            c = np.polyfit(x[m], a[i, m], 1)
            out[i] = a[i] - np.polyval(c, x)
    return out


VARIANTS = {"raw": v_raw, "plane": v_plane, "flat0": v_flat0, "flat1": v_flat1}
VARIANT_PRIORITY = {"raw": 0, "plane": 1, "flat0": 2, "flat1": 3}


# ========================================================= orientations
ORIENTATIONS = ("igor", "flipud", "fliplr", "rot180", "transpose", "rot90ccw", "rot90cw", "antitranspose")
TRANSPOSING = {"transpose", "rot90ccw", "rot90cw", "antitranspose"}
# for .ibw files: the same orientation expressed as a matplotlib call on the stored matrix M = wData[:, :, k]
IBW_ORIENT_DESC = {"igor": "Igor/AR view = imshow(M.T, origin='lower')",
                   "flipud": "imshow(M.T) (origin upper)",
                   "fliplr": "Igor view mirrored left-right",
                   "rot180": "Igor view rotated 180 deg",
                   "transpose": "Igor view transposed",
                   "rot90ccw": "imshow(M, origin='lower')",
                   "rot90cw": "imshow(M) (origin upper)",
                   "antitranspose": "Igor view anti-transposed"}


def orient(a, o):
    """Apply one of the eight dihedral transforms to the last two axes."""
    if o in TRANSPOSING:
        a = np.swapaxes(a, -1, -2)
        o = {"transpose": "igor", "rot90ccw": "flipud", "rot90cw": "fliplr", "antitranspose": "rot180"}[o]
    if o == "flipud":
        return a[..., ::-1, :]
    if o == "fliplr":
        return a[..., :, ::-1]
    if o == "rot180":
        return a[..., ::-1, ::-1]
    return a


def orient_inverse(a, o):
    inv = {"rot90ccw": "rot90cw", "rot90cw": "rot90ccw"}.get(o, o)
    return orient(a, inv)


def oriented_shape(shape, o):
    return (shape[1], shape[0]) if o in TRANSPOSING else tuple(shape[:2])


# ============================================================ resampling
def area_matrix(n_in, n_out):
    """(n_out x n_in) area-averaging resampling matrix (box filter)."""
    e_in = np.arange(n_in + 1) / n_in
    e_out = np.arange(n_out + 1) / n_out
    R = np.maximum(0.0, np.minimum(e_out[1:, None], e_in[None, 1:]) - np.maximum(e_out[:-1, None], e_in[None, :-1]))
    return (R / R.sum(axis=1, keepdims=True)).astype(np.float32)


def resample(a, shape):
    """Resize a 2-D array to (rows, cols); area averaging when shrinking."""
    h, w = shape
    if a.shape == (h, w):
        return a.astype(np.float32, copy=False)
    interp = cv2.INTER_AREA if (h <= a.shape[0] and w <= a.shape[1]) else cv2.INTER_LINEAR
    return cv2.resize(np.ascontiguousarray(a, dtype=np.float32), (int(w), int(h)), interpolation=interp)


def resample_masked(a, valid, shape, min_cover=0.5):
    """Masked area average: cells are valid when >= min_cover of their area is valid."""
    num = resample(np.where(valid, a, 0.0), shape)
    den = resample(valid.astype(np.float32), shape)
    out = np.where(den > 1e-6, num / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    return out, den >= min_cover


def masked_r(x, y, m):
    """Pearson r of x and y over mask m (nan if degenerate)."""
    if m.sum() < 10:
        return np.nan
    a, b = x[m].astype(np.float64), y[m].astype(np.float64)
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / den) if den > 0 else np.nan


# ============================================================ raw library
def discover_raw_files(roots):
    """Recursive listing: (supported files, {file: reason} for skipped ones)."""
    files, skipped = [], {}
    for root in roots:
        root = Path(root)
        if not root.exists():
            continue
        for p in sorted(root.rglob("*")):
            if not p.is_file():
                continue
            ext = p.suffix.lower()
            if ext in SUPPORTED_EXT:
                files.append((root, p))
            elif ext in KNOWN_UNSUPPORTED or _BRUKER_NUM.match(ext):
                skipped[str(p)] = f"unsupported format ({KNOWN_UNSUPPORTED.get(ext, 'Bruker NanoScope')})"
            elif ext not in (".png", ".jpg", ".jpeg", ".pdf", ".xlsx", ".xls", ".docx", ".pptx", ".md", ".json",
                             ".ini", ".log", ".zip"):
                skipped[str(p)] = "unknown file type"
    return files, skipped


class _LRU:
    """Least-recently-used cache with a memory budget (bytes of the numpy arrays it holds)."""

    def __init__(self, max_bytes):
        self.max_bytes, self.nbytes, self.d = max_bytes, 0, OrderedDict()

    @staticmethod
    def _size(v):
        if isinstance(v, np.ndarray):
            return v.nbytes
        return sum(it[1].nbytes for it in v if isinstance(it, tuple) and isinstance(it[1], np.ndarray))

    def get(self, key):
        if key in self.d:
            self.d.move_to_end(key)
            return self.d[key]
        return None

    def put(self, key, value):
        self.d[key] = value
        self.nbytes += self._size(value)
        while self.nbytes > self.max_bytes and len(self.d) > 1:
            _, old = self.d.popitem(last=False)
            self.nbytes -= self._size(old)
        return value


class RawLibrary:
    """All 2-D layers of the raw files under one or more root folders.

    Layer metadata are read once (``scan``); arrays are re-read on demand
    through memory-limited LRU caches (loaded files, processed variants)."""

    def __init__(self, roots, cache_mb=1500, log=print):
        self.roots = [Path(r) for r in roots]
        self.layers: list[RawLayer] = []
        self.skipped: dict = {}
        self._files = _LRU(0.4 * cache_mb * 2 ** 20)
        self._vars = _LRU(0.6 * cache_mb * 2 ** 20)
        self.log = log

    # ---------------------------------------------------------------- loading
    def _load_file(self, path):
        key = str(path)
        items = self._files.get(key)
        if items is None:
            items = self._files.put(key, LOADERS[Path(path).suffix.lower()](path))
        return items

    def scan(self):
        files, self.skipped = discover_raw_files(self.roots)
        t0 = time.time()
        for root, p in files:
            try:
                items = self._load_file(p)
            except Exception as e:  # noqa: BLE001
                self.skipped[str(p)] = f"read error: {type(e).__name__}: {str(e)[:120]}"
                continue
            if not items:
                self.skipped[str(p)] = "no 2-D image layer"
                continue
            try:
                rel = str(p.relative_to(root))
            except ValueError:
                rel = p.name
            for k, (ch, arr, unit, src, phys, meta) in enumerate(items):
                fin = np.isfinite(arr)
                if fin.mean() < 0.5 or np.ptp(arr[fin]) == 0:
                    continue                         # empty / constant layer
                self.layers.append(RawLayer(len(self.layers), p, rel, ch, k, unit, src, arr.shape, phys,
                                            p.suffix.lower().lstrip("."), dict(meta, nan_frac=float(1 - fin.mean()))))
        self.log(f"  raw library: {len(files)} files, {len(self.layers)} layers "
                 f"({len(self.skipped)} skipped) in {time.time() - t0:.1f}s")
        return self

    def array(self, lid):
        """Display-oriented layer in SI units (may contain NaN)."""
        L = self.layers[lid]
        return self._load_file(L.file)[L.index][1]

    def variant(self, lid, v):
        """Processed variant (NaN filled) of a layer, display orientation, SI units."""
        out = self._vars.get((lid, v))
        if out is None:
            a = self.array(lid)
            out = self._vars.put((lid, v), _fill(VARIANTS[v](a) if v != "raw" else a))
        return out


@dataclass
class Candidate:
    cid: int
    lid: int
    variant: str


# ------------------------------------------------------- robust "detail" maps
def detrend(a, m=None):
    """Remove a least-squares plane (fitted on the cells of mask m)."""
    h, w = a.shape
    if m is None or m.all():
        # full grid: centred x and y regressors are orthogonal -> closed form
        a = a.astype(np.float64) - a.mean()
        xc, yc = np.arange(w) - (w - 1) / 2.0, np.arange(h) - (h - 1) / 2.0
        bx = (a * xc[None, :]).sum() / max(1e-12, h * (xc * xc).sum())
        by = (a * yc[:, None]).sum() / max(1e-12, w * (yc * yc).sum())
        return (a - bx * xc[None, :] - by * yc[:, None]).astype(np.float32)
    yy, xx = np.mgrid[0:h, 0:w]
    A = np.stack([np.ones(int(m.sum())), xx[m] / w, yy[m] / h], 1)
    c, *_ = np.linalg.lstsq(A, a[m].astype(np.float64), rcond=None)
    return (a - (c[0] + c[1] * xx / w + c[2] * yy / h)).astype(np.float32)


def winsorize(a, m=None, q=(1.0, 99.0)):
    v = a[m] if m is not None else a
    if v.size == 0:
        return a
    lo, hi = np.percentile(v, q)
    return np.clip(a, lo, hi).astype(np.float32)


def detail(a, m=None, q=(1.0, 99.0)):
    """Identification transform: plane removed, then winsorised. A tilt plane
    (common to many raw height scans) and isolated outliers (spikes in the
    raw data, scale bars or labels mapped onto the colour bar in the figure)
    would otherwise dominate the correlation."""
    return winsorize(detrend(a, m), m, q)


class CandidateBank:
    """Mid-resolution (M x M, stretched to square) "detail" copies of every
    candidate (layer x processing variant) for the coarse full-scan screening,
    plus robust value statistics (SI units) of each candidate.

    'raw' and 'plane' have identical detail maps (the plane is removed for
    identification anyway), so the bank holds raw, flat0 and flat1; the plane
    variant is re-introduced in the fine stage."""

    def __init__(self, lib: RawLibrary, variants=("raw", "flat0", "flat1"), mid=64, log=print):
        self.lib, self.mid = lib, mid
        self.cands: list[Candidate] = []
        mids, stats = [], []
        t0 = time.time()
        by_file = {}
        for L in lib.layers:
            by_file.setdefault(str(L.file), []).append(L)
        for _, layers in by_file.items():         # file by file: each file is read once
            for L in layers:
                for v in variants:
                    a = lib.variant(L.lid, v)
                    self.cands.append(Candidate(len(self.cands), L.lid, v))
                    mids.append(detail(resample(a, (mid, mid)), q=(2.0, 98.0)))
                    stats.append(np.percentile(a, [0, 0.5, 1, 25, 50, 75, 99, 99.5, 100]))
        self.mids = np.stack(mids).astype(np.float32) if mids else np.zeros((0, mid, mid), np.float32)
        self.stats = np.array(stats) if stats else np.zeros((0, 9))
        self.lids = np.array([c.lid for c in self.cands], dtype=int)
        self.files = np.array([str(lib.layers[c.lid].file) for c in self.cands])
        self.variants = np.array([c.variant for c in self.cands])
        log(f"  candidate bank: {len(self.cands)} candidates ({len(lib.layers)} layers x {len(variants)} variants) "
            f"in {time.time() - t0:.1f}s")

    def describe(self, cid):
        c = self.cands[cid]
        L = self.lib.layers[c.lid]
        return {"file": L.rel, "channel": L.channel, "variant": c.variant, "lid": c.lid}


# ================================================================ matching
@dataclass
class MatchParams:
    coarse_grid: int = 48           # long side of the coarse comparison grid (cells)
    coarse_border: float = 0.04     # fraction of the panel ignored at each edge in the coarse stage
    n_groups: int = 12              # (file, orientation) groups re-scored at the fine level
    min_files: int = 3              # ... covering at least this many files (chance level / runner-up)
    n_full_files: int = 3           # leading files whose every channel/variant is re-scored
    crop_expand_files: int = 5      # ... the same for the crop stage
    max_shift_frac: float = 0.04    # integer-shift search range (fraction of the grid size); also
                                    # absorbs the trace/retrace lateral offset (piezo hysteresis)
    fine_max: int = 384             # cap of the fine comparison grid (long side, px)
    crop_trigger: float = 0.85      # skip the crop search if the full-scan score reaches this ...
    crop_skip_contrast: float = 0.25  # ... or stands out from the chance level by this much
    crop_margin: float = 0.03       # a crop must beat the full-scan score by this much to be preferred
    crop_zooms: tuple = (0.9, 0.76, 0.64, 0.54, 0.46, 0.39, 0.33, 0.28, 0.24, 0.2)
    crop_template: int = 20         # long side of the coarse crop template (cells)
    crop_top: int = 16              # coarse crop hypotheses ((file, orientation) groups) refined
    crop_variants: tuple = ("raw", "flat1")
    crop_max_candidates: int = 20000
    ambiguous_margin: float = 0.05  # score margin below which another FILE is "ambiguous"
    channel_margin: float = 0.03    # score margin below which another channel of the same file is "ambiguous"
    variant_tie: float = 0.01       # variants within this plain r of the best are tied
    min_score: float = 0.25         # a match needs at least this identification score ...
    min_contrast: float = 0.08      # ... and this margin over the chance level (unrelated hypotheses)


def panel_grid_shape(shape, long_side):
    h, w = shape
    if w >= h:
        return int(np.clip(round(long_side * h / w), 4, long_side)), long_side
    return long_side, int(np.clip(round(long_side * w / h), 4, long_side))


def coarse_full_scan(bank: CandidateBank, vm, valid, allowed=None, grid=48, border=0.04):
    """Coarse full-scan screening: Pearson r between the panel's detail map and
    every candidate's detail map stretched to the panel grid, for all eight
    orientations. Returns r[n_cand, 8] (nan for excluded candidates)."""
    gh, gw = panel_grid_shape(vm.shape, grid)
    P, M = resample_masked(vm, valid, (gh, gw))
    b_h, b_w = max(1, int(round(border * gh))), max(1, int(round(border * gw)))
    m = np.zeros_like(M)
    m[b_h:gh - b_h, b_w:gw - b_w] = True
    m &= M
    out = np.full((len(bank.cands), len(ORIENTATIONS)), np.nan, dtype=np.float32)
    if m.sum() < 20 or len(bank.cands) == 0:
        return out
    idx = np.arange(len(bank.cands)) if allowed is None else np.flatnonzero(allowed)
    if len(idx) == 0:
        return out
    yy, xx = np.mgrid[0:gh, 0:gw]
    Q, _ = np.linalg.qr(np.stack([np.ones(int(m.sum())), xx[m] / gw, yy[m] / gh], 1))
    p = detail(P, m, q=(2.0, 98.0))[m].astype(np.float64)
    p -= Q @ (Q.T @ p)
    pn = np.linalg.norm(p)
    if pn == 0:
        return out
    Mid = bank.mid
    for chunk in np.array_split(idx, max(1, len(idx) // 4000 + 1)):
        mids = bank.mids[chunk]
        # straight orientations: rows -> gh, cols -> gw ; transposing ones: rows -> gw, cols -> gh
        C1 = np.matmul(np.matmul(area_matrix(Mid, gh), mids), area_matrix(Mid, gw).T)
        C2 = np.matmul(np.matmul(area_matrix(Mid, gw), mids), area_matrix(Mid, gh).T)
        for k, o in enumerate(ORIENTATIONS):
            X = orient(C2 if o in TRANSPOSING else C1, o)[:, m].astype(np.float64)
            X -= (X @ Q) @ Q.T
            xn = np.linalg.norm(X, axis=1)
            r = (X @ p) / np.maximum(xn * pn, 1e-30)
            r[xn == 0] = 0.0
            out[chunk, k] = r
    return out


def _fine_grid(panel_shape, region_shape, cap):
    hc = min(panel_shape[0], region_shape[0])
    wc = min(panel_shape[1], region_shape[1])
    s = min(1.0, cap / max(hc, wc))
    return max(8, int(round(hc * s))), max(8, int(round(wc * s)))


class PanelFine:
    """The panel prepared for fine comparisons with regions of a given native
    shape (cached per comparison grid)."""

    def __init__(self, vm, valid, params: MatchParams):
        self.vm, self.valid, self.p = vm.astype(np.float32), valid.astype(bool), params
        self._cache = {}

    def at(self, grid):
        if grid not in self._cache:
            P, M = resample_masked(self.vm, self.valid, grid)
            s = max(2, int(round(self.p.max_shift_frac * max(grid))))
            D = detail(P, M) if M.sum() >= 30 else P
            W = winsorize(P, M) if M.sum() >= 30 else P
            self._cache[grid] = (P, M, D, W, s)
        return self._cache[grid]


def _match_template(img, T, Tm):
    """Masked TM_CCOEFF_NORMED (= masked Pearson r at every position)."""
    img = np.ascontiguousarray(img, np.float32)
    T, Tm = np.ascontiguousarray(T, np.float32), np.ascontiguousarray(Tm, np.float32)
    try:
        res = cv2.matchTemplate(img, T, cv2.TM_CCOEFF_NORMED, mask=Tm)
    except cv2.error:
        # OpenCV 4.10 fails for some result shapes (e.g. 4 x 1) with a mask: pad the image by one
        # replicated row and column (positions of the original result are unaffected) and crop back
        rh, rw = img.shape[0] - T.shape[0] + 1, img.shape[1] - T.shape[1] + 1
        pad = cv2.copyMakeBorder(img, 0, 1, 0, 1, cv2.BORDER_REPLICATE)
        res = cv2.matchTemplate(pad, T, cv2.TM_CCOEFF_NORMED, mask=Tm)[:rh, :rw]
    return np.nan_to_num(res, nan=-1.0, posinf=-1.0, neginf=-1.0)


def fine_compare(pf: PanelFine, A, shift=None, search=1):
    """Compare the panel with a raw region A that shows the same field of view.

    Both are resampled to the coarser of the two native grids. The integer
    shift (grid pixels, A relative to the panel) maximising the identification
    score is searched within +/- max_shift_frac of the grid, or within
    +/- ``search`` around a given ``shift``.

    Returns dict(score = detail r, r_plain = winsorised r without detrending,
    shift, grid)."""
    grid = _fine_grid(pf.vm.shape, A.shape, pf.p.fine_max)
    P, M, D, W, s = pf.at(grid)
    hc, wc = grid
    T_m = M[s:hc - s, s:wc - s]
    if T_m.sum() < 30 or hc - 2 * s < 4 or wc - 2 * s < 4:
        return {"score": np.nan, "r_plain": np.nan, "shift": (0, 0), "grid": grid}
    Ac = resample(A, grid)
    Ad = detail(Ac)
    if shift is None:
        y0, x0, img = 0, 0, Ad
    else:
        y0, x0 = max(0, s + shift[0] - search), max(0, s + shift[1] - search)
        y1, x1 = min(hc, s + shift[0] + search + hc - 2 * s), min(wc, s + shift[1] + search + wc - 2 * s)
        img = Ad[y0:y1, x0:x1]
        if img.shape[0] < hc - 2 * s or img.shape[1] < wc - 2 * s:
            y0, x0, img = 0, 0, Ad
    res = _match_template(img, D[s:hc - s, s:wc - s], T_m)
    j = int(np.argmax(res))
    yy, xx = divmod(j, res.shape[1])
    dy, dx = yy + y0 - s, xx + x0 - s
    win = winsorize(Ac)[s + dy:s + dy + hc - 2 * s, s + dx:s + dx + wc - 2 * s]
    r_plain = masked_r(W[s:hc - s, s:wc - s], win, T_m) if win.shape == T_m.shape else np.nan
    return {"score": float(res[yy, xx]), "r_plain": r_plain, "shift": (int(dy), int(dx)), "grid": grid}


def region_of(lib: RawLibrary, lid, variant, orientation, crop=None):
    """Raw region shown by a panel, SI units: processing variant -> crop
    (x0, y0, x1, y1) in pixels of the display-oriented layer -> orientation."""
    a = lib.variant(lid, variant)
    if crop is not None:
        x0, y0, x1, y1 = [int(round(v)) for v in crop]
        a = a[max(0, y0):max(y0 + 1, y1), max(0, x0):max(x0 + 1, x1)]
    return np.ascontiguousarray(orient(a, orientation))


def _plausible_for_crop(bank: CandidateBank, unit_factor, unit_base, span_si):
    """Unit / magnitude prior used to limit the (expensive) crop search."""
    ok = np.ones(len(bank.cands), bool)
    if unit_base is None:
        return ok
    for c in bank.cands:
        L = bank.lib.layers[c.lid]
        if L.unit is None or L.unit == "V" and unit_base != "V":
            continue                                   # unknown / uncalibrated: keep
        if L.unit != unit_base:
            ok[c.cid] = False
            continue
        st = bank.stats[c.cid]
        cspan = st[7] - st[1]                          # p99.5 - p0.5 (SI)
        if span_si is not None and cspan > 0 and not (0.01 <= span_si / cspan <= 3.0):
            ok[c.cid] = False
    return ok


def _box_from(x, y, tw, th, Wz, Hz, W, H):
    return (x / Wz * W, y / Hz * H, (x + tw) / Wz * W, (y + th) / Hz * H)


def crop_search(bank: CandidateBank, pf: PanelFine, allowed, params: MatchParams):
    """Find the panel as a zoomed sub-region of a candidate.

    Coarse stage: the panel detail map (long side ``crop_template`` cells) in
    each of the eight orientations is matched (normalised cross-correlation)
    against every allowed candidate resized so that the template covers a
    fraction z of its width, for z in ``crop_zooms``. The best hypothesis of
    each (file, orientation) group - at least ``min_files`` files - is
    refined by a local scale/position search and scored with
    ``fine_compare`` on the crop. Returns hypothesis dicts, best first."""
    lib = bank.lib
    th0, tw0 = panel_grid_shape(pf.vm.shape, params.crop_template)
    P0, M0 = resample_masked(pf.vm, pf.valid, (th0, tw0))
    if M0.sum() < 0.5 * M0.size:
        return []
    # invalid cells set to 0 (the detail map is ~zero-mean): unmasked NCC is ~3x faster
    D0 = np.where(M0, detail(P0, M0, q=(2.0, 98.0)), 0.0).astype(np.float32)
    templates = {o: orient_inverse(D0, o) for o in ORIENTATIONS}
    sel = [c for c in bank.cands if (allowed is None or allowed[c.cid]) and c.variant in params.crop_variants]
    sel = sel[:params.crop_max_candidates]
    hyps = []
    for c in sel:
        A = lib.variant(c.lid, c.variant)
        H, W = A.shape
        resized = {}
        for o in ORIENTATIONS:
            T = templates[o]
            th, tw = T.shape
            best = None
            for z in params.crop_zooms:
                Wz = int(round(tw / z))
                Hz = int(round(Wz * H / W))
                if Hz < th + 1 or Wz < tw + 1:
                    continue
                if (Hz, Wz) not in resized:
                    resized[(Hz, Wz)] = detail(resample(A, (Hz, Wz)), q=(2.0, 98.0))
                res = cv2.matchTemplate(np.ascontiguousarray(resized[(Hz, Wz)], np.float32),
                                        np.ascontiguousarray(T, np.float32), cv2.TM_CCOEFF_NORMED)
                j = int(np.argmax(res))
                yy, xx = divmod(j, res.shape[1])
                v = float(res[yy, xx]) if np.isfinite(res[yy, xx]) else -1.0
                if best is None or v > best[0]:
                    best = (v, z, _box_from(xx, yy, tw, th, Wz, Hz, W, H))
            if best is not None:
                hyps.append({"cid": c.cid, "lid": c.lid, "variant": c.variant, "orientation": o,
                             "coarse": best[0], "zoom": best[1], "box": best[2]})
    hyps.sort(key=lambda h: -h["coarse"])
    seen, files, top = set(), set(), []
    for h in hyps:
        f = str(lib.layers[h["lid"]].file)
        key = (f, h["orientation"])
        if key in seen:
            continue
        if len(top) >= params.crop_top:
            if len(files) >= params.min_files:
                break
            if f in files:
                continue
        seen.add(key)
        files.add(f)
        top.append(h)
    out = []
    for h in top:
        r = refine_crop(lib, pf, h["lid"], h["variant"], h["orientation"], h["box"])
        if r is not None:
            out.append({**h, **r})
    out.sort(key=lambda h: -np.nan_to_num(h["score"], nan=-2))
    return out


CROP_ROUNDS = (((0.88, 0.94, 1.0, 1.06, 1.12), 96, 0.15), ((0.97, 0.985, 1.0, 1.015, 1.03), 192, 0.05))
CROP_FINE_ROUNDS = (((0.97, 0.985, 1.0, 1.015, 1.03), 192, 0.05),)
FULL_ROUNDS = (((0.95, 0.97, 0.985, 1.0), 128, 0.03), ((0.99, 0.995, 1.0, 1.005, 1.01), 256, 0.015))


def refine_crop(lib, pf: PanelFine, lid, variant, o, box, rounds=CROP_ROUNDS):
    """Local scale/position refinement of the field of view shown by a panel
    (``box`` in pixels of the display-oriented layer), then fine scoring.

    Each round tries the given scale factors around the current box; for each,
    the panel (template, long side <= res px) is located by masked NCC inside
    the box enlarged by ``margin``."""
    A = lib.variant(lid, variant)
    H, W = A.shape
    Pinv, Minv = orient_inverse(pf.vm, o), orient_inverse(pf.valid, o)
    x0, y0, x1, y1 = box
    for scales, res_px, margin in rounds:
        th, tw = panel_grid_shape(Pinv.shape, int(min(res_px, max(Pinv.shape))))
        Tp, Tmask = resample_masked(Pinv, Minv, (th, tw))
        if Tmask.sum() < 30:
            return None
        Td = detail(Tp, Tmask)
        best = None
        cx, cy, bw, bh = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        for s in scales:
            w, h = bw * s, bh * s
            if w > W + 1e-6 or h > H + 1e-6:
                continue
            k = tw / w                                           # layer px -> template px
            mx, my = margin * w + 2, margin * h + 2
            rx0, ry0 = max(0, int(np.floor(cx - w / 2 - mx))), max(0, int(np.floor(cy - h / 2 - my)))
            rx1, ry1 = min(W, int(np.ceil(cx + w / 2 + mx))), min(H, int(np.ceil(cy + h / 2 + my)))
            Rh, Rw = int(round((ry1 - ry0) * k)), int(round((rx1 - rx0) * k))
            if Rh < th or Rw < tw:
                continue
            img = detail(resample(A[ry0:ry1, rx0:rx1], (Rh, Rw)))
            res = _match_template(img, Td, Tmask)
            j = int(np.argmax(res))
            yy, xx = divmod(j, res.shape[1])
            if best is None or res[yy, xx] > best[0]:
                bx0, by0 = rx0 + xx / k, ry0 + yy / k
                best = (float(res[yy, xx]), (bx0, by0, bx0 + tw / k, by0 + th / k))
        if best is None:
            return None
        x0, y0, x1, y1 = best[1]
    x0, y0 = max(0.0, x0), max(0.0, y0)
    x1, y1 = min(float(W), x1), min(float(H), y1)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None
    region = region_of(lib, lid, variant, o, (x0, y0, x1, y1))
    fc = fine_compare(pf, region)
    return {"score": fc["score"], "r_plain": fc["r_plain"], "shift": fc["shift"], "grid": fc["grid"],
            "crop": (x0, y0, x1, y1), "crop_frac": float(np.sqrt((x1 - x0) * (y1 - y0) / (W * H)))}


def _row(lib, lid, variant, o, mode, crop, crop_frac, coarse, fc):
    L = lib.layers[lid]
    return {"lid": lid, "file": L.rel, "path": str(L.file), "channel": L.channel, "variant": variant,
            "orientation": o, "mode": mode, "crop": crop, "crop_frac": crop_frac, "coarse": coarse,
            "score": fc["score"], "r_plain": fc["r_plain"], "shift": fc["shift"], "grid": fc["grid"]}


def _expand_file(bank, pf, ref, allowed, mode):
    """Every allowed channel x variant of the file of hypothesis ``ref``, with
    the same orientation and field of view. Each gets its own shift search:
    trace and retrace images are laterally offset by a few pixels (piezo
    hysteresis), so a shared alignment would favour the reference channel."""
    lib = bank.lib
    L0 = lib.layers[ref["lid"]]
    rows = []
    for L in lib.layers:
        if L.file != L0.file or L.shape != L0.shape:
            continue
        if allowed is not None and not _allowed_layer(bank, allowed, L.lid):
            continue
        for v in VARIANTS:
            if L.lid == ref["lid"] and v == ref["variant"]:
                continue
            fc = fine_compare(pf, region_of(lib, L.lid, v, ref["orientation"], ref["crop"]))
            rows.append(_row(lib, L.lid, v, ref["orientation"], mode, ref["crop"], ref["crop_frac"], np.nan, fc))
    return rows


def _resolve_file(bank, pf, ref, allowed, mode):
    """Score all channels/variants of ref's file; if another channel wins, the
    field of view is re-refined on that channel and the file re-scored once."""
    lib = bank.lib
    rows = [ref] + _expand_file(bank, pf, ref, allowed, mode)
    best = max(rows, key=lambda r: np.nan_to_num(r["score"], nan=-2))
    if best["lid"] == ref["lid"]:
        return rows
    H, W = lib.layers[best["lid"]].shape
    if mode == "full":
        rf = refine_crop(lib, pf, best["lid"], best["variant"], best["orientation"], (0, 0, W, H), FULL_ROUNDS)
    else:
        rf = refine_crop(lib, pf, best["lid"], best["variant"], best["orientation"], best["crop"], CROP_FINE_ROUNDS)
    if rf is None or not np.isfinite(rf["score"]) or rf["score"] <= best["score"]:
        return rows
    new = dict(best)
    new.update({k: rf[k] for k in ("score", "r_plain", "shift", "grid", "crop", "crop_frac")})
    if mode == "full" and new["crop_frac"] >= 0.999:
        new["crop"] = None
    return [new] + _expand_file(bank, pf, new, allowed, mode)


def _best_and_null(rows):
    """Best row and the best score of an unrelated hypothesis (another file,
    or the same file in another orientation): the empirical chance level."""
    rows = [r for r in rows if np.isfinite(r["score"])]
    if not rows:
        return None, np.nan
    best = max(rows, key=lambda r: r["score"])
    others = [r["score"] for r in rows if r["path"] != best["path"] or r["orientation"] != best["orientation"]]
    return best, (max(others) if others else 0.0)


def match_panel(bank: CandidateBank, vm, valid, params: MatchParams = None, allowed=None,
                unit=(None, None), span_si=None):
    """Identify the raw scan, channel, processing variant, orientation and
    field of view shown by a reconstructed panel.

    Parameters
    ----------
    bank : CandidateBank of the paper's raw files
    vm, valid : reconstructed value map and its valid mask
    allowed : optional boolean mask over bank candidates (hints)
    unit : (factor, base) of the colour-bar unit (crop-search prior only)
    span_si : vmax - vmin of the colour bar in SI units (crop-search prior only)

    Returns a dict: ``best`` (hypothesis row + score_ident, variant_ties),
    ``runner_file`` (best hypothesis of another file), ``runner_channel``
    (best other channel of the same file), ``null`` and ``contrast`` (score of
    the best unrelated hypothesis and the margin over it), ``table`` (all
    scored hypotheses) and timings.
    """
    params = params or MatchParams()
    lib = bank.lib
    t0 = time.time()
    pf = PanelFine(vm, valid, params)
    # ---- 1. coarse full-scan screening -> (file, orientation) groups
    R = np.nan_to_num(coarse_full_scan(bank, vm, valid, allowed, params.coarse_grid, params.coarse_border), nan=-2.0)
    groups, files_seen = OrderedDict(), set()
    for j in np.argsort(-R.ravel()):
        cid, k = divmod(int(j), len(ORIENTATIONS))
        if R[cid, k] <= -2:
            break
        f = bank.files[cid]
        key = (f, ORIENTATIONS[k])
        if key in groups:
            continue
        if len(groups) >= params.n_groups:
            # past n_groups only groups of new files are added, until min_files files are covered
            if len(files_seen) >= params.min_files:
                break
            if f in files_seen:
                continue
        groups[key] = (cid, float(R[cid, k]))
        files_seen.add(f)
    # ---- 2. fine score of each group's reference candidate
    rows = []
    for (f, o), (cid, rc) in groups.items():
        c = bank.cands[cid]
        rows.append(_row(lib, c.lid, c.variant, o, "full", None, 1.0, rc,
                         fine_compare(pf, region_of(lib, c.lid, c.variant, o))))
    rows.sort(key=lambda r: -np.nan_to_num(r["score"], nan=-2))
    # ---- 3. leading files: refine the field of view (detector boxes are not pixel-exact),
    #         then score every channel and variant at that alignment
    full_rows, done, expanded_keys = [], [], set()
    for r in rows:
        if r["path"] in done or not np.isfinite(r["score"]):
            continue
        done.append(r["path"])
        expanded_keys.add((r["path"], r["orientation"]))
        H, W = lib.layers[r["lid"]].shape
        ref = dict(r)
        rf = refine_crop(lib, pf, r["lid"], r["variant"], r["orientation"], (0, 0, W, H), FULL_ROUNDS)
        if rf is not None and np.isfinite(rf["score"]) and rf["score"] > r["score"]:
            ref.update({k: rf[k] for k in ("score", "r_plain", "shift", "grid", "crop", "crop_frac")})
        full_rows += _resolve_file(bank, pf, ref, allowed, "full")
        if len(done) >= params.n_full_files:
            break
    # the other groups (other files, or other orientations of the leading files) keep their reference
    # score: they are the unrelated hypotheses that define the chance level
    full_rows += [r for r in rows if (r["path"], r["orientation"]) not in expanded_keys]
    best_full, null_full = _best_and_null(full_rows)
    s_full = best_full["score"] if best_full else np.nan
    c_full = s_full - null_full if best_full else np.nan
    t_full = time.time() - t0
    # ---- 4. crop search, unless the full-scan match is already convincing
    crop_rows = []
    need_crop = best_full is None or not (s_full >= params.crop_trigger or c_full >= params.crop_skip_contrast)
    if need_crop:
        plausible = _plausible_for_crop(bank, unit[0], unit[1], span_si)
        allow_c = plausible if allowed is None else (plausible & allowed)
        for h in crop_search(bank, pf, allow_c, params):
            crop_rows.append(_row(lib, h["lid"], h["variant"], h["orientation"], "crop", h["crop"], h["crop_frac"],
                                  h["coarse"], h))
        # the coarse crop search cannot tell the channels of one file apart (e.g. Amplitude1 / Amplitude2):
        # every channel and variant of the leading files is scored at their refined crop
        expanded, done, keys = [], [], set()
        for h in [r for r in crop_rows if np.isfinite(r["score"])]:
            if h["path"] in done:
                continue
            done.append(h["path"])
            keys.add((h["path"], h["orientation"]))
            expanded += _resolve_file(bank, pf, h, allowed, "crop")
            if len(done) >= params.crop_expand_files:
                break
        crop_rows = expanded + [r for r in crop_rows if (r["path"], r["orientation"]) not in keys]
    best_crop, null_crop = _best_and_null(crop_rows)
    s_crop = best_crop["score"] if best_crop else np.nan
    c_crop = s_crop - null_crop if best_crop else np.nan
    t_crop = time.time() - t0 - t_full
    # ---- 5. decision: a crop must score higher AND stand out more from its own chance level
    use_crop = best_crop is not None and (best_full is None or (s_crop > s_full + params.crop_margin and c_crop > c_full))
    pool = crop_rows if use_crop else full_rows
    pool = [r for r in pool if np.isfinite(r["score"])]
    table = sorted([r for r in full_rows + crop_rows if np.isfinite(r["score"])], key=lambda r: -r["score"])
    base = {"table": table, "crop_searched": need_crop, "best_full_score": s_full, "best_crop_score": s_crop,
            "contrast_full": c_full, "contrast_crop": c_crop, "t_full": t_full, "t_crop": t_crop}
    if not pool:
        return {**base, "best": None, "runner_file": None, "runner_channel": None, "null": np.nan, "contrast": np.nan}
    top = max(pool, key=lambda r: r["score"])
    # processing variant: among the variants of the identified (file, channel, orientation, field of
    # view) the one with the highest plain (not detrended) r; near-ties are resolved by the caller
    same = [r for r in pool if r["lid"] == top["lid"] and r["orientation"] == top["orientation"]]
    same.sort(key=lambda r: (-np.nan_to_num(r["r_plain"], nan=-2), VARIANT_PRIORITY[r["variant"]]))
    best = dict(same[0])
    best["score_ident"] = top["score"]
    rp0 = np.nan_to_num(same[0]["r_plain"], nan=-2)
    best["variant_ties"] = [r["variant"] for r in same if np.nan_to_num(r["r_plain"], nan=-2) >= rp0 - params.variant_tie]
    others_f = [r for r in pool if r["path"] != top["path"]]          # same hypothesis type (full / crop)
    others_c = [r for r in pool if r["path"] == top["path"] and r["channel"] != top["channel"]]
    null = null_crop if use_crop else null_full
    return {**base, "best": best,
            "runner_file": max(others_f, key=lambda r: r["score"]) if others_f else None,
            "runner_channel": max(others_c, key=lambda r: r["score"]) if others_c else None,
            "null": null, "contrast": top["score"] - null}


def _allowed_layer(bank: CandidateBank, allowed, lid):
    """A layer is allowed if any of its bank candidates is."""
    return bool(np.any(allowed[bank.lids == lid]))
