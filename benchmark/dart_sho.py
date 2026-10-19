"""SHO maps from Asylum Research DART (dual AC resonance tracking) images.

Published DART-ESM/PFM amplitude images are usually not the measured amplitudes but the drive
amplitudes recovered with a simple-harmonic-oscillator (SHO) model from the amplitudes A1, A2 and
phases phi1, phi2 measured at the two drive frequencies f1 = fc - df/2 and f2 = fc + df/2
(B. Gannepalli, D. G. Yablon, A. H. Tsou and R. Proksch, Nanotechnology 22, 355705 (2011));
the Asylum Research software does this automatically. To compare such figures with the raw data,
this script computes, pixel by pixel, the SHO resonance frequency f0, quality factor Q, drive
amplitude A_d = A1 |D(f1)| / f0^2 and drive phase from the raw channels. It uses the amplitude
ratio and the phase difference, so the result does not depend on the phase offset:

    D(f) = f0^2 - f^2 + i f f0 / Q,   D(f2) / D(f1) = (A1 / A2) exp(i (phi2 - phi1)),

which is linear in a = f0^2 and b = f0 / Q. Pixels without a physical solution (a <= 0, b <= 0)
are NaN, as are pixels with Q outside 1-1000 (the SHO "errors" that published images often mark).
The maps are written as Gwyddion simple-field files (.gsf) with their units, which
tier3_evaluate.py reads like raw data.

Usage: python benchmark/dart_sho.py <file.ibw | folder> [...] --out <folder>
"""
import argparse, sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tier3_match import load_ibw, _ibw_note_kv  # noqa: E402


def sho(A1, A2, p1_deg, p2_deg, fc, df):
    f1, f2 = fc - df / 2, fc + df / 2
    z = (A1 / A2) * np.exp(1j * np.deg2rad(p2_deg - p1_deg))
    zr, zi = z.real, z.imag
    det = (1 - zr) * (f2 - zr * f1) + zi ** 2 * f1
    a = ((f2 ** 2 - zr * f1 ** 2) * (f2 - zr * f1) + zi ** 2 * f1 ** 3) / det
    b = zi * (f2 ** 2 - f1 ** 2) / det
    ok = np.isfinite(a) & np.isfinite(b) & (a > 0) & (b > 0)
    f0 = np.where(ok, np.sqrt(np.where(ok, a, 1)), np.nan)
    Q = np.where(ok, f0 / np.where(ok, b, 1), np.nan)
    ok &= (Q > 1) & (Q < 1000)
    D1 = np.hypot(a - f1 ** 2, b * f1)
    Ad = np.where(ok, A1 * D1 / np.where(ok, a, 1), np.nan)
    phd = np.where(ok, p1_deg - np.rad2deg(np.arctan2(b * f1, a - f1 ** 2)), np.nan)
    f0 = np.where(ok, f0, np.nan); Q = np.where(ok, Q, np.nan)
    return {"Amplitude_SHO": (Ad, "m"), "Frequency_SHO": (f0, "Hz"), "Q_SHO": (Q, ""), "Phase_SHO": (phd, "deg")}


def write_gsf(path, arr, xreal, zunit, title):
    hdr = (f"Gwyddion Simple Field 1.0\nXRes = {arr.shape[1]}\nYRes = {arr.shape[0]}\nXReal = {xreal}\n"
           f"YReal = {xreal}\nXYUnits = m\nZUnits = {zunit}\nTitle = {title}\n").encode()
    pad = 4 - len(hdr) % 4
    path.write_bytes(hdr + b"\0" * pad + np.nan_to_num(arr, nan=np.nan).astype("<f4").tobytes())


def process(ibw, out):
    import igor2.binarywave as bw
    layers = {name: arr for name, arr, *_ in load_ibw(ibw)}
    kv = _ibw_note_kv(bw.load(str(ibw))["wave"].get("note", b""))
    df = float(kv.get("DFRTFrequencyWidth", "nan"))
    scan = float(kv.get("ScanSize", "nan"))
    if not np.isfinite(df):
        raise SystemExit(f"{ibw.name}: no DFRTFrequencyWidth in the wave note (not a DART image?)")
    rows = []
    for d in ("Trace", "Retrace"):
        need = [f"Amplitude1{d}", f"Amplitude2{d}", f"Phase1{d}", f"Phase2{d}", f"Frequency{d}"]
        if not all(n in layers for n in need):
            continue
        res = sho(*(layers[n] for n in need), df)
        for name, (arr, unit) in res.items():
            p = out / f"{ibw.stem}__{name}{d}.gsf"
            write_gsf(p, arr, scan, unit, f"{name}{d}")
            v = arr[np.isfinite(arr)]
            rows.append((p.name, unit, float(np.mean(np.isfinite(arr))), float(np.median(v)) if v.size else np.nan))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", type=Path, nargs="+")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    files = [f for i in a.inputs for f in (sorted(i.rglob("*.ibw")) if i.is_dir() else [i])]
    for f in files:
        for name, unit, frac, med in process(f, a.out):
            print(f"{name:70s} {unit:4s} valid {frac:6.1%}  median {med:.4g}")


if __name__ == "__main__":
    main()
