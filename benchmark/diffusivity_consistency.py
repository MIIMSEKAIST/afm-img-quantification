"""Reconstruction-to-analysis consistency at the diffusivity level.

For every benchmark case, convert both the original and the reconstructed
amplitude map to an apparent-diffusivity map (the amplitude enters
quadratically, so reconstruction error is amplified) and compare scan-median
apparent diffusivities. Reproduces Supplementary Fig. S3:

    scan-median deviation 0.90 +/- 0.77 %, pixel log-diffusivity r = 0.91 +/- 0.08

Because f, Q, beta and V_ac multiply the original and reconstructed maps
identically, the ratio D_recon / D_raw is independent of those parameters; only
the absolute value (cm^2/s) depends on f and Q. This makes the consistency check
a clean probe of reconstruction fidelity alone.

Inputs: per-case .npz files (gt, recon, valid) written by
`reproduce_benchmark.py --recon-dir`, plus the IBW scans (for the drive
frequency / Q-factor parsed from the Igor note).

Usage
-----
    python benchmark/reproduce_benchmark.py --ibw-dir data/esm_ibw \
        --out results/benchmark104.csv --recon-dir runs/recon_maps
    python benchmark/diffusivity_consistency.py --recon-dir runs/recon_maps \
        --ibw-dir data/esm_ibw --out results/diffusivity_recon_vs_raw.csv

Source: migrated from scripts/diffusivity_from_recon.py (logic preserved).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd
import igor2.binarywave as bw

V_AC = 1.0
# nominal Vegard strain coefficients (material-specific)
BETA = {"LMNO": 0.025, "NCM": 0.023, "LICGC": 0.015}
# fallback drive frequencies (Hz) if absent from the IBW note
F_FALLBACK = {"LMNO": 395153.0, "NCM": 353816.0, "LICGC": 311811.0}
Q_NOTE_KEYS = ("DFRTQ", "QFactor", "Q")


def material_of(stem: str) -> str:
    for key in BETA:
        if stem.upper().startswith(key):
            return key
    raise ValueError(f"could not identify material from: {stem}")


def parse_note(ibw_path: Path) -> dict:
    note = bw.load(str(ibw_path))["wave"]["note"]
    if isinstance(note, bytes):
        note = note.decode("utf-8", errors="ignore")
    kv = {}
    for line in re.split(r"[\r\n]+", note):
        k, sep, v = line.partition(":")
        if sep:
            kv[k.strip()] = v.strip()
    return kv


def load_scan_params(scan_stems, ibw_dir: Path, params_csv: Path | None):
    """Per-scan (f, Q). Priority: params_csv > IBW note > fallback (Q=1)."""
    params, missing_q, override = {}, [], {}
    if params_csv and params_csv.exists():
        for _, r in pd.read_csv(params_csv).iterrows():
            override[str(r["scan"])] = (float(r["f_Hz"]), float(r["Q"]))
        print(f"[params] using {params_csv} ({len(override)} scans)")

    for stem in scan_stems:
        if stem in override:
            params[stem] = override[stem]
            continue
        mat = material_of(stem)
        f, Q = F_FALLBACK[mat], None
        p = ibw_dir / f"{stem}.ibw"
        if p.exists():
            kv = parse_note(p)
            for key in ("DriveFrequency", "Drive Frequency"):
                if key in kv:
                    try:
                        f = float(kv[key]); break
                    except ValueError:
                        pass
            for key in Q_NOTE_KEYS:
                if key in kv:
                    try:
                        Q = float(kv[key]); break
                    except ValueError:
                        pass
        if Q is None:
            Q = 1.0
            missing_q.append(stem)
        params[stem] = (f, Q)

    if missing_q:
        print(f"[warn] Q=1.0 substituted for {len(missing_q)} scans; absolute D "
              f"values are nominal (ratio metrics are unaffected).")
    return params


def diffusivity(amp, f, Q, beta, vac=V_AC):
    """D_app = pi*f*(A_corr/(V_ac*beta))^2 with A_corr = A/Q; ->cm^2/s (x1e4)."""
    a = np.clip(amp, 0.0, None) / Q
    return np.pi * f * (a / (vac * beta)) ** 2 * 1e4


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recon-dir", type=Path, default=Path("runs/recon_maps"))
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--params-csv", type=Path, default=None,
                    help="optional CSV with columns scan,f_Hz,Q")
    ap.add_argument("--out", type=Path, default=Path("results/diffusivity_recon_vs_raw.csv"))
    args = ap.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(args.recon_dir.glob("*.npz"))
    if not npz_files:
        raise SystemExit(f"{args.recon_dir} is empty — run reproduce_benchmark.py "
                         f"with --recon-dir first.")
    scan_stems = sorted({p.stem.rsplit("__", 1)[0] for p in npz_files})
    params = load_scan_params(scan_stems, args.ibw_dir, args.params_csv)
    print(f"{len(npz_files)} cases / {len(scan_stems)} scans")

    rows = []
    for npz_path in npz_files:
        case_id = npz_path.stem
        stem, cmap_name = case_id.rsplit("__", 1)
        mat = material_of(stem)
        f, Q = params[stem]

        d = np.load(npz_path)
        gt, recon, valid = d["gt"].astype(float), d["recon"].astype(float), d["valid"]
        D_raw = diffusivity(gt, f, Q, BETA[mat])
        D_rec = diffusivity(recon, f, Q, BETA[mat])
        g, r = D_raw[valid], D_rec[valid]
        med_g, med_r = float(np.median(g)), float(np.median(r))
        mean_g, mean_r = float(g.mean()), float(r.mean())

        pos = (g > 0) & (r > 0)
        r_log = (float(np.corrcoef(np.log10(g[pos]), np.log10(r[pos]))[0, 1])
                 if pos.sum() > 10 else np.nan)

        rows.append({
            "case_id": case_id, "scan": stem, "material": mat, "colormap": cmap_name,
            "f_Hz": f, "Q": Q,
            "D_median_raw_cm2s": med_g, "D_median_recon_cm2s": med_r,
            "median_dev_pct": (med_r / med_g - 1.0) * 100,
            "mean_dev_pct": (mean_r / mean_g - 1.0) * 100,
            "pixel_r_logD": r_log,
            "valid_ratio": float(valid.mean()),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)

    print("\n" + "=" * 60)
    print("RECONSTRUCTED vs RAW DIFFUSIVITY - SUMMARY")
    print("=" * 60)
    print(f"Total cases:        {len(df)}")
    print(f"|median deviation|: {df.median_dev_pct.abs().mean():.2f} +/- {df.median_dev_pct.abs().std():.2f} %")
    print(f"pixel r(logD):      {df.pixel_r_logD.mean():.3f} +/- {df.pixel_r_logD.std():.3f}")
    print(f"\nSaved: {args.out}")


if __name__ == "__main__":
    main()
