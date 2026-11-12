"""Reproduce every number reported in the manuscript from the deposited data.

Runs the benchmark scripts in order and collects the headline quantities into
``results/numbers_manifest.json`` (machine-readable) and
``results/numbers_manifest.md`` (human-readable), so that the manuscript text
can be checked against the code output line by line.

Usage
-----
    python reproduce_all.py --ibw-dir data/esm_ibw            # engine benchmarks (~5 min) + OCR test
    python reproduce_all.py --ibw-dir data/esm_ibw --weights detection/weights/afm_detect_best.pt \
        --data path/to/afm_detection/data.yaml                 # + detector metrics and the realistic-figure
                                                               #   (Tier 2) benchmark (~40 min, GPU recommended)
    python reproduce_all.py --skip tier2_render tier2_eval detection detection_submitted   # skip the slow / data-dependent steps
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
PY = sys.executable
COLORMAPS = ["copper", "hot", "jet", "viridis"]
MATS = ["LMNO", "LICGC", "NCM811"]

STEPS = [  # (key, command)
    ("benchmark", ["benchmark/reproduce_benchmark.py", "--out", "results/benchmark104.csv",
                   "--recon-dir", "runs/recon_maps"]),
    ("scanlevel", ["analysis/scan_level_stats.py", "--recon-dir", "runs/recon_maps",
                   "--out", "results/meta_scanlevel.csv"]),
    ("diffusivity", ["benchmark/diffusivity_consistency.py", "--recon-dir", "runs/recon_maps",
                     "--out", "results/diffusivity_recon_vs_raw.csv"]),
    ("robustness", ["benchmark/robustness_sweep.py", "--out", "results/robustness_sweep.csv"]),
    ("subsampling", ["benchmark/subsampling_sweep.py", "--out", "results/subsampling_sweep.csv"]),
    ("subsampling_cielab", ["benchmark/subsampling_sweep.py", "--color-space", "cielab",
                            "--out", "results/subsampling_sweep_cielab.csv"]),
    ("subsampling_rgb", ["benchmark/subsampling_sweep.py", "--color-space", "rgb",
                         "--out", "results/subsampling_sweep_rgb.csv"]),
    ("baseline", ["benchmark/baseline_comparison.py", "--out", "results/baseline_comparison.csv"]),
    ("jpeg_noise", ["benchmark/jpeg_noise.py", "--out", "results/jpeg_noise.csv"]),
    ("colormaps", ["benchmark/colormap_analysis.py", "--out", "results/colormap_properties.csv"]),
    # --- stages added in revision R1 (colour-bar reading, detection, end-to-end) ---
    ("ocr", ["benchmark/ocr_synthetic_test.py", "--n", "300", "--out", "results/ocr_synthetic_test.csv"]),
    ("detection", ["detection/evaluate.py", "--out", "results/detection_eval.json"]),
    ("detection_submitted", ["detection/evaluate.py", "--weights", "detection/weights/afm_detect_v13.pt",
                             "--out", "results/detection_eval_v13.json"]),
    ("tier2_render", ["benchmark/tier2_render.py", "--out", "runs/tier2"]),
    ("tier2_eval", ["benchmark/tier2_evaluate.py", "--manifest", "runs/tier2/manifest.json",
                    "--out", "results/tier2_results.csv"]),
]
NEEDS_IBW = {"benchmark", "scanlevel", "diffusivity", "robustness", "subsampling", "subsampling_cielab", "subsampling_rgb",
             "baseline", "jpeg_noise", "tier2_render"}
NEEDS_WEIGHTS = {"detection", "tier2_eval"}      # detector weights (released with v1.1.0)
NEEDS_DATA = {"detection", "detection_submitted"}                        # labelled detection dataset (Zenodo deposit)
NEEDS_PDFF = {"tier2_render"}                     # pdffigures2 jar (optional; PyMuPDF fallback)


def r(x, n=3):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), n)


def build_manifest():
    R = ROOT / "results"
    M = {}

    b = pd.read_csv(R / "benchmark104.csv")
    M["benchmark"] = {
        "n_cases": int(len(b)), "n_scans": int(b.scan.nunique()),
        "nMAE_mean": r(b.nMAE_pct.mean()), "nMAE_std": r(b.nMAE_pct.std()),
        "nMAE_min": r(b.nMAE_pct.min()), "nMAE_max": r(b.nMAE_pct.max()),
        "SSIM_mean": r(b.SSIM.mean()), "SSIM_std": r(b.SSIM.std()), "SSIM_min": r(b.SSIM.min()),
        "mean_error_mean": r(b.mean_error_pct.mean()),
        "std_error_abs_mean": r(b.std_error_pct.mean()), "std_error_abs_max": r(b.std_error_pct.max()),
        "EMD_mean": r(b.normalized_EMD_pct.mean()),
        "valid_ratio_mean": r(b.valid_pixel_ratio.mean(), 4), "valid_ratio_min": r(b.valid_pixel_ratio.min(), 4),
        "by_colormap": {c: {"nMAE": r(g.nMAE_pct.mean()), "SSIM": r(g.SSIM.mean())}
                        for c, g in b.groupby("colormap")},
        "by_material": {m: {"nMAE": r(g.nMAE_pct.mean()), "SSIM": r(g.SSIM.mean())}
                        for m, g in b.groupby("material")},
    }
    if "nMAE_IQR_pct" in b:
        M["benchmark_extended"] = {
            "nMAE_IQR_mean": r(b.nMAE_IQR_pct.mean()), "nMAE_IQR_median": r(b.nMAE_IQR_pct.median()),
            "nMAE_IQR_max": r(b.nMAE_IQR_pct.max()),
            "nMAE_SD_mean": r(b.nMAE_SD_pct.mean()), "nMAE_SD_median": r(b.nMAE_SD_pct.median()),
            "std_error_signed_mean": r(b.std_error_signed_pct.mean()),
            "std_error_negative_fraction": r((b.std_error_signed_pct < 0).mean()),
            "mean_error_signed_mean": r(b.mean_error_signed_pct.mean()),
            "nMAE_allpix_mean": r(b.nMAE_allpix_pct.mean()),
            "IQR_over_range_median": r(b.IQR_over_range.median()),
            "levels_in_IQR_median": r(b.levels_in_IQR.median(), 1),
            "levels_in_IQR_min": r(b.levels_in_IQR.min(), 1),
            "by_colormap_nMAE_IQR": {c: r(g.nMAE_IQR_pct.mean()) for c, g in b.groupby("colormap")},
        }

    if (R / "meta_scanlevel.csv").exists():
        s = pd.read_csv(R / "meta_scanlevel.csv")
        s["material"] = s.material.replace({"NCM": "NCM811"})
        dev = s.med_dev_pct.abs()
        sys.path.insert(0, str(ROOT))
        from analysis.meta_analysis import corpus_stats  # noqa: E402
        gt0 = s[s.colormap == COLORMAPS[0]]
        corp = {"ground_truth": corpus_stats({m: gt0[gt0.material == m].gt_med.values * 1e12 for m in MATS})}
        for c in COLORMAPS:
            sc = s[s.colormap == c]
            corp[c] = corpus_stats({m: sc[sc.material == m].rec_med.values * 1e12 for m in MATS})
        M["scan_level"] = {
            "median_dev_abs_mean": r(dev.mean(), 2), "median_dev_abs_std": r(dev.std(), 2),
            "median_dev_abs_max": r(dev.max(), 2),
            "median_dev_by_material": {m: r(dev[s.material == m].mean(), 2) for m in MATS},
            "table1_gt_median_pm": {m: r(np.median(gt0[gt0.material == m].gt_med) * 1e12, 0) for m in MATS},
            "table1_rec_median_pm_mean_over_cmaps": {
                m: r(np.mean([np.median(s[(s.colormap == c) & (s.material == m)].rec_med) * 1e12
                              for c in COLORMAPS]), 0) for m in MATS},
            "heterogeneity_r": r(np.corrcoef(s.gt_cv, s.rec_cv)[0, 1]),
            "heterogeneity_dev_abs_mean": r(s.cv_dev_pct.abs().mean(), 1),
            "heterogeneity_dev_signed_mean": r(s.cv_dev_pct.mean(), 1),
            "qc_flags": {m: int(gt0[gt0.material == m].qc_flag.sum()) for m in MATS},
            "corpora": {k: {"ranking": v["ranking"],
                            "cliffs_delta": {p: r(x, 3) for p, x in v["cliffs_delta"].items()},
                            "mannwhitney_p": {p: r(x, 4) for p, x in v["mannwhitney_p"].items()}}
                        for k, v in corp.items()},
            "ranking_preserved_all_corpora": len({v["ranking"] for v in corp.values()}) == 1,
        }

    if (R / "diffusivity_recon_vs_raw.csv").exists():
        d = pd.read_csv(R / "diffusivity_recon_vs_raw.csv")
        M["diffusivity"] = {"median_dev_abs_mean": r(d.median_dev_pct.abs().mean(), 2),
                            "median_dev_abs_std": r(d.median_dev_pct.abs().std(), 2),
                            "median_dev_abs_max": r(d.median_dev_pct.abs().max(), 2),
                            "r_logD_mean": r(d.pixel_r_logD.mean(), 2), "r_logD_std": r(d.pixel_r_logD.std(), 2)}

    if (R / "robustness_sweep.csv").exists():
        rb = pd.read_csv(R / "robustness_sweep.csv")
        q = rb[rb.sweep == "quality"].pivot_table(index="param", columns="colormap", values="nMAE", aggfunc="mean")
        res = rb[rb.sweep == "resolution"].pivot_table(index="param", columns="colormap", values="nMAE", aggfunc="mean")
        order_ok = {int(p): list(q.loc[p].sort_values().index) == COLORMAPS for p in q.index}
        M["robustness"] = {
            "n_cases": int(len(rb)),
            "quality_mean_nMAE": {int(p): {c: r(q.loc[p, c]) for c in COLORMAPS} for p in q.index},
            "resolution_mean_nMAE": {int(p): {c: r(res.loc[p, c]) for c in COLORMAPS} for p in res.index},
            "group_mean_nMAE_min": r(min(q.values.min(), res.values.min()), 2),
            "group_mean_nMAE_max": r(max(q.values.max(), res.values.max()), 2),
            "quality_group_mean_range": [r(q.values.min(), 2), r(q.values.max(), 2)],
            "worst_single_case_nMAE": r(rb.nMAE.max(), 2),
            "colormap_order_copper<hot<jet<viridis_by_quality": order_ok,
        }

    for key, fn in (("subsampling", "subsampling_sweep.csv"), ("subsampling_cielab", "subsampling_sweep_cielab.csv"),
                    ("subsampling_rgb", "subsampling_sweep_rgb.csv")):
        if (R / fn).exists():
            ss = pd.read_csv(R / fn)
            M[key] = {metric: {v: {c: r(ss[(ss.variant == v) & (ss.colormap == c)][metric].mean())
                                   for c in COLORMAPS} | {"mean": r(ss[ss.variant == v][metric].mean())}
                               for v in ss.variant.unique()}
                      for metric in ("nMAE_pct", "SSIM", "std_error_signed_pct")}

    if (R / "baseline_comparison.csv").exists():
        bl = pd.read_csv(R / "baseline_comparison.csv")
        M["baseline"] = {f"{m}[{k}]": {"nMAE": r(g.nMAE_pct.mean()), "nMAE_IQR": r(g.nMAE_IQR_pct.mean()),
                                       "SSIM": r(g.SSIM.mean())}
                         for (m, k), g in bl.groupby(["method", "mask"])}
        M["baseline_by_colormap_nMAE_allpix"] = {
            m: {c: r(g[g.colormap == c].nMAE_pct.mean()) for c in COLORMAPS}
            for m, g in bl[bl["mask"] == "all"].groupby("method")}

    if (R / "jpeg_noise.csv").exists():
        jn = pd.read_csv(R / "jpeg_noise.csv")
        M["jpeg_noise"] = {c: {k: r(g[k].mean(), 2) for k in ("sd_Y", "sd_Cb", "sd_Cr", "chroma_to_luma")}
                           for c, g in jn.groupby("colormap")}

    if (R / "colormap_properties.csv").exists():
        cp = pd.read_csv(R / "colormap_properties.csv")
        M["colormap_properties"] = {row.colormap: {k: r(getattr(row, k)) for k in
                                                   ("path_dE", "path_dL", "path_dab", "luminance_share",
                                                    "min_step_dE", "mean_chroma")}
                                    | {"L_monotonic": bool(row.L_monotonic)}
                                    for row in cp.itertuples()}
    if (R / "ocr_synthetic_test.csv").exists():
        o = pd.read_csv(R / "ocr_synthetic_test.csv")
        ok = o[o.ok.astype(bool)]
        err = pd.concat([ok.err_lo_rel, ok.err_hi_rel]) * 100
        M["ocr_synthetic"] = {
            "n_cases": int(len(o)), "scale_read_frac": r(o.ok.astype(bool).mean(), 4),
            "orientation_ok_frac": r(o.orient_ok.astype(bool).mean(), 4),
            "success_2pct_frac": r(o.success.astype(bool).mean(), 4),
            "success_5pct_frac": r(o.success_5pct.astype(str).eq("True").mean(), 4),
            "unit_ok_frac": r(o.unit_ok.astype(str).eq("True").mean(), 4),
            "end_error_median_pct": r(err.median()), "end_error_p90_pct": r(err.quantile(0.9)),
            "end_error_p95_pct": r(err.quantile(0.95)),
            "success_by_orientation": {("vertical" if v else "horizontal"): r(g.success.astype(bool).mean(), 4)
                                       for v, g in o.groupby("vertical")},
            "success_by_nticks": {int(k): r(g.success.astype(bool).mean(), 4) for k, g in o.groupby("nticks")},
            "success_by_fontsize": {int(k): r(g.success.astype(bool).mean(), 4) for k, g in o.groupby("fontsize")},
            "failure_reasons": {str(k): int(v) for k, v in
                                o[~o.success.astype(bool)].reason.fillna("end_error>tol").value_counts().items()},
        }

    if (R / "detection_eval.json").exists():
        M["detection"] = json.loads((R / "detection_eval.json").read_text(encoding="utf-8")).get("splits")
    if (R / "detection_eval_v13.json").exists():   # the submitted detector, for the comparison in Section 2.1
        M["detection_submitted"] = json.loads((R / "detection_eval_v13.json").read_text(encoding="utf-8")).get("splits")

    if (R / "tier2_results.csv").exists():
        t = pd.read_csv(R / "tier2_results.csv")
        M["tier2"] = {}
        for cs, tc in t.groupby("color_space"):
            esm = tc[tc.panel.str.startswith("esm")]          # esm_b: right-hand map of the two-map figures
            okp = esm[esm.status == "ok"]
            by_var = {}
            for v, g in esm.groupby("variant"):
                go = g[g.status == "ok"]
                by_var[v] = {"n": int(len(g)), "detected": r((g.status != "map_not_detected").mean(), 4),
                             "bar_read": r(g.status.isin(["ok", "low_valid_ratio", "wrong_scale"]).mean(), 4),
                             "quantified": r((g.status == "ok").mean(), 4),
                             "scale_err_median_pct": r(go.scale_err.median() * 100) if len(go) else None,
                             "nMAE_mean": r(go.nMAE_pct.mean()) if len(go) else None,
                             "nMAE_IQR_mean": r(go.nMAE_IQR_pct.mean()) if len(go) else None,
                             "median_dev_abs_mean": r(go.median_dev_pct.abs().mean()) if len(go) else None,
                             "SSIM_mean": r(go.SSIM.mean()) if len(go) else None,
                             "unit_ok": r(go.unit_ok.astype(float).mean(), 4) if len(go) else None,
                             "status_counts": {str(k): int(n) for k, n in g.status.value_counts().items()},
                             "pairing_correct": (r(g.pairing_ok.dropna().astype(float).mean(), 4)
                                                 if "pairing_ok" in g and g.pairing_ok.notna().any() else None)}
            topo = tc[tc.panel == "topo"]
            tg = topo[topo.status == "ok"]
            M["tier2"][cs] = {
                "n_panels_esm": int(len(esm)), "esm_ok_frac": r((esm.status == "ok").mean(), 4),
                "esm_ok_frac_excluding_hbar": r((esm[esm.variant != "hbar"].status == "ok").mean(), 4),
                "esm_ok_scale_err_median_pct": r(okp.scale_err.median() * 100),
                "esm_ok_scale_err_p95_pct": r(okp.scale_err.quantile(0.95) * 100),
                "esm_ok_nMAE_mean": r(okp.nMAE_pct.mean()), "esm_ok_nMAE_std": r(okp.nMAE_pct.std()),
                "esm_ok_nMAE_IQR_mean": r(okp.nMAE_IQR_pct.mean()),
                "esm_ok_median_dev_abs_mean": r(okp.median_dev_pct.abs().mean()),
                "esm_ok_median_dev_abs_max": r(okp.median_dev_pct.abs().max()),
                "esm_ok_SSIM_mean": r(okp.SSIM.mean()),
                "by_variant": by_var,
                "topo": ({"n": int(len(topo)), "quantified": r((topo.status == "ok").mean(), 4),
                          "nMAE_mean": r(tg.nMAE_pct.mean()) if len(tg) else None,
                          "median_dev_range_abs_mean": (r(tg.median_dev_range_pct.abs().mean())
                                                        if len(tg) and "median_dev_range_pct" in tg else None),
                          "SSIM_mean": r(tg.SSIM.mean()) if len(tg) else None} if len(topo) else None),
            }
    return M


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ibw-dir", type=Path, default=Path("data/esm_ibw"))
    ap.add_argument("--weights", type=Path, default=Path("detection/weights/afm_detect_best.pt"),
                    help="detector weights (detection, tier2_eval)")
    ap.add_argument("--data", type=Path, default=Path("detection/data.yaml"),
                    help="detection dataset yaml (detection)")
    ap.add_argument("--pdffigures2", type=Path, default=Path("tools/pdffigures2/pdffigures2.jar"))
    ap.add_argument("--skip", nargs="*", default=[], help="step keys to skip")
    ap.add_argument("--manifest-only", action="store_true")
    args = ap.parse_args()

    if not args.manifest_only:
        for key, cmd in STEPS:
            if key in args.skip:
                print(f"\n[skip] {key}")
                continue
            full = ([PY, *cmd] + (["--ibw-dir", str(args.ibw_dir)] if key in NEEDS_IBW else [])
                    + (["--weights", str(args.weights)] if key in NEEDS_WEIGHTS else [])
                    + (["--data", str(args.data)] if key in NEEDS_DATA else [])
                    + (["--pdffigures2", str(args.pdffigures2)] if key in NEEDS_PDFF else []))
            print("\n$", " ".join(full), flush=True)
            subprocess.run(full, cwd=ROOT, check=True)

    M = build_manifest()
    out = ROOT / "results" / "numbers_manifest.json"
    out.write_text(json.dumps(M, indent=2), encoding="utf-8")
    md = ["# Numbers manifest (generated by reproduce_all.py)\n"]
    for section, body in M.items():
        md.append(f"## {section}\n```json\n{json.dumps(body, indent=2)}\n```\n")
    (ROOT / "results" / "numbers_manifest.md").write_text("\n".join(md), encoding="utf-8")
    print(f"\nManifest written: {out}")


if __name__ == "__main__":
    main()
