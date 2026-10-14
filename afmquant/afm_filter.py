"""AFM-specificity filter (Referee 1, comment 1).

The detector localises colour-mapped image panels with a colour bar; it has no
notion of the measurement technique. Technique attribution therefore uses the
textual context that accompanies every mined figure:

* the figure caption (from the PDF extraction step), and
* the physical unit read from the colour bar.

``classify_panel`` returns one of
    "afm"               caption names an AFM mode (or AFM itself) and the unit
                        is compatible with AFM observables, or the unit alone
                        is AFM-typical and the caption does not name another
                        colour-mapped technique
    "non_afm_colormap"  caption names another technique that produces
                        colour-mapped images (STM, EDS, Raman, simulation, ...)
    "unknown"           neither

The rule set is deliberately simple and transparent; its precision/recall are
measured against human judgement on the corpus sample (results/yield_*.csv).
"""
from __future__ import annotations

import re

AFM_MODES = re.compile(
    r"\b(AFM|atomic[- ]force|topograph\w*|height (?:image|map)|PFM|piezoresponse|ESM|electrochemical strain|"
    r"KPFM|Kelvin[- ]probe|C-?AFM|conductive[- ]AFM|current (?:map|image)|MFM|magnetic[- ]force|"
    r"adhesion (?:map|image|force)|friction (?:map|image)|force (?:map|volume)|phase (?:image|map|contrast)|"
    r"(?:Young'?s )?modulus map|stiffness map|deformation map|dissipation|amplitude (?:image|map)|"
    r"scanning probe|SPM|tapping[- ]mode|contact[- ]mode|cantilever|tip-sample)\b", re.I)
NON_AFM = re.compile(
    r"\b(STM|scanning tunnel\w*|EDS|EDX|XRF|Raman (?:map|imag)\w*|PL (?:map|imag)\w*|photoluminescence map|"
    r"fluorescence|confocal|hyperspectral|CL (?:map|imag)\w*|cathodoluminescence|EBSD|XRD map|"
    r"simulat\w*|FEM|finite[- ]element|COMSOL|DFT|molecular dynamics|phase[- ]field|heat ?map|"
    r"thermal (?:map|imag)\w*|infrared (?:map|imag)\w*|SEM|TEM|STEM|EELS|SIMS|ToF-SIMS|MRI|CT|ultrasound|"
    r"satellite|weather|elevation|DEM)\b", re.I)
AFM_UNITS = {"nm", "um", "µm", "pm", "Å", "mv", "v", "pa", "na", "deg", "°", "rad", "mrad",
             "n/m", "nn", "pn", "a.u.", "a.u", "arb", "arb.", "mpa", "gpa", "hz", "khz", "pm/v", "mv/v", "nm/v"}


def classify_panel(caption: str | None, unit: str | None) -> dict:
    cap = caption or ""
    afm_hits = sorted({m.group(0).lower() for m in AFM_MODES.finditer(cap)})
    non_hits = sorted({m.group(0).lower() for m in NON_AFM.finditer(cap)})
    u = (unit or "").strip().lower()
    unit_afm = u in AFM_UNITS
    if afm_hits and not non_hits:
        label = "afm"
    elif afm_hits and non_hits:
        # mixed figures (e.g. AFM next to SEM): keep the panel if its unit is AFM-typical
        label = "afm" if unit_afm else "unknown"
    elif non_hits:
        label = "non_afm_colormap"
    else:
        label = "afm" if unit_afm and u not in {"a.u.", "a.u", "arb", "arb."} else "unknown"
    return {"label": label, "afm_keywords": afm_hits, "non_afm_keywords": non_hits,
            "unit": u or None, "unit_afm_typical": unit_afm}
