"""Workspace paths of the corpus-sample evaluations.

The literature figures are copyrighted and not redistributed, so these scripts run on the authors' workspace
(or on a workspace rebuilt from the Zenodo deposit: sample manifests, verification workbooks, pipeline results).
Set AFM_DB_ROOT to that workspace; the repository itself is located relative to this file.
"""
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
for p in (REPO, REPO / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
_root = os.environ.get("AFM_DB_ROOT")
if not _root:
    raise SystemExit("AFM_DB_ROOT is not set: these scripts need the workspace that holds the corpus samples "
                     "(revision/detection/yield_sample*, datasets/, runs/); see scripts/corpus/README.md")
ROOT = Path(_root)
