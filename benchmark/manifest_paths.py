"""Paths named in a benchmark manifest (kept free of heavy imports, so that the unit tests can use it)."""
from __future__ import annotations

import re
from pathlib import Path


def resolve_path(path, manifest_dir):
    """File named in a manifest. The manifest stores the paths of the layout in which the benchmark was generated
    (``<root>/<variant>/<file>``, ``<root>/gt/<file>``); in the data deposit the same files lie under
    ``<manifest folder>/figures/<variant or gt>/<file>``. The path is used as written when it exists, otherwise it
    is looked up next to the manifest."""
    p = Path(path)
    if p.exists():
        return p
    parts = [x for x in re.split(r"[\\/]+", str(path)) if x]
    manifest_dir = Path(manifest_dir)
    for cand in (manifest_dir.joinpath(*parts), manifest_dir.joinpath("figures", *parts[-2:]), manifest_dir.joinpath(*parts[-2:])):
        if cand.exists():
            return cand
    return p
