# Literature collection pipeline (EXPERIMENTAL)

> The validated contribution of the paper is the quantification **engine** and
> the ground-truth **benchmark**. Large-scale detection and database
> construction over the full literature corpus are **ongoing / experimental**;
> this folder is provided for transparency of the end-to-end workflow.

| step | script | role |
|------|--------|------|
| 0 | `step0_collect.py`   | query Crossref for candidate papers (metadata only) |
| 1 | `step1_download.py`  | download + validate PDFs |
| 2 | `step2_extract.py`   | extract figures + captions (uses pdffigures2) |
| 3 | `step3_decompose.py` | split composite figures into panels |

The scripts are migrated faithfully from the original `src/` pipeline. They read
configuration/paths from a project config and the (copyright-restricted) corpus,
so they are provided for transparency rather than turn-key execution; expect to
adjust paths for your own corpus.

## ⚠️ Copyright

This pipeline downloads and processes third-party publications. The resulting
**PDFs and extracted figures are copyrighted and are NOT redistributed** in this
repository. Only the collection *code* is shared. External dependency:
[`pdffigures2`](https://github.com/allenai/pdffigures2).
