# Corpus-sample evaluations (revision R1)

These scripts produce the in-the-wild numbers of the revised manuscript (Sections 2.1, 2.3, 2.5, 2.8 and
Supplementary Notes 7 and 9): the pipeline runs on random samples of literature figures and its output is
compared with human verification. The figures themselves are copyrighted and are **not redistributed**; the
sample manifests (figure identifiers and DOIs), the pipeline results and the verification workbooks are part
of the Zenodo deposit, so every number can be checked without the images. To re-run the pipeline on the
figures, rebuild the sample from the identifiers with the `collection/` scripts.

All scripts take the workspace root from the environment variable `AFM_DB_ROOT` (see `_paths.py`); the
expected layout below the root is `revision/detection/yield_sample` (1,000-figure sample, submitted version),
`revision/detection/yield_sample_v2` (fresh held-out sample), `datasets/afm_detection[_v2]` and
`runs/detect/runs/afm_detect/<version>/weights/best.pt`.

| script | purpose |
|---|---|
| `run_sample.py --config baseline\|evidence\|final4` | run a pipeline configuration on the held-out sample (resumable); `baseline` = submitted pipeline, `final4` = released pipeline (`final`, `final2` are the same configuration with earlier revisions of the reader; the manuscript reports `final4`) |
| `make_verification_sheet.py` | rating workbook + HTML galleries showing the baseline and released outputs side by side |
| `aggregate_verification.py --new final6 --before-sheet <xlsx> --before final4` | before/after summary of the human ratings with bootstrap confidence intervals (Section 2.8); `--before-sheet` keeps the rating of the pipeline before its final round next to the re-rated released pipeline |
| `make_rerating_sheet.py --old <xlsx> --before final4 --new final6` | re-rating workbook: only the figures whose result changed between two runs are rated again |
| `sample_corpus_v3.py --n 1000 --seed 2` | third sample: excludes the papers of both earlier samples and of the detector training sets |
| `make_verification_sheet_v3.py final6` / `aggregate_verification_v3.py` | rating workbook and summary for the third sample (released pipeline only, held-out) |
| `make_detection_sheet.py` | detection-only rating workbook (200 figures) |
| `eval_reader.py --reader v1\|v2` | bar-level evaluation of the colour-bar reader on the hard corpus bars (`data/corpus_sample/bar_truth.csv`) and the human-confirmed bars (Section 2.3) |
| `eval_pairing.py --reader v2 --pairing evidence --unit lexicon` | figure-level pairing / scale evaluation on the 88 figures quantified by the submitted pipeline (Section 2.5) |
| `build_dataset_v2.py` | detector training set v2: original labels + human-verified corpus figures (verified positives, hard negatives) |
| `finetune_v15.py` | fine-tune the submitted detector (v13) on dataset v2 → released detector (v15); held-out metrics of both |
| `compare_detectors_heldout.py --conf 0.5 0.25` | submitted vs released detector on the verified corpus figures not used for training (Section 2.1) |
