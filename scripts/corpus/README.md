# Corpus samples (revision R1)

These scripts draw the samples of the literature corpus and run the pipeline on them (manuscript Section 2.8). The
figures themselves are copyrighted and are **not redistributed**; the sample manifests (figure identifiers and DOIs)
and the pipeline output of every state on every sample are part of the Zenodo deposit, so the numbers of Section 2.8 can be checked without the images. To re-run the pipeline on the figures, rebuild a sample from its identifiers with the
`collection/` scripts.

**Where the accuracies of the article come from.** The numbers of Section 2.8 (Table 2, Supplementary Note 9 and
Table S4) are the joint rating of two of the authors, made with a browser tool that shows one object per screen. The
item lists, the answers with their time log, the tool and the script that summarises them (`rating_collect.py`) are in
the Zenodo deposit (`corpus_sample/rating/`), not in this repository.

**First pass.** Before that rating, the same figures were rated in a first pass whose verdicts were produced by an AI
coding assistant, in the workbooks written by `make_verification_sheet*.py`, `make_rerating_sheet*.py` and
`make_detection_sheet.py` and summarised by `aggregate_verification*.py`. The first pass guided the development of the
reader and the pairing (Supplementary Note 10) and the choice of the figures for fine-tuning the detector; no accuracy
of the article is taken from it. Its workbooks are deposited for the record (`corpus_sample/first_pass_ai_assisted/`).
In the scripts, names such as `verified_positive` and the "rater" of the workbooks refer to this first pass.

All scripts take the workspace root from the environment variable `AFM_DB_ROOT` (see `_paths.py`); the expected layout
below the root is `revision/detection/yield_sample` (first sample), `yield_sample_v2`, `yield_sample_v3` and
`yield_sample_v4` (second to fourth samples), `datasets/afm_detection[_v2]` and
`runs/detect/runs/afm_detect/<version>/weights/best.pt`.

Pipeline states of the article and the configurations of `run_sample.py`: initial = `baseline` (first and second
samples); first revision = `final4` (second sample); second revision = `final6` (third sample); released = `final7`
(fourth sample). `final8` is `final7` with the two changes made after the fourth sample had been rated (map crop
trimmed to the panel, luma-weighted lookup distance), which are the defaults of the released code; they change no
pairing, no read scale and no quantified status. `final`, `final2`, `final3` and `final5` are intermediate revisions
of the reader.

| script | purpose |
|---|---|
| `run_sample.py --config <name> [--sample-dir …]` | run a pipeline configuration on a sample (resumable) |
| `sample_corpus_v3.py --n 1000 --seed 2` | third sample: excludes the papers of both earlier samples and of the detector training sets |
| `sample_corpus_v4.py --n 1000 --per-paper 2 --seed 3` | fourth sample: the remaining papers, up to two figures per paper (held-out test of the released pipeline) |
| `build_dataset_v2.py` | detector training set v2: original labels + figures of the first sample chosen with the first-pass rating (91 with their detections as labels, 198 map-free negatives) |
| `finetune_v15.py` | fine-tune the submitted detector (v13) on dataset v2 → released detector (v15); held-out metrics of both |
| `eval_reader.py --reader v1\|v2` | development: bar-level evaluation of the colour-bar reader on the regression set of corpus colour bars (`data/corpus_sample/bar_truth.csv`; its reference readings were prepared with the help of an AI assistant, and no accuracy is claimed from it) |
| `eval_pairing.py --reader v2 --pairing evidence --unit lexicon` | development: figure-level pairing / scale evaluation on the 88 figures of the first sample quantified by the initial pipeline, against the first-pass verdicts |
| `compare_detectors_heldout.py --conf 0.5 0.25` | development: submitted vs released detector against the first-pass counts on figures of the first sample not used for training (the article reports the rating of Section 2.8 instead) |
| `make_verification_sheet.py`, `make_verification_sheet_v3.py final7 [--sample-dir …] [--gallery-only]` | first pass: rating workbooks and HTML galleries (second sample: initial and first-revision outputs side by side; third and fourth samples: one pipeline state) |
| `make_rerating_sheet.py`, `make_rerating_sheet_v3.py --old <xlsx> --before final6 --new final7` | first pass: re-rating workbooks, in which only the figures whose result changed between two runs are rated again |
| `make_detection_sheet.py` | first pass: detection-only rating workbook (200 figures) |
| `aggregate_verification.py`, `aggregate_verification_v3.py` | first pass: summaries of the workbooks with bootstrap confidence intervals |
