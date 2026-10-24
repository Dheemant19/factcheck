# configs

One YAML per run: dataset, model, method, hyperparameters, seed (plan Section 16).
Run with `python -m factcheck.train --config configs/<file>.yaml`.

`template.yaml` lists the fields every run needs; copy it to start a new run.

`lstm/` holds the Bi-LSTM baselines (T05), one per dataset. Settings a config leaves out
take the plan Section 10 defaults in `factcheck/models/lstm.py`.

`encoder/` holds the base encoder baselines (T06): DistilBERT and BERT on COVID, LIAR and
PUBHEALTH. Defaults are in `factcheck/models/encoder.py`. Add `data: {subset: 300}` for a quick
debug run; the app and the specs check leave subset runs out. T17 adds RoBERTa, DeBERTa-v3-base
and ModernBERT, and every base encoder on ISOT with source shortcuts removed
(`*_isot_clean.yaml`, head 128 + tail 382 tokens). Run the whole sweep, 3 seeds each, with
`jobs/queue.sh jobs/t17_encoders.txt` (one worker per GPU).

`llm/` holds the prompted Qwen3-8B runs (T07 zero-shot). They train nothing and need a
cluster GPU (about 17 GB in bf16).
