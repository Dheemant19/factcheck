# configs

One YAML per run: dataset, model, method, hyperparameters, seed (plan Section 16).
Run with `python -m factcheck.train --config configs/<file>.yaml`.

`template.yaml` lists the fields every run needs; copy it to start a new run.

`lstm/` holds the Bi-LSTM baselines (T05), one per dataset. Settings a config leaves out
take the plan Section 10 defaults in `factcheck/models/lstm.py`.

`encoder/` holds the base encoder baselines (T06): DistilBERT and BERT on COVID, LIAR and
PUBHEALTH. Defaults are in `factcheck/models/encoder.py`. Add `data: {subset: 300}` for a quick
debug run; the app and the specs check leave subset runs out.
