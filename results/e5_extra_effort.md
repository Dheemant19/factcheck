# E5: what each extra adds (test macro-F1, mean over seeds)

Each step is measured against the run it changes (tuning: against the default
DeBERTa-v3-base; every other step: against the tuned one). The steps were trained one
at a time, so the gains do not add up. ± is one standard error of the difference.
Written by `python -m factcheck.extra_effort`.

| Dataset | Step | Macro-F1 | Reference | Gain | p (Holm) |
|---|---|---|---|---|---|
| COVID | Optuna tuning | 0.972 | 0.971 | +0.001 ± 0.002 | 1.000 |
| COVID | + speaker metadata | not applicable | | | |
| COVID | + evidence sentences | not applicable | | | |
| COVID | + class weights | 0.971 | 0.972 | -0.001 ± 0.000 | 1.000 |
| COVID | domain model instead | 0.981 | 0.972 | +0.008 ± 0.001 | 0.097 |
| COVID | top-3 ensemble instead | not run yet | | | |
| LIAR | Optuna tuning | 0.275 | 0.278 | -0.003 ± 0.012 | 1.000 |
| LIAR | + speaker metadata | 0.296 | 0.275 | +0.021 ± 0.010 | 1.000 |
| LIAR | + evidence sentences | not applicable | | | |
| LIAR | + class weights | 0.277 | 0.275 | +0.002 ± 0.009 | 1.000 |
| LIAR | domain model instead | not applicable | | | |
| LIAR | top-3 ensemble instead | not run yet | | | |
| PUBHEALTH | Optuna tuning | 0.503 | 0.465 | +0.039 ± 0.060 | 1.000 |
| PUBHEALTH | + speaker metadata | not applicable | | | |
| PUBHEALTH | + evidence sentences | 0.562 | 0.503 | +0.059 ± 0.017 | 1.000 |
| PUBHEALTH | + class weights | 0.535 | 0.503 | +0.032 ± 0.010 | 1.000 |
| PUBHEALTH | domain model instead | 0.509 | 0.503 | +0.005 ± 0.013 | 1.000 |
| PUBHEALTH | top-3 ensemble instead | not run yet | | | |
