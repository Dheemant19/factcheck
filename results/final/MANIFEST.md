# Results freeze (T43)

* Frozen file: `results/final/runs_final.csv` (502 rows, 125 runs)
* SHA-256: `3cb37c4d7f61ec952ebf9a169c7f264cae4d3795dc3edcfa62a07426ec092f7e`
* Frozen on 2026-11-11 00:41 at commit `70960af`

Check it with `python -c "import hashlib; print(hashlib.sha256(open('results/final/runs_final.csv','rb').read()).hexdigest())"`. After the freeze, any change to a reported number needs the whole team's agreement.

## Runs per family and dataset (test split)

| family | climate_fever | covid | isot | liar | pubhealth |
|---|---|---|---|---|---|
| classic | 1 | 5 | 6 | 5 | 5 |
| encoder | 2 | 13 | 8 | 22 | 23 |
| knn | 0 | 1 | 1 | 1 | 1 |
| llm | 3 | 3 | 1 | 3 | 4 |
| lstm | 0 | 5 | 0 | 5 | 5 |

## Warnings at freeze time

* 2 trained configs with fewer than 3 seeds: deberta_v3_base_pubhealth_tuned_evidence_to_climate_fever, scibert_pubhealth_to_climate_fever
* 255 of 502 rows were logged with uncommitted code changes (-dirty), from 76 runs
