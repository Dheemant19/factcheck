# configs

One YAML per run: dataset, model, method, hyperparameters, seed (plan Section 16).
Run with `python -m factcheck.train --config configs/<file>.yaml`.

`template.yaml` lists the fields every run needs; copy it to start a new run.

`classic/` holds the traditional baselines (T54, added after the Week 12 consultation):
`tfidf_nb`, `tfidf_lr` and `tfidf_svm` (TF-IDF word 1-2 grams plus character 2-5 grams, word
n-grams only on ISOT) and `glove_svm` (the mean GloVe 300-d vector of the words), one per
dataset, plus `tfidf_lr_isot_raw` for the raw vs clean comparison. The regularisation is picked
on validation from the grid in `factcheck/models/classic.py`; fitting is deterministic, so they
run once (seed 42), on a CPU, in seconds to minutes. Each run also writes `top_words.csv`.

`lstm/` holds the Bi-LSTM baselines (T05), one per dataset. Settings a config leaves out
take the plan Section 10 defaults in `factcheck/models/lstm.py`.

`encoder/` holds the base encoder baselines (T06): DistilBERT and BERT on COVID, LIAR and
PUBHEALTH. Defaults are in `factcheck/models/encoder.py`. Add `data: {subset: 300}` for a quick
debug run; the app and the specs check leave subset runs out. T17 adds RoBERTa, DeBERTa-v3-base
and ModernBERT, and every base encoder on ISOT with source shortcuts removed
(`*_isot_clean.yaml`, head 128 + tail 382 tokens). Run the whole sweep, 3 seeds each, with
`jobs/queue.sh jobs/t17_encoders.txt` (one worker per GPU).

`llm/` holds the prompted Qwen3-8B runs (T07 zero-shot). They train nothing and need a
cluster GPU (about 17 GB in bf16). T20 adds `method: knn` (the 8 most similar training claims
and their verdicts in the prompt) on all four datasets, and `method: rag` with
`features.use_evidence` (evidence sentences in the prompt) on PUBHEALTH and Climate-FEVER, own
evidence and open retrieval from the pool. The ISOT runs set `data.llm_subset: true` and score
only the fixed 2,000-article test subset. Run them all with `jobs/queue.sh jobs/t20_llm.txt`.
T27 adds `method: qlora` (`qwen3_8b_qlora_*.yaml`): Qwen3-8B in 4-bit with LoRA and a
classification head, trained through the encoder trainer; `model/` holds the adapters and head.

`encoder/lora/` holds the T26 LoRA sweep (`method: lora`, `lora_r`, `lora_targets: attention`
or `all_linear`) on roberta-large and deberta-v3-large, LIAR and PUBHEALTH. They use lr 1e-4
and 10% warmup; roberta-large at rank 64 also needs `lora_alpha: 32` (the default 2 x rank
collapsed it to one class).

`encoder/*_tuned_{meta,evidence,cw}.yaml` are the T29 ablations: a T18 tuned config with one
switch on (metadata, evidence, or `train.class_weights: balanced`). `roberta_isot_raw.yaml`
and `modernbert_covid_clean.yaml` are the other halves of the T28 raw vs clean comparison.

`features` switches the two input extras of the ablation (E5) on and off without code changes
(T15; both are off when left out). `use_metadata` (LIAR only) prefixes each statement with
`[speaker: ... | party: ... | subject: ... | context: ...]`. `use_evidence` (PUBHEALTH and
Climate-FEVER only) pairs each claim with its `evidence_k` most similar evidence sentences from
the T08 retrieval cache. Every model family reads its inputs through
`factcheck.inputs.load_inputs`, so a switch means the same for the Bi-LSTM, the encoders and
the LLM (for the LLM, evidence is the RAG mode of T20). Examples:
`encoder/deberta_v3_base_liar_meta.yaml` and `encoder/deberta_v3_base_pubhealth_evidence.yaml`.

`encoder/*_tuned.yaml` are written by the Optuna search (T18), not by hand: the best trial of
`python -m factcheck.tune` as a run config (`jobs/t18_tune.sh` runs the search on the cluster;
the studies are in `results/optuna/`). They add two settings the search varies,
`model.dropout` and `model.freeze_layers`.

`knn/` holds the kNN-vote baselines (T08): no training, just the similarity-weighted verdict
of the 8 most similar training claims.
