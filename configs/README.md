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

`features` switches the two input extras of the ablation (E5) on and off without code changes
(T15; both are off when left out). `use_metadata` (LIAR only) prefixes each statement with
`[speaker: ... | party: ... | subject: ... | context: ...]`. `use_evidence` (PUBHEALTH and
Climate-FEVER only) pairs each claim with its `evidence_k` most similar evidence sentences from
the T08 retrieval cache. Every model family reads its inputs through
`factcheck.inputs.load_inputs`, so a switch means the same for the Bi-LSTM, the encoders and
the LLM (for the LLM, evidence is the RAG mode of T20). Examples:
`encoder/deberta_v3_base_liar_meta.yaml` and `encoder/deberta_v3_base_pubhealth_evidence.yaml`.

`knn/` holds the kNN-vote baselines (T08): no training, just the similarity-weighted verdict
of the 8 most similar training claims.
