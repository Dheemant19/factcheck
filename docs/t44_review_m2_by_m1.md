# T44 code review: M2's code, reviewed by M1

Plan Section 19 pairs M1 and M2 as each other's reviewers; M2's review of M1 is
`docs/t44_review_m1_by_m2.md`. This covers M2's code on `main` at `2227fb5`: the kNN and RAG
modes of `factcheck/models/llm.py` (T20), LoRA, QLoRA and class weights in
`factcheck/models/encoder.py` (T26, T27, T29), `factcheck/models/llm_explain.py` (T30), the
recurrent cells and gradient logging in `factcheck/models/lstm.py` (T55),
`scripts/t55_gradients.py`, their tests, and the T55 section of `docs/m2_interpretation.md`.

Findings come first, most serious first. Each names the file and line, what goes wrong, how it
was checked and a suggested fix. Fixes are left to M2. None of them changes a number in the
frozen `runs.csv` (`v1.0-results`); finding 1 changes how one result should be described in
the report and slides.

## Findings

### 1. The vanishing-gradient numbers hold before training, not after it (T55 write-up)

`docs/m2_interpretation.md`, section 7. The table and the text use the gradient 16 words before
the end as a share of the last word's, **before training**: 0.03% for the plain RNN against
0.3% for the GRU and LSTM, "so the early words barely affect learning". The same runs also
logged the share **after training** (`d16_over_d0_trained` in `results/t55_gradients.csv`), and
there the plain RNN is not cut off at all:

| Dataset | Plain RNN, before → after | GRU, before → after | LSTM, before → after |
|---|---|---|---|
| COVID | 0.03% → 39% | 0.26% → 74% | 0.29% → 116% |
| LIAR | 0.03% → 66% | 0.31% → 54% | 0.33% → 43% |
| PUBHEALTH | 0.03% → 13% | 0.28% → 83% | 0.31% → 37% |

So the trained plain RNN does pass signal back 16 words, as much as the LSTM on LIAR. What
separates it is earlier in training: it stopped after 5.7 to 7.3 epochs on COVID and PUBHEALTH
(best epoch, then patience) against 11 to 14 for the GRU, while clipping cut 80% to 99% of its
updates. That reads as unstable early training (vanishing at the start, exploding steps
throughout) rather than a model that cannot reach early words.

Suggested fix: say "at the start of training" next to the 0.03% figure, add the after-training
column, and give the short training and the clipping as the reason the plain RNN scores lower.
The Week 5 story still holds, and the after-training numbers make it more accurate.

### 2. `scripts/t55_gradients.py` crashes where the T05 Bi-LSTM runs exist

`scripts/t55_gradients.py:46-56`. The script reads every `outputs/lstm_<dataset>/seed*/train_summary.json`
and expects the T55 fields (`grad_norm_mean`, `grad_by_distance_init`). The T05 Bi-LSTM runs on
the laptop predate T55 and do not have them, so on that machine the script stops with
`KeyError: 'grad_norm_mean'` (checked on the laptop; `results/t55_gradients.csv` is left as
committed). It only works on gpu55, where the Bi-LSTM was re-run with T55's code.

Suggested fix: skip summaries without `grad_by_distance_init` and print which runs were
skipped, or say in the docstring that the table needs the gpu55 outputs.

### 3. Smaller points

* **The after-training gradient is from the last epoch, not the saved one**
  (`scripts/t55_gradients.py:76`). `history[-1]` is the epoch early stopping ended on, two or
  three epochs after the best epoch that `model.pt` keeps. The docstring says "after the last
  epoch", so this is consistent; it is worth one word in the write-up if finding 1's column is
  added.
* **Latency for kNN and RAG runs keys contexts by text** (`factcheck/models/llm.py:272`,
  `context_of = dict(zip(test_texts, test_contexts))`). Two identical test texts share one
  context. They get the same retrieval anyway, so nothing changes; a list of indices would make
  that explicit.
* **`load_4bit_model` pins the model to GPU 0** (`factcheck/models/encoder.py:281`, `device_map={"": 0}`). The job queue sets
  `CUDA_VISIBLE_DEVICES`, so this is right on the cluster, but a machine without a GPU gets a
  CUDA error instead of a clear message. A check with "QLoRA needs a CUDA GPU" would help a
  reader who tries it.

## Checked and fine

* **Class-weighted loss (T29).** `balanced_weights` is n / (classes × count) and refuses a class
  with no rows; `WeightedLossTrainer` uses the weighted mean cross-entropy and is covered by a
  hand-computed test. With `grad_accum: 2` (only `deberta_v3_base_covid_tuned_cw`), I checked
  that its gradient matches the plain `Trainer`'s exactly (all weights 1, tiny BERT, CPU).
  Both scale with `grad_accum`: transformers 5 does not divide a sequence-classification loss
  by the accumulation steps, because these models' `forward` takes `**kwargs`. That is in the
  shared trainer setup (M1's), affects every run with `grad_accum` above 1 the same way, and is
  harmless with AdamW, which normalises each update by its own running size.
* **LoRA (T26).** Targets are full module names inside the encoder, so a head layer named like
  an attention projection (RoBERTa's `classifier.dense`) never gets an adapter; the head and
  DeBERTa's top-level pooler train in full through `modules_to_save`; the adapters are merged
  before saving, so a LoRA run loads like any other model (tested).
* **QLoRA (T27).** NF4 with double quantisation and bf16 compute; the pad token is set before
  the head is built, so the head scores the last real token; `load_qlora` reads the base model
  from the adapter config and loads it with `for_training=False` for inference; the
  `family: llm` override keeps QLoRA rows in the LLM family.
* **kNN and RAG prompts (T20).** Retrieval sees the claim alone, never a metadata prefix; the
  kNN index is built on training claims only (`tests/test_retrieval.py`); kNN refuses a
  non-raw variant; RAG and `use_evidence` must be switched on together; ISOT's 2,000-article
  subset is scored as `test_llm_subset` with no full-test row; `use_cache=False` keeps long
  ISOT prompts inside GPU memory; retrieval time is logged apart from scoring.
* **QLoRA SHAP (T30).** The explainer tokenises exactly as training did (`encode_texts`, same
  truncation and head tokens) and runs in bf16, on M4's fixed 50 test examples, and checks its
  probabilities against the logged test predictions before explaining.
* **Recurrent cells (T55).** `CELLS` swaps the cell while keeping the attribute name, so T05
  checkpoints still load; the classifier width follows `bidirectional`; the gradient probe uses
  a fixed batch built outside the DataLoader, so it does not change the shuffle order, runs
  without dropout and cuDNN, and does not touch the parameter gradients; `clipped_fraction`
  compares the pre-clipping norm with the limit, as `clip_grad_norm_` does.
* Tests and lint pass on `main` (207 tests).
