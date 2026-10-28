# Prompts and input switches: what we did and why

Notes for M3 on T14 (retrieval input builders) and T15 (LIAR metadata and evidence switches).
M1 + M2 use them in T20 (LLM retrieval modes) and T29 (extra-effort ablation, E5).

## In short

* **T14** turns retrieval results into model inputs: kNN few-shot prompts and evidence (RAG)
  prompts for Qwen3-8B (`factcheck/prompts.py`), and (claim, evidence) pairs for encoders.
  `factcheck.retrieval.knn_examples` and `evidence_for` fetch what goes into them.
* **T15** adds a `features` block to run configs: `use_metadata` (LIAR) and `use_evidence`
  (PUBHEALTH, Climate-FEVER), both off by default. `factcheck.inputs.load_inputs(cfg)` applies
  them in one place, and the Bi-LSTM, encoder and LLM trainers all load data through it.

## 1. The prompts (T14)

| Builder | What the model sees |
|---|---|
| `zero_shot_messages` (T07, moved here) | task, label definitions, the claim, the answer instruction |
| `knn_messages` | the same, plus 8 similar training claims with their verdicts, most similar last |
| `rag_messages` | the same, plus numbered evidence sentences and "judge using only this evidence" |
| `encoder_pair` | (claim, evidence joined), for `tokenizer(claim, evidence, truncation="only_second")` |

| Decision | Why |
|---|---|
| **Answer words are the label names** (`pants-fire`, `mixture`, ...), not letters | The T07 scorer reads each label's first token, and M1 checked they all differ for Qwen3, LIAR included. Keeping the same words means zero-shot, kNN and RAG differ only in the added context. Climate-FEVER's labels were added and also differ (`supports`, `ref`, `not`, `dis`). |
| **Chat messages, not a string** | `factcheck.models.llm` applies the chat template with thinking off, so the same builders work for Llama. |
| **Same closing instruction everywhere** | "Reply with the label only, in lowercase, ..." is what lifted label coverage from 0.53 to 1.00 on PUBHEALTH in T07. |
| **Most similar example last** | It sits right above the claim being rated (the ordering the M3 guide asks for; the most relevant example is the nearest context). |
| **Examples cut to 60 words** | 8 ISOT articles would otherwise not fit; with 60 words, 8 examples plus a 512-token article stay under 2,000 tokens (tested with the Qwen3 tokenizer). |
| **Only train claims as examples; a train claim never gets itself** | Otherwise a prompt could show the answer. Tested on LIAR. |
| **Encoders get a tokenizer pair, not a literal `[SEP]`** | BERT, RoBERTa and DeBERTa use different separators; the tokenizer inserts the right one. `only_second` cuts the evidence, never the claim, and since evidence is ordered most similar first, the least similar sentence is cut first. |

The prompt text moved from `factcheck/models/llm.py` to `factcheck/prompts.py`, which needs no
PyTorch, so the app and tests can import it cheaply; `llm.py` imports it back, unchanged.

## 2. The switches (T15)

```yaml
features:
  use_metadata: true      # LIAR only
  use_evidence: false     # PUBHEALTH / Climate-FEVER only
  evidence_k: 5
  evidence_source: own    # own (the claim's article / given sentences) or pool
```

**Metadata.** `[speaker: rick perry | party: republican | subject: immigration | context: a
radio interview] Building a wall ...`. Missing fields read `unknown`. Job title and state are
left out (28% and 22% missing, T09), and the five credit-history count columns are never even
loaded: they count the speaker's past verdicts and can leak the answer.

**Evidence.** The claim stays `text`; the evidence goes in a separate `text_pair` column. The
encoder tokenises them as a pair, the Bi-LSTM reads them joined (`single_text`), and for the
LLM it is the RAG prompt (T20). PUBHEALTH's evidence is the claim's own fact-check article,
as in the dataset paper, so results compare with it; note that the article was written by the
fact-checker and can state the verdict, which makes this an optimistic setting.
Climate-FEVER's `pool` source retrieves from every evidence sentence instead, which is the
honest setting.

**Wrong switches fail loudly**: metadata on a dataset without metadata, evidence on a dataset
without evidence, or a misspelt switch raise a clear error instead of silently doing nothing.

**What changes when a switch is off: nothing.** `load_inputs` then returns exactly what
`load_splits` returns (tested for all five datasets), and the encoder's tokenised-data cache
key is built as before, so existing caches stay valid.

## 3. What is left for others

* **T20 (M1 + M2):** `score_llm` still runs zero-shot only. A kNN run calls `knn_examples` and
  `knn_messages`; a RAG run calls `evidence_for` and `rag_messages`. `use_evidence` with an
  LLM config currently stops with a message pointing there.
* **T29 (M1 + M2):** example configs `configs/encoder/deberta_v3_base_liar_meta.yaml` and
  `deberta_v3_base_pubhealth_evidence.yaml`; swap in T18's tuned settings.
* **App and SHAP (M5, M4):** a model trained with metadata or evidence expects the same input
  shape; `build_inputs` produces it from a row.

## 4. Checking

```bash
pytest tests/test_prompts.py tests/test_inputs.py
```

A one-epoch run of a tiny BERT with each switch (LIAR metadata, PUBHEALTH evidence) trained,
logged and saved its `features` block in `outputs/<run>/seed<N>/config.yaml`.
