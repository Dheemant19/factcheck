# What the training-team results mean (M2)

Interpretation paragraphs for M4 (report and slides) on M2's experiments: T20 (Qwen3-8B with
kNN examples and evidence), T26 (LoRA), T27 (QLoRA), T28 (raw vs cleaned text), T29
(improvements on and off), T30 (SHAP on the QLoRA model) and T55 (recurrent cells). Every
number is test macro-F1 from `results/runs.csv` unless it says otherwise. "3 seeds" means a mean
over seeds 42, 13 and 2026; prompted Qwen3-8B runs are deterministic and have one run, and the
LoRA and QLoRA runs have one seed (seed 42), so differences under about 0.02 there are within
seed noise (the 3-seed encoder runs have standard deviations of 0.01 to 0.03 on LIAR and
PUBHEALTH).

## In short

* **Prompting alone is not enough, and the model is badly calibrated.** Zero-shot Qwen3-8B is
  far below every fine-tuned model on LIAR (0.179) and PUBHEALTH (0.275). Eight similar
  training claims in the prompt lift it a lot (0.259 and 0.457), but its probabilities stay
  overconfident (ECE 0.37 to 0.63).
* **Fine-tuning the same 8B model with QLoRA puts it level with the best encoders** on COVID
  (0.979), LIAR (0.289) and ISOT (1.000), and makes it the best model that reads the claim alone
  on PUBHEALTH (0.515). It trains 0.5% of the weights and fits in 7 to 9 GB.
* **Evidence is the single largest improvement** on PUBHEALTH (0.504 to 0.562 for the tuned
  DeBERTa-v3-base). What counts is how good the evidence is: on Climate-FEVER, evidence found
  by open search keeps only about 40% of the gain that the dataset's own evidence gives.
* **LoRA matches full fine-tuning on roberta-large with half the memory, but not on
  deberta-v3-large,** and it does not save time.
* **The gates matter more than the second direction.** A plain RNN's gradient barely reaches
  words more than a few positions from the end. A GRU or one-direction LSTM recovers most of
  the gap to the Bi-LSTM.

## 1. Prompted Qwen3-8B: zero-shot, kNN examples, evidence (T20)

| Dataset | Zero-shot | kNN (8 examples) | Evidence (RAG) | kNN vote, no LLM (T08) | Best fine-tuned |
|---|---|---|---|---|---|
| COVID | 0.806 | 0.937 | | 0.915 | 0.980 (COVID-Twitter-BERT) |
| LIAR | 0.179 | 0.259 | | 0.241 | 0.296 (DeBERTa-v3-large; DeBERTa-v3-base + metadata) |
| PUBHEALTH | 0.275 | 0.457 | 0.420 | 0.419 | 0.562 (DeBERTa-v3-base + evidence) |
| ISOT (2,000-article subset) | 0.853 | 0.953 | | 0.932 | 1.000 |
| Climate-FEVER | 0.398 (no evidence) | | 0.479 (own evidence), 0.429 (open search) | | |

Without examples, Qwen3-8B has to guess what each dataset means by its labels, and the labels
are not self-explanatory: LIAR's six grades from "pants-fire" to "true" and PUBHEALTH's
"mixture" and "unproven" are editorial conventions. Showing the model the 8 most similar
training claims with their verdicts teaches it those conventions, and it gains 0.08 to 0.18
macro-F1 on every dataset. Much of that gain comes from the examples themselves: a plain
similarity-weighted vote of the same neighbours (no LLM) already scores 0.241 on LIAR and 0.419
on PUBHEALTH, so the LLM adds only 0.02 to 0.04 on top of the neighbours. Prompting also leaves
the probabilities badly calibrated. On LIAR and PUBHEALTH the expected calibration error is 0.37
to 0.63, against 0.05 to 0.19 for fine-tuned encoders, so a prompted model's confidence should
not be shown to users as a probability. Label coverage was 1.000 in every run, so these are real
judgements, not answers the scorer failed to read.

Evidence helps the prompted model less than examples do on PUBHEALTH (0.420 vs 0.457). The
retrieved sentences tell it what the article says, but not how PUBHEALTH's fact-checkers turn
that into "mixture" or "unproven". The same evidence helps a fine-tuned encoder far more (§5),
because the encoder learns from training data how evidence maps to each label.

## 2. Evidence quality: own evidence vs open search (T20, Climate-FEVER)

Climate-FEVER has no training split, so it is a pure test of reading evidence. Without
evidence, Qwen3-8B scores 0.398. With the 5 evidence sentences the dataset gives for each claim
it scores 0.479, and with the 5 best sentences found by open search over the whole pool of
Climate-FEVER evidence it scores 0.429. Open search therefore keeps about 40% of the gain
(0.031 of 0.081). Retrieval is the bottleneck: the model can use evidence once it has the
right evidence, and a deployed fact-checker would get only what search finds. This argues for
spending effort on the retriever (a stronger embedding model, re-ranking) before using a larger
reader.

## 3. LoRA vs full fine-tuning (T26)

| Model | Dataset | Full fine-tuning (3 seeds) | LoRA, best setting (1 seed) | LoRA range over 6 settings | Peak memory, full vs LoRA |
|---|---|---|---|---|---|
| roberta-large | LIAR | 0.285 ± 0.015 | 0.289 (r64, attention) | 0.274 to 0.289 | 6.3 GB vs 3.3 to 4.4 GB |
| roberta-large | PUBHEALTH | 0.457 ± 0.026 | 0.482 (r64, all layers) | 0.436 to 0.482 | 6.6 GB vs 2.7 to 3.5 GB |
| deberta-v3-large | LIAR | 0.296 ± 0.012 | 0.287 (r64, all layers) | 0.217 to 0.287 | 9.8 GB vs 6.1 to 7.2 GB |
| deberta-v3-large | PUBHEALTH | 0.478 ± 0.028 | 0.420 (r16, all layers) | 0.365 to 0.420 | 8.8 GB vs 4.3 to 5.0 GB |

LoRA freezes the pretrained weights and trains small low-rank updates (rank 4 to 64, 0.5% to 8%
of the parameters). On roberta-large every LoRA setting lands within the seed spread of full
fine-tuning, at about half the memory. It is also more stable: full fine-tuning of roberta-large
collapsed to predicting a single class on some seeds and needed 10% warmup to train reliably
(T39). On deberta-v3-large, LoRA
falls short, by up to 0.08 on LIAR and 0.06 to 0.11 on PUBHEALTH. Adapting only the attention
layers is clearly worse there (LIAR 0.217 to 0.221 vs 0.275 to 0.287 when all linear layers
are adapted). A plausible reading is that DeBERTa's disentangled attention and its feed-forward
layers both need adapting for a new label scheme, while RoBERTa's pretrained features already
fit the task. Rank matters less than which layers are adapted: rank 4 is often as good as rank
64. LoRA saves memory, not time: every step still runs the full forward and backward pass through
the frozen model, so wall-clock time was similar or longer (157 to 704 s vs 148 to 443 s).

## 4. QLoRA Qwen3-8B (T27)

QLoRA loads Qwen3-8B in 4-bit, adds LoRA adapters (rank 16) on every linear layer and a
classification head, and trains only those: 43.7M parameters, 0.5% of the model. It scores
COVID 0.979, LIAR 0.289, PUBHEALTH 0.515 and ISOT 1.000 (one seed each). That puts it level
with the best encoder on COVID (COVID-Twitter-BERT 0.980) and LIAR (0.296), and it is the best
claim-only model on PUBHEALTH (SciBERT 0.509, DeBERTa-v3-large 0.478). Only the encoder that
also reads evidence beats it (0.562). It is also well calibrated (ECE 0.02 to 0.14), unlike the
prompted model. The difference from prompting the same model is large: on PUBHEALTH, zero-shot
gives 0.275, kNN 0.457 and QLoRA 0.515. What the 8B model knows helps once it learns the
dataset's labels from training data, but not from instructions alone. The costs are 7 to 9 GB of
GPU memory, 15 to 20 minutes of training per dataset (2 hours on ISOT), and 75 to 210 ms per
claim, against 4 to 20 ms for base encoders. For the app that is affordable for single claims
but not for bulk scoring.

## 5. What each improvement adds (T29)

Each row switches one thing on in the tuned DeBERTa-v3-base (3 seeds each):

| Change | Dataset | Without | With | Gain |
|---|---|---|---|---|
| Evidence (5 retrieved sentences) | PUBHEALTH | 0.504 | 0.562 | +0.058 (ROC-AUC 0.903 to 0.951) |
| Class weights (balanced) | PUBHEALTH | 0.504 | 0.535 | +0.031 |
| Speaker metadata | LIAR | 0.275 | 0.296 | +0.021 (ECE 0.129 to 0.046) |
| Class weights (balanced) | LIAR | 0.275 | 0.277 | +0.002 |
| Class weights (balanced) | COVID | 0.972 | 0.971 | −0.001 |

Evidence is the biggest single gain, because PUBHEALTH's verdicts are about what the evidence
says, which the claim alone does not show. Class weights help only where the classes are
unbalanced and the metric is macro-F1: PUBHEALTH's rare "unproven" and "mixture" classes gain,
while balanced COVID does not change. Speaker metadata raises LIAR to the level of the much
larger deberta-v3-large (0.296) and makes the model better calibrated. But it works because
some speakers are rated false far more often than others, so it is a prior about the speaker,
not a check of the claim (the plan already leaves out the "credit history" counts because they
leak labels). We would present it as a useful but slightly unfair advantage.

## 6. Raw vs cleaned text (T28)

Removing the known source give-aways barely changes the scores. On ISOT, RoBERTa scores 1.0000
on raw text and 0.9996 on cleaned text (3 seeds), and ModernBERT on COVID scores 0.966 on raw
and 0.968 on cleaned. This does not show that the models are free of shortcuts. ISOT stays
almost perfectly separable after cleaning (TF-IDF + logistic regression 0.988), so other style
differences between the Reuters articles and the fake-news sites remain. M1's style-stripped
COVID runs (T59) show the same thing from the other side: removing formatting costs DistilBERT
0.021 and COVID-Twitter-BERT 0.012. The honest reading is that ISOT measures whether a model can
tell the outlets apart, not whether it can check facts, and the cross-domain tests are the
better evidence of real ability.

## 7. Plain RNN, GRU, LSTM and Bi-LSTM (T55)

| Model (frozen GloVe, 2 layers, 3 seeds) | COVID | LIAR | PUBHEALTH | Gradient 16 words before the end, as a share of the last word's, before training |
|---|---|---|---|---|
| Plain RNN, no clipping | 0.838 | 0.207 | 0.389 | 0.03% |
| Plain RNN | 0.866 | 0.195 | 0.403 | 0.03% |
| GRU | 0.964 | 0.231 | 0.457 | 0.3% |
| LSTM (one direction) | 0.958 | 0.227 | 0.420 | 0.3% |
| Bi-LSTM | 0.954 | 0.242 | 0.463 | 7% to 25% |

This is the Week 5 vanishing-gradient story measured on our data. Before any training, the loss
gradient reaching a word 16 positions before the end of a plain RNN's input is 0.03% of the
gradient at the last word, so the early words barely affect learning. The GRU and LSTM gates
pass ten times more signal back that far. The Bi-LSTM passes far more again, because its
backward direction reads the claim from the end, so the start of the claim is close to one of
its two final states. The scores follow: the gated cells beat the plain RNN by 0.09 on COVID and
0.02 to 0.05 elsewhere. A GRU or one-direction LSTM gets most of the way to the Bi-LSTM, so the
gates matter more than the second direction. Gradients also explode now and then in the plain
RNN: clipping cut 80% to 99% of its updates, and the largest gradient norm was 326 on COVID.
Without clipping it did worse on COVID and PUBHEALTH (−0.028 and −0.014) and varied far more
between seeds on PUBHEALTH (standard deviation 0.052 vs 0.009). It did slightly better on LIAR
(+0.012), where every recurrent model is near the floor. Adam already rescales each update,
which explains why clipping helps less here than it does with plain SGD. `python
scripts/t55_gradients.py` rebuilds the table, with the gradient sizes after training as well.

## 8. What the QLoRA model reads on LIAR (T30)

The SHAP explanation for the QLoRA LIAR model (`outputs/explanations/qwen3_8b_qlora_liar_seed42_shap.*`,
the same 50 fixed test claims as M4's encoder plots) reproduces the logged probabilities to
within 0.018, so it explains the model that was evaluated. Summed over the 50 claims, ignoring
stop words and words that appear only once, the words that push most towards the three "false"
labels are *tax*, *Walker*, *people*, *Scott*, *income* and *America*. Those that push most
towards the "true" side are *percent*, *McCain*, *Romney*, *Iraq*, *billion*, *said* and
*tuition*. Two patterns stand out. First, specific numbers ("percent", "billion") and attributed
statements ("said") make the model lean true. This matches how PolitiFact grades statements, as
precise, checkable statistics are more often rated true than sweeping claims about "people" or
"America". Second, names (Scott Walker vs McCain and Romney) push both ways: the model has
learned which politicians and topics tend to get which verdict. This is the same speaker prior
that the metadata switch exploits (§5), and it will not carry over to new politicians or to
other countries. Only 50 claims are explained and most words appear 2 to 11 times, so this
supports a slide example, not a statistical claim. On these 50 claims the model gets 36% right,
a little above its overall LIAR accuracy (0.285). They are the fixed sample shared with the
encoder plots, not hand-picked.

## Caveats to carry into the report

* LoRA, QLoRA and prompted-LLM rows have one run each. After the Week 12 consultation we kept it
  that way, since the prompted runs are deterministic and the LoRA sweep compares many settings
  rather than seeds. Treat LoRA differences under about 0.02 as ties.
* The Bi-LSTM seed-42 rows were re-run on the cluster for T55 (same code, different GPU), which
  moved them by 0.008 to 0.013 from the original laptop run.
* ISOT LLM rows (zero-shot, kNN) are scored on the fixed 2,000-article subset
  (`split = test_llm_subset`), so compare them with the other models' `test_llm_subset` rows,
  not their full-test rows.
