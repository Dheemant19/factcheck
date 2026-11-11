# What the baselines, encoders and probes mean (M1)

Interpretation paragraphs for M4 (report and slides) on M1's experiments: T54 (TF-IDF and
GloVe baselines), T05 (Bi-LSTM), T06, T17 and T19 (encoders), T18 (Optuna), T38 (Climate-FEVER)
and T58 / T59 (style probes and style-stripped COVID). Each section says what we saw, why, and
what it cost. Every number is test macro-F1 from the frozen `results/final/runs_final.csv`
(`v1.0-results`) unless it says otherwise. Encoders are means over seeds 42, 13 and 2026; the
TF-IDF and GloVe baselines are deterministic and have one run. M2 covers the recurrent cells
(T55), raw vs cleaned text (T28) and the improvements (T29); M3 covers the data, retrieval,
topics and significance tests.

## In short

* **TF-IDF with logistic regression is the baseline to beat, and it is hard to beat.** It is
  within 0.03 of the best encoder on COVID (0.953 vs 0.980), PUBHEALTH (0.487 vs 0.509) and
  ISOT (0.988 vs 1.000). Only on LIAR do transformers pull clearly ahead (0.237 vs 0.296).
* **Its top words show what all the models read: how a dataset's sources write, not whether a
  claim is true.** "rt", "drtedros" and "data" mean real on COVID; "said on tuesday" means real
  on ISOT.
* **Going up the ladder buys little per step and costs a lot.** Averaged GloVe is below TF-IDF
  everywhere, the Bi-LSTM only matches it, and a large encoder needs up to 1,700 times the
  weights and 50 times the time per claim for a few points.
* **Tuning made the encoders steadier, not better.** It changed nothing on COVID and LIAR. Its
  0.04 gain on PUBHEALTH comes from one default seed that collapsed, and the seed spread there
  fell from 0.103 to 0.016.
* **LIAR stays near 0.30 and PUBHEALTH's rare labels stay near 0.3 F1 for every model**,
  because the text alone does not hold the answer.
* **Models trained on one dataset do not check facts elsewhere.** On Climate-FEVER they score
  at chance, while Qwen3-8B with no training gets 0.75 ROC-AUC from what it already knows.

## 1. TF-IDF + Naive Bayes, logistic regression or SVM (T54; Week 3)

**What we saw.** Logistic regression was the best of the three on validation on every dataset:
0.953 COVID, 0.237 LIAR, 0.487 PUBHEALTH and 0.988 ISOT (cleaned). Its strongest words per label
(`outputs/tfidf_lr_*/seed42/top_words.csv`) are mostly about the source and its style:

| Dataset | Push towards false / fake | Push towards true / real |
|---|---|---|
| COVID | news, corona, covid 19, trump, coronavirusfacts, fact | rt, drtedros, via, states, cases, data, tests |
| ISOT (cleaned) | read more, gop, just, president trump, watch, obama | said, said on, president donald, on tuesday, on wednesday |
| PUBHEALTH | obama, says, the coronavirus, gates, post says | health, case, uk, ebola, report, outbreak |

**Why.** TF-IDF can only count words, so it scores well exactly when the labels are tied to
words. On COVID the real posts come from health agencies ("@DrTedros", case counts) and the
fake ones from fact-check roundups ("#coronavirusfacts"). On ISOT the real articles are Reuters
copy, and "said on tuesday" is Reuters house style. The PUBHEALTH words for false ("post
says", "obama", "gates") are the topics fact-checkers pick, not signs of falsehood. On LIAR a
claim is one short sentence with no source attached, so there is little to count.

**What it cost.** Seconds to a minute on a laptop CPU, 260,000 to 385,000 weights, and under
1 ms per claim. It is also the only model whose explanation is exact: each word's weight is
the model, which is why the app uses it to show word pushes.

**A lesson from the SVM.** The calibrated SVM first scored 0.400 on PUBHEALTH, against 0.467
for the plain SVM on validation. The sigmoid calibration that turns SVM margins into
probabilities had been fitted without the class weights, so it undid them and the model went
back to predicting the common labels. With the calibration weighted too it scores 0.446. A
model's reported probabilities are part of the model, not a free extra.

## 2. Averaged GloVe vectors + SVM (T54; Weeks 4.1 and 4.2)

**What we saw.** Below TF-IDF on every dataset: 0.916 COVID, 0.221 LIAR, 0.411 PUBHEALTH and
0.975 ISOT.

**Why.** Averaging 300-d word vectors keeps the general meaning of a text and loses the few
words that carry the label. "@DrTedros" and "#coronavirusfacts" become a small nudge in a
300-number average, next to every ordinary word. The vectors are also generic (trained on
Wikipedia and news), so the dataset's own markers have no special place in them. Embeddings
help when similar words should count the same; here the exact words were the signal.

**What it cost.** About as cheap as TF-IDF (301 weights per label), but the 400,000-word GloVe
file has to be loaded first.

## 3. Bi-LSTM on frozen GloVe (T05; Week 5)

**What we saw.** 0.954 ± 0.004 COVID, 0.242 ± 0.013 LIAR and 0.463 ± 0.021 PUBHEALTH: level with
TF-IDF on COVID and LIAR, and below it on PUBHEALTH.

**Why.** Reading words in order should help when meaning depends on order ("not safe" vs
"safe"). But the claims are short, most of the signal is in single words and pairs that
TF-IDF already counts, and with frozen generic vectors and 6,000 to 10,000 training claims the
LSTM cannot learn much more. M2's T55 runs show the gates matter more than the second
direction.

**What it cost.** 2.7 million weights, 12 to 18 seconds per run on a GPU and about 1 ms per
claim: cheap, but 7 to 10 times TF-IDF's weights for no gain.

## 4. Transformer encoders: base, large and domain (T06, T17, T19)

**What we saw.** The best base encoder per dataset scores 0.966 COVID, 0.288 LIAR, 0.496
PUBHEALTH and 1.000 ISOT; the best large or domain encoder 0.980, 0.296, 0.509 and 1.000. The
largest gain over TF-IDF is on LIAR (+0.06 macro-F1, and ROC-AUC 0.733 vs 0.639). Going from
base to large or domain encoders adds only 0.008 to 0.014 on COVID, LIAR and PUBHEALTH, close to
the 0.01 to 0.03 that seeds alone move a score. Bigger is not always better: deberta-v3-large
scores 0.478 on PUBHEALTH, below SciBERT's 0.509, which is smaller but pretrained on scientific
text.

**Why.** Pretraining on billions of words gives the encoders a sense of meaning and context
that counting cannot, which helps most on LIAR, where the label depends on what the sentence
says rather than on source markers. Where the source markers already decide the label (COVID,
ISOT), there is little left to gain. Domain pretraining helps more than size when the
vocabulary is specialised (SciBERT on health claims, COVID-Twitter-BERT on tweets).

**What it cost.** 110 to 435 million weights, 1 to 7 minutes per run on an RTX 3090 (up to an
hour on ISOT's long articles), 2.2 to 9.8 GB of GPU memory and 8 to 40 ms per claim, against
TF-IDF's few hundred thousand weights, a CPU and under 1 ms. For COVID, PUBHEALTH and ISOT, that
is a few points for 400 to 1,700 times the weights.

## 5. Optuna tuning of DeBERTa-v3-base (T18)

**What we saw.** Tuned vs default test macro-F1: 0.972 vs 0.971 COVID, 0.275 vs 0.278 LIAR and
0.504 vs 0.465 PUBHEALTH. The PUBHEALTH gain comes from one default seed (42) that collapsed to
0.348, while the other two scored 0.542 and 0.504. What changed was stability: the PUBHEALTH
spread across seeds fell from 0.103 to 0.016. The most important
setting (fANOVA, `results/optuna/*_importance.json`) was dropout on LIAR (0.51) and PUBHEALTH
(0.68), and the warm-up share on COVID (0.45).

**Why.** The default settings were already sensible, so tuning could not find much more score.
On the small, imbalanced PUBHEALTH set some default seeds collapsed to the common labels;
the tuned dropout and warm-up stopped that, which is a gain in reliability that a single-seed
score would hide.

**What it cost.** 90 trials on 4 GPUs in 33 minutes, plus three seeds of each tuned config.

## 6. Why LIAR stays near 0.30

**What we saw.** Every model on LIAR scores between 0.18 and 0.30. The best are
deberta-v3-large (0.296) and the tuned DeBERTa-v3-base with speaker metadata (0.296, up from
0.275 without it).

**Why.** A LIAR claim is one short political sentence, and its six labels ("barely true" vs
"half true") are judged against facts and context that are not in the sentence. Metadata
(speaker, party, context) adds 0.02 because who says something is a real clue to how it will
be rated, but that is a prior about the speaker, not a check of the claim. The ceiling is in
the data, not the model.

## 7. Why PUBHEALTH's rare labels fail

**What we saw.** Per-label F1 on the 1,233 test claims (388 false, 201 mixture, 599 true, 45
unproven), mean over seeds:

| Model | false | mixture | true | unproven |
|---|---|---|---|---|
| TF-IDF + LR | 0.56 | 0.35 | 0.76 | 0.28 |
| SciBERT | 0.63 | 0.34 | 0.79 | 0.28 |
| Tuned DeBERTa-v3-base | 0.64 | 0.32 | 0.79 | 0.26 |
| + class weights (M2, T29) | 0.63 | 0.35 | 0.79 | 0.38 |
| + evidence (M2, T29) | 0.71 | 0.36 | 0.83 | 0.35 |

**Why.** "Mixture" and "unproven" are not properties of the wording: a claim is unproven
because no evidence exists, and mixture because part of it holds. The text of the claim cannot
show either, and with only 45 unproven test claims each mistake moves its F1 a lot. Class
weights help the rarest label most, and evidence helps every label, because it brings in the
missing information. The published 0.705 (Kotonya & Toni 2020) is still far above our 0.562.

## 8. The models learn a dataset's style, not falsehood (T58 probes, T59)

**What we saw.** Hand-written probes on the two COVID app models, P(fake):

| Probe | TF-IDF + LR | COVID-Twitter-BERT |
|---|---|---|
| "WHO: Washing your hands ... helps stop the spread of #COVID19." | 0.02 | 0.00 |
| The same tweet, but washing hands "spreads #COVID19" | 0.27 | 0.96 |
| "WHO: Drinking bleach for 20 seconds helps stop the spread of #COVID19." | 0.05 | 1.00 |
| "CDC data: an adult human skeleton has 206 bones." (true) | 0.71 | 1.00 |
| The same with "2006 bones" | 0.65 | 1.00 |

Stripping the surface style (links, @mentions, "RT", emoji, hashtags, "!", capitals) before
training (T59) costs 1 to 3 points on COVID: TF-IDF 0.953 to 0.927, DistilBERT 0.966 to 0.945,
COVID-Twitter-BERT 0.980 to 0.968. The same stripping changes nothing on the other datasets
(LIAR 0.237 to 0.250, PUBHEALTH 0.487 to 0.479, ISOT 0.988 to 0.988).

**Why.** TF-IDF calls the bleach tweet true because it is written like a WHO tweet. The
encoder catches swapped facts inside its own topic, but calls a true fact about bones fake,
with full confidence, because it is off-topic for COVID tweets. Neither model has a notion of
true; each has learned what true and false posts looked like in its training data. Formatting
explains only part of that (1 to 3 points); the rest is wording, which stripping cannot
remove.

**What it cost.** Five sentences and a minute in the app; style stripping is one setting
(`features: strip_style: true`) and one re-run per model.

## 9. Climate-FEVER: does any of it transfer? (T38)

**What we saw.** PUBHEALTH-trained models scored on Climate-FEVER with no Climate-FEVER
training are at chance: DeBERTa-v3-base with evidence 0.185 macro-F1 (supports-vs-refutes
ROC-AUC 0.597), SciBERT 0.169 (0.506) and TF-IDF 0.177 (0.533). Qwen3-8B zero-shot, with no
evidence and no training, gets 0.398 (0.750), and with retrieved evidence (M2's RAG run) 0.479
(0.862).

**Why.** A classifier trained on PUBHEALTH learned which health-claim wordings tend to be true;
none of that applies to climate claims. The LLM brings general knowledge from pretraining and
can weigh evidence it is given, which is closer to what checking a claim means. This is the
strongest evidence in the project that high in-domain scores do not mean the models check
facts.

**What it cost.** Inference only: a few minutes per model, and the Qwen3-8B runs about 130 ms
per claim on a 24 GB GPU.

## Caveats to carry into the report

* TF-IDF, GloVe and prompted Qwen3-8B runs are deterministic (one run); the encoder numbers are
  3-seed means, with standard deviations of 0.01 to 0.03 on LIAR and PUBHEALTH, so differences
  under about 0.02 are within seed noise. M3's significance tests (T31) have not been run on
  the full predictions yet; once they are, use them before calling any difference real.
* The probes in section 8 are five hand-written sentences: they illustrate the failure mode,
  they do not measure how often it happens.
* "Weights" counts trainable parameters as logged; the calibrated SVMs log three times their
  real size (three calibration folds).
