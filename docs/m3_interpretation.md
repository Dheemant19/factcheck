# What the data, retrieval and topic results mean (M3)

Interpretation paragraphs for M4 (T57, report and slides) on M3's part: cleaning and shortcut
removal (T13), duplicates and leakage (T13), retrieval (T08), evidence and metadata (T14,
T15, measured by M2 in T29 and charted in T32), LDA topics (T56) and the significance tests
(T31). Each finding says what we saw, why, and what it cost, and names the lecture it builds
on. Numbers are test macro-F1 from `results/runs.csv` (mean over 3 seeds) unless they say
otherwise; probes are scored on validation.

## In short

* **ISOT is easy for the wrong reasons, and cleaning does not fix that.** Removing the
  Reuters dateline and 15 other collection fingerprints barely moves any model, because the
  two classes also differ in topic (LDA alone predicts the label at 0.89 macro-F1) and in
  headline style (capitalisation alone: 97.9% accuracy). ISOT measures "Reuters or not", not
  "true or not".
* **Copies across splits were real but small**, except on COVID: 6.4% of its training posts
  repeated a test or validation post, mostly reposts.
* **Similar past fact-checks carry signal where claims repeat** (kNN vote 0.915 on COVID)
  and almost none where each claim is new (0.241 on LIAR).
* **The one improvement that changes what the model can know, evidence, helps most** (+0.059
  on PUBHEALTH); tuning helps least. But after correcting for 22 comparisons, almost no gap
  in this project is beyond test-set noise on macro-F1 (Section 6).

## 1. Source fingerprints, and why removing them changes so little (T13; Weeks 1, 3, 4.1)

**What we saw.** A TF-IDF + logistic-regression probe on raw ISOT scores 0.995 on validation,
and its top words are not about the news at all: "reuters", "(", "washington" for real;
"via", "image", "getty", "pic.twitter.com" for fake, and even broken contractions ("didn",
"doesn") because the fake articles were collected with their apostrophes stripped. We wrote 16
named rules that delete these (the dateline opens 98% of real articles and no fake one; image
credits close 52% of fake ones). After cleaning, the same probe still scores 0.984, now from
style words ("said", "u.s." and weekdays against "gop", "hillary"). Transformers do not move
either: RoBERTa scores 1.000 on raw and 0.9996 on cleaned ISOT (T28).

**Why.** The fingerprints were only the most obvious difference. The two classes come from
different sources, so they also differ in what they cover (Section 5) and how they write:
Reuters headlines are in sentence case and the fake sites use Title Case, and two numbers per
headline (share of capitalised and of all-caps words) predict the label with 97.9% accuracy.
TF-IDF lower-cases its input, so its probes cannot see this; case-sensitive transformers can.

**What it cost.** The rules are cheap (regular expressions, seconds) and easy to audit, but
they only remove what you think of. The honest conclusion is about the dataset, not the model:
a near-perfect ISOT score says a model can tell Reuters from US partisan blogs, not that it
can tell true from false. A fair test needs real and fake text collected the same way
(PUBHEALTH and LIAR are, and there the best models stay at 0.30 and 0.56).

## 2. Duplicates and leakage (T13; Week 1, data preparation)

**What we saw.** Before T13, 93 COVID, 49 ISOT, 5 LIAR and 2 PUBHEALTH test items had their
exact text (ignoring case, punctuation and links) in the training set. We removed 831 training
rows that copied or nearly copied a validation or test row: 6.4% of COVID's training set, about
1% or less elsewhere. Validation and test were never changed, so the test sets stay comparable
with the published baselines.

**Why.** Social media reposts: the same fake claim circulates many times, and official
accounts post templated daily updates. A near-copy rule on embeddings alone (cosine >= 0.95)
would have removed 13% of COVID's training posts, mostly updates like "States reported 941 new
deaths" vs "2,492 new deaths", which are 0.97 similar but report different facts. Requiring the
same numbers as well kept those and removed only true copies.

**What it cost.** A slightly smaller training set (COVID loses 410 posts) and a judgement call
about what counts as "the same claim". The gain is that a COVID score no longer rewards
remembering a test post seen in training. On LIAR, 7 statements appeared twice in training with
different verdicts; no model can get both right.

## 3. Retrieval: are similar past fact-checks informative? (T08; Weeks 4.2, 7.1, 7)

**What we saw.** Each claim and every evidence sentence is embedded once with bge-small (a
sentence-embedding model; cosine similarity is a dot product). Labelling a claim with the
similarity-weighted verdict of its 8 nearest training claims, with no training at all, gives
0.915 on COVID, 0.925 on ISOT, 0.419 on PUBHEALTH and 0.241 on LIAR (macro-F1; always guessing
the bigger class scores 0.34, 0.35, 0.16 and 0.06).

**Why.** Retrieval works when claims repeat: COVID misinformation circulates in near-identical
versions, so the nearest neighbour is often the same claim already fact-checked. LIAR's
political statements are mostly new, and two statements about the same topic ("taxes went up")
can have opposite verdicts, so a similar claim says little about this one. That is also why
the LLM's kNN prompts help it most on COVID.

**What it cost.** The embeddings take minutes on a GPU but about six hours on a laptop CPU
(370,507 PUBHEALTH evidence sentences); they are computed once and cached. Only training claims
are searchable, so no prompt can show a test verdict.

## 4. What each extra adds (T14, T15, T29, T32; Weeks 6, 7)

**What we saw.** Against the Optuna-tuned DeBERTa-v3-base, each extra switched on alone:
evidence on PUBHEALTH +0.059 (0.504 to 0.562), class weights on PUBHEALTH +0.032, speaker
metadata on LIAR +0.021, COVID-Twitter-BERT on COVID +0.008. Tuning itself added about 0 on
COVID and LIAR. Class weights did nothing on balanced COVID.

**Why.** Evidence is the only extra that gives the model information it did not have: a
health claim cannot be judged from its wording, but it can be judged next to the sentences of
the fact-check. Class weights help where a class is rare (PUBHEALTH "unproven" is 3% of
claims). Metadata helps because LIAR's verdicts depend on who said it and where, which the
statement alone does not show. Tuning changes how the model learns, not what it can see, so on
a near-ceiling (COVID) or information-poor (LIAR) task it has little to work with.

**What it cost.** Evidence needs the retrieval cache and inputs of up to 384 tokens (a typical
claim is about 16 tokens, claim plus evidence about 250), so training is slower; and PUBHEALTH's evidence comes from the
fact-checker's own article, which can state the verdict, so this is an optimistic setting
(the dataset paper does the same). Metadata ties the model to a known speaker list, so it will
not help for a new account. The gains are not additive: each was trained alone, and no
combination was run.

## 5. LDA topics: do real and fake differ by subject? (T56; Week 3)

**What we saw.** LDA with 10 topics on the training text. On ISOT, the topic proportions
alone predict real vs fake with 0.89 macro-F1 (0.35 for always guessing the bigger class), and
56% of articles sit in topics that are at least 80% one label: "donald, hillary, news, obama,
clinton, media" is 34% of fake text but 3% of real; "china, north korea, nuclear, foreign minister"
is 21% of real text but 3% of fake. On PUBHEALTH the same probe gets 0.47 against 0.38, with an
accuracy half a point above guessing, and no topic is one-sided.

**Why.** ISOT's real articles are Reuters world and politics news; its fake ones come from US
partisan sites that write about US personalities and culture-war stories. A model can learn
"world news means real" without judging any claim. PUBHEALTH's true and false claims are
about the same health subjects, chosen by fact-checkers, so the subject says almost nothing
about the verdict.

**What it cost.** About two minutes on a laptop CPU, no labels needed. The lesson carries
beyond this dataset: before trusting a high score, check whether the labels can be predicted
from the subject alone.

## 6. Which differences are real? (T31; statistics)

**What we saw.** 22 planned comparisons on four datasets, each model's 3 seeds averaged,
tested with a paired bootstrap on test macro-F1 and with McNemar's test on right-or-wrong
counts, both Holm-corrected (`results/significance.csv`, run 11 Nov). On macro-F1 only one
gap survives: DeBERTa-v3-large with LoRA is worse than full fine-tuning on PUBHEALTH (−0.096,
95% interval −0.137 to −0.054). Nothing else does: not tuning, not speaker metadata on LIAR
(+0.020, interval −0.007 to +0.046), not evidence on PUBHEALTH (+0.042, −0.012 to +0.095),
not class weights, not the domain models, and not the best encoder against the best Qwen3-8B
mode on any dataset. McNemar adds two PUBHEALTH results: the evidence model gets more claims
right than the same model without evidence (Holm p = 0.046) and than BERT-base (p = 0.011).

**Why.** Macro-F1 gives each class the same weight, so a few rows of a rare class move it a
lot: PUBHEALTH's test set has 45 "unproven" claims out of 1,233, LIAR's has 92 "pants-fire"
out of 1,283. Resampling the test set therefore spreads the score widely, and with 22 tests
Holm asks the smallest p-value to be below 0.05 / 22 ≈ 0.002. McNemar counts each claim once,
so many small wins add up; that is why evidence passes it but not the macro-F1 test. The
gains here are a little smaller than in the E5 chart (evidence +0.042 against +0.059)
because the tests average each model's seeds into one prediction before scoring, while the
chart averages the three seeds' scores. The LoRA runs have one seed each.

**What it cost.** About a minute on a CPU, once the per-row predictions were collected from
the laptop and both GPU servers. For the report and video: call only the PUBHEALTH LoRA gap
significant; say evidence makes the model right on more claims, but its macro-F1 gain, like
every other extra, is within test-set noise.

## For the video (M3 segment, 1:00 to 2:30)

Show `factcheck/data/shortcuts.py` (the dateline and image-credit rules) next to one raw and
one cleaned ISOT article, then the T56 topic chart. The line to land: "We removed the words
that name the source, and accuracy did not move, because the real and fake articles are about
different things and written differently. A perfect ISOT score tells you a model can spot
Reuters, not the truth." Figures: `results/figures/t56_lda_topics.png`,
`results/figures/e5_extra_effort.png`, `results/figures/eda_isot_shortcuts.png`.
