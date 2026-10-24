# ISOT and the 80/10/10 split: what we did and why

Notes for M3 (data and retrieval lead) on task T03, also useful for the M5 review, the
Week 12 consultation and the 1:20–1:45 part of the video. Every number below was measured on
the Kaggle release of ISOT with the code on branch `t03-loaders-splits`.

## In short

* ISOT is the only one of our datasets with **no official split**, so we make one.
* We build each article as `title. body`, drop the 631 articles with an empty body, drop
  5,609 exact duplicates, and keep **38,658 articles**.
* We split them **80/10/10, stratified by label, seed 42**: 30,926 train, 3,866 validation,
  3,866 test. All three splits are 45.2% fake.
* From the test split we take a fixed, stratified **2,000-article subset** for the 8B LLM.
* The split is saved in `data/splits/isot.csv` and committed. Loaders read that file instead of
  re-splitting, so every machine uses exactly the same rows.

## 1. What ISOT is

The ISOT Fake and Real News dataset (Ahmed et al., 2017; University of Victoria) has two files:

| File | Articles | Where they come from | Dates |
|---|---|---|---|
| `True.csv` | 21,417 real | Reuters news wire | Jan 2016 – Dec 2017 |
| `Fake.csv` | 23,481 fake | Sites flagged by PolitiFact and Wikipedia as unreliable | Mar 2015 – Feb 2018 |

Each row has `title`, `text` (the article body), `subject` and `date`. Articles are long: the
median is 381 words and the 95th percentile 891 words, which is why ISOT uses 512 tokens
(head 128 + tail 382) instead of the 128 we use for COVID posts and LIAR claims.

**Why it is in the project.** It is long-form news, unlike our other datasets, which are
short claims. It is also well known for being easy for the wrong reasons. Published models
score close to 100%, largely because the two classes come from different sources that leave
fingerprints (Section 6). That makes it the centre of our shortcut study (E7, T13) and of
M4's interpretation segment.

## 2. From raw files to 38,658 articles

The code is `read_isot` in `factcheck/data/readers.py`. The steps run in this order:

| Step | Fake | Real | Total |
|---|---|---|---|
| Raw rows | 23,481 | 21,417 | 44,898 |
| Drop articles whose body is empty | −630 | −1 | −631 |
| Drop exact duplicates (same title and body) | −5,389 | −220 | −5,609 |
| **Kept** | **17,462** | **21,196** | **38,658** |

**Text = `title + ". " + body`.** The headline carries much of the claim, and the app and
the LLM see the headline first. The same string is kept untouched as `text_raw`; `text` is
the lightly cleaned version (encoding repair, HTML, whitespace).

**Empty bodies are dropped.** 630 fake articles but only 1 real article have no body, just
a headline. If we kept them, "headline only" would almost always mean fake, and a model could
learn that pattern instead of anything about the content. The M3 guide said to keep or drop
them and write the decision down; this is the decision.

**Each article's ID is a fingerprint of its text.** ISOT has no IDs, so we use the first 16
hex characters of the SHA-1 hash of `text_raw`. This has three useful properties:

* the same article always gets the same ID, whatever order the rows are in;
* the ID comes from the raw text, so the raw and cleaned (T13) versions of an article share an
  ID, and E7 can compare them row by row;
* two different articles getting the same ID is practically impossible (about 4 in 100
  billion for 39,000 articles). The loader checks IDs are unique anyway and stops if they
  are not.

**Exact duplicates are dropped *before* splitting.** Fake-news sites repost the same story,
so 5,389 fake rows are copies. If one copy landed in train and another in test, the model
would be graded on an article it had already memorised, which inflates the test score.
Removing copies first guarantees no article appears in two splits. No duplicate pair had
conflicting labels.

## 3. Why 80/10/10

Each split has one job:

| Split | Size | Used for |
|---|---|---|
| Train (80%) | 30,926 | Learning the model weights |
| Validation (10%) | 3,866 | Early stopping, Optuna tuning, choosing the top 3 models for the ensemble (T31) |
| Test (10%) | 3,866 | The final score, used **once per final model** |

The proportions trade off two things: more training data versus more precise evaluation.

**Training data.** 80% already gives about 31,000 training articles, which is a lot for
fine-tuning a pre-trained encoder. Going to 90% would add about 3,900 (13% more), which is
unlikely to matter much at this size. The stretch experiment E9 (training on 1%, 10%, 30%
and 100% of the data) can show this directly.

**Evaluation precision.** The test score is an estimate, and a smaller test set makes it
noisier. For an accuracy *p* on *n* articles, the 95% margin of error is about
1.96 × √(p(1−p)/n):

| Split | Test articles | Margin at 95% accuracy | Margin at 99% accuracy |
|---|---|---|---|
| 90/5/5 | 1,933 | ±0.97 pp | ±0.44 pp |
| **80/10/10 (ours)** | **3,866** | **±0.69 pp** | **±0.31 pp** |
| 70/15/15 | 5,799 | ±0.56 pp | ±0.26 pp |
| 60/20/20 | 7,732 | ±0.49 pp | ±0.22 pp |

ISOT models score in the high 90s, so differences between them are small. With 3,866 test
articles the margin is well under 1 percentage point, which is enough to check spec S5 (≥ 0.95
on the cleaned text) and to separate models, together with the paired bootstrap and McNemar
tests in T31. Going to 70/15/15 would narrow the margin by only about 0.1 pp and cost about
3,900 training articles.

**Consistency with the other datasets.** LIAR (10,269 / 1,284 / 1,283) and PUBHEALTH
(9,832 / 1,225 / 1,235 in the source) are also 80/10/10, so all our datasets except COVID
(official 60/20/20) use the same proportions. It is also the split the plan fixed in Section 6.

### Alternatives we did not use

| Option | Why not |
|---|---|
| k-fold cross-validation | Five folds means five times the training cost for every encoder, LoRA and QLoRA run, which breaks the compute budget (spec S9). The dataset is large enough that one held-out test set is reliable. |
| Split by date (train on older, test on newer) | The classes cover different periods: 2,891 fake articles are older than the first real one. A date split would give splits with very different fake shares, so it would measure the date gap, not the model. |
| Split by `subject` (topic) | Subject values never overlap between classes (fake: News, politics, left-news…; real: politicsNews, worldnews), so a subject split would separate the classes themselves. |
| No validation set (tune on test) | Choosing settings on the test set makes the test score look better than it is. The validation set exists so the test set is only used at the end. |

## 4. Stratified, seed 42, and a committed file

**Stratified** means each split keeps the same fake/real ratio as the whole dataset. A plain
random split of 38,658 rows would land close anyway, but stratifying makes it exact for free:
the measured fake shares are 45.17% train, 45.16% validation and 45.19% test. Equal class
balance means accuracy and F1 mean the same thing in every split.

**Seed 42** is an arbitrary but fixed number for the random shuffle; any fixed number would
do. What matters is that it never changes.

**The committed file, not the seed, decides the split.** In principle the seed makes the
split reproducible, but the result also depends on row order and on the scikit-learn and
pandas versions. The cluster and five laptops won't all have identical versions. So the split
was made once, with `python scripts/make_splits.py`, and the result is committed as
`data/splits/isot.csv` (columns `id, split, in_llm_subset`). `load_splits("isot")` reads that
file and **refuses to load** if the raw data has articles the file doesn't list, or vice
versa, instead of silently re-splitting. Re-making the file changes every ISOT number, so it
needs the whole team's agreement.

The other datasets have split files too, even though they keep their official splits. That
way every loader works the same way, and the files double as a check that everyone has the
same download.

## 5. The 2,000-article LLM subset

The 8B LLM (Qwen3-8B) is by far the slowest model to run on long articles, especially with
retrieved examples in the prompt. Instead of the full 3,866-article test set, LLM runs on ISOT
use a fixed subset of **2,000 test articles** (904 fake, 1,096 real: 45.2% fake, the same as
the full test set).

* It is drawn **from the test split only**, after the split, so train and validation are
  untouched (`in_llm_subset` is never true outside test).
* It is **stratified**, so its class balance matches the full test set.
* **Every encoder is also scored on the same 2,000 articles**, so the LLM vs encoder
  comparison uses identical articles (plan Section 6). Encoders can still be scored on the
  full test set as well.
* At 2,000 articles the margin of error is about ±0.96 pp at 95% accuracy: a little wider than
  the full test set but still under 1 pp, for roughly half the LLM compute.

Use it with `test[test["in_llm_subset"]]`.

## 6. Known caveats (useful for T13 and M4)

These are properties of ISOT, not bugs in our split, but they decide what our ISOT numbers
mean.

**Source fingerprints (shortcuts).**

| Cue | Real | Fake |
|---|---|---|
| Body contains `(Reuters)` | 99.2% | 0.04% |
| `subject` values | politicsNews, worldnews | News, politics, left-news, Government News, US_News, Middle-east |
| `date` format | Always `December 31, 2017 ` (full month name, trailing space) | Never a trailing space; 49% use a short month (`Dec 1, 2016`) |
| Empty body | 1 | 630 (dropped, Section 2) |

`subject` and `date` are kept as columns for analysis and the app's shortcut demo, but they
are **never model input**. The Reuters dateline is inside the text itself, so the `raw`
variant still contains it; T13's `clean` variant removes it, along with the other
collection fingerprints (`docs/m3_retrieval_and_shortcuts.md`). The gap between raw and
clean accuracy is the result of experiment E7.

**Near-duplicates (handled in T13).** Exact copies were removed before splitting, but some
articles are near-copies, such as the same headline with a slightly edited body, or a Reuters
story updated later. T13 found 335 training articles that copy a validation/test article
apart from case, punctuation or a few words (embedding similarity ≥ 0.95 and the same
numbers). They are listed in `data/splits/exclusions.csv` and dropped from train; the test
split is unchanged. Details in `docs/m3_retrieval_and_shortcuts.md`.

**A random split tests the same news period.** Train and test articles come from the same
2015–2018 news cycle, so ISOT test scores measure performance on familiar topics and sources.
How well models do on new domains is what the cross-domain experiments (E8) and Climate-FEVER
are for.

## 7. Checking and reproducing

```bash
python scripts/make_splits.py --check        # reloads every dataset and checks it matches its split file
pytest tests/test_data.py                     # 24 tests, including the ISOT split checks
```

```python
from factcheck.data import load_splits
isot = load_splits("isot")                    # {"train", "validation", "test"}
llm_test = isot["test"][isot["test"]["in_llm_subset"]]
```

Where things are: `read_isot` (cleanup and IDs) in `factcheck/data/readers.py`;
`stratified_split` and the split-file checks in `factcheck/data/prepare.py`; the tests
`test_isot_split` and `test_stratified_split_sizes_and_llm_subset` in `tests/test_data.py`.

## 8. Questions you may be asked

**"Why not just use the 44,898 articles everyone else uses?"** About 14% of them have an
empty body or are exact duplicates. Duplicates across train and test inflate accuracy; empty bodies are a fake-only
cue. Our number is lower but honest, and the drops are listed in Section 2.

**"Is ISOT too easy to be useful?"** On the raw text, nearly. That is why we use it: the
raw-vs-clean gap (E7) measures how much of the accuracy came from source fingerprints.

**"Why is the test set used only once?"** Every look at the test set that changes a decision
tunes the model to it a little. Choices are made on validation; test scores are reported at
the end (results freeze, T43).

**"Would a different split seed change the results?"** A little: a different random test set
moves scores by roughly the margins in Section 3. We keep one fixed split so every model is
compared on the same articles, train final models with 3 training seeds, and use
significance tests (T31) to judge which differences are real.
