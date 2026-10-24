# Retrieval, shortcuts and leakage: what we did and why

Notes for M3 on tasks T08 (retrieval index) and T13 (shortcut removal and leakage checks),
written so M5 can review them and M4 can use the results in the shortcut study (T34). T09's
findings are in `notebooks/01_eda.ipynb`. Every number here was measured on the committed
data with the code in this branch.

## In short

* **Retrieval (T08).** Every training claim, and every PUBHEALTH / Climate-FEVER evidence
  sentence, is embedded once with `BAAI/bge-small-en-v1.5` and cached. A claim can then look
  up its most similar *already fact-checked* claims, or the best evidence sentences, with one
  matrix multiply. Only train claims are searchable, so a prompt can never show a validation
  or test verdict.
* **kNN-vote baseline.** Labelling each claim with the verdict of its 8 nearest training
  claims already gets **91.5%** accuracy on COVID test (majority class 52%). It costs
  nothing to train, so every model should beat it.
* **Shortcuts (T13).** On raw ISOT a bag-of-words model scores **99.5%** on validation,
  mostly from collection fingerprints (the Reuters dateline, image credits, embedded tweets,
  links, stripped apostrophes). `variant="clean"` removes them with 16 named rules; the same
  model then scores **98.4%**, now from writing style and vocabulary.
* **Leakage (T13).** **831 training rows** (6.4% of COVID, about 1% or less elsewhere) were
  copies or near-copies of a validation/test row. They are listed in
  `data/splits/exclusions.csv` and dropped from train. Validation and test never change.

## 1. The retrieval index (T08)

### What is embedded

| Index | Rows | Used for |
|---|---|---|
| Claim index (one per dataset) | train claims of COVID, LIAR, PUBHEALTH, ISOT, minus T13 exclusions | kNN few-shot prompts (T14), the kNN-vote baseline, the app's "similar fact-checks" panel, near-duplicate search |
| PUBHEALTH evidence | 370,507 sentences from the fact-check article of every claim (about 30 per article) | Top-5 evidence for evidence RAG and claim + evidence encoder inputs |
| Climate-FEVER evidence | 5,240 distinct evidence sentences (5 per claim, some shared) | The claim's own 5 sentences, or open retrieval from the whole pool |

Climate-FEVER has no claim index because it has no train split; it is only ever a test set.

### Decisions and why

| Decision | Why |
|---|---|
| Model `bge-small-en-v1.5` (33M parameters, 384-d) | Plan Section 8.4. Small enough for a laptop CPU, strong for its size. |
| Unit-length vectors; search = one matrix multiply in NumPy | Cosine similarity becomes a dot product. The largest index (ISOT, about 30,600 × 384) is about 47 MB, so FAISS is not needed. |
| Index **train claims only** | Otherwise a few-shot prompt could show the verdict of a validation or test claim. A test checks no held-out id is in any index. |
| A training claim used as a query never retrieves itself (`exclude_ids`) | Its nearest neighbour would be itself, with the answer attached. |
| ISOT is represented by the title + first 100 words | The lead paragraph identifies a news story. Full 512-token articles embed at about 10 per second on a CPU; 100 words is about twice as fast. |
| No query instruction for evidence search | bge offers a prefix ("Represent this sentence for searching relevant passages: "). On 150 PUBHEALTH validation claims it changed nothing: the top-5 evidence was equally close to the expert explanation (0.6961 vs 0.6963) and 90% the same sentences. |
| Evidence sentences: nltk punkt, at least 5 words | Shorter "sentences" are captions, bylines and fragments. |
| Cache: float16 `.npy` in `data/cache/retrieval/`, file name = hash of model + exact ids + texts | A change in the split, the cleaning or the model rebuilds automatically; an unchanged one loads in seconds. The full train split is always what gets embedded; T13 exclusions are filtered afterwards, so writing exclusions never forces a re-embed. |

**Cost.** On a laptop CPU (no GPU) the whole build took about 2 hours, almost all of it the
PUBHEALTH evidence sentences. On a cluster GPU it takes minutes. Either way it runs once:
`python -m factcheck.retrieval build`.

### Does retrieval carry signal? The kNN vote

Each claim gets the similarity-weighted majority verdict of its 8 nearest training claims
(`configs/knn/`, logged to `results/runs.csv` like every other model):

| Dataset | Majority class (test) | kNN accuracy (test) | kNN macro-F1 (test) |
|---|---|---|---|
| COVID | 52.3% | 91.5% | 0.915 |
| LIAR (6 classes) | 20.8% | 25.4% | 0.241 |
| PUBHEALTH (4 classes) | 48.6% | 59.1% | 0.419 |
| ISOT (raw text) | 54.8% | 92.7% | 0.925 |

COVID is high because many posts repeat a template or an earlier claim. LIAR is hard for
everyone (6 classes, one-sentence claims); the kNN vote is a little below the best model in
the LIAR paper (27.4% test accuracy). PUBHEALTH's macro-F1 is low because the vote gets few
of the rarer classes right: 29 of 201 mixture and 6 of 45 unproven test claims. These are free baselines for M4's table,
and they show that "similar past fact-checks" carry real signal for the LLM's retrieval mode.

The index is built on the `raw` text. For ISOT that includes the source fingerprints, which
is one reason its kNN vote is high; a clean-variant index for ISOT prompts can be added with
T14 if the LLM runs use the clean variant.

## 2. Shortcut removal (T13)

### How the rules were found

`python scripts/find_shortcuts.py` fits TF-IDF + logistic regression on ISOT train, scores
validation (never test) and prints the 30 words with the largest weight for each class. On
raw ISOT the top words were fingerprints of the source, not the content:

* real: `reuters`, `(`, `)`, `washington`, and intact contractions (`it's`, `don't`);
* fake: `via`, `image`, `getty`, `pic.twitter.com`, `@`, `https`, `video`, and broken ones
  (`didn`, `doesn`, `re`, `ve`).

That last pair was a surprise: the fake articles were collected with their apostrophes
stripped ("didn t", "Trump s"), so a model can tell the classes apart by how `don't` was
tokenised. One rule was added per fingerprint, then the probe was run again, until the top
words were about content and style.

### The rules

Each rule is a named regex in `factcheck/data/shortcuts.py`. Shares are of all ISOT articles
that the rule changes (from `results/leakage_report.csv`):

| Rule | What it removes | Fake | Real |
|---|---|---|---|
| `agency_dateline` | "WASHINGTON (Reuters) - ", "SOCHI, Russia (Reuters) - " | 0.0% | 98.1% |
| `agency_tag_dash` | other "(Reuters) - " openings (after bylines) | 0.0% | 1.1% |
| `agency_name` | any other "(Reuters)" or "Reuters" ("told Reuters") | 1.3% | 23.0% |
| `site_boilerplate` | 21st Century Wire sign-offs and membership plugs | 4.0% | 0.0% |
| `repost_credit` | "Via: Breitbart News", "Continue this story at The Guardian" | 13.1% | 0.0% |
| `tweet_signature` | "(@realDonaldTrump) December 23, 2017" | 24.9% | 0.0% |
| `link` | URLs, pic.twitter.com, bit.ly | 24.5% | 1.0% |
| `handle` | remaining @handles | 12.0% | 1.3% |
| `image_credit` | "Featured image via Win McNamee/Getty Images" | 51.5% | 0.1% |
| `getty_images` | stray "Getty Images", "screengrab" | 1.6% | 0.0% |
| `media_tag` | "[VIDEO]", "(TWEETS)" | 38.9% | 0.0% |
| `title_prefix` | "BREAKING:", "WATCH:" at the start | 11.6% | 0.0% |
| `time_stamp` | Reuters' "(1600 GMT)", "[1814 EST]" | 0.0% | 1.5% |
| `stripped_nt`, `stripped_s`, `stripped_contraction` | restore "didn't", "Trump's", "we're" | 67%, 92%, 52% | 8%, 42%, 6% |

Two principles:

* **Delete or normalise, never insert a placeholder.** A placeholder that one class gets far
  more often (say `HTTPURL`, which 25% of fake and 1% of real articles would get) is just a
  new shortcut.
* **Keep writing style.** Reuters writes "said", "U.S." and weekdays; fake sites use more
  capitals and words like "gop" and "hillary". That is how the text is written, not how it
  was collected, so a model using it is reading the text.

### Result

| Dataset | Validation accuracy, raw | Validation accuracy, clean | Top words after cleaning |
|---|---|---|---|
| ISOT | 99.5% | 98.4% | fake: gop, this, us, just, obama, even, mr, sen, rep, hillary; real: said, u.s, on, wednesday, tuesday, thursday, republican, nov, minister, told |
| COVID | 93.3% | 93.2% | fake: coronavirus, 19, news, corona, coronavirusfacts, trump, fact, factcheck, misinformation, false; real: httpurl, rt, covid19, indiafightscorona, states, cases, data, today |

**Accuracy stays high, and that is the honest finding.** Removing the fingerprints costs a
bag-of-words model only about a point on ISOT, because the two sources also write very
differently. The interesting comparison for E7 is how much each *transformer* drops, and
whether its SHAP explanations move from "Reuters" to content words.

**COVID.** The clean variant masks links and handles (`HTTPURL`, `@USER`, as COVID-Twitter-BERT
does), which changes little: real posts still have links more often (69% vs 32%). The bigger
COVID cue is in the content: many "fake" posts are themselves fact-check posts that say
`false`, `misinformation` or `factcheck`. That cannot be removed without destroying the text,
so M4 should look for it in the SHAP examples.

**Known leftovers** (all under 0.5% of articles): a "Via:" credit can also remove the next
capitalised word; two credits glued together can leave a stray ": Truthfeed".

## 3. Leakage: duplicates across splits (T13)

### The rule

A **training** row is excluded if, compared with any validation or test row:

1. its normalised text is the same (lower-case; links, punctuation and spacing removed), or
2. it is a near-copy: embedding cosine similarity ≥ 0.95 **and the same numbers**.

Repeats *inside* train keep their first copy; if the copies disagree on the label, all are
excluded.

**Why "and the same numbers".** The guide's 0.95 threshold alone would exclude 856 COVID
training posts (13.3%), 74% of them real. Reading them showed why: official accounts post
templates, so "States reported 941 new deaths" and "States reported 2492 new deaths" are 0.97
similar but report different facts. They are not copies. Two texts that state the same
numbers and mean the same thing are: "Bill Gates predicted the coronavirus outbreak" vs "Bill
Gates predicted coronavirus", or LIAR's "Social Security is indeed a Ponzi scheme" vs "Social
Security is a Ponzi scheme".

**Why only train rows.** Removing test rows would make our test sets differ from the papers'
and break the comparison with published baselines. Removing the train copy closes the leak
just as well.

### What was excluded

| Dataset | Same text as val/test | Near-copy | Repeat in train | Conflicting labels | Total | Share of train |
|---|---|---|---|---|---|---|
| COVID | 182 | 137 | 91 | 0 | 410 | 6.4% |
| LIAR | 12 | 21 | 12 | 14 | 59 | 0.6% |
| PUBHEALTH | 2 | 13 | 4 | 8 | 27 | 0.3% |
| ISOT | 96 | 70 | 169 | 0 | 335 | 1.1% |

Validation rows that also appear in test are reported but kept (COVID 28, ISOT 7,
PUBHEALTH 1, LIAR 0), because validation and test are never changed.

ISOT's exact duplicates were already removed before splitting (T03); these 335 are copies
that differ in case, punctuation or a few words, which the T03 hash could not see.

## 4. What this changes for the team

* **M1 + M2.** Train sets are slightly smaller than in the T05–T07 baselines (the
  exclusions load automatically). Use `variant: clean` for ISOT in the E1 sweep (T17), as the
  plan says. kNN-vote runs are a new `family: knn` in `factcheck.train`.
* **M4.** `results/leakage_report.csv` (rows per rule and per check),
  `results/shortcuts/` (before/after accuracy and top words) and the COVID caveat above feed
  the shortcut study (T34).
* **M5.** The app's shortcut demo can now load `load_splits("isot", "clean")`; the
  retrieval index can power a "similar fact-checks" and "evidence" panel.

## 5. Reproducing

```bash
python -m factcheck.retrieval build        # embeddings (once)
python -m factcheck.retrieval check        # neighbours, kNN-vote accuracy, results/retrieval_duplicate_flags.csv
python scripts/find_shortcuts.py           # results/shortcuts/
python -m factcheck.data.leakage           # data/splits/exclusions.csv, results/leakage_report.csv
pytest tests/test_retrieval.py tests/test_shortcuts.py tests/test_data.py
```
