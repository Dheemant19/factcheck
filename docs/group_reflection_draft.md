# Group reflection (DRAFT for approval)

**Status: draft by M3 (T50), 11 Nov 2026.** Each member, please check your own paragraph and
reply with corrections by **Thu 12 Nov**; signatures are collected on **Fri 13 Nov**. Every
statement below comes from `docs/contribution_log.md`; nothing is added from memory. Before
submitting, confirm the course's reflection template (plan "Things to confirm") and paste the
approved text into it. Replace the slot names with your names.

## Project in one paragraph

We asked how well deep learning catches misinformation and whether it is right for the right
reasons, on four datasets (COVID-19 posts, LIAR political statements, PUBHEALTH health claims,
ISOT news articles) plus Climate-FEVER for extrapolation. After the Week 12 consultation we
reframed the work as a ladder from TF-IDF and word embeddings through the RNN family to
transformers, with the emphasis on interpreting each step. Everything runs from one Python
package: fixed splits, one metrics function, one results table (`results/runs.csv`, frozen as
`v1.0-results`), and a Streamlit app.

## Individual contributions

### M1 (training team), name: ____________

Set up the repository and the shared package (T01, taken over from M5), downloaded the
datasets and set up the cluster and laptop environments (T02), and wrote the shared metrics
and specs module (T04, taken over from M4) and the one training entry point. Trained the
Bi-LSTM, encoder and zero-shot Qwen3-8B baselines (T05 to T07), the 90-run base, large and
domain encoder sweeps (T17, T19) and the Optuna search (T18), and added the traditional TF-IDF
and GloVe baselines (T54). Built results table v1 and the consultation pack (T21, T24, taken
over from M4 and M5), the app and its redesign (T22, T35, T58, taken over from M5), the
Climate-FEVER and style-stripped COVID studies (T38, T59) and the re-run queue (T39). Reviewed
and merged M3's data and retrieval work.

### M2 (training team), name: ____________

Added the kNN few-shot and evidence (RAG) modes to the Qwen3-8B scorer (T20), the LoRA sweep
on the large encoders (T26) and QLoRA fine-tuning of Qwen3-8B (T27), and ran the raw versus
cleaned shortcut check (T28) and the extra-effort ablations with class weights (T29). Explained
the QLoRA model with SHAP (T30), built the RNN / GRU / LSTM ladder with gradient logging and
the clipping study (T55), and fixed the collapsed roberta-large runs (T39). Wrote the
interpretation paragraphs for the training-team results and reviewed M1's code (T44, five
findings, several fixed before the freeze).

### M3 (data and retrieval), name: ____________

Wrote the dataset loaders and the fixed, committed splits (T03), the data exploration notebook
(T09) and the shortcut removal and leakage checks (T13: 16 rules for cleaned ISOT, 831 copied
training rows excluded). Built the retrieval index and the kNN-vote baseline (T08), the prompt
and input builders used by the LLM runs (T14) and the metadata and evidence switches (T15).
Wrote the significance tests and optional ensemble (T31; the run on the saved predictions is
still to do on the cluster), the extra-effort chart (T32) and the LDA topic study (T56).
Wrote the interpretation paragraphs for this part (T57), reviewed M4's code (T44) and froze the
results (T43).

### M4 (evaluation and interpretation), name: ____________

Built the first baseline table against the published results (T10) and the SHAP explanation
pipeline (T16). Staged the model checkpoints and set up the explanation comparison and error
analysis (T33), the shortcut study (T34), the cross-dataset transfer matrix (T36), the
cross-domain heatmap and "beyond misinformation" section (T41) and the final figures and specs
table (T42). The metrics module (T04) and results table v1 (T21) were taken over by M1 by
agreement. *M4: the T33 to T42 work is on a branch with review findings (see
`docs/t44_review_m4_by_m3.md`); please update this paragraph once it is merged.*

### M5 (product and presentation), name: ____________

*M5: there are no rows for you in the contribution log yet, so this paragraph is left for you.
By agreement, M1 took over T01 (repository), T11, T22 and T35 (app) and T24 (consultation
pack). Your remaining tasks in the plan are T12 (slide template), T23 (storyboard), T45
(README), T46 (assembling the slides), T48 (recording and editing the video). Please add rows
to the contribution log for what you have done and write two or three factual sentences here.*

## How the work was split, and how it went

**How we split it.** Five slots from the plan (Section 19): a two-person training team (M1, M2)
running everything that needs a GPU, and one owner each for data and retrieval (M3),
evaluation and interpretation (M4) and product and presentation (M5). Every task had an ID, an
owner, a finish-by date one day before its deadline, and a reviewer. Several tasks moved
between slots by agreement, and each move is in the contribution log.

**What went well.**

* One package, one metrics function and one results table meant every number in the report
  comes from the same code, and the app reads the same table.
* Fixing the data first (committed splits, leakage checks, shortcut rules) made later
  comparisons fair and let us explain why ISOT is easy for the wrong reasons.
* Code reviews between members caught real bugs before the freeze (an app verdict that
  counted "mixture" as true, SVM settings picked on the wrong model, reversed labels in the
  cross-dataset matrix).

**What we would change.**

* Start with the traditional baselines. The consultation asked for TF-IDF and plain RNNs,
  which we added in the last two weeks; they are cheap and make the deep models easier to
  interpret.
* Log every run from committed code: 255 of the 502 frozen result rows were logged with
  uncommitted changes, so the exact code behind them cannot be named by a single commit.
* Put the saved predictions where everyone can reach them; the significance tests (T31) had to
  wait because the predictions stayed on the cluster.
* Spread the load more evenly and keep the contribution log current: one member took over
  several other members' tasks, and one member has no log entries yet.

## Signatures

| Slot | Name | Signature | Date |
|---|---|---|---|
| M1 | | | |
| M2 | | | |
| M3 | | | |
| M4 | | | |
| M5 | | | |
