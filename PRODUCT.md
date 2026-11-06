# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

Primary audience for the app (confirmed): the EE6405 markers and viewers of the team's 10-minute video. They see the app in the live-demo segment (7:45 to 9:15, presented by M5) and judge the Presentation rubric row: "impressive demonstration and intuitive front end". They watch a screen recording, so every result must read at a glance on a video frame.

The audience the product is framed for: fact-checking teams, newsroom editors, content-moderation staff and public-health communicators who need triage ("which of today's posts should a human check first, and what evidence is relevant?").

## Product Purpose

Fact-Check Lab: paste a claim, get an explained verdict. It is the front end of a course project comparing a Bi-LSTM, fine-tuned transformer encoders, LoRA and a retrieval-augmented 8B LLM on four misinformation datasets (COVID-19 fake news, LIAR, PUBHEALTH, ISOT; Climate-FEVER for extrapolation). Success: a marker understands the verdict, the evidence and the reason within seconds, and remembers the shortcut demo.

## Positioning

Triage support for human fact-checkers, not a judge of truth. It shows why a model decided (SHAP highlights, retrieved evidence, similar past fact-checks), where models disagree, and exposes shortcut learning (the "WASHINGTON (Reuters) -" flip) rather than hiding it.

## Operating Context

Runs locally on a laptop RTX 5060 (8 GB). Models are found under `outputs/<run>/seed<N>/model`; results are read live from `results/runs.csv`; specs S1 to S10 come from `configs/specs.yaml` via `factcheck.evaluate.check_specs`. Team of five; the app (T11, T22, T35) is built by M1 (taken over from M5, 6 Nov), on M3's retrieval index and M4's SHAP masker; the inference engine is `factcheck/serve.py`.

## Capabilities and Constraints

- Views from the plan (Section 15): Check a claim (verdict badge Likely true / Mixed / Likely false, confidence, response time against spec S7 <= 5 s), Evidence panel, Why panel (SHAP: red pushes towards false, green towards true), Compare models (fast encoder vs best encoder vs RAG LLM), Shortcut demo, Results explorer (charts + specs pass/fail), example claims pre-loaded in a sidebar.
- Working (6 Nov): every view. Check: ladder verdict, S6 encoder time, evidence (similar fact-checked training claims with their verdicts, closest PUBHEALTH and Climate-FEVER evidence sentences), Why (SHAP in log-odds of false, words tinted red/green, top-words bar chart), optional Qwen3-8B QLoRA second opinion with the S7 full-response timer, live GPU memory against S8 in the sidebar. Compare, Shortcut demo (raw vs cleaned ISOT RoBERTa), Results (specs board, mean-over-seeds chart, all runs). Links: `?claim=...`, `?view=...`, `?llm=1`.
- Confirmed: unfinished panels get designed empty states that say what will appear and who or what fills it; no fabricated numbers.
- Confirmed: styling uses the native Streamlit theme (`.streamlit/config.toml`) and native components only, no custom CSS.
- Streamlit 1.66, Python 3.11, installed with pip.

## Evidence on Hand

166 logged runs in `results/runs.csv`; 12 trained models plus 4 QLoRA adapters on the laptop under `outputs/` (copied from the MLDA servers, not in git). Example claims are in `app/app.py`. Do not invent accuracies, latencies or verdicts: every number on screen is read from runs.csv or measured live.

## Product Principles

- One action: paste a claim, get an explained verdict.
- Show the reason and the evidence next to the verdict, never a bare label.
- Honesty over spectacle: uncertainty, disagreement and shortcuts are shown, not smoothed away.
- Every number on screen is read from the project's own results, never typed in.
