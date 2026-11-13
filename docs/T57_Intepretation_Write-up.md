# Project Interpretation Write-Up (T57): From TF-IDF and RNNs to Transformers

Consolidated interpretation across M1–M4 (`v1.0-results`, frozen `results/final/runs_final.csv`). All scores are test macro-F1 (mean over seeds 13, 42, 2026 for neural/transformer models; single deterministic runs for TF-IDF, GloVe, and prompted LLMs; single seed 42 for LoRA/QLoRA) unless noted otherwise.

## In Short

* **TF-IDF + logistic regression is a remarkably strong yardstick**, landing within 0.03 macro-F1 of the best transformer on COVID (0.953 vs. 0.980), PUBHEALTH (0.487 vs. 0.509 claim-only), and cleaned ISOT (0.988 vs. 1.000), while running in 0.7 ms on a CPU[cite: 34, 41].
* **High in-domain accuracy often reflects source and style shortcuts rather than truth verification.** Stripping 16 publisher fingerprints from ISOT barely lowers accuracy (RoBERTa 1.0000 $\to$ 0.9996; TF-IDF 0.994 $\to$ 0.988) because the two classes still differ by topic (LDA alone predicts real vs. fake at 0.885 macro-F1) and headline casing (97.9% accuracy)[cite: 35, 37, 41].
* **Up the sequence ladder, gating matters far more than bidirectionality.** A plain RNN suffers severe vanishing/exploding gradients (0.03% gradient retention 16 tokens back; norms spiking to 326), whereas GRU/LSTM gates retain 10$\times$ more signal and recover nearly the entire gap to Bi-LSTM[cite: 35].
* **External evidence is the single most valuable addition (+0.059 macro-F1 on PUBHEALTH)** because missing facts—not model capacity—create the ceiling on LIAR (~0.30) and PUBHEALTH's rare *unproven* class (45 test rows)[cite: 35, 36, 37].
* **Retrieval quality, not reader size, bottlenecks zero-shot extrapolation.** On unseen Climate-FEVER claims, fine-tuned PUBHEALTH classifiers perform at chance (0.17–0.19), while Qwen3-8B rises from 0.398 (no evidence) to 0.429 (open retrieval) vs. 0.479 (gold evidence)—capturing only 39% of the gold-evidence gain[cite: 34, 35, 41].

---

## 1. Data Leakage & Publisher Shortcuts (`T13`, `T28`, `T34`; Weeks 1, 3, 4.1)

**What we saw.** Before cleaning, 831 training rows copied or nearly copied validation/test rows (cosine $\ge 0.95$ with identical numbers), led by COVID reposts (410 rows, 6.4% of train) and ISOT near-duplicates (335 rows, 1.1%)[cite: 37, 40]. On raw ISOT, TF-IDF + logistic regression achieves 0.994 test accuracy (0.995 validation) primarily from collection artifacts such as `"reuters"`, `"washington"`, `"via"`, `"getty"`, and stripped apostrophes (`"didn"`)[cite: 37, 40, 41]. Removing 16 named fingerprint rules drops TF-IDF accuracy by only 0.6 points (to 0.988) and leaves RoBERTa-base virtually unchanged (1.0000 $\to$ 0.9996)[cite: 35, 41].

**Why.** Deleting explicit datelines and image credits removes surface tags, but ISOT's real articles (Reuters wire) and fake articles (US partisan blogs) still differ systematically in topic and formatting—for example, Reuters uses sentence-case headlines while fake sites use Title Case, which alone predicts the label with 97.9% validation accuracy[cite: 37, 38]. Near-perfect ISOT scores therefore measure outlet classification rather than factual verification[cite: 35, 37, 41].

**What it cost.** Regex cleaning and leak filtering run in seconds on a CPU and reduce training size slightly (e.g., −410 COVID posts), while preserving untouched validation/test splits for fair comparison against published baselines[cite: 37, 40].

---

## 2. Unsupervised Topic Separation (`T56`; Week 3)

**What we saw.** Fitting a 10-topic LDA model on training text shows that topic proportions alone separate ISOT real vs. fake news at 0.885–0.89 macro-F1 (vs. 0.35 majority baseline), with 56% of articles falling into topics that are $\ge 80\%$ one label (e.g., `"donald, hillary, news, obama, clinton, media"` makes up 34% of fake text vs. 3% of real; `"china, north korea, nuclear, foreign minister"` makes up 21% of real vs. 3% of fake)[cite: 37, 41]. On PUBHEALTH, LDA achieves only 0.47 macro-F1 (vs. 0.38 baseline) with no label-skewed topics[cite: 37].

**Why.** ISOT collects world wire news for real stories and US domestic culture-war blogs for fake stories, allowing a model to rely on subject matter as a proxy for veracity[cite: 37]. PUBHEALTH pairs true and false claims within the exact same public-health subjects, forcing classifiers beyond topic shortcuts[cite: 37].

**What it cost.** Roughly two minutes on a laptop CPU with zero label supervision, providing an essential pre-modeling audit for dataset bias[cite: 37].

---

## 3. Traditional Baselines: TF-IDF vs. Averaged GloVe (`T54`; Weeks 3, 4.1, 4.2)

**What we saw.** TF-IDF + logistic regression outperforms Naive Bayes and linear SVM across all four datasets, scoring 0.953 on COVID, 0.237 on LIAR, 0.487 on PUBHEALTH, and 0.988 on cleaned ISOT[cite: 34]. Conversely, averaged 300-d GloVe vectors + SVM underperform TF-IDF everywhere (0.916 COVID, 0.221 LIAR, 0.411 PUBHEALTH, 0.975 ISOT)[cite: 34]. Additionally, unweighted sigmoid probability calibration on the PUBHEALTH SVM initially degraded macro-F1 from 0.467 to 0.400 by washing out class weights (recovering to 0.446 once calibration was weighted)[cite: 34].

**Why.** Where labels correlate with specific lexical or source markers (`"rt"`, `"drtedros"`, `"data"` on COVID; `"said on tuesday"` on ISOT), sparse n-gram weights preserve exact high-signal tokens, whereas averaging 300-d embeddings dilutes rare markers into generic Wikipedia/news semantics[cite: 34]. On LIAR, single-sentence claims lack source markers, leaving bag-of-words methods near 0.24[cite: 34].

**What it cost.** Seconds to one minute on a laptop CPU, 260k–385k parameters, and 0.7 ms/claim latency—making TF-IDF + LR the cheapest, fastest, and only exact token-additive model for the app[cite: 34, 41].

---

## 4. Recurrent Ladder & Gradient Dynamics (`T05`, `T55`; Weeks 5.1, 5.2)

**What we saw.** Across 2-layer models on frozen 300-d GloVe embeddings (3 seeds), a plain RNN without gradient clipping scores 0.838 (COVID), 0.207 (LIAR), and 0.389 (PUBHEALTH, with high seed variance $\text{SD} = 0.052$)[cite: 35]. Adding gradient clipping stabilizes PUBHEALTH ($\text{SD} = 0.009$) and lifts COVID to 0.866 and PUBHEALTH to 0.403[cite: 35]. Switching to gated cells yields a major jump: GRU reaches 0.964 / 0.231 / 0.457 and unidirectional LSTM reaches 0.958 / 0.227 / 0.420, closely approaching the 2-layer Bi-LSTM (0.954 / 0.242 / 0.463)[cite: 35].

**Why.** Measuring gradient norms before training reveals that only 0.03% of the final-step loss gradient reaches 16 tokens earlier in a plain RNN, while occasional spikes reach norm 50 unclipped (and 326 pre-clipping, triggering clipping on 80%–99% of updates)[cite: 35, 41]. GRU and LSTM constant-error carousels/gates transmit 10$\times$ more gradient (0.3%) back 16 steps, and Bi-LSTM's backward pass places early tokens near a terminal state (7%–25% gradient retention)[cite: 35]. Gating provides the bulk of the performance gain over word order alone[cite: 35].

**What it cost.** 2.7M trainable parameters, 12–18 seconds per training run on GPU, and ~1 ms/claim inference—7–10$\times$ more weights than TF-IDF for roughly equal accuracy[cite: 34].

---

## 5. Transformer Encoders: Scale vs. Domain Pretraining (`T06`, `T17`, `T19`; Week 6)

**What we saw.** Fine-tuned base encoders reach 0.966 (COVID), 0.288 (LIAR), 0.496 (PUBHEALTH), and 1.000 (ISOT), while large or domain-specific encoders reach 0.980 (COVID-Twitter-BERT), 0.296 (DeBERTa-v3-large), 0.509 (SciBERT), and 1.000 (ISOT)[cite: 34]. The largest gain over TF-IDF occurs on LIAR (+0.059 macro-F1; ROC-AUC 0.733 vs. 0.639)[cite: 34]. On PUBHEALTH, smaller domain-pretrained SciBERT (110M, 0.509) outperforms larger general-purpose DeBERTa-v3-large (435M, 0.478)[cite: 34].

**Why.** Contextual self-attention and billion-word pretraining help most on LIAR, where verdicts depend on syntactic nuance rather than keyword frequency[cite: 34]. On specialized biomedical claims, domain vocabulary alignment (SciBERT) matters more than raw parameter count[cite: 34].

**What it cost.** 110M–435M parameters (400$\times$–1,700$\times$ TF-IDF), 2.2–9.8 GB VRAM, 1–7 minutes per run on an RTX 3090 (up to 2.76 hours on long ISOT articles), and 5.1–45.7 ms/claim latency[cite: 34, 41].

---

## 6. Hyperparameter Tuning & Seed Stability (`T18`; Weeks 5, 6)

**What we saw.** Optuna TPE tuning on DeBERTa-v3-base barely changes mean test macro-F1 on COVID (0.972 vs. 0.971 default) or LIAR (0.275 vs. 0.278), but lifts PUBHEALTH from 0.465 to 0.504 by preventing single-seed collapse (default seed 42 collapsed to 0.348 vs. 0.504 and 0.542 on other seeds), shrinking across-seed standard deviation from 0.103 to 0.016[cite: 34]. fANOVA importance identifies dropout on PUBHEALTH (0.68) and LIAR (0.51), and warmup ratio on COVID (0.45), as the dominant hyperparameters[cite: 34].

**Why.** Default AdamW schedules are already near-optimal on balanced datasets, so tuning primarily regularizes optimization dynamics on small, class-imbalanced splits like PUBHEALTH[cite: 34].

**What it cost.** 90 pruned trials across 4 GPUs in 33 minutes, plus 3-seed confirmation runs[cite: 34].

---

## 7. Retrieval & Single-Factor Extra-Effort Ablations (`T08`, `T14`, `T15`, `T29`, `T32`; Weeks 4.2, 6, 7)

**What we saw.** A training-free kNN vote over top-8 `bge-small-en-v1.5` training neighbors already achieves 0.915 on COVID and 0.925 on ISOT, vs. 0.419 on PUBHEALTH and 0.241 on LIAR[cite: 40]. In single-factor ablations on tuned DeBERTa-v3-base, adding **retrieved evidence sentences on PUBHEALTH** produces the project's largest improvement (+0.059 macro-F1, 0.504 $\to$ 0.562; ROC-AUC 0.903 $\to$ 0.951), followed by **balanced class weights on PUBHEALTH** (+0.032, 0.504 $\to$ 0.535) and **speaker/context metadata on LIAR** (+0.021, 0.275 $\to$ 0.296; ECE improves 0.129 $\to$ 0.046)[cite: 35, 36].

**Why.** Unlike tuning, retrieval supplies missing external facts needed to verify health claims[cite: 35, 37]. Class weights counteract extreme rarity on PUBHEALTH's *unproven* class (3% of data) while leaving balanced COVID unchanged (−0.001)[cite: 35, 37]. Speaker metadata improves LIAR by encoding historical speaker priors rather than verifying claim text[cite: 35, 37].

**What it cost.** Precomputing 370,507 PUBHEALTH sentence embeddings takes ~6 hours on a laptop CPU (minutes on GPU) and expands encoder inputs from ~16 to 384 tokens, while speaker metadata ties predictions to known politicians[cite: 37, 40].

---

## 8. LoRA vs. Full Fine-Tuning & Qwen3-8B Modes (`T20`, `T26`, `T27`; Weeks 6, 7)

**What we saw.** LoRA matches full fine-tuning on `roberta-large` (0.289 vs. 0.285 LIAR; 0.482 vs. 0.457 PUBHEALTH) while cutting peak VRAM in half (2.7–4.4 GB vs. 6.3–6.6 GB), but falls short on `deberta-v3-large` (0.420 vs. 0.478 PUBHEALTH), especially when restricted to attention layers alone (0.217–0.221 vs. 0.275–0.287 on LIAR)[cite: 35]. For `Qwen3-8B`, zero-shot prompting struggles on dataset-specific schemas (0.179 LIAR, 0.275 PUBHEALTH) with severe overconfidence (ECE 0.37–0.63)[cite: 35]. Adding 8 kNN examples boosts prompting by +0.08 to +0.18 (0.259 LIAR, 0.457 PUBHEALTH), while 4-bit QLoRA fine-tuning (43.7M trainable weights, 0.5%) matches top encoders (0.979 COVID, 0.289 LIAR, 0.515 claim-only PUBHEALTH, 1.000 ISOT) and restores calibration (ECE 0.02–0.14)[cite: 35].

**Why.** DeBERTa's disentangled attention and feed-forward projections both require adaptation for multi-class veracity boundaries, whereas RoBERTa's representations transfer with low-rank updates[cite: 35]. Prompted LLMs lack calibration on editorial labels like *pants-fire* or *mixture* until supervised QLoRA aligns their classification head[cite: 35].

**What it cost.** LoRA saves memory but not runtime (157–704 s vs. 148–443 s) because full forward/backward passes still traverse frozen layers[cite: 35]. QLoRA requires 7–9 GB VRAM, 15–20 minutes training per dataset (2 hours on ISOT), and 75–210.6 ms/claim inference—suitable for interactive single-claim checks but expensive for bulk triage[cite: 35, 41].

---

## 9. Dataset Ceilings: Why LIAR (~0.30) and PUBHEALTH Rare Classes Fail (`T33`, `T42`; Week 9)

**What we saw.** Every model plateaus between 0.18 and 0.30 macro-F1 on LIAR (best 6-class accuracy 0.299, passing spec S3 $\ge 0.28$)[cite: 34, 41]. On PUBHEALTH, even the best evidence-augmented model reaches 0.562 macro-F1 (missing spec S4 $\ge 0.65$) because *mixture* (201 test claims) and *unproven* (45 test claims) achieve only 0.36 and 0.35 F1 compared to 0.83 on *true* and 0.71 on *false*[cite: 34, 41].

**Why.** UMAP projections of `[CLS]` embeddings show clean binary separation on COVID and ISOT but heavy overlap across LIAR's six adjacent grades (*barely-true* vs. *half-true*) and PUBHEALTH's four classes[cite: 41]. *Unproven* claims are defined by the *absence* of corroborating evidence—a meta-property invisible in claim phrasing and noisy to learn from 45 test instances[cite: 34].

**What it cost.** Chasing higher macro-F1 via larger backbones yields diminishing returns; improving rare-class recall requires targeted evidence retrieval and class weighting rather than deeper encoders[cite: 34, 35].

---

## 10. Explainability, Adversarial Style Probes & Calibration (`T16`, `T30`, `T33`, `T58`, `T59`; Week 9)

**What we saw.** Comparing TF-IDF weights with transformer SHAP on cleaned ISOT shows TF-IDF relies on partisan/topic words (`"gop"`, `"obama"`), while RoBERTa SHAP highlights journalistic attribution verbs (`"said"`, `"urged"`) and entities (`"trump"`, `"tillerson"`)[cite: 41]. On LIAR (50 fixed claims), QLoRA SHAP shows numeric/attribution cues (`"percent"`, `"billion"`, `"said"`) push toward *true*, while politician names (`"Walker"` vs. `"McCain"`, `"Romney"`) encode speaker priors[cite: 35]. In hand-written COVID probes, TF-IDF rates `"WHO: Drinking bleach for 20 seconds helps stop the spread of #COVID19"` at only 0.05 $P(\text{fake})$, while COVID-Twitter-BERT rates the true off-topic statement `"CDC data: an adult human skeleton has 206 bones"` at 1.00 $P(\text{fake})$[cite: 34]. Stripping surface formatting (`@mentions`, links, hashtags, `"RT"`, caps) lowers COVID macro-F1 by 1.2–2.6 points across models (COVID-Twitter-BERT 0.980 $\to$ 0.968; TF-IDF 0.953 $\to$ 0.927) while leaving other datasets unchanged[cite: 34]. Calibration follows task complexity: binary models are well calibrated (ISOT ECE $\le 0.029$; COVID-Twitter-BERT 0.015), whereas multi-class models are overconfident (median ECE 0.084 on PUBHEALTH, 0.075 on LIAR)[cite: 41].

**Why.** Models learn stylistic and topical distributions of their training corpora—authoritative WHO phrasing looks "true" to TF-IDF, while non-COVID medical facts look out-of-distribution ("fake") to a COVID-specialized encoder[cite: 34].

**What it cost.** SHAP attribution runs in minutes for encoders (longer for 8B LLMs on a 50-claim audit set), exposing failure modes that aggregate accuracy hides[cite: 35, 37].

---

## 11. Cross-Domain Extrapolation & Beyond Misinformation (`T36`, `T38`, `T41`; Week 9 & Rubric Row 5)

**What we saw.** In binary cross-domain transfer, models trained on formal claims transfer partially between LIAR and PUBHEALTH (0.40–0.59 macro-F1 vs. 0.63–0.77 in-domain), whereas COVID-trained models collapse on formal news (0.315 on LIAR, 0.284 on PUBHEALTH—worse than majority guessing)[cite: 41]. On unseen Climate-FEVER claims, PUBHEALTH-trained classifiers perform at chance (DeBERTa + evidence 0.185, SciBERT 0.169, TF-IDF 0.177 macro-F1), whereas zero-shot Qwen3-8B achieves 0.398 macro-F1 (0.750 ROC-AUC) from pretraining alone, 0.429 (+0.032; 0.794 ROC-AUC) with top-5 open-search `bge-small` evidence, and 0.479 (+0.082; 0.862 ROC-AUC) with gold evidence[cite: 34, 35, 41].

**Why.** Discriminative classifiers overfit to domain-specific style (COVID hashtags), failing completely out-of-domain[cite: 41]. Generative LLMs extrapolate via broad parametric knowledge and evidence-entailment reasoning, but open-corpus retrieval captures only **39%** of the gain achieved with gold evidence—proving that **retriever quality** (dense embeddings and re-ranking), not reader size, is the primary bottleneck[cite: 35, 41]. Beyond misinformation, these exact mechanics carry over to **phishing/scam detection** (urgency cues act as publisher shortcuts), **review/spam fraud** (style vs. content separation), and **scientific claim verification** (mandatory evidence retrieval), while warning against speaker-metadata overfitting and temporal knowledge staleness[cite: 41].

**What it cost.** Inference-only evaluation (~130 ms/claim for Qwen3-8B on a 24 GB GPU), demonstrating that reliable deployment in new domains requires upgrading retrieval before scaling the classifier[cite: 34, 35].

---

## 12. Statistical Significance & System Deployment Specs (`T31`, `T40`, `T42`)

**What we saw.** Across 22 Holm-corrected paired bootstrap comparisons on test macro-F1, only DeBERTa-v3-large LoRA vs. full fine-tuning on PUBHEALTH (−0.096, 95% CI [−0.137, −0.054]) is statistically significant; all other macro-F1 differences (tuning, metadata, domain models, QLoRA vs. encoders) lie within test-set resampling noise[cite: 37]. However, McNemar's test confirms that adding evidence on PUBHEALTH gets significantly more individual claims right than both the no-evidence model (Holm $p = 0.046$) and BERT-base ($p = 0.011$)[cite: 37]. On the laptop GPU (RTX 5060 8 GB), the full Fact-Check Lab app (encoder + 4-bit Qwen3-8B + retrieval) runs in **5.6 GB VRAM** (pass S8 $\le 8\text{ GB}$), with **18 ms median** encoder latency (pass S6 $\le 50\text{ ms}$) and **0.16 s median** full response time (pass S7 $\le 5\text{ s}$)[cite: 41].

**Why.** Macro-F1 weights rare classes equally (45 *unproven* claims on PUBHEALTH; 92 *pants-fire* on LIAR), creating wide bootstrap confidence intervals that fail strict 22-test Holm thresholds ($p < 0.0023$), whereas claim-level McNemar tests capture consistent accuracy gains[cite: 37].

**What it cost.** ~1 minute on CPU for 10,000 bootstrap resamples, providing statistical discipline so the team avoids overclaiming small score fluctuations[cite: 36, 37].

---
