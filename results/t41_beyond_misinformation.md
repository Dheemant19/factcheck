# Section 13: Interpreting the Models & Beyond Misinformation

## 1. Cross-Domain Generalization & Transferability
- **Formal Claim Transfer (LIAR <-> PUBHEALTH):** Models transfer well between formal political claims and public health articles (~0.397 - 0.592 Macro-F1) due to shared declarative syntax and professional tone.
- **Social Media Overfitting (COVID -> Formal News):** Models trained on COVID Twitter posts drop significantly (~0.315 to 0.284 Macro-F1) on longer articles, as they overfit to hashtags, slang, and pandemic terminology rather than general fact-verification logic.

## 2. Climate-FEVER Zero-Shot Extrapolation & Retrieval Bottleneck
- **No Evidence (Parametric Only):** 0.398 Macro-F1 (LLMs suffer from hallucination and temporal decay).
- **Open Search Evidence (Top-5):** 0.429 Macro-F1 (+0.032 gain).
- **Gold Evidence (Dataset Provided):** 0.479 Macro-F1 (+0.082 gain).
- **Key Finding:** Open retrieval captures only **39.0%** of the potential gain offered by gold evidence (+0.032 vs +0.082). This proves that **retrieval quality (dense embeddings & re-ranking)** is the primary system bottleneck, rather than downstream model reading capacity.

## 3. Where Findings Carry Over (Beyond Misinformation)
- **Phishing & Scam Detection:** Urgency words, emotional manipulation, and source-header shortcuts operate identically in email phishing filters.
- **Scientific Claim Verification:** External evidence retrieval is mandatory for specialized domain claims; accuracy cannot be judged from text style alone.

## 4. Failure Modes & Non-Transferable Scenarios
- **Speaker & Context Priors:** Exploiting speaker history metadata fails completely when deployed to unindexed social media users or emerging public figures.
- **Temporal Knowledge Staleness:** Static parametric weights degrade as new public figures, events, and scientific facts emerge post-training.
