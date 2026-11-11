TF-IDF baselines (T54), from M1, 10 Nov.

top_words/<run>/top_words.csv: per label, the 30 word n-grams and 30 character n-grams (quoted)
  that push hardest towards it (weight = the model's coefficient). Word n-grams are the ones
  to compare with SHAP (T33). tfidf_lr_* is the model in the results table and the app.
  *_plain runs were trained on style-stripped text (T59).
top_words/<run>/train_summary.json: the validation grid and the chosen setting.
predictions/<run>/<split>.csv: per-row label, prediction and probabilities (seed 42; TF-IDF
  is deterministic). Same format as M2's prediction folders.

The SVM rows were re-run on 10 Nov after the T44 fix, so they are newer than any older copy.
