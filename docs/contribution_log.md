# Contribution log

One line per piece of work, added by the person who did it. This feeds the signed group
reflection and keeps the peer evaluation fair. Newest at the bottom.

| Date | Member | Task | Files / PR |
|------|--------|------|------------|
| 2026-10-20 | M1 | T01 repo skeleton (Section 16 layout, stubs), T02 dataset download script and cluster/laptop setup scripts, early Streamlit app (T11) | repo, `scripts/`, `jobs/setup_cluster.sh`, `app/` |
| 2026-10-20 | M1 | T02 laptop part: download script, check_env, smoke_test.py (DistilBERT 0.938 val acc on the RTX 5060, driver 610.88), docs/environments.md, setup_cluster.sh for any Linux GPU server, zipped raw data for the shared drive | `scripts/`, `jobs/setup_cluster.sh`, `docs/environments.md` |
| 2026-10-20 | M1 | T02 cluster part: MLDA gpu21 set up (conda Python 3.11 + venv, torch cu126), all five datasets on the server, smoke test PASS (0.936 on an RTX 3090), libstdc++ fix for optuna in setup_cluster.sh | `jobs/setup_cluster.sh`, `docs/environments.md` |
