# Checkpoints

Not committed (see `.gitignore`). Two options for submission:

1. **Git LFS** — `git lfs install && git lfs track "project/*/checkpoints/*.pth"`
2. **External link** — upload to Kaggle/Drive and put a permanent link in
   `README.md` under "Checkpoints", with a note that it stays live during
   evaluation.

Files produced by training: `<step>.pth` (acoustic model), `hifigan/<step>` (vocoder).
