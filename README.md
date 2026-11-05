# FastSpeech 2 — English (Svarah) + Telugu (IndicTTS Telugu)

Submission for the assignment *Implementation of FastSpeech 2-Based Text-to-Speech Synthesis*
(deadline 07-11-2026). Two independently trained models, one shared harness, one technical report.

Everything the brief asks for lives under **`project/`**; this top-level directory only adds the
harness that was used to produce it.

```
.
├── project/                 << the submission >>
│   ├── English/             train.py  preprocess.py  inference.py  config.yaml  checkpoints/
│   ├── Telugu/              same, plus data/vocab.json (grapheme inventory)
│   ├── Results/English|Telugu/
│   │   ├── training_loss.png  validation_loss.png
│   │   ├── test_generated_audios/{E1,E2a,E2b,E3,E4} or {T1,T2a,T2b,T3,T4}.wav
│   │   └── variance_adaptor_plots/{duration,pitch,energy}/*.png  +  per-sentence *.csv
│   ├── Report/Technical_Report.pdf
│   ├── patches/             modified upstream files, per language (diff + plain files)
│   ├── README.md            submission map, data policy, checksums
│   └── requirements.txt
├── common/                  shared harness: config, manifest/split, MMS_FA align, staging,
│                            Telugu front-end, repo patching, training and inference runners
├── notebooks/               the four Kaggle notebooks used for the runs
├── third_party/             upstream clone (gitignored, see below)
└── requirements.txt
```

## Results

| | English | Telugu |
|---|---|---|
| Reference | ming024/FastSpeech2 `d4e79eb` | same |
| Dataset | Svarah | IndicTTS Telugu |
| Train / val audio | ≈4.0 h / ≈1.0 h (proportional, seed 1234) | 5.002 h / 1.000 h (hour-greedy, seed 1234) |
| Input units | ARPAbet phones, 360 symbols | grapheme clusters, 1044 symbols |
| Training | 40 000 steps, 2×T4 | 40 000 steps, 5 h 50 m, 2×T4 |
| Training loss (40k) | 0.9856 | 0.8352 |
| Validation loss (40k) | 4.5144 | 4.5053 |
| Audio | 5 wavs, 3.48–9.18 s | 5 wavs, 3.77–11.00 s |
| Variance plots | 5 × 3 | 5 × 3 |

Both models were trained **from scratch** — no fine-tuning from the published LJSpeech checkpoint —
so that the English/Telugu comparison in the report is apples-to-apples. The only pretrained
component is the HiFi-GAN vocoder (LJSpeech generator for English, `universal` for Telugu).

The loss curves show clean convergence and clear overfitting: validation loss is 4.6× (English) and
5.4× (Telugu) the training loss at step 40 000, concentrated almost entirely in the pitch and energy
heads. This is analysed in `project/Report/Technical_Report.pdf` §5.

## Reference implementation

[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2), commit
`d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08). It is **not vendored**: clone it next to
the harness, once per language, so the Telugu symbol table cannot shift the English ids:

```bash
git clone --depth 1 https://github.com/ming024/FastSpeech2 third_party/FastSpeech2_English
git clone --depth 1 https://github.com/ming024/FastSpeech2 third_party/FastSpeech2_Telugu
```

The edits the reference needs on a modern stack (torch ≥ 2.6 `weights_only`, librosa ≥ 1.x mel
kwargs, device-agnostic STFT, short-clip guards, UTF-8 lab files, Telugu graphemes) are applied
idempotently by `common/patch_repo.py` and are also shipped as plain diffs in `project/patches/`:

```bash
git -C third_party/FastSpeech2_Telugu apply project/patches/Telugu/modified_files.patch
```

Both patches are verified to apply cleanly to a pristine clone of `d4e79eb`.

## Reproduce

```bash
pip install -r requirements.txt
sudo apt-get install -y espeak-ng            # English phonemisation only

# Telugu (vocabulary first: the symbol patch is generated from it)
python project/Telugu/preprocess.py --stage vocab
python project/Telugu/preprocess.py --stage all      # split -> align (MMS_FA) -> features -> stage
python project/Telugu/train.py                       # 40k steps, val + checkpoint every 2k
python project/Telugu/inference.py                    # 5 wavs + 15 variance-adaptor plots

# English: same three commands under project/English (no vocab stage needed)
```

A fast sanity pass before a full run:

```bash
python project/Telugu/preprocess.py --stage align --limit 20
python project/Telugu/train.py --steps 2000
```

## Checkpoints

`project/<lang>/checkpoints/40000.pth.tar`, 350,875,787 bytes each — above GitHub's 100 MB
per-file limit, so they are **not** in the git history. The assignment allows an external link in
that case; point the link at these files and verify with the SHA-256 values listed in
`project/README.md`.

## Kaggle runbook

Sessions are ~9 h, `/kaggle/temp` is wiped, and only `/kaggle/working` survives a commit. A
different notebook is a different session, so `/kaggle/working` written by notebook 3 is **not**
visible to notebook 4.

| # | Notebook | Produces |
|---|---|---|
| 1 | `01_setup_data.ipynb` | clone, deps, dataset download, `metadata.csv`, 5 h/1 h split |
| 2 | `02_preprocess.ipynb` | TextGrids, `vocab.json`, preprocessed tensors, staged `train.txt`/`val.txt` |
| 3 | `03_train.ipynb` | **everything**: checkpoints, loss curves, test wavs, variance plots, `telugu_bundle.zip` + `Telugu_40000.pth.tar` staged for download |
| 4 | `04_infer_plots.ipynb` | optional inference-only path (needs a checkpoint uploaded into that session) |

All four notebooks default to `LANG = "Telugu"` in cell 0; set it to `"English"` to redo that
model. Notebook 3 ends with inference, verification and download staging because a later notebook
cannot see its checkpoints — **download the two files it prints before closing the tab.**

## Design decisions

1. **MMS_FA instead of MFA.** No Telugu acoustic model or pronunciation dictionary ships with MFA,
   and one aligner for both languages keeps the two models comparable.
2. **Grapheme-level Telugu.** The script is close to phonemic and there is no usable G2P, so the
   unit inventory is the set of grapheme clusters of `U+0C00–U+0C7F` (684 units observed, all
   codepoints of the block always included so inference cannot go out of vocabulary). Documented
   cost: homographs such as `పట్టు` (T2a/T2b) get identical inputs.
3. **Text normalisation is part of the model.** E4/T4 numerals, ordinals and times, and the broken
   spacing in E3/T3, are the point of the test set; `common/textnorm.py` (English) and
   `common/telugu.py` (Telugu) are where they are handled.
4. **Two clones, one harness.** Language-specific behaviour lives in new modules — `common/telugu.py`,
   `common/align.py`, the symbol generator in `common/patch_repo.py` — so the English clone stays a
   faithful copy of the published code.
