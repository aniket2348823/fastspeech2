# project/ — submission contents

FastSpeech 2 TTS for **English (Svarah)** and **Telugu (IndicTTS Telugu)**, implemented on top of
[ming024/FastSpeech2](https://github.com/ming024/FastSpeech2) (commit `d4e79eb52e8b01d24703b2dfc0385544092958f3`,
2021-07-08). Two independent clones of the reference are used, one per language, so that the
Telugu symbol-table extension cannot shift the English symbol ids.

## Structure (exactly as required by the assignment)

```
project/
├── English/
│   ├── train.py  preprocess.py  inference.py  config.yaml
│   ├── checkpoints/40000.pth.tar          trained English model
│   ├── fs2_config/                        rendered preprocess/model/train YAML for the reference
│   └── output/log/{train,val}/log.txt     raw loss logs (source of every number in the report)
├── Telugu/
│   ├── train.py  preprocess.py  inference.py  config.yaml
│   ├── checkpoints/40000.pth.tar          trained Telugu model
│   ├── data/{vocab.json,train.txt,val.txt,split_stats.json}
│   ├── fs2_config/
│   └── output/log/{train,val}/log.txt
├── Results/
│   ├── English/
│   │   ├── training_loss.png  validation_loss.png
│   │   ├── inference_summary.json
│   │   ├── test_generated_audios/          E1 E2a E2b E3 E4 (.wav, 22.05 kHz mono)
│   │   └── variance_adaptor_plots/
│   │       ├── duration/  pitch/  energy/  {E1,E2a,E2b,E3,E4}.png
│   │       └── {E1,E2a,E2b,E3,E4}.csv      the numbers the plots are drawn from
│   └── Telugu/                             same layout, T1 T2a T2b T3 T4
├── Report/
│   └── Technical_Report.pdf                15 pages
├── patches/                                modified upstream files, per language
├── README.md                               this file
└── requirements.txt
```

## Modified files of the reference implementation

`patches/<Lang>/` holds them two ways:

| Path | Content |
|---|---|
| `patches/<Lang>/modified_files.patch` | `git diff` against upstream `d4e79eb` |
| `patches/<Lang>/files/…` | the patched files themselves, same relative paths |
| `patches/<Lang>/README.md` | upstream commit, file list, re-apply command |

Both languages ship the same six compatibility/correctness patches (`audio/audio_processing.py`,
`audio/stft.py`, `dataset.py`, `preprocessor/preprocessor.py`, `utils/model.py`, `utils/tools.py`).
Telugu additionally ships **`text/symbols.py`** — the 1044-entry table
(360 upstream symbols + 684 `@<grapheme>` entries) that was actually used for training and
inference. It is generated from `project/Telugu/data/vocab.json`, so the pair
`vocab.json` + `symbols.py` is self-consistent and a reviewer can run inference without
re-running preprocessing.

## Checkpoints

Each `checkpoints/` directory holds the final step, `40000.pth.tar` (350,875,787 bytes ≈ 335 MiB),
which is above GitHub's 100 MB per-file limit. As the assignment permits, they are provided
through an external download link / Git LFS rather than committed to the repository history.
Verify a download with:

| Model | SHA-256 |
|---|---|
| English `40000.pth.tar` | `df91015427e4b69fd5e3f932d0391a525e3b008a07ecf14c5f3698731bb5bffd` |
| Telugu `40000.pth.tar` | `cf66de638c0a7f1d88719d715a3a08626697f4c8d5b5377127810ed95f297327` |

```bash
sha256sum project/Telugu/checkpoints/40000.pth.tar
```

Each checkpoint stores `{"model": state_dict, "optimizer": state_dict}` only.

## Reproducing

```bash
pip install -r requirements.txt

# per language: LANG = English | Telugu
python project/$LANG/preprocess.py          # metadata split -> align (MMS_FA) -> features -> stage
python project/$LANG/train.py               # 40k steps, validate + checkpoint every 2k
python project/$LANG/inference.py           # 5 test sentences + variance-adaptor plots
python project/$LANG/train.py --plot-only   # redraw the two loss curves from the logs
```

Telugu needs the vocabulary stage before the symbol patch:

```bash
python project/Telugu/preprocess.py --stage vocab    # writes data/vocab.json
python project/Telugu/preprocess.py --stage patch    # appends @<grapheme> entries to text/symbols.py
```

The Kaggle notebooks under `notebooks/` run exactly these commands: `01` fetches the data,
`02` preprocesses, `03` trains and produces the results bundle, `04` is the optional
inference-only path.

## Data policy

| | English | Telugu |
|---|---|---|
| Dataset | Svarah (`ai4bharat/Svarah`) | IndicTTS Telugu (`SPRINGLab/IndicTTS_Telugu`) |
| Available | ≈5.0 h, 2 speakers | 8.74 h (8,576 utts), 2 speakers |
| Filter | 0.4–20 s per utterance | 0.4–20 s → 2,236 dropped, 6,340 usable |
| Split rule | proportional, `train_ratio: 0.8` | hour-greedy, 5 h / 1 h |
| Used | ≈4.0 h train / ≈1.0 h val | 5.002 h train (1,907 utts) / 1.000 h val (373 utts) |
| Unused | — | 2.74 h |

Splits come from one seeded shuffle (`seed 1234`) followed by a single cut, so they are
disjoint by construction; `split_stats.json` records `"overlap": 0`.

## Results at a glance

| | English | Telugu |
|---|---|---|
| Training loss (step 40k) | 0.9856 | 0.8352 |
| Validation loss (step 40k) | 4.5144 | 4.5053 |
| Validation / training ratio | 4.58× | 5.39× |
| Predicted speech, 5 sentences | 3.48–9.18 s | 3.77–11.00 s |
| Units per sentence | 39–85 (phones) | 17–61 (graphemes) |

The full analysis is in `Report/Technical_Report.pdf` §5 (loss curves), §6 (generated speech)
and §7 (variance adaptor).
