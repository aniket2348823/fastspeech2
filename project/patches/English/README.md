# English - modifications to the reference implementation

Upstream: https://github.com/ming024/FastSpeech2
Commit  : `d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08 - update README and demo page)

`modified_files.patch` is `git diff` of the clone against that commit;
`files/` holds the same files so they can be read without applying a patch.

| file | bytes |
|---|---|
| `audio/audio_processing.py` | 2833 |
| `audio/stft.py` | 6472 |
| `dataset.py` | 8850 |
| `preprocessor/preprocessor.py` | 12152 |
| `utils/model.py` | 3008 |
| `utils/tools.py` | 10520 |

Re-apply from a fresh clone:

```bash
git clone --depth 1 https://github.com/ming024/FastSpeech2 third_party/FastSpeech2_English
git -C third_party/FastSpeech2_English apply project/patches/English/modified_files.patch
```

On Kaggle the same edits are applied at runtime by `common/patch_repo.py` (notebook 02 cell 3), which is idempotent.
