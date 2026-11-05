# Telugu - modifications to the reference implementation

Upstream: https://github.com/ming024/FastSpeech2
Commit  : `d4e79eb52e8b01d24703b2dfc0385544092958f3` (2021-07-08)

`modified_files.patch` is `git diff` of the clone against that commit;
`files/` holds the same files so they can be read without applying a patch.

| file | bytes | change |
|---|---|---|
| `audio/audio_processing.py` | 2833 | `pad_center` guard, float32 saves |
| `audio/stft.py` | 6472 | mel-basis device fix, updated librosa keywords |
| `dataset.py` | 8223 | device-agnostic tensors, pitch/energy bounds checks |
| `preprocessor/preprocessor.py` | 12152 | UTF-8 `.lab` handling, bounds + padding guards |
| `text/symbols.py` | 8222 | **Telugu grapheme table, 1044 entries** |
| `utils/model.py` | 3008 | `weights_only=False`, `map_location` |
| `utils/tools.py` | 10520 | float32 loss saves, device-agnostic plotting |

## The Telugu symbol table

`text/symbols.py` ships in its final form so inference works out of the box:

```
360 upstream symbols (LJSpeech/ARPAbet table, incl. 296 "@" entries)
+ 684 Telugu units, each prefixed with "@"  ->  1044 symbols
```

The upstream encoder allocates `len(symbols) + 1 = 1045` embedding rows
(the extra row is the padding index), which is exactly the shape of
`encoder.src_word_emb.weight` in `checkpoints/40000.pth.tar`:
`(1045, 256)`.

The 684 units come from `project/Telugu/data/vocab.json`, which is the
grapheme inventory built from the corpus:

* every codepoint of U+0C00–U+0C7F whose Unicode category is L/M/P/S (83 units),
* every multi-codepoint cluster that occurs in the text (596 units),
* the five punctuation marks `. , ? ! -`.

`vocab.json` = `["_", "~"] + 83 + 596 + 5` = 686 entries; `_` and `~` are
PAD/EOS and are stripped before the `@` prefix is applied, leaving 684.

Regenerate both files deterministically with:

```bash
python project/Telugu/preprocess.py --stage vocab   # writes data/vocab.json
python project/Telugu/preprocess.py --stage patch   # rewrites text/symbols.py
```

## Re-apply from a fresh clone

```bash
git clone --depth 1 https://github.com/ming024/FastSpeech2 third_party/FastSpeech2_Telugu
git -C third_party/FastSpeech2_Telugu apply project/patches/Telugu/modified_files.patch
```

On Kaggle the same edits are applied at runtime by `common/patch_repo.py`
(notebook 02), which is idempotent: the symbol block is delimited by
`# --- SSTF PATCH BEGIN/END (telugu graphemes) ---` and replaced, never duplicated.
