# Checkpoints

`40000.pth.tar` — the final Telugu model, 350,875,787 bytes.

Above GitHub's 100 MB per-file limit, so it is hosted on Google Drive (public, no sign-in):

https://drive.google.com/file/d/1jadSC2tEHPyiSpmlFVmbI859Lm4qZYvY/view?usp=drive_link

Drive cannot virus-scan files this large, so the download page shows a "can't scan for
viruses" warning — click **Download anyway**. Verify with:

```bash
sha256sum Telugu_40000.pth.tar
# cf66de638c0a7f1d88719d715a3a08626697f4c8d5b5377127810ed95f297327
```

Rename it to `40000.pth.tar` and place it in this directory (`checkpoints/`), then run
`python project/Telugu/inference.py --step 40000`.

The file is a `torch.save` dict with two keys, `model` and `optimizer`. Its
`encoder.src_word_emb.weight` is `(1045, 256)` — 1044 symbols from
`project/patches/Telugu/files/text/symbols.py` plus the padding row.

The HiFi-GAN vocoder is not stored here — it is downloaded by `utils.model.get_vocoder`
from the reference repository on first use (`generator_universal` for Telugu, see
`config.yaml` → `vocoder.speaker`).
