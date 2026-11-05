# Checkpoints

`40000.pth.tar` — the final English model, 348,774,539 bytes.

Above GitHub's 100 MB per-file limit, so it is hosted on Google Drive (public, no sign-in):

https://drive.google.com/file/d/1uacreqMA7ErGy6GdGdLx3ZTChCwmoCde/view?usp=drive_link

Drive cannot virus-scan files this large, so the download page shows a "can't scan for
viruses" warning — click **Download anyway**. Verify with:

```bash
sha256sum English_40000.pth.tar
# df91015427e4b69fd5e3f932d0391a525e3b008a07ecf14c5f3698731bb5bffd
```

Rename it to `40000.pth.tar` and place it in this directory (`checkpoints/`), then run
`python project/English/inference.py --step 40000`.

The file is a `torch.save` dict with two keys, `model` and `optimizer`. Its
`encoder.src_word_emb.weight` is `(361, 256)` — the 360-symbol upstream table plus the
padding row.

The HiFi-GAN vocoder is not stored here — it is downloaded by `utils.model.get_vocoder`
from the reference repository on first use (`generator_LJSpeech` for English, see
`config.yaml` → `vocoder.speaker`).
