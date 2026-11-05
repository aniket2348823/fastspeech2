# Checkpoints

`40000.pth.tar` — the final English model, 350,875,787 bytes.

Not committed to git: it is above GitHub's 100 MB per-file limit and the assignment allows an
external download link in that case. Put the link in the repository `README.md` (and in
`project/README.md`), keep it live for the evaluation window, and verify the download with:

```bash
sha256sum 40000.pth.tar
# df91015427e4b69fd5e3f932d0391a525e3b008a07ecf14c5f3698731bb5bffd
```

The file is a `torch.save` dict with two keys, `model` and `optimizer`, exactly as written by the
reference training loop. It restores with:

```bash
python project/English/inference.py --step 40000
```

The HiFi-GAN vocoder is not stored here — it is downloaded by `utils.model.get_vocoder` from the
reference repository on first use (`generator_LJSpeech` for English).
