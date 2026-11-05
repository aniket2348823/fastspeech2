# Checkpoints

`40000.pth.tar` — the final Telugu model, 350,875,787 bytes.

Not committed to git: it is above GitHub's 100 MB per-file limit and the assignment allows an
external download link in that case. Put the link in the repository `README.md` (and in
`project/README.md`), keep it live for the evaluation window, and verify the download with:

```bash
sha256sum 40000.pth.tar
# cf66de638c0a7f1d88719d715a3a08626697f4c8d5b5377127810ed95f297327
```

The file is a `torch.save` dict with two keys, `model` and `optimizer`. Its
`encoder.src_word_emb.weight` is `(1045, 256)` — 1044 symbols from
`project/patches/Telugu/files/text/symbols.py` plus the padding row — which is the quickest way to
confirm you have the right vocabulary when the model loads. It restores with:

```bash
python project/Telugu/inference.py --step 40000
```

The HiFi-GAN vocoder is not stored here — it is downloaded by `utils.model.get_vocoder` from the
reference repository on first use (`generator_universal` for Telugu, see `config.yaml` →
`vocoder.speaker`).
