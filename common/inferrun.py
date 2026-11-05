"""Synthesise the assignment's test sentences and plot the variance adaptor.

Everything here runs inside the upstream clone (``cwd=<repo>``) because
``utils.model.get_vocoder`` resolves ``hifigan/config.json`` relative to the
working directory and ``text``/``model`` are top-level packages of the clone.

Outputs, exactly where the brief asks for them::

    project/Results/<lang>/test_generated_audios/<ID>.wav
    project/Results/<lang>/variance_adaptor_plots/duration/<ID>.png
    project/Results/<lang>/variance_adaptor_plots/pitch/<ID>.png
    project/Results/<lang>/variance_adaptor_plots/energy/<ID>.png
    project/Results/<lang>/variance_adaptor_plots/<ID>.csv   (values behind the plots)

The model is the reference FastSpeech2: encoder -> variance adaptor -> decoder.
With ``mels=None`` the variance adaptor predicts duration, pitch and energy
instead of using ground truth, which is what makes the analysis possible.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from . import config, fs2cfg, g2p, patch_repo, telugu, textnorm


def _repo_text(lang: str, sentence: str, lexicon) -> tuple[str, list[str], list[str]]:
    """(``{unit unit ...}`` string, display units, romanised units)."""
    text = textnorm.normalize(sentence, lang)
    if lang == "English":
        units = g2p.phone_list(text, lexicon)
        return "{" + " ".join(units) + "}", units, units
    units = telugu.units(text)
    return "{" + " ".join(units) + "}", units, [telugu.romanise(u) for u in units]


def _plot(feature: str, values: list[float], labels: list[str], title: str,
          ylabel: str, path: Path, color: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(max(7, 0.22 * len(values) + 3), 3.6))
    x = list(range(len(values)))
    if feature == "duration":
        ax.bar(x, values, color=color, width=0.8)
    else:
        ax.plot(x, values, color=color, linewidth=1.6, marker="o", ms=2.5)
    step = max(1, len(values) // 40)
    ax.set_xticks(x[::step])
    ax.set_xticklabels(labels[::step], rotation=90, fontsize=6)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("unit")
    ax.set_title(title)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run(lang: str, step: int | None = None, only: str | None = None) -> dict:
    cfg = config.load(lang)
    repo = config.require_repo(lang)
    patch_repo.apply(lang)
    fs2cfg.render(lang)
    config.ensure_dirs(lang)

    import yaml
    p = fs2cfg.paths(lang)
    ckpt_dir = config.lang_dir(lang) / "output" / "ckpt"
    restore = step or cfg["inference"].get("restore_step") or 0
    if not restore:
        steps = []
        for f in ckpt_dir.glob("*.pth.tar"):
            try:
                steps.append(int(f.name.split(".")[0]))
            except ValueError:
                pass
        if not steps:
            raise SystemExit(f"no checkpoints in {ckpt_dir}; train first")
        restore = max(steps)

    os.chdir(repo)
    sys.path.insert(0, str(repo))
    import numpy as np
    import torch
    from text import text_to_sequence
    from utils.model import get_model, get_vocoder, vocoder_infer

    preprocess_config = yaml.safe_load(open(p["preprocess"], encoding="utf-8"))
    model_config = yaml.safe_load(open(p["model"], encoding="utf-8"))
    train_config = yaml.safe_load(open(p["train"], encoding="utf-8"))
    train_config["path"]["ckpt_path"] = str(ckpt_dir)
    configs = (preprocess_config, model_config, train_config)

    class _Args:
        restore_step = restore

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_model(_Args(), configs, device, train=False)
    vocoder = get_vocoder(model_config, device)

    stats_path = Path(preprocess_config["path"]["preprocessed_path"]) / "stats.json"
    stats = json.loads(stats_path.read_text(encoding="utf-8"))
    pitch_mean, pitch_std = stats["pitch"][2], stats["pitch"][3]
    energy_mean, energy_std = stats["energy"][2], stats["energy"][3]
    hop = preprocess_config["preprocessing"]["stft"]["hop_length"]
    sr = preprocess_config["preprocessing"]["audio"]["sampling_rate"]
    cleaners = preprocess_config["preprocessing"]["text"]["text_cleaners"]

    lexicon = None
    if lang == "English":
        lexicon = g2p.load_lexicon(repo / "lexicon" / "librispeech-lexicon.txt")

    import soundfile as sf
    res = config.results_dir(lang)
    sentences = cfg["inference"]["sentences"]
    if only:
        sentences = [s for s in sentences if s["id"] == only]

    summary = []
    for item in sentences:
        sid, sentence = item["id"], item["text"]
        repo_text, units, roman = _repo_text(lang, sentence, lexicon)
        seq = text_to_sequence(repo_text, cleaners)
        if not seq:
            print(f"[infer] {sid}: empty sequence, skipped", flush=True)
            continue

        texts = torch.from_numpy(np.array([seq], dtype=np.int64)).long().to(device)
        src_lens = torch.tensor([len(seq)], dtype=torch.long).to(device)
        speakers = torch.zeros(1, dtype=torch.long).to(device)

        with torch.no_grad():
            out = model(speakers, texts, src_lens, len(seq))
        mel, p_pred, e_pred, d_rounded = out[0], out[2], out[3], out[5]

        # FastSpeech2 outputs mel as [batch, time, 80]; HiFi-GAN needs [batch, 80, time]
        mel = mel.transpose(1, 2)

        wav = vocoder_infer(mel, vocoder, model_config, preprocess_config)[0]
        wav_path = res / "test_generated_audios" / f"{sid}.wav"
        sf.write(str(wav_path), wav, sr)

        n = min(len(units), d_rounded.shape[1])
        dur_frames = d_rounded[0, :n].cpu().numpy().tolist()
        dur_sec = [f * hop / sr for f in dur_frames]
        pitch_hz = [float(v) * pitch_std + pitch_mean for v in p_pred[0, :n].cpu().numpy()]
        energy = [float(v) * energy_std + energy_mean for v in e_pred[0, :n].cpu().numpy()]
        labels = units[:n] if lang == "English" else roman[:n]

        # csv module, not f-strings: the "," unit would otherwise add a field and
        # silently break every row after it (seen on Telugu T4).
        import csv
        rows = [["unit", "roman", "duration_frames", "duration_sec", "pitch_hz", "energy"]]
        for i in range(n):
            rows.append([units[i], roman[i], f"{dur_frames[i]:.0f}",
                         f"{dur_sec[i]:.4f}", f"{pitch_hz[i]:.2f}", f"{energy[i]:.4f}"])
        with open(res / "variance_adaptor_plots" / f"{sid}.csv", "w",
                  newline="", encoding="utf-8") as fh:
            csv.writer(fh).writerows(rows)

        plots = res / "variance_adaptor_plots"
        _plot("duration", dur_sec, labels, f"{lang} {sid} - predicted duration",
              "seconds", plots / "duration" / f"{sid}.png", "C0")
        _plot("pitch", pitch_hz, labels, f"{lang} {sid} - predicted F0",
              "Hz", plots / "pitch" / f"{sid}.png", "C3")
        _plot("energy", energy, labels, f"{lang} {sid} - predicted energy",
              "energy", plots / "energy" / f"{sid}.png", "C4")

        info = {
            "id": sid,
            "text": sentence,
            "normalised": textnorm.normalize(sentence, lang),
            "units": n,
            "wav": str(wav_path),
            "wav_sec": round(len(wav) / sr, 3),
            "pred_sec": round(sum(dur_sec), 3),
            "pitch_mean_hz": round(sum(pitch_hz) / max(1, n), 2),
            "energy_mean": round(sum(energy) / max(1, n), 4),
        }
        summary.append(info)
        print("[infer]", json.dumps(info, ensure_ascii=False), flush=True)

    (res / "inference_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"restore_step": restore, "sentences": len(summary), "out": str(res)}
