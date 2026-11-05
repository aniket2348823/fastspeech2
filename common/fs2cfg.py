"""Render the upstream repo's three YAML files from project/<lang>/config.yaml.

The reference implementation takes three config files and nothing else:

* ``preprocess.yaml`` -> top-level ``dataset`` plus ``path`` and ``preprocessing``
* ``model.yaml``      -> transformer / variance predictor / ``multi_speaker``
* ``train.yaml``      -> optimizer and step schedule plus output paths

Absolute paths are emitted on purpose: the scripts are launched with
``cwd=<repo>``, so relative paths would resolve against a different tree.
"""
from __future__ import annotations

from pathlib import Path

import yaml

from . import config


def _cleaners(value) -> list[str]:
    if not value:
        return []
    return list(value) if isinstance(value, (list, tuple)) else [value]


def paths(lang: str) -> dict[str, Path]:
    d = config.lang_dir(lang) / "fs2_config"
    return {name: d / f"{name}.yaml" for name in ("preprocess", "model", "train")}


def render(lang: str) -> Path:
    cfg = config.load(lang)
    repo = config.require_repo(lang)
    out_dir = config.lang_dir(lang) / "fs2_config"
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = cfg["dataset"]["name"]
    raw = (repo / "raw_data" / ds).resolve()
    pre = (repo / "preprocessed_data" / ds).resolve()
    exp = config.lang_dir(lang) / "output"

    preprocess = {
        "dataset": ds,
        "path": {
            "corpus_path": str(raw),
            "raw_path": str(raw),
            "preprocessed_path": str(pre),
            "lexicon_path": str((repo / "lexicon" / "librispeech-lexicon.txt").resolve()),
        },
        "preprocessing": {
            # the reference shuffles and slices val_size utterances; we write our
            # own deterministic, disjoint train.txt / val.txt afterwards
            "val_size": 0,
            "text": {
                "text_cleaners": _cleaners(cfg["text"].get("cleaner")),
                "language": cfg["text"]["language"],
            },
            "audio": {
                "sampling_rate": cfg["audio"]["sampling_rate"],
                "max_wav_value": 32768.0,
            },
            "stft": {
                "filter_length": cfg["audio"]["fft_size"],
                "hop_length": cfg["audio"]["hop_length"],
                "win_length": cfg["audio"]["win_length"],
            },
            "mel": {
                "n_mel_channels": cfg["audio"]["n_mel_channels"],
                "mel_fmin": cfg["audio"]["mel_fmin"],
                "mel_fmax": cfg["audio"]["mel_fmax"],
            },
            "pitch": {
                "feature": cfg["pitch"]["feature"],
                "normalization": bool(cfg["pitch"]["normalization"]),
            },
            "energy": {
                "feature": cfg["energy"]["feature"],
                "normalization": bool(cfg["energy"]["normalization"]),
            },
        },
    }

    model = {
        "transformer": cfg["model"]["transformer"],
        "variance_predictor": {
            "filter_size": cfg["model"]["variance_predictor_filter_size"],
            "kernel_size": cfg["model"]["variance_predictor_kernel_size"],
            "dropout": cfg["model"]["variance_predictor_dropout"],
        },
        "variance_embedding": {
            "pitch_quantization": "linear",
            "energy_quantization": "linear",
            "n_bins": cfg["model"].get("n_bins", 256),
        },
        "multi_speaker": bool(cfg["model"].get("multi_speaker", False)),
        "max_seq_len": cfg["model"]["max_seq_len"],
        "vocoder": {
            "model": "HiFi-GAN",
            "speaker": cfg["vocoder"]["speaker"],
        },
    }

    tr = cfg["train"]
    train = {
        "path": {
            "ckpt_path": str((exp / "ckpt").resolve()),
            "log_path": str((exp / "log").resolve()),
            "result_path": str((exp / "result").resolve()),
        },
        "optimizer": {
            "batch_size": tr["batch_size"],
            "betas": [0.9, 0.98],
            "eps": 1e-9,
            "weight_decay": 0.0,
            "grad_clip_thresh": 1.0,
            "grad_acc_step": tr["grad_accum_step"],
            "warm_up_step": tr["warm_up_step"],
            "anneal_steps": tr["anneal_steps"],
            "anneal_rate": tr.get("anneal_rate", 0.3),
        },
        "step": {
            "total_step": tr["max_steps"],
            "log_step": tr["log_every"],
            "synth_step": tr["val_every"],
            "val_step": tr["val_every"],
            "save_step": tr["save_every"],
        },
    }

    for name, payload in (("preprocess", preprocess), ("model", model), ("train", train)):
        (out_dir / f"{name}.yaml").write_text(
            yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return out_dir


def set_total_steps(lang: str, steps: int) -> None:
    """Override ``step.total_step`` (smoke runs, or a shorter budget)."""
    p = paths(lang)["train"]
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    data["step"]["total_step"] = int(steps)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")


def set_schedule(
    lang: str,
    log: int | None = None,
    val: int | None = None,
    save: int | None = None,
) -> None:
    """Override the log / validation / checkpoint cadence.

    Only used by the smoke test, which has to touch all three code paths
    (TensorBoard + ``log.txt``, ``evaluate()`` on ``val.txt``, ``torch.save``)
    in a handful of steps instead of after 2000.
    """
    p = paths(lang)["train"]
    data = yaml.safe_load(p.read_text(encoding="utf-8"))
    if log:
        data["step"]["log_step"] = int(log)
    if val:
        data["step"]["val_step"] = int(val)
        data["step"]["synth_step"] = int(val)
    if save:
        data["step"]["save_step"] = int(save)
    p.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")
