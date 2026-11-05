"""Locations and config loading for the FastSpeech2 assignment harness.

Layout (identical on this laptop and on Kaggle)::

    <root>/
        common/          shared code
        project/         the deliverable (English/, Telugu/, Results/, Report/)
        third_party/     upstream FastSpeech2 clones (not part of the deliverable)
        data/            datasets (symlink to /kaggle/working/data on Kaggle)

One upstream clone per language: ``third_party/FastSpeech2_English`` and
``third_party/FastSpeech2_Telugu``.  Telugu needs extra entries in
``text/symbols.py``; keeping two clones means that patch can never shift the
English symbol ids (which would invalidate an English checkpoint).
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT / "project"

LANGS = ("English", "Telugu")


def lang_dir(lang: str) -> Path:
    return PROJECT / lang


def results_dir(lang: str) -> Path:
    return PROJECT / "Results" / lang


def repo_dir(lang: str) -> Path:
    return ROOT / "third_party" / f"FastSpeech2_{lang}"


def require_repo(lang: str) -> Path:
    repo = repo_dir(lang)
    if not (repo / "train.py").exists():
        raise SystemExit(
            f"upstream FastSpeech2 not found at {repo}\n"
            "run the setup cell of notebooks/01_setup_data.ipynb, or:\n"
            f"  git clone --depth 1 https://github.com/ming024/FastSpeech2 {repo}"
        )
    return repo


def load(lang: str) -> dict:
    with open(lang_dir(lang) / "config.yaml", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    cfg["lang"] = lang
    return cfg


def resolve(path: str | Path) -> Path:
    """Expand env vars; relative paths are relative to the project root.

    Kaggle keeps datasets outside the code directory, so if the project-relative
    path is absent the same relative path under /kaggle/working is tried.
    """
    p = Path(os.path.expandvars(str(path))).expanduser()
    if p.is_absolute():
        return p
    primary = ROOT / p
    if primary.exists():
        return primary
    alt = Path("/kaggle/working") / p
    return alt if alt.exists() else primary


def ensure_dirs(lang: str) -> None:
    """Create every output directory the assignment's folder tree requires."""
    res = results_dir(lang)
    for sub in (
        res / "test_generated_audios",
        res / "variance_adaptor_plots" / "duration",
        res / "variance_adaptor_plots" / "pitch",
        res / "variance_adaptor_plots" / "energy",
        lang_dir(lang) / "checkpoints",
        lang_dir(lang) / "data",
        lang_dir(lang) / "alignments" / "train",
        lang_dir(lang) / "alignments" / "val",
        lang_dir(lang) / "output" / "ckpt",
        lang_dir(lang) / "output" / "log",
        lang_dir(lang) / "fs2_config",
    ):
        sub.mkdir(parents=True, exist_ok=True)
