"""Stage data into the layout the upstream preprocessor reads, then run it.

The reference preprocessor's contract (verified against the source):

    raw_path/<speaker>/<name>.wav
    raw_path/<speaker>/<name>.lab                                  # display text
    preprocessed_path/TextGrid/<speaker>/<name>.TextGrid          # "phones" tier

It never reads ``.lab`` for training: the model's input string is built from the
TextGrid labels (``text = "{" + " ".join(phone) + "}"``).  ``.lab`` only feeds the
``raw_text`` column that evaluation prints.

The preprocessor also shuffles and slices ``val_size`` utterances on its own.
We set ``val_size: 0`` and rewrite ``train.txt`` / ``val.txt`` from our recorded
split, so the assignment's "no overlap" requirement is enforced by our code and
is reproducible.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

from . import align, config, fs2cfg, patch_repo


def read_split(lang: str, split: str) -> list[tuple[str, str, str]]:
    path = config.lang_dir(lang) / "data" / f"{split}.txt"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel, spk, text = line.split("|", 2)
            rows.append((rel, spk, text))
    return rows


def _link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists() or dst.is_symlink():
        return
    try:
        os.symlink(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def _measure(cfg, pre: Path, rows_by_split: dict[str, list[str]]) -> dict[str, float]:
    """Hours per split from the duration features the model will train on.

    ``duration`` holds one frame count per phoneme, so ``sum()`` is the number of
    frames (``len()`` would be the phoneme count and under-report by ~10x).
    """
    import numpy as np

    hop, sr = cfg["audio"]["hop_length"], cfg["audio"]["sampling_rate"]
    hours = {}
    for split, rows in rows_by_split.items():
        frames = 0
        for line in rows:
            name, spk = line.split("|")[0], line.split("|")[1]
            npy = pre / "duration" / f"{spk}-duration-{name}.npy"
            if npy.is_file():
                frames += int(np.load(npy).sum())
        hours[split] = round(frames * hop / sr / 3600, 3)
    return hours


def stage(lang: str, force: bool = False) -> dict:
    cfg = config.load(lang)
    repo = config.require_repo(lang)
    ds = cfg["dataset"]["name"]
    audio_dir = config.resolve(cfg["dataset"]["audio_dir"])
    align_root = config.lang_dir(lang) / "alignments"

    raw = repo / "raw_data" / ds
    pre = repo / "preprocessed_data" / ds
    marker = pre / "_stage_complete.json"

    splits = {s: read_split(lang, s) for s in ("train", "val")}
    stems = {s: {Path(rel).stem for rel, _, _ in rows} for s, rows in splits.items()}
    dup = stems["train"] & stems["val"]
    if dup:
        raise SystemExit(f"duplicate stems across train/val: {sorted(dup)[:5]}")
    if len(stems["train"]) != len(splits["train"]):
        raise SystemExit("duplicate file names inside the train split; rename them first")

    if marker.exists() and not force:
        info = json.loads(marker.read_text(encoding="utf-8"))
        if "train_hours" in info:
            print(f"[stage] {lang}: already preprocessed ({info}); pass --force to redo")
            return info
        # marker written by an older build: measure what is already on disk
        # instead of redoing six minutes of feature extraction
        rows_by_split = {
            s: [l for l in (pre / f"{s}.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
            for s in ("train", "val") if (pre / f"{s}.txt").exists()
        }
        if len(rows_by_split) == 2:
            merged = {line.split("|")[0] for rows in rows_by_split.values() for line in rows}
            dropped = [
                {"split": s, "name": n}
                for s in ("train", "val")
                for n in sorted(stems[s])
                if n not in merged
            ]
            hours = _measure(cfg, pre, rows_by_split)
            info.update({
                "train": len(rows_by_split["train"]),
                "val": len(rows_by_split["val"]),
                "train_hours": hours["train"],
                "val_hours": hours["val"],
                "dropped_by_preprocessor": len(dropped),
                "dropped": dropped,
            })
            marker.write_text(json.dumps(info, indent=2), encoding="utf-8")
            print(f"[stage] {info}", flush=True)
            return info
        print("[stage] incomplete preprocessed data - redoing it", flush=True)
        marker.unlink(missing_ok=True)

    # Ensure TextGrid alignments exist before staging; auto-run align if missing.
    missing_grids = [(s, r) for s in ("train", "val") for r in splits[s]
                     if not (align_root / s / f"{Path(r[0]).stem}.TextGrid").is_file()]
    if missing_grids:
        print(f"[stage] {len(missing_grids)} utterances lack alignments - running align...",
              flush=True)
        align.run(lang, ("train", "val"), limit=0)

    missing_audio = missing_tg = 0
    staged = 0
    for split, rows in splits.items():
        for rel, spk, text in rows:
            spk_safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in spk) or "spk1"
            spk_dir = raw / spk_safe
            tg_dir = pre / "TextGrid" / spk_safe
            spk_dir.mkdir(parents=True, exist_ok=True)
            tg_dir.mkdir(parents=True, exist_ok=True)

            name = Path(rel).stem
            src_wav = audio_dir / rel
            if not src_wav.is_file():
                missing_audio += 1
                continue
            _link_or_copy(src_wav, spk_dir / f"{name}.wav")

            lab = spk_dir / f"{name}.lab"
            if not lab.exists():
                lab.write_text(text.replace("|", " ").replace("\n", " ") + "\n",
                               encoding="utf-8")

            src_tg = align_root / split / f"{name}.TextGrid"
            if not src_tg.is_file():
                missing_tg += 1
                continue
            dst_tg = tg_dir / f"{name}.TextGrid"
            if not dst_tg.exists():
                shutil.copy2(src_tg, dst_tg)
            staged += 1

    if missing_audio:
        raise SystemExit(f"{missing_audio} audio files missing under {audio_dir}")
    if missing_tg:
        raise SystemExit(f"{missing_tg} utterances have no TextGrid - run the align stage first")
    if staged == 0:
        raise SystemExit("nothing staged")

    patch_repo.apply(lang)
    fs2cfg.render(lang)
    p = fs2cfg.paths(lang)
    cmd = [sys.executable, "preprocess.py", str(p["preprocess"])]
    print("[stage] $", " ".join(cmd), flush=True)
    if subprocess.run(cmd, cwd=repo).returncode != 0:
        raise SystemExit("upstream preprocess.py failed (see the traceback above)")

    if not (pre / "train.txt").exists():
        raise SystemExit(
            f"upstream preprocess.py wrote no metadata at {pre / 'train.txt'}\n"
            "this almost always means the align stage produced no TextGrids. "
            "Run the align stage first:\n"
            f"  python project/{lang}/preprocess.py --stage align"
        )

    emitted = {}
    for line in (pre / "train.txt").read_text(encoding="utf-8").splitlines():
        if line.strip():
            emitted[line.split("|")[0]] = line
    print(f"[stage] preprocessor emitted {len(emitted)} utterances "
          f"of {len(stems['train']) + len(stems['val'])} staged", flush=True)

    kept = {}
    splits_rows = {}
    for split in ("train", "val"):
        rows = [emitted[n] for n in sorted(stems[split]) if n in emitted]
        kept[split] = len(rows)
        splits_rows[split] = rows
        (pre / f"{split}.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")

    dropped = [
        {"split": split, "name": n}
        for split in ("train", "val")
        for n in sorted(stems[split])
        if n not in emitted
    ]

    # Hours straight from the duration features the model will actually train
    # on (per-phoneme frames -> seconds), not from the manifest estimate.
    hours = _measure(cfg, pre, splits_rows)

    info = {
        "language": lang,
        "preprocessed": str(pre),
        "staged": staged,
        "train": kept["train"],
        "val": kept["val"],
        "train_hours": hours["train"],
        "val_hours": hours["val"],
        "dropped_by_preprocessor": len(dropped),
        "dropped": dropped,
    }
    marker.write_text(json.dumps(info, indent=2), encoding="utf-8")
    print(f"[stage] {info}", flush=True)
    return info
