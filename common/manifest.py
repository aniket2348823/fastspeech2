"""Choose ~5 h train / ~1 h validation from a metadata CSV and record the policy.

Input columns are configured in ``project/<lang>/config.yaml``; the file needs a
relative audio path, a transcript and optionally a speaker label.

Selection policy (documented in the report):

* Every utterance outside ``[min_sec, max_sec]`` is dropped first, so a single
  40 s clip cannot dominate the hour budget and very short clips cannot floor it.
* The remaining pool is shuffled once with a fixed seed.
* Walk the shuffled pool: fill train until ``train_hours``, then val until
  ``val_hours``, then stop.  One row can be consumed once, so the two splits are
  disjoint by construction (asserted).
* The chosen rows are written in the upstream metadata format
  ``<file>|<speaker>|<text>`` and summarised in ``split_stats.json``.
"""
from __future__ import annotations

import csv
import json
import random
from pathlib import Path

from . import config, textnorm


def _read_rows(path: Path, text_col: str, path_col: str, spk_col: str | None) -> list[dict]:
    delim = "\t" if path.suffix.lower() in (".tsv",) else ","
    with open(path, encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh, delimiter=delim)
        if reader.fieldnames is None:
            raise SystemExit(f"{path}: empty file")
        for col in (text_col, path_col):
            if col not in reader.fieldnames:
                raise SystemExit(f"{path}: no column {col!r} (have {reader.fieldnames})")
        rows = []
        for rec in reader:
            speaker = (rec.get(spk_col) or "spk1").strip() if spk_col else "spk1"
            rows.append({
                "file": (rec.get(path_col) or "").strip(),
                "text": (rec.get(text_col) or "").strip(),
                "speaker": speaker or "spk1",
            })
    return [r for r in rows if r["file"] and r["text"]]


def _duration(path: Path) -> float | None:
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        pass
    try:                                     # stdlib fallback, wav only
        import wave
        with wave.open(str(path)) as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def build(lang: str) -> dict:
    cfg = config.load(lang)
    ds = cfg["dataset"]
    csv_path = config.resolve(ds["metadata_csv"])
    audio_dir = config.resolve(ds["audio_dir"])
    cols = ds.get("columns", {})
    split = cfg["split"]

    if not csv_path.exists():
        raise SystemExit(f"{csv_path} not found - export the dataset first (see README)")
    if not audio_dir.is_dir():
        raise SystemExit(f"{audio_dir} not found - export the dataset first (see README)")

    rows = _read_rows(csv_path, cols.get("text", "text"),
                      cols.get("path", "file"), cols.get("speaker"))
    dropped = {"missing_audio": 0, "out_of_range": 0, "empty_text": 0}
    kept: list[dict] = []

    for r in rows:
        wav = audio_dir / r["file"]
        if not wav.is_file():
            dropped["missing_audio"] += 1
            continue
        dur = _duration(wav)
        if dur is None:
            dropped["missing_audio"] += 1
            continue
        if not (split["min_sec"] <= dur <= split["max_sec"]):
            dropped["out_of_range"] += 1
            continue
        r["dur"] = dur
        r["text_norm"] = textnorm.normalize(r["text"], lang)
        if not r["text_norm"]:
            dropped["empty_text"] += 1
            continue
        kept.append(r)

    random.Random(split["seed"]).shuffle(kept)

    # Use proportional split if hour budgets exceed available data.
    # Falls back to hour-based greedy when enough data exists.
    total_h = sum(r["dur"] for r in kept) / 3600.0
    train_ratio = split.get("train_ratio", 0.8)

    if total_h < split["train_hours"] + split["val_hours"]:
        # Not enough data for hour budgets -> proportional split
        split_idx = int(len(kept) * train_ratio)
        train = kept[:split_idx]
        val = kept[split_idx:]
        t_h = sum(r["dur"] for r in train) / 3600.0
        v_h = sum(r["dur"] for r in val) / 3600.0
        print(f"[manifest] insufficient data ({total_h:.2f}h < {split['train_hours']+split['val_hours']}h) -> proportional {int(train_ratio*100)}/{int((1-train_ratio)*100)} split: {t_h:.2f}h train / {v_h:.2f}h val")
    else:
        train: list[dict] = []
        val: list[dict] = []
        t_h = v_h = 0.0
        for r in kept:
            if t_h < split["train_hours"]:
                train.append(r)
                t_h += r["dur"] / 3600.0
            elif v_h < split["val_hours"]:
                val.append(r)
                v_h += r["dur"] / 3600.0
            else:
                break

        # Guarantee at least one validation utterance if we have data
        if not val and train:
            val.append(train.pop())
            v_h = val[-1]["dur"] / 3600.0
            t_h -= v_h

    overlap = {r["file"] for r in train} & {r["file"] for r in val}
    assert not overlap, f"train/val overlap: {sorted(overlap)[:5]}"
    if not train or not val:
        raise SystemExit(
            f"empty split (train={len(train)}, val={len(val)}, usable={len(kept)}); "
            f"check audio_dir={audio_dir}"
        )

    out = config.lang_dir(lang) / "data"
    out.mkdir(parents=True, exist_ok=True)
    for name, rows_ in (("train.txt", train), ("val.txt", val)):
        with open(out / name, "w", encoding="utf-8", newline="") as fh:
            for r in rows_:
                fh.write(f"{r['file']}|{r['speaker']}|{r['text_norm']}\n")

    stats = {
        "language": lang,
        "metadata_csv": str(csv_path),
        "audio_dir": str(audio_dir),
        "pool_size": len(rows),
        "usable": len(kept),
        "dropped": dropped,
        "train_utts": len(train),
        "val_utts": len(val),
        "train_hours": round(t_h, 3),
        "val_hours": round(v_h, 3),
        "train_speakers": len({r["speaker"] for r in train}),
        "mean_utt_sec": round(sum(r["dur"] for r in train + val) / len(train + val), 2),
        "seed": split["seed"],
        "min_sec": split["min_sec"],
        "max_sec": split["max_sec"],
        "policy": "drop out-of-range, global seeded shuffle, greedy fill to hour budget",
        "overlap": 0,
    }
    (out / "split_stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    return stats
