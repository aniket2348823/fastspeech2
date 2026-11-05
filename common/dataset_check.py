"""Cheap audit of the preprocessed features, run right before training.

Two invariants have to hold for every utterance or the 5-hour run dies at the
first batch that contains it:

* ``len(text_to_sequence(text)) == len(duration)`` and one pitch/energy value
  per phone.  A phone that received zero frames during alignment leaves the
  stored pitch/energy vectors shorter than the phone sequence, and
  ``x + pitch_embedding`` then fails with
  ``The size of tensor a (98) must match the size of tensor b (69)``.
* ``int(duration.sum()) == mel.shape[0]`` -- the length regulator expands the
  encoder output to ``sum(duration)`` frames, which must equal the mel length.

Both mismatches are repairable without re-aligning (2 h) or re-extracting
features: the frame counts are rescaled proportionally to the mel length and
the short pitch/energy tails are padded with 0 (unvoiced / silent).  Only a
phone count that disagrees with the stored duration is fatal, because no
repair can invent the missing targets.

Run it standalone with::

    python -m common.dataset_check --lang English
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml

from . import config, fs2cfg

SPLITS = ("train", "val")


def _preprocessed_config(lang: str) -> dict:
    pre = fs2cfg.paths(lang)["preprocess"]
    if not pre.exists():
        fs2cfg.render(lang)
    return yaml.safe_load(pre.read_text(encoding="utf-8"))


def _rescale(duration: np.ndarray, frames: int) -> np.ndarray | None:
    """Spread ``frames`` frames over the phones, proportional to ``duration``."""
    weights = np.asarray(duration, dtype=np.float64)
    total = weights.sum()
    if total <= 0 or frames <= 0:
        return None
    weights = weights / total * frames
    new = np.floor(weights).astype(np.int64)
    shortfall = frames - int(new.sum())
    if shortfall > 0:
        # largest remainder: keeps the total exact without changing the shape
        order = np.argsort(-(weights - new))
        new[order[:shortfall]] += 1
    return new


def audit(lang: str, repair: bool = True) -> dict:
    """Check every utterance; repair frame counts and short feature vectors."""
    repo = config.require_repo(lang)
    cfg = _preprocessed_config(lang)
    pp = Path(cfg["path"]["preprocessed_path"])
    cleaners = cfg["preprocessing"]["text"]["text_cleaners"]

    import sys

    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    from text import text_to_sequence  # the reference repo's package

    report: dict = {"preprocessed_path": str(pp), "splits": {},
                    "n_rescaled": 0, "n_padded": 0}

    for split in SPLITS:
        meta = pp / f"{split}.txt"
        if not meta.exists():
            raise SystemExit(f"missing {meta}; run 02_preprocess.ipynb first")
        rows = [line.split("|") for line in
                meta.read_text(encoding="utf-8").splitlines() if line.strip()]
        bad_len, bad_frames, rescaled, padded = [], [], [], []
        speakers = set()

        for name, spk, text, _raw in rows:
            speakers.add(spk)
            dur_path = pp / "duration" / f"{spk}-duration-{name}.npy"
            duration = np.load(dur_path)
            mel = np.load(pp / "mel" / f"{spk}-mel-{name}.npy", mmap_mode="r")
            n_phones = len(text_to_sequence(text, cleaners))

            if n_phones != len(duration):
                bad_len.append({"id": name, "tokens": n_phones,
                                "duration": int(len(duration))})
                continue

            frames = int(mel.shape[0])
            if int(duration.sum()) != frames:
                fixed = _rescale(duration, frames) if repair else None
                if fixed is None:
                    bad_frames.append({"id": name, "sum_duration": int(duration.sum()),
                                       "mel_frames": frames})
                    continue
                rescaled.append({"id": name, "before": int(duration.sum()),
                                 "after": frames})
                duration = fixed
                np.save(dur_path, duration)

            for feature in ("pitch", "energy"):
                path = pp / feature / f"{spk}-{feature}-{name}.npy"
                arr = np.load(path)
                if len(arr) == n_phones:
                    continue
                fixed = np.zeros(n_phones, dtype=np.float32)
                keep = min(len(arr), n_phones)
                fixed[:keep] = arr[:keep]
                if repair:
                    np.save(path, fixed)
                padded.append(f"{name}:{feature} {len(arr)}->{n_phones}")

        report["splits"][split] = {
            "utts": len(rows),
            "speakers": len(speakers),
            "n_bad_length": len(bad_len),
            "n_bad_frames": len(bad_frames),
            "n_rescaled": len(rescaled),
            "n_padded": len(padded),
            "bad_length": bad_len[:10],
            "bad_frames": bad_frames[:10],
            "rescaled": rescaled[:5],
            "padded": padded[:5],
        }
        report["n_rescaled"] += len(rescaled)
        report["n_padded"] += len(padded)

    # read original split (before preprocessor filtering) to know expected counts
    orig = config.lang_dir(lang) / "data"
    orig_train = {line.split("|")[0] for line in
                  (orig / "train.txt").read_text(encoding="utf-8").splitlines() if line.strip()}
    orig_val = {line.split("|")[0] for line in
                (orig / "val.txt").read_text(encoding="utf-8").splitlines() if line.strip()}

    def ids(split: str) -> set[str]:
        return {line.split("|")[0] for line in
                (pp / f"{split}.txt").read_text(encoding="utf-8").splitlines() if line.strip()}

    pre_train = ids("train")
    pre_val = ids("val")

    report["overlap"] = sorted(pre_train & pre_val)
    # train must have utterances after preprocessing; val may be empty if preprocessor
    # dropped all (rare with real data, common with tiny test sets).  Fail only if
    # original val was non-empty but preprocessed val is empty AND train is also empty.
    train_ok = bool(pre_train and report["splits"]["train"]["utts"] > 0
                    and report["splits"]["train"]["n_bad_length"] == 0
                    and report["splits"]["train"]["n_bad_frames"] == 0)
    val_ok = bool(report["splits"]["val"]["n_bad_length"] == 0
                  and report["splits"]["val"]["n_bad_frames"] == 0)
    # if original val had utterances but none survived preprocessing, flag it
    if orig_val and not pre_val:
        report["val_dropped_all"] = True
    report["ok"] = train_ok and val_ok
    return report


def main(argv: list[str] | None = None) -> None:
    import argparse
    import json

    ap = argparse.ArgumentParser(description="audit preprocessed features")
    ap.add_argument("--lang", required=True, choices=list(config.LANGS))
    ap.add_argument("--no-repair", action="store_true")
    args = ap.parse_args(argv)
    report = audit(args.lang, repair=not args.no_repair)
    print(json.dumps(report, indent=2))
    if not report["ok"]:
        raise SystemExit("dataset audit failed")


if __name__ == "__main__":
    main()
