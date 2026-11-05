"""Telugu preprocessing: vocabulary -> metadata split -> alignment -> features.

The extra ``vocab`` stage is Telugu-specific: the grapheme inventory is collected
from every transcript in the pool (plus the assignment's test sentences, so the
fixed test set cannot contain an out-of-vocabulary character) and then installed
into the Telugu clone's ``text/symbols.py`` before the feature extractor runs.

Usage
-----
    python project/Telugu/preprocess.py --stage all
    python project/Telugu/preprocess.py --stage vocab
    python project/Telugu/preprocess.py --stage align --limit 20
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import align, config, fs2stage, manifest, patch_repo, telugu, textnorm  # noqa: E402

LANG = "Telugu"


def build_vocab() -> dict:
    cfg = config.load(LANG)
    csv_path = config.resolve(cfg["dataset"]["metadata_csv"])
    audio_dir = config.resolve(cfg["dataset"]["audio_dir"])
    cols = cfg["dataset"]["columns"]

    texts: list[str] = []
    if csv_path.exists():
        with open(csv_path, encoding="utf-8", newline="") as fh:
            for rec in csv.DictReader(fh):
                text = (rec.get(cols.get("text", "text")) or "").strip()
                if not text:
                    continue
                rel = (rec.get(cols.get("path", "file")) or "").strip()
                if rel and not (audio_dir / rel).is_file():
                    continue
                texts.append(text)

    # the fixed test sentences must never be out of vocabulary
    texts += [s["text"] for s in cfg["inference"]["sentences"]]

    vocab = telugu.build_vocab([textnorm.normalize(t, LANG) for t in texts])
    path = config.lang_dir(LANG) / "data" / "vocab.json"
    telugu.save_vocab(vocab, path)

    oov = set()
    for s in cfg["inference"]["sentences"]:
        for unit in telugu.units(textnorm.normalize(s["text"], LANG)):
            if unit not in vocab:
                oov.add(unit)
    return {"vocab": str(path), "size": len(vocab), "sources": len(texts),
            "test_oov": sorted(oov)}


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{LANG} preprocessing")
    ap.add_argument("--stage", default="all",
                    choices=["all", "vocab", "patch", "manifest", "align", "repo"])
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run feature extraction")
    args = ap.parse_args()

    config.ensure_dirs(LANG)

    if args.stage in ("all", "vocab", "repo"):
        print(json.dumps(build_vocab(), indent=2, ensure_ascii=False), flush=True)

    if args.stage in ("all", "patch"):
        print(json.dumps(patch_repo.apply(LANG), indent=2, ensure_ascii=False), flush=True)

    if args.stage in ("all", "manifest"):
        print(json.dumps(manifest.build(LANG), indent=2), flush=True)

    if args.stage in ("all", "align"):
        splits = ("train", "val") if args.split == "all" else (args.split,)
        print(json.dumps(align.run(LANG, splits, args.limit), indent=2), flush=True)

    if args.stage in ("all", "repo"):
        print(json.dumps(fs2stage.stage(LANG, force=args.force), indent=2), flush=True)


if __name__ == "__main__":
    main()
