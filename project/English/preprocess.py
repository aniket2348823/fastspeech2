"""English preprocessing: metadata split -> MMS alignment -> upstream preprocessed data.

Usage
-----
    python project/English/preprocess.py --stage all
    python project/English/preprocess.py --stage manifest
    python project/English/preprocess.py --stage align --limit 20   # smoke test
    python project/English/preprocess.py --stage repo --force       # redo features
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import align, config, fs2stage, manifest, patch_repo  # noqa: E402

LANG = "English"


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{LANG} preprocessing")
    ap.add_argument("--stage", default="all",
                    choices=["all", "patch", "manifest", "align", "repo"])
    ap.add_argument("--split", default="all", choices=["all", "train", "val"])
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="re-run feature extraction")
    args = ap.parse_args()

    config.ensure_dirs(LANG)

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
