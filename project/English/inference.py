"""English inference: synthesise every test sentence and plot the variance adaptor.

    python project/English/inference.py
    python project/English/inference.py --step 40000
    python project/English/inference.py --only E2a

Writes ``Results/English/test_generated_audios/<ID>.wav`` plus
``Results/English/variance_adaptor_plots/{duration,pitch,energy}/<ID>.png``.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config, inferrun  # noqa: E402

LANG = "English"


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{LANG} inference")
    ap.add_argument("--step", type=int, default=None,
                    help="checkpoint step to load (default: latest)")
    ap.add_argument("--only", default=None, help="single sentence id, e.g. E2a")
    args = ap.parse_args()
    config.ensure_dirs(LANG)
    print(json.dumps(inferrun.run(LANG, args.step, args.only), indent=2))


if __name__ == "__main__":
    main()
