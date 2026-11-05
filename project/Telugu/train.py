"""Telugu training entry point.

    python project/Telugu/train.py                 # full run (config train.max_steps)
    python project/Telugu/train.py --steps 2000    # smoke test
    python project/Telugu/train.py --restore-step 20000
    python project/Telugu/train.py --plot-only

After training it (re)writes ``Results/Telugu/training_loss.png`` and
``Results/Telugu/validation_loss.png`` from the run's logs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common import config, trainrun  # noqa: E402

LANG = "Telugu"


def main() -> None:
    ap = argparse.ArgumentParser(description=f"{LANG} training")
    ap.add_argument("--steps", type=int, default=None,
                    help="override train.max_steps (smoke test)")
    ap.add_argument("--restore-step", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()

    config.ensure_dirs(LANG)
    if not args.plot_only:
        code = trainrun.run(LANG, args.steps, args.restore_step,
                            args.log_every, args.val_every, args.save_every)
        if code != 0:
            raise SystemExit(f"train.py exited with code {code}")
    print(json.dumps(trainrun.plot_losses(LANG), indent=2))


if __name__ == "__main__":
    main()
