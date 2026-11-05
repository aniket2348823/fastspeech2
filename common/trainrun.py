"""Launch the upstream training script and turn its logs into the loss plots.

The reference `train.py` appends one human-readable line per log step to
``output/log/train/log.txt`` and one per validation to ``output/log/val/log.txt``.
Parsing those two files is more robust than reading TensorBoard event files
(no protobuf/tensorboard dependency, and it works after an interrupted run).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

from . import config, fs2cfg, patch_repo

TRAIN_RE = re.compile(
    r"Step (\d+)/\d+, Total Loss: ([\d.]+), Mel Loss: ([\d.]+), "
    r"Mel PostNet Loss: ([\d.]+), Pitch Loss: ([\d.]+), "
    r"Energy Loss: ([\d.]+), Duration Loss: ([\d.]+)")
VAL_RE = re.compile(
    r"Validation Step (\d+), Total Loss: ([\d.]+), Mel Loss: ([\d.]+), "
    r"Mel PostNet Loss: ([\d.]+), Pitch Loss: ([\d.]+), "
    r"Energy Loss: ([\d.]+), Duration Loss: ([\d.]+)")

NAMES = ["total", "mel", "mel_postnet", "pitch", "energy", "duration"]


def _parse(path: Path, pattern: re.Pattern) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {"step": []}
    for name in NAMES:
        out[name] = []
    if not path.exists():
        return out
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        m = pattern.search(line)
        if not m:
            continue
        out["step"].append(int(m.group(1)))
        for i, name in enumerate(NAMES):
            out[name].append(float(m.group(i + 2)))
    return out


def log_paths(lang: str) -> tuple[Path, Path]:
    log = fs2cfg.paths(lang)["train"]
    import yaml
    try:
        data = yaml.safe_load(log.read_text(encoding="utf-8"))
        base = Path(data["path"]["log_path"])
    except Exception:
        # No config rendered yet (e.g. --plot-only before training): fall back to
        # the standard log dir so plotting shows an empty "no training log yet".
        base = config.lang_dir(lang) / "output" / "log"
    return base / "train" / "log.txt", base / "val" / "log.txt"


def latest_checkpoint(lang: str) -> int:
    ckpt_dir = config.lang_dir(lang) / "output" / "ckpt"
    steps = []
    for p in ckpt_dir.glob("*.pth.tar"):
        try:
            steps.append(int(p.name.split(".")[0]))
        except ValueError:
            pass
    return max(steps) if steps else 0


def ensure_vocoder(lang: str) -> Path:
    cfg = config.load(lang)
    repo = config.require_repo(lang)
    speaker = cfg["vocoder"]["speaker"]
    target = repo / "hifigan" / f"generator_{speaker}.pth.tar"
    if target.exists():
        return target
    zipped = target.with_suffix(".tar.zip")
    if zipped.exists():
        import zipfile
        with zipfile.ZipFile(zipped) as zf:
            zf.extractall(repo / "hifigan")
        if target.exists():
            return target
    raise SystemExit(
        f"HiFi-GAN checkpoint missing: {target}\n"
        "expected the repo's own hifigan/generator_{}.pth.tar.zip".format(speaker))


def prune_checkpoints(lang: str, keep: int = 2) -> list[str]:
    """Delete all but the newest ``keep`` checkpoints (each is ~350 MB)."""
    ckpt_dir = config.lang_dir(lang) / "output" / "ckpt"
    files = []
    for p in ckpt_dir.glob("*.pth.tar"):
        try:
            files.append((int(p.name.split(".")[0]), p))
        except ValueError:
            pass
    files.sort()
    removed = []
    for _, p in files[:-keep] if keep else files:
        p.unlink()
        removed.append(p.name)
    return removed


def run(
    lang: str,
    steps: int | None = None,
    restore_step: int = 0,
    log_every: int | None = None,
    val_every: int | None = None,
    save_every: int | None = None,
) -> int:
    repo = config.require_repo(lang)
    cfg = config.load(lang)
    patch_repo.apply(lang)
    ensure_vocoder(lang)
    fs2cfg.render(lang)
    if steps:
        fs2cfg.set_total_steps(lang, steps)
    if log_every or val_every or save_every:
        fs2cfg.set_schedule(lang, log_every, val_every, save_every)
    p = fs2cfg.paths(lang)

    cmd = [sys.executable, "train.py",
           "-p", str(p["preprocess"]),
           "-m", str(p["model"]),
           "-t", str(p["train"])]
    if restore_step:
        cmd += ["--restore_step", str(restore_step)]
    (config.lang_dir(lang) / "output" / "ckpt").mkdir(parents=True, exist_ok=True)
    (config.lang_dir(lang) / "output" / "result").mkdir(parents=True, exist_ok=True)
    print("[train] $", " ".join(cmd), flush=True)
    print("[train] cwd =", repo, flush=True)
    code = subprocess.run(cmd, cwd=repo).returncode
    if code == 0:
        # A finished run only ever needs the newest checkpoints; 10 x 350 MB
        # would otherwise fill /kaggle/working and slow the version commit.
        removed = prune_checkpoints(lang, keep=2)
        if removed:
            print("[train] pruned old checkpoints:", ", ".join(removed), flush=True)
    return code


def plot_losses(lang: str) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    train_txt, val_txt = log_paths(lang)
    tr = _parse(train_txt, TRAIN_RE)
    va = _parse(val_txt, VAL_RE)
    out = config.results_dir(lang)
    out.mkdir(parents=True, exist_ok=True)
    made = {}

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if tr["step"]:
        axes[0].plot(tr["step"], tr["total"], color="C0")
        axes[0].set_title("training total loss")
        for i, name in enumerate(NAMES[1:], start=1):
            axes[1].plot(tr["step"], tr[name], label=name, linewidth=1)
        axes[1].set_title("training loss components")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "no training log yet", ha="center", va="center")
    fig.suptitle(f"{lang} - training loss")
    fig.tight_layout()
    made["training_loss"] = out / "training_loss.png"
    fig.savefig(made["training_loss"], dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    if va["step"]:
        if tr["step"]:
            axes[0].plot(tr["step"], tr["total"], color="C0", alpha=0.5, label="train")
        axes[0].plot(va["step"], va["total"], color="C3", marker="o", ms=3, label="validation")
        axes[0].set_title("total loss: train vs validation")
        axes[0].legend(fontsize=8)
        for i, name in enumerate(NAMES[1:], start=1):
            axes[1].plot(va["step"], va[name], label=name, linewidth=1)
        axes[1].set_title("validation loss components")
        axes[1].legend(fontsize=8)
        for ax in axes:
            ax.set_xlabel("step")
            ax.grid(alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "no validation log yet", ha="center", va="center")
    fig.suptitle(f"{lang} - validation loss")
    fig.tight_layout()
    made["validation_loss"] = out / "validation_loss.png"
    fig.savefig(made["validation_loss"], dpi=150)
    plt.close(fig)

    # copy the checkpoints the report/submission needs next to the config
    if latest_checkpoint(lang):
        print(f"[plot] latest checkpoint: {latest_checkpoint(lang)}", flush=True)
    return {k: str(v) for k, v in made.items()}


def main() -> None:
    ap = argparse.ArgumentParser(description="train one language")
    ap.add_argument("--lang", required=True, choices=list(config.LANGS))
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--restore-step", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=None)
    ap.add_argument("--val-every", type=int, default=None)
    ap.add_argument("--save-every", type=int, default=None)
    ap.add_argument("--plot-only", action="store_true")
    args = ap.parse_args()
    if not args.plot_only:
        code = run(args.lang, args.steps, args.restore_step,
                   args.log_every, args.val_every, args.save_every)
        if code != 0:
            raise SystemExit(f"train.py exited with {code}")
    print(plot_losses(args.lang))
