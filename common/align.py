"""Forced alignment -> phone-level TextGrids, for both languages.

Why this exists
---------------
The reference implementation expects every utterance to come with a TextGrid
whose ``phones`` tier holds the exact units the model consumes.  Montreal
Forced Aligner is the usual tool, but it ships no Telugu acoustic model or
dictionary.  ``torchaudio.pipelines.MMS_FA`` is a character-level CTC model
covering 1100+ languages, so a single aligner serves both languages.

Units
-----
* English: ARPAbet phones (``common.g2p``).  MMS aligns *words* (Latin
  characters); each word span is then split across its phones.  Word timing is
  measured, timing inside a word is uniform (documented approximation).
* Telugu: grapheme clusters (``common.telugu``).  Each cluster is romanised and
  aligned on its own, so cluster boundaries are measured, not interpolated.

Output::

    project/<lang>/alignments/<split>/<name>.TextGrid   # tier "phones"

The file is written in Praat *short* text format directly.  ``tgt`` 1.4 and 1.5
have different writer APIs (``write_textgrid`` vs ``write_to_file``), and the
format is 15 lines, so writing it here removes that dependency entirely.
"""
from __future__ import annotations

import argparse
import traceback
from pathlib import Path

from . import config, g2p, telugu

SILENCE = "sp"


class _Aligner:
    """MMS_FA wrapper; loads the model once per process."""

    def __init__(self, device: str | None = None):
        import torch
        import torchaudio

        self.torch = torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.bundle = torchaudio.pipelines.MMS_FA
        self.model = self.bundle.get_model(with_star=False).to(self.device).eval()
        self.vocab = self.bundle.get_dict()
        self.aligner = self.bundle.get_aligner()
        self.sr = self.bundle.sample_rate

    def token_ids(self, roman: list[str]) -> tuple[list[int], list[tuple[int, int]]]:
        """Flat token ids plus the [lo, hi) slice of ids each roman unit owns."""
        ids: list[int] = []
        chunks: list[tuple[int, int]] = []
        for text in roman:
            lo = len(ids)
            for ch in text:
                idx = self.vocab.get(ch)
                if idx is not None:
                    ids.append(idx)
            chunks.append((lo, len(ids)))
        return ids, chunks

    def load(self, path: Path):
        import torchaudio
        wav, sr = torchaudio.load(str(path))
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        if sr != self.sr:
            wav = torchaudio.functional.resample(wav, sr, self.sr)
        return wav

    def spans(self, wav, roman: list[str]) -> tuple[list[tuple[float, float]], float]:
        """[(start, end)] in seconds per roman unit, and the clip duration."""
        duration = wav.shape[1] / self.sr
        ids, chunks = self.token_ids(roman)
        if not ids:
            return [], duration
        with self.torch.inference_mode():
            emission, _ = self.model(wav.to(self.device))
            token_spans = self.aligner(emission[0], self.torch.tensor(ids))
        ratio = wav.shape[1] / emission.shape[1] / self.sr
        out: list[tuple[float, float]] = []
        for lo, hi in chunks:
            if hi <= lo or hi > len(token_spans):
                out.append((0.0, 0.0))
            else:
                out.append((float(token_spans[lo].start) * ratio,
                            float(token_spans[hi - 1].end) * ratio))
        return out, duration


def _proportional(weights: list[float], duration: float) -> list[tuple[float, float]]:
    total = sum(weights) or 1.0
    out, pos = [], 0.0
    for w in weights:
        step = duration * (w / total)
        out.append((pos, pos + step))
        pos += step
    return out


def _units_for(lang: str, text: str) -> tuple[list[str], list[str]]:
    """(model units, romanisation handed to the aligner)."""
    if lang == "English":
        ws = g2p.words(text)
        return ws, [w.lower() for w in ws]
    us = telugu.units(text)
    return us, [telugu.romanise(u) for u in us]


def _labels_for(lang: str, units: list[str], lexicon) -> list[list[str]]:
    if lang == "English":
        return [g2p.phones_for_word(u, lexicon) for u in units]
    return [[u] for u in units]


def align_utterance(al: _Aligner, lang: str, wav_path: Path, text: str,
                    lexicon) -> list[tuple[float, float, str]]:
    wav = al.load(wav_path)
    duration = wav.shape[1] / al.sr
    units, roman = _units_for(lang, text)
    if not units:
        return [(0.0, duration, SILENCE)]
    labels = _labels_for(lang, units, lexicon)

    try:
        unit_spans, duration = al.spans(wav, roman)
    except Exception:
        unit_spans = []

    if len(unit_spans) != len(units) or not any(e > s for s, e in unit_spans):
        unit_spans = _proportional([max(1, len(r)) for r in roman], duration)

    # gapless, monotonic sequence; a gap between units becomes "sp"
    seq: list[tuple[float, float, str]] = []
    cursor = 0.0
    for (start, end), labs in zip(unit_spans, labels):
        start, end = max(start, cursor), max(end, cursor)
        if not labs or end - start <= 1e-4:
            continue
        if start - cursor > 0.01:
            seq.append((cursor, start, SILENCE))
        step = (end - start) / len(labs)
        for i, lab in enumerate(labs):
            seq.append((start + i * step, start + (i + 1) * step, lab))
        cursor = end
    if duration - cursor > 0.01:
        seq.append((cursor, duration, SILENCE))
    return seq or [(0.0, duration, SILENCE)]


def _escape(text: str) -> str:
    return text.replace('"', '""')


def write_textgrid(segs: list[tuple[float, float, str]], path: Path,
                   min_sec: float = 0.02) -> None:
    """Praat short TextGrid, single IntervalTier named ``phones``."""
    fixed: list[tuple[float, float, str]] = []
    cursor = 0.0
    for start, end, label in segs:
        start = max(float(start), cursor)
        end = max(float(end), start + min_sec)
        fixed.append((start, end, label or SILENCE))
        cursor = end
    if not fixed:
        raise ValueError("no segments to write")

    lines = [
        'File type = "ooTextFile"',
        'Object class = "TextGrid"',
        "",
        "0",
        f"{cursor:.6f}",
        "<exists>",
        "1",
        '"IntervalTier"',
        '"phones"',
        "0",
        f"{cursor:.6f}",
        str(len(fixed)),
    ]
    for start, end, label in fixed:
        lines += [f"{start:.6f}", f"{end:.6f}", f'"{_escape(label)}"']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def split_rows(lang: str, split: str) -> list[tuple[str, str, str]]:
    path = config.lang_dir(lang) / "data" / f"{split}.txt"
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rel, spk, text = line.split("|", 2)
            rows.append((rel, spk, text))
    return rows


def run(lang: str, splits: tuple[str, ...] = ("train", "val"), limit: int = 0) -> dict:
    cfg = config.load(lang)
    audio_dir = config.resolve(cfg["dataset"]["audio_dir"])
    out_root = config.lang_dir(lang) / "alignments"
    lexicon = None
    if lang == "English":
        lexicon = g2p.load_lexicon(
            config.require_repo(lang) / "lexicon" / "librispeech-lexicon.txt")

    al = _Aligner()
    print(f"[align] {lang}: device={al.device} sr={al.sr}", flush=True)
    done = cached = failed = 0
    for split in splits:
        rows = split_rows(lang, split)
        if limit:
            rows = rows[:limit]
        for rel, _spk, text in rows:
            name = Path(rel).stem
            dst = out_root / split / f"{name}.TextGrid"
            if dst.exists():
                cached += 1
                continue
            try:
                write_textgrid(align_utterance(al, lang, audio_dir / rel, text, lexicon), dst)
                done += 1
            except Exception as exc:
                failed += 1
                print(f"[align] {split}/{rel}: {type(exc).__name__}: {exc}", flush=True)
                if failed <= 2:
                    traceback.print_exc()
            if (done + failed) % 200 == 0 and (done + failed):
                print(f"[align] {split}: {done} new, {cached} cached, {failed} failed", flush=True)
    info = {"aligned": done, "cached": cached, "failed": failed, "out": str(out_root)}
    print(info, flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="MMS_FA forced alignment -> TextGrids")
    ap.add_argument("--lang", required=True, choices=list(config.LANGS))
    ap.add_argument("--split", default="all", choices=["train", "val", "all"])
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    splits = ("train", "val") if args.split == "all" else (args.split,)
    run(args.lang, splits, args.limit)
