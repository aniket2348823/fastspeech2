"""Every source edit the reference implementation needs on a modern stack.

All edits are idempotent and are applied to both language clones by
``apply()``.  ``fs2stage.stage`` and the train/infer entry points call
``apply()`` themselves, so a freshly cloned (unpatched) repo can never reach the
upstream scripts.

1. ``compat`` - torch >= 2.6
   ``torch.load`` defaults to ``weights_only=True``; the reference loads plain
   ``{"model": ..., "optimizer": ...}`` dicts and the HiFi-GAN vocoder
   checkpoint, both of which fail under that default.

2. ``stft`` - librosa >= 0.10 / 1.0
   ``librosa.util.pad_center`` made ``size`` keyword-only, so the reference call
   ``pad_center(fft_window, filter_length)`` raises
   ``TypeError: pad_center() takes 1 positional argument but 2 were given``.
   ``librosa.filters.mel`` likewise requires ``sr=`` as a keyword.  Both calls
   are rewritten (the pad becomes plain ``np.pad``, so it is version-proof).

3. ``audio_processing`` - same ``pad_center`` bug inside ``window_sumsquare``.

4. ``preprocessor`` - two crash guards around short utterances
   (``IndexError: index 69 is out of bounds ... size 69``) plus explicit UTF-8
   when reading ``.lab`` files (Telugu).

5. ``symbols`` - Telugu only
   The preprocessor writes metadata text as ``{unit unit ...}`` and
   ``dataset.py`` tokenises it through ``_arpabet_to_sequence``, which prefixes
   every unit with ``"@"``.  Telugu therefore needs ``"@క"``, ``"@ా"``, ...
   appended to ``text/symbols.py``.

6. ``preflight`` - build a ``TacotronSTFT`` and run one mel pass in a child
   process.  This converts any remaining numeric/audio-stack incompatibility
   into a loud, early failure instead of a stack trace halfway through 4300
   utterances.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys

from . import config, telugu

BEGIN = "# --- SSTF PATCH BEGIN (telugu graphemes) ---"
END = "# --- SSTF PATCH END (telugu graphemes) ---"
COMPAT_MARK = "# SSTF: torch>=2.6 compat"

_TORCH_LOAD = [
    ("ckpt = torch.load(ckpt_path)", "ckpt = torch.load(ckpt_path, weights_only=False)"),
    (
        'torch.load("hifigan/generator_LJSpeech.pth.tar")',
        'torch.load("hifigan/generator_LJSpeech.pth.tar", weights_only=False)',
    ),
    (
        'torch.load("hifigan/generator_universal.pth.tar")',
        'torch.load("hifigan/generator_universal.pth.tar", weights_only=False)',
    ),
]


def _pad_center_call(target: str, size: str, indent: str) -> str:
    """Center-pad ``target`` to ``size`` samples without touching librosa."""
    return (
        f"{indent}{target} = np.pad(\n"
        f"{indent}    {target},\n"
        f"{indent}    (\n"
        f"{indent}        ({size} - len({target})) // 2,\n"
        f"{indent}        {size} - len({target}) - ({size} - len({target})) // 2,\n"
        f"{indent}    ),\n"
        f"{indent})"
    )


# --------------------------------------------------------------------------- #
# 1. torch.load compatibility
# --------------------------------------------------------------------------- #
def compat(lang: str) -> dict:
    """Make ``utils/model.py`` load checkpoints on torch >= 2.6."""
    repo = config.require_repo(lang)
    model_py = repo / "utils" / "model.py"
    src = model_py.read_text(encoding="utf-8")
    changed = 0
    for old, new in _TORCH_LOAD:
        if new in src:
            continue
        if old in src:
            src = src.replace(old, new)
            changed += 1
    # A checkpoint written on a GPU box cannot be read on a CPU-only one, and
    # ``get_vocoder`` runs at the very start of training, so without
    # map_location the job dies before step 1 whenever CUDA is missing.
    src, n_map = re.subn(
        r"torch\.load\((?![^()]*map_location)(?P<arg>[^()]*?),\s*weights_only=False\s*\)",
        lambda m: f"torch.load({m.group('arg')}, map_location=device, weights_only=False)",
        src,
    )
    changed += n_map
    if changed and COMPAT_MARK not in src:
        src = src.rstrip("\n") + f"\n\n{COMPAT_MARK}: weights_only=False on torch.load\n"
    model_py.write_text(src, encoding="utf-8")
    return {"file": str(model_py), "patched_calls": changed}


# --------------------------------------------------------------------------- #
# 2. librosa pad_center / mel keyword arguments
# --------------------------------------------------------------------------- #
_PAD_STFT = re.compile(
    r"(?m)^(?P<ind>[ \t]*)fft_window = pad_center\(fft_window, *filter_length\)[ \t]*$"
)
_MEL_CALL = re.compile(r"(?s)(?P<ind>[ \t]*)mel_basis = librosa_mel_fn\((?P<args>.*?)\)")

# The reference hard-codes ``.cuda()`` inside the mel transform, so feature
# extraction cannot run on a CPU-only box (and device placement is fragile on
# multi-GPU).  Make every tensor follow the module's own device.
_DEVICE_PAIRS = [
    ("input_data.cuda(),", "input_data.to(self.forward_basis.device),"),
    (
        "torch.autograd.Variable(self.forward_basis, requires_grad=False).cuda(),",
        "self.forward_basis.to(input_data.device),",
    ),
    (
        "torch.autograd.Variable(self.inverse_basis, requires_grad=False),",
        "self.inverse_basis.to(recombine_magnitude_phase.device),",
    ),
    (
        "window_sum = window_sum.cuda() if magnitude.is_cuda else window_sum",
        "window_sum = window_sum.to(magnitude.device)",
    ),
]



def stft(lang: str) -> dict:
    """Make ``audio/stft.py`` work on librosa >= 0.10 (incl. 1.x)."""
    repo = config.require_repo(lang)
    stft_py = repo / "audio" / "stft.py"
    src = stft_py.read_text(encoding="utf-8")
    changes = []

    m = _PAD_STFT.search(src)
    if m:
        src = src[: m.start()] + _pad_center_call("fft_window", "filter_length", m.group("ind")) + src[m.end():]
        changes.append("pad_center->np.pad")

    m = _MEL_CALL.search(src)
    if m and "sr=" not in m.group("args"):
        ind = m.group("ind")
        new = (
            f"{ind}mel_basis = librosa_mel_fn(\n"
            f"{ind}    sr=sampling_rate,\n"
            f"{ind}    n_fft=filter_length,\n"
            f"{ind}    n_mels=n_mel_channels,\n"
            f"{ind}    fmin=mel_fmin,\n"
            f"{ind}    fmax=mel_fmax,\n"
            f"{ind})"
        )
        src = src[: m.start()] + new + src[m.end():]
        changes.append("librosa_mel_fn keywords")

    for old, new in _DEVICE_PAIRS:
        if new in src:
            continue
        if old in src:
            src = src.replace(old, new)
            changes.append(f"device: {old.strip()[:28]}")

    if changes:
        stft_py.write_text(src, encoding="utf-8")
    _compile_check(stft_py)

    # hard guarantee: the upstream call that broke Kaggle must be gone
    after = stft_py.read_text(encoding="utf-8")
    if "pad_center(fft_window" in after:
        raise SystemExit(
            f"{stft_py} still calls pad_center positionally - the patch did not apply.\n"
            "Check that the file is writable and not re-cloned after this step.")
    return {"file": str(stft_py), "changes": changes}


# --------------------------------------------------------------------------- #
# 3. same pad_center bug in window_sumsquare
# --------------------------------------------------------------------------- #
_PAD_WINSUM = re.compile(
    r"(?m)^(?P<ind>[ \t]*)win_sq = librosa_util\.pad_center\(win_sq, *n_fft\)[ \t]*$"
)


def audio_processing(lang: str) -> dict:
    repo = config.require_repo(lang)
    path = repo / "audio" / "audio_processing.py"
    src = path.read_text(encoding="utf-8")
    m = _PAD_WINSUM.search(src)
    changed = False
    if m:
        src = src[: m.start()] + _pad_center_call("win_sq", "n_fft", m.group("ind")) + src[m.end():]
        path.write_text(src, encoding="utf-8")
        changed = True
    _compile_check(path)
    return {"file": str(path), "patched": changed}


# --------------------------------------------------------------------------- #
# 4b. Tensor dtypes: pitch/energy .npy come off disk as float64
# --------------------------------------------------------------------------- #
# ``utils/tools.py`` casts pits but not energies to float32, so the energy
# target stays double and ``total_loss.backward()`` dies with
# ``RuntimeError: Found dtype Double but expected Float``.
_TENSOR_CASTS = [
    (
        "energies = torch.from_numpy(energies).to(device)",
        "energies = torch.from_numpy(energies).float().to(device)",
    ),
]


def tensors(lang: str) -> dict:
    """Force float32 targets in ``utils/tools.py`` (fixes the Double/Float crash)."""
    repo = config.require_repo(lang)
    path = repo / "utils" / "tools.py"
    src = path.read_text(encoding="utf-8")
    changes = []
    for old, new in _TENSOR_CASTS:
        if new in src:
            continue
        if old in src:
            src = src.replace(old, new)
            changes.append("energies -> float32")
    if changes:
        path.write_text(src, encoding="utf-8")
    _compile_check(path)
    return {"file": str(path), "changes": changes}


# --------------------------------------------------------------------------- #
# 4c. dataset.py: the .npy features are float64 except mel
# --------------------------------------------------------------------------- #
_DATASET_CASTS = [
    ("mel = np.load(mel_path)", "mel = np.load(mel_path).astype(np.float32)"),
    ("pitch = np.load(pitch_path)", "pitch = np.load(pitch_path).astype(np.float32)"),
    ("energy = np.load(energy_path)", "energy = np.load(energy_path).astype(np.float32)"),
]


def dataset_dtypes(lang: str) -> dict:
    """Load pitch/energy/mel as float32 so no double reaches the loss."""
    repo = config.require_repo(lang)
    path = repo / "dataset.py"
    src = path.read_text(encoding="utf-8")
    changes = []
    for old, new in _DATASET_CASTS:
        if new in src:
            continue
        if old in src:
            src = src.replace(old, new)
            changes.append(new.split(".")[0])
    if changes:
        path.write_text(src, encoding="utf-8")
    _compile_check(path)
    return {"file": str(path), "changes": changes}


# --------------------------------------------------------------------------- #
# 4. preprocessor guards (short utterances) + UTF-8 labels
# --------------------------------------------------------------------------- #
def _phoneme_guard(src: str, var: str) -> tuple[str, bool]:
    if f"if i >= len({var})" in src:
        return src, False
    old = (
        "            for i, d in enumerate(duration):\n"
        "                if d > 0:\n"
        f"                    {var}[i] = np.mean({var}[pos : pos + d])"
    )
    new = (
        "            for i, d in enumerate(duration):\n"
        f"                if i >= len({var}):\n"
        "                    break\n"
        "                if d > 0:\n"
        f"                    end = min(pos + d, len({var}))\n"
        "                    if end > pos:\n"
        f"                        {var}[i] = np.mean({var}[pos:end])\n"
        "                    else:\n"
        f"                        {var}[i] = 0"
    )
    if old not in src:
        return src, False
    return src.replace(old, new, 1), True


def preprocessor(lang: str) -> dict:
    """Stop the upstream preprocessor dying on short / mismatched utterances."""
    repo = config.require_repo(lang)
    path = repo / "preprocessor" / "preprocessor.py"
    src = path.read_text(encoding="utf-8")
    changes = []

    old_open = 'with open(text_path, "r") as f:'
    if 'with open(text_path, "r", encoding="utf-8") as f:' not in src and old_open in src:
        src = src.replace(old_open, 'with open(text_path, "r", encoding="utf-8") as f:', 1)
        changes.append("utf-8 label read")

    for var in ("pitch", "energy"):
        src, done = _phoneme_guard(src, var)
        if done:
            changes.append(f"{var} bounds guard")

    # A phone can end up with zero frames (very short clip, or two phones on the
    # same alignment boundary).  ``x[: len(duration)]`` then yields a target
    # vector shorter than the phoneme sequence and training dies with
    # "The size of tensor a (98) must match the size of tensor b (69)".
    for var in ("pitch", "energy"):
        old = f"            {var} = {var}[: len(duration)]"
        new = (
            f"            if len({var}) < len(duration):\n"
            f"                {var} = np.pad({var}, (0, len(duration) - len({var})))\n"
            f"            {var} = {var}[: len(duration)]"
        )
        if new not in src and old in src:
            src = src.replace(old, new, 1)
            changes.append(f"{var} length pad")

    # Frames and phones must describe the same audio: sum(duration) is the mel
    # length the length regulator will produce.
    guard_mark = "# SSTF: phones and frames must describe the same audio"
    if guard_mark not in src:
        anchor = "        # Save files\n        dur_filename"
        if anchor in src:
            src = src.replace(
                anchor,
                "        " + guard_mark + "\n"
                "        if len(duration) == 0 or int(np.sum(duration)) != mel_spectrogram.shape[1]:\n"
                "            return None\n\n"
                + anchor,
                1,
            )
            changes.append("frame/phone count guard")

    # save float32 so the training loader never sees float64 targets
    saves = [
        ('np.save(os.path.join(self.out_dir, "pitch", pitch_filename), pitch)',
         'np.save(os.path.join(self.out_dir, "pitch", pitch_filename), '
         'np.asarray(pitch, dtype=np.float32))'),
        ('np.save(os.path.join(self.out_dir, "energy", energy_filename), energy)',
         'np.save(os.path.join(self.out_dir, "energy", energy_filename), '
         'np.asarray(energy, dtype=np.float32))'),
    ]
    for old, new in saves:
        if new in src:
            continue
        if old in src:
            src = src.replace(old, new)
            changes.append("float32 features")

    if changes:
        path.write_text(src, encoding="utf-8")
    _compile_check(path)
    return {"file": str(path), "changes": changes}


def _compile_check(path) -> None:
    out = subprocess.run(
        [sys.executable, "-m", "py_compile", str(path)], capture_output=True, text=True
    )
    if out.returncode != 0:
        raise SystemExit(f"patched file does not compile: {path}\n{out.stderr[-1500:]}")


# --------------------------------------------------------------------------- #
# 5. Telugu grapheme symbols
# --------------------------------------------------------------------------- #
def symbols(lang: str = "Telugu") -> dict:
    repo = config.require_repo(lang)
    symbols_py = repo / "text" / "symbols.py"
    vocab_path = config.lang_dir(lang) / "data" / "vocab.json"
    if not vocab_path.exists():
        raise SystemExit(
            f"missing {vocab_path}; run:\n"
            f"  python project/{lang}/preprocess.py --stage vocab")
    vocab = [u for u in telugu.load_vocab(vocab_path) if u not in (telugu.PAD, telugu.EOS)]

    src = symbols_py.read_text(encoding="utf-8")
    if BEGIN in src:
        head, rest = src.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
        src = head + tail.lstrip("\n")

    block = (
        f"{BEGIN}\n"
        f"_telugu_units = {json.dumps(vocab, ensure_ascii=False)}\n"
        "_telugu_symbols = [\"@\" + u for u in _telugu_units]\n"
        "symbols = symbols + [s for s in _telugu_symbols if s not in symbols]\n"
        f"{END}\n"
    )
    symbols_py.write_text(src.rstrip("\n") + "\n\n" + block, encoding="utf-8")

    code = (
        "import sys; sys.path.insert(0, %r);"
        "from text.symbols import symbols;"
        "print(len(symbols));"
        "print(len([s for s in symbols if s.startswith('@')]))" % str(repo)
    )
    out = subprocess.run([sys.executable, "-c", code], cwd=repo,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit("symbol patch does not import:\n" + out.stderr[-1500:])
    total, n_at = (int(x) for x in out.stdout.split()[:2])
    return {"file": str(symbols_py), "n_symbols": total,
            "n_at_symbols": n_at, "telugu_units": len(vocab)}


# --------------------------------------------------------------------------- #
# 6. preflight: prove the patched audio stack actually runs here
# --------------------------------------------------------------------------- #
_PREFLIGHT = r"""
import torch
from audio.stft import TacotronSTFT

stft = TacotronSTFT(1024, 256, 1024, 80, 22050, 0.0, 8000.0)
y = (torch.rand(1, 16000, dtype=torch.float32) - 0.5) * 1.8
mel, energy = stft.mel_spectrogram(y)
assert tuple(mel.shape)[1] == 80, mel.shape
rec = stft.stft_fn.inverse(*stft.stft_fn.transform(y))
assert rec.shape[-1] > 0, rec.shape
print("ok", tuple(mel.shape), tuple(energy.shape), tuple(rec.shape))
"""


def preflight(lang: str) -> dict:
    """Instantiate the patched STFT and run one mel pass in a child process."""
    repo = config.require_repo(lang)
    out = subprocess.run([sys.executable, "-c", _PREFLIGHT], cwd=repo,
                         capture_output=True, text=True)
    if out.returncode != 0:
        raise SystemExit(
            "audio stack still incompatible after patching:\n"
            + (out.stderr or out.stdout)[-2000:]
        )
    return {"file": "audio/stft.py", "check": out.stdout.strip()}


# --------------------------------------------------------------------------- #
def apply(lang: str, verbose: bool = False) -> dict:
    """Apply every patch this language needs, then verify the result runs."""
    from . import __version__

    info = {
        "version": __version__,
        "compat": compat(lang),
        "stft": stft(lang),
        "audio_processing": audio_processing(lang),
        "preprocessor": preprocessor(lang),
        "dataset": dataset_dtypes(lang),
        "tensors": tensors(lang),
    }
    if lang == "Telugu":
        info["symbols"] = symbols(lang)
    info["preflight"] = preflight(lang)
    if verbose:
        print(json.dumps(info, indent=2, ensure_ascii=False), flush=True)
    return info


def main() -> None:
    ap = argparse.ArgumentParser(description="apply the documented source patches")
    ap.add_argument("--lang", required=True, choices=list(config.LANGS))
    args = ap.parse_args()
    apply(args.lang, verbose=True)


if __name__ == "__main__":
    main()
