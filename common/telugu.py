"""Telugu front-end: graphemes, romanisation, numbers and the symbol inventory.

Why graphemes and not phonemes
-----------------------------
There is no usable Telugu G2P (no MFA acoustic model, no pronunciation
dictionary, espeak's Telugu is unreliable).  Telugu script, however, is close to
phonemic: a base character plus its vowel sign *is* one syllable-sized unit.
So the unit inventory is the set of grapheme clusters (U+0C00..U+0C7F), and the
model learns the mapping from cluster to acoustics directly.  That removes G2P
from the pipeline entirely and is the main "modification for Telugu support"
the assignment asks to document.

Romanisation exists only so torchaudio's MMS_FA aligner (a Latin-character CTC
model) can be used for forced alignment.  It never reaches the model.
"""
from __future__ import annotations

import json
import unicodedata
from pathlib import Path

TELUGU_START, TELUGU_END = 0x0C00, 0x0C7F
PUNCT = ".,?!-"
PAD, EOS = "_", "~"

# --- romanisation -----------------------------------------------------------
_VOWELS = {
    "అ": "a", "ఆ": "aa", "ఇ": "i", "ఈ": "ii", "ఉ": "u", "ఊ": "uu",
    "ఋ": "ru", "ౠ": "ruu", "ఌ": "lu", "ౡ": "luu",
    "ఎ": "e", "ఏ": "ee", "ఐ": "ai", "ఒ": "o", "ఓ": "oo", "ఔ": "au",
}
_CONSONANTS = {
    "క": "k", "ఖ": "kh", "గ": "g", "ఘ": "gh", "ఙ": "ng",
    "చ": "ch", "ఛ": "chh", "జ": "j", "ఝ": "jh", "ఞ": "ny",
    "ట": "t", "ఠ": "th", "డ": "d", "ఢ": "dh", "ణ": "n",
    "త": "t", "థ": "th", "ద": "d", "ధ": "dh", "న": "n",
    "ప": "p", "ఫ": "ph", "బ": "b", "భ": "bh", "మ": "m",
    "య": "y", "ర": "r", "ఱ": "r", "ల": "l", "ళ": "l", "ఴ": "l",
    "వ": "v", "శ": "sh", "ష": "sh", "స": "s", "హ": "h",
    "ౘ": "ts", "ౙ": "dz", "ౚ": "r",
}
_MATRAS = {
    "ా": "aa", "ి": "i", "ీ": "ii", "ు": "u", "ూ": "uu", "ృ": "ru",
    "ౄ": "ruu", "ె": "e", "ే": "ee", "ై": "ai", "ొ": "o", "ో": "oo", "ౌ": "au",
}
_SIGNS = {"ం": "n", "ః": "h", "ఁ": "n", "్": "", "ఽ": "", "ౕ": "", "ౖ": ""}

_TE_DIGITS = {chr(0x0C66 + d): str(d) for d in range(10)}


def is_telugu(ch: str) -> bool:
    return TELUGU_START <= ord(ch) <= TELUGU_END


def graphemes(text: str) -> list[str]:
    """Cluster each base character with the combining marks that follow it.

    Telugu vowel signs are category ``Mc`` (spacing combining marks) and the
    virama is ``Mn``; a plain ``list(text)`` would tear ``కా`` apart.
    """
    out: list[str] = []
    for ch in unicodedata.normalize("NFC", text):
        if out and unicodedata.category(ch) in ("Mn", "Mc") and out[-1].strip():
            out[-1] += ch
        else:
            out.append(ch)
    return out


def romanise(unit: str) -> str:
    """ASCII-only ISO-15919-flavoured romanisation, used by the aligner."""
    if not unit:
        return ""
    chars = list(unit)
    if len(chars) == 1 and chars[0] in _CONSONANTS:
        return _CONSONANTS[chars[0]] + "a"          # inherent vowel
    out = []
    for ch in chars:
        if ch in _CONSONANTS:
            out.append(_CONSONANTS[ch])
        elif ch in _VOWELS:
            out.append(_VOWELS[ch])
        elif ch in _MATRAS:
            out.append(_MATRAS[ch])
        elif ch in _SIGNS:
            out.append(_SIGNS[ch])
        elif ch in _TE_DIGITS:
            out.append(_TE_DIGITS[ch])
        elif is_telugu(ch):
            out.append("")                           # unknown mark: ignore
        else:
            out.append(ch.lower() if ch.isascii() else "")
    return "".join(out)


# --- number spelling --------------------------------------------------------
_UNITS = ["సున్నా", "ఒకటి", "రెండు", "మూడు", "నాలుగు", "ఐదు", "ఆరు",
          "ఏడు", "ఎనిమిది", "తొమ్మిది", "పది", "పదకొండు", "పన్నెండు",
          "పదమూడు", "పద్నాలుగు", "పదిహేను", "పదహారు", "పదిహేడు",
          "పద్దెనిమిది", "పంతొమ్మిది"]
_TENS = {2: "ఇరవై", 3: "ముప్పై", 4: "నలభై", 5: "యాభై",
         6: "అరవై", 7: "డెబ్బై", 8: "ఎనభై", 9: "తొంభై"}


def _below_100(n: int) -> str:
    if n < 20:
        return _UNITS[n]
    t, u = divmod(n, 10)
    return _TENS[t] + ((" " + _UNITS[u]) if u else "")


def _below_1000(n: int) -> str:
    if n < 100:
        return _below_100(n)
    h, rest = divmod(n, 100)
    if rest == 0:
        head = "వంద" if h == 1 else _UNITS[h] + " వందలు"
        return head
    head = ("వంద" if h == 1 else _UNITS[h] + " వందల")
    return head + " " + _below_100(rest)


def number_to_words(n: int) -> str:
    """0..99,99,999 -> Telugu words ("2026" -> "రెండు వేల ఇరవై ఆరు")."""
    if n < 0:
        return "మైనస్ " + number_to_words(-n)
    if n < 1000:
        return _below_1000(n)
    if n < 100000:
        k, rest = divmod(n, 1000)
        if rest == 0:
            head = "వెయ్యి" if k == 1 else _below_1000(k) + " వేలు"
            return head
        head = ("వెయ్యి" if k == 1 else _below_1000(k) + " వేల")
        return head + " " + _below_1000(rest)
    return " ".join(number_to_words(int(d)) for d in str(n))


def ordinal_to_words(n: int) -> str:
    word = number_to_words(n)
    if word == "ఒకటి":
        return "మొదటి"
    return word[:-1] + "వ" if word.endswith("ు") else word + "వ"


# --- cleaning + vocabulary --------------------------------------------------
def clean(text: str) -> str:
    """Keep Telugu, spaces and basic punctuation; collapse whitespace."""
    t = unicodedata.normalize("NFC", text)
    for zw in ("\u200c", "\u200d", "\ufeff"):
        t = t.replace(zw, "")
    kept = [ch for ch in t if is_telugu(ch) or ch.isspace() or ch in PUNCT]
    return " ".join("".join(kept).split())


def units(text: str) -> list[str]:
    """Model input units for already-normalised Telugu text."""
    return [u for u in graphemes(clean(text)) if not u.isspace()]


def build_vocab(texts: list[str]) -> list[str]:
    """Every Telugu codepoint plus the clusters that actually occur.

    Including the whole block guarantees zero OOV at inference time for any
    Telugu sentence, which matters because the test sentences are fixed but the
    vocabulary is built from the training split only.
    """
    clusters: set[str] = set()
    for text in texts:
        clusters.update(units(text))
    base = [chr(c) for c in range(TELUGU_START, TELUGU_END + 1)
            if unicodedata.category(chr(c))[0] in ("L", "M", "P", "S")]
    extra = sorted(c for c in clusters if len(c) > 1)
    return [PAD, EOS] + base + extra + list(PUNCT)


def save_vocab(vocab: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(vocab, ensure_ascii=False, indent=1), encoding="utf-8")


def load_vocab(path: Path) -> list[str]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_repo_text(text: str) -> str:
    """The `{unit unit ...}` form the upstream preprocessor/dataset speaks."""
    return "{" + " ".join(units(text)) + "}"
