"""English text -> ARPAbet phones (the same front-end for training and inference).

The reference `synthesize.py` looks words up in `lexicon/librispeech-lexicon.txt`
and falls back to `g2p_en`.  Training and inference must agree on the phone
string, otherwise the acoustic model is asked to pronounce symbols it never saw,
so both `common/align.py` and `common/inferrun.py` call this module and nothing
else.

Three fallbacks, in order, each of which can fail without breaking the run:
    lexicon -> g2p_en -> letter-name spelling -> "sp"
"""
from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

# ARPAbet phones the repo's symbol table knows: one or two capitals + stress digit
_PHONE_RE = re.compile(r"^[A-Z]{1,2}[0-2]?$")
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z'\-]*")
_SILENCE = "sp"

# last-resort pronunciation when neither the lexicon nor g2p_en knows the word
_LETTERS = {
    "a": "AH", "b": "B", "c": "K", "d": "D", "e": "EH", "f": "F", "g": "G",
    "h": "HH", "i": "IH", "j": "JH", "k": "K", "l": "L", "m": "M", "n": "N",
    "o": "OW", "p": "P", "q": "K", "r": "R", "s": "S", "t": "T", "u": "UH",
    "v": "V", "w": "W", "x": "K", "y": "Y", "z": "Z",
}

_G2P = None
_LEXICONS: dict[str, dict[str, list[str]]] = {}


def is_phone(token: str) -> bool:
    return bool(_PHONE_RE.match(token))


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def load_lexicon(path: str | Path) -> dict[str, list[str]]:
    key = str(path)
    if key not in _LEXICONS:
        lex: dict[str, list[str]] = {}
        p = Path(path)
        if p.exists():
            with open(p, encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    parts = re.split(r"\s+", line.strip())
                    if len(parts) > 1 and parts[0].lower() not in lex:
                        phones = [q for q in parts[1:] if is_phone(q)]
                        if phones:
                            lex[parts[0].lower()] = phones
        _LEXICONS[key] = lex
    return _LEXICONS[key]


def _g2p():
    """Import g2p_en lazily: it downloads NLTK data on first use."""
    global _G2P
    if _G2P is None:
        from g2p_en import G2p
        _G2P = G2p()
    return _G2P


@lru_cache(maxsize=200_000)
def _g2p_tuple(word: str) -> tuple[str, ...]:
    try:
        return tuple(p for p in _g2p()(word) if is_phone(p))
    except Exception:
        return ()


def phones_for_word(word: str, lexicon: dict[str, list[str]] | None = None) -> list[str]:
    key = re.sub(r"[^A-Za-z'\-]", "", word).lower()
    if not key:
        return [_SILENCE]
    if lexicon and key in lexicon:
        return list(lexicon[key])
    phones = list(_g2p_tuple(key))
    if phones:
        return phones
    spelled = [_LETTERS[ch] for ch in key if ch in _LETTERS]
    return spelled or [_SILENCE]


def phone_list(text: str, lexicon: dict[str, list[str]] | None = None) -> list[str]:
    out: list[str] = []
    for word in words(text):
        out.extend(phones_for_word(word, lexicon))
    return out


def to_repo_text(text: str, lexicon: dict[str, list[str]] | None = None) -> str:
    """The `{HH AH0 L OW1}` form the upstream dataset tokeniser expects."""
    return "{" + " ".join(phone_list(text, lexicon)) + "}"
