"""Text normalisation front-end for the assignment test sentences.

The test set is deliberately adversarial, so normalisation is not optional:

  E1  "antidisestablishmentarianism"    -> very long OOV word
  E2  "conduct" (noun vs verb)          -> heteronym, same graphemes
  E3  "rhyth  m"                        -> broken spacing / typo
  E4  "andaroo", "10:45 AM", "12 November 2026" -> code-switch + numerals
  T4  "2026 నవంబర్ 12వ తేదీ", "10:45"    -> Telugu numerals/time

Every function is pure: text in, text out. The original sentence is kept
alongside in results/ so the report can quote the input verbatim.
"""
from __future__ import annotations

import re
import unicodedata

from num2words import num2words

from . import telugu

MONTHS_EN = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]
_MONTH_ALT = "|".join(MONTHS_EN)

_TE_DIGITS = {
    "0": "సున్నా", "1": "ఒకటి", "2": "రెండు", "3": "మూడు", "4": "నాలుగు",
    "5": "ఐదు", "6": "ఆరు", "7": "ఏడు", "8": "ఎనిమిది", "9": "తొమ్మిది",
}


def _num(value, lang: str) -> str:
    try:
        return num2words(int(value), lang=lang)
    except Exception:
        return str(value)


def _year_en(year: int) -> str:
    """2026 -> 'twenty twenty six' (how a year is actually read aloud)."""
    if 1100 <= year <= 2099:
        hi, lo = divmod(year, 100)
        if lo == 0:
            return f"{_num(hi, 'en')} hundred"
        lo_words = _num(lo, "en") if lo >= 10 else f"oh {_num(lo, 'en')}"
        return f"{_num(hi, 'en')} {lo_words}"
    return _num(year, "en")


def _digits_one_by_one(digits: str, lang: str) -> str:
    """Spell a digit string (used for digit-by-digit numbers such as 10:45)."""
    if lang == "te":
        return " ".join(_TE_DIGITS.get(d, d) for d in digits)
    words = ["zero", "one", "two", "three", "four", "five", "six", "seven",
             "eight", "nine"]
    return " ".join(words[int(d)] if d.isdigit() else d for d in digits)


# ---------------------------------------------------------------- English ----
def normalize_english(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    # 1,200 -> 1200 so the comma is not treated as punctuation
    t = re.sub(r"(?<=\d),(?=\d{3}(\D|$))", "", t)

    # times: 10:45 AM / 10.45 am / 10:45
    def _time(m: re.Match) -> str:
        hh, mm, mer = m.group(1), m.group(2), (m.group(3) or "").upper()
        if mm == "00":
            out = _num(int(hh), "en") + ("" if mer else " o'clock")
        else:
            out = f"{_num(int(hh), 'en')} {_num(int(mm), 'en')}"
        if mer:
            out += " " + " ".join(mer + "M")        # "A" -> "A M"
        return out

    t = re.sub(r"\b(\d{1,2})[:.](\d{2})\s*(?:([AaPp])\.?\s*[Mm]\.?)?", _time, t)

    # dates: 12 November 2026 / November 12, 2026 / 15 August
    def _date_dmy(m: re.Match) -> str:
        d, mon, y = int(m.group(1)), m.group(2), int(m.group(3))
        return f"the {num2words(d, lang='en', to='ordinal')} of {mon} {_year_en(y)}"

    def _date_mdy(m: re.Match) -> str:
        mon, d, y = m.group(1), int(m.group(2)), int(m.group(3))
        return f"{mon} {num2words(d, lang='en', to='ordinal')} {_year_en(y)}"

    def _date_dm(m: re.Match) -> str:
        d, mon = int(m.group(1)), m.group(2)
        return f"the {num2words(d, lang='en', to='ordinal')} of {mon}"

    t = re.sub(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\s+(\d{{4}})\b", _date_dmy, t)
    t = re.sub(rf"\b({_MONTH_ALT})\s+(\d{{1,2}}),?\s+(\d{{4}})\b", _date_mdy, t)
    t = re.sub(rf"\b(\d{{1,2}})\s+({_MONTH_ALT})\b", _date_dm, t)

    # ordinals, decimals, year-style 4-digit numbers, remaining integers
    t = re.sub(r"\b(\d+)(?:st|nd|rd|th)\b",
               lambda m: num2words(int(m.group(1)), lang="en", to="ordinal"), t)
    t = re.sub(r"\b(\d+)\.(\d+)\b",
               lambda m: f"{_num(m.group(1), 'en')} point {_digits_one_by_one(m.group(2), 'en')}", t)
    t = re.sub(r"\b(\d{4})\b", lambda m: _year_en(int(m.group(1))), t)
    t = re.sub(r"\b(\d+)\b", lambda m: _num(m.group(1), "en"), t)

    # symbols that the phonemizer chokes on
    for src, dst in (("&", " and "), ("%", " percent "), ("+", " plus "),
                     ("=", " equals "), ("@", " at "), ("#", " number ")):
        t = t.replace(src, dst)
    t = re.sub(r"[^\w\s',.\-?;:!]", " ", t)
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return re.sub(r"\s+", " ", t).strip()


# ----------------------------------------------------------------- Telugu ----
def _te_year(year: int) -> str:
    return telugu.number_to_words(year)


def normalize_telugu(text: str) -> str:
    t = unicodedata.normalize("NFC", text)
    # zero-width joiners/non-joiners confuse both the aligner and the vocab
    t = t.replace("\u200c", "").replace("\u200d", "").replace("\ufeff", "")
    t = t.replace("।", ".").replace("॥", ".")

    # decimals before anything else: 3.5 -> మూడు పాయింట్ ఐదు
    t = re.sub(r"\b(\d+)\.(\d+)\b",
               lambda m: f"{telugu.number_to_words(int(m.group(1)))} పాయింట్ "
                         f"{telugu.number_to_words(int(m.group(2)))}", t)

    # 10:45 -> పది నలభై ఐదు  (the surrounding "గంటలకు" is kept)
    t = re.sub(r"\b(\d{1,2}):(\d{2})\b",
               lambda m: f"{telugu.number_to_words(int(m.group(1)))} "
                         f"{telugu.number_to_words(int(m.group(2)))}", t)

    # 12వ -> పన్నెండవ
    t = re.sub(r"\b(\d+)వ\b",
               lambda m: telugu.ordinal_to_words(int(m.group(1))), t)

    # "2026 నవంబర్" reads as a year, other 4-digit runs are also years
    t = re.sub(r"\b(\d{4})\b", lambda m: _te_year(int(m.group(1))), t)
    t = re.sub(r"\b(\d+)\b", lambda m: telugu.number_to_words(int(m.group(1))), t)

    t = unicodedata.normalize("NFC", t)
    t = re.sub(r"[^\u0C00-\u0C7F\s.,?!\-]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# ------------------------------------------------------------- dispatcher ----
_CLEANERS = {"en": normalize_english, "te": normalize_telugu}
_ALIASES = {"English": "en", "Telugu": "te", "en": "en", "te": "te"}


def normalize(text: str, lang: str) -> str:
    """`lang` accepts either the language name ('English') or its code ('en')."""
    return _CLEANERS[_ALIASES.get(lang, lang)](text)


if __name__ == "__main__":  # quick self-test against the assignment sentences
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
    samples = {
        "en": [
            "The meeting will start at 10:45 AM on 12 November 2026, so andaroo should arrive on time.",
            "The rhyth  m of the orchestra changed unexpectedly.",
            "The antidisestablishmentarianism movement gained considerable momentum during the parliamentary proceedings.",
        ],
        "te": [
            "ఈ సమావేశం 2026 నవంబర్ 12వ తేదీ ఉదయం 10:45 గంటలకు ప్రారంభమవుతుంది, సమయానికి సిద్ధంగా ఉంచండి.",
        ],
    }
    for lang, rows in samples.items():
        for row in rows:
            print(f"[{lang}] {normalize(row, lang)}")
