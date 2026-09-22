"""Regex sentence splitter with abbreviation, initial and decimal guards, plus a prose
filter that keeps carousel headlines, timestamps and label lines out of cited passages."""

import re

_ABBREVIATIONS = (
    "Inc",
    "Corp",
    "Co",
    "Ltd",
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Jr",
    "Sr",
    "St",
    "vs",
    "approx",
    "No",
    "Jan",
    "Feb",
    "Mar",
    "Apr",
    "Jun",
    "Jul",
    "Aug",
    "Sep",
    "Sept",
    "Oct",
    "Nov",
    "Dec",
    "U.S",
    "U.K",
    "e.g",
    "i.e",
)
_DOT = "\u0001"  # placeholder for a protected full stop
_ABBR_RE = re.compile(r"\b(" + "|".join(re.escape(a) for a in _ABBREVIATIONS) + r")\.")
_INITIAL_RE = re.compile(r"\b([A-Z])\.(?=\s[A-Z])")
_DECIMAL_RE = re.compile(r"(?<=\d)\.(?=\d)")
_SPLIT_RE = re.compile(r"(?:(?<=[.!?])|(?<=[.!?][\"']))\s+(?=[\"(\[$A-Z0-9])")
_TIME_RE = re.compile(r"\b\d{1,2}:\d{2}\b")


def split_sentences(text: str) -> list[str]:
    protected = _ABBR_RE.sub(lambda m: m.group(1).replace(".", _DOT) + _DOT, text)
    protected = _INITIAL_RE.sub(lambda m: m.group(1) + _DOT, protected)
    protected = _DECIMAL_RE.sub(_DOT, protected)
    return [p.replace(_DOT, ".").strip() for p in _SPLIT_RE.split(protected) if p.strip()]


def is_prose(sentence: str, min_words: int = 8, max_words: int = 80) -> bool:
    words = sentence.split()
    if not min_words <= len(words) <= max_words:
        return False
    if "•" in sentence or _TIME_RE.search(sentence) or sentence.endswith(":"):
        return False
    alpha = [w for w in words if w[0].isalpha()]
    if not alpha:
        return False
    # Headlines and nav text are Title Case; prose is not.
    return sum(w[0].isupper() for w in alpha) / len(alpha) < 0.6
