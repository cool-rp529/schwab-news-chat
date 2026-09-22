"""Company/product names → tickers. Whole-word, case-insensitive. The ticker key in the
corpus is unreliable, so this table is how a question about "the iPhone" reaches AAPL."""

import re

from newschat.text.tokenize import normalize_text, tokenize

ALIASES: dict[str, tuple[str, ...]] = {
    "AAPL": (
        "apple",
        "iphone",
        "ipad",
        "macbook",
        "apple tv",
        "tim cook",
        "cupertino",
        # Apple's primary contract manufacturer; the corpus files its India-assembly
        # article under AAPL even though the ticker key alone never mentions Apple.
        "foxconn",
    ),
    "AMZN": ("amazon", "aws", "amazon web services", "andy jassy", "prime video", "whole foods"),
    "IBM": ("ibm", "international business machines", "watsonx", "arvind krishna", "red hat"),
    "INTC": ("intel", "intel foundry", "pat gelsinger", "lip-bu tan"),
    "MSFT": ("microsoft", "azure", "satya nadella", "openai", "copilot", "xbox", "linkedin"),
    "NFLX": ("netflix", "ted sarandos"),
    "NVDA": ("nvidia", "jensen huang", "geforce", "blackwell", "cuda"),
}

_ALIAS_PATTERNS: dict[str, re.Pattern[str]] = {
    ticker: re.compile(r"\b(?:" + "|".join(re.escape(n) for n in names) + r")\b", re.IGNORECASE)
    for ticker, names in ALIASES.items()
}
_SYMBOLS = "|".join(ALIASES)
_BARE_RE = re.compile(r"(?<![A-Za-z$])(" + _SYMBOLS + r")\b")  # case-sensitive: NVDA, not "nvda"
_DOLLAR_RE = re.compile(r"\$(" + _SYMBOLS + r")\b", re.IGNORECASE)


def find_tickers(text: str) -> list[str]:
    text = normalize_text(text)
    found = {t for t, pattern in _ALIAS_PATTERNS.items() if pattern.search(text)}
    found.update(m.group(1).upper() for m in _BARE_RE.finditer(text))
    found.update(m.group(1).upper() for m in _DOLLAR_RE.finditer(text))
    return sorted(found)


def mentions(text: str, ticker: str) -> bool:
    text = normalize_text(text)
    if _ALIAS_PATTERNS[ticker].search(text):
        return True
    return re.search(r"(?<![A-Za-z])" + ticker + r"\b", text) is not None


def alias_tokens() -> frozenset[str]:
    return frozenset(tok for names in ALIASES.values() for n in names for tok in tokenize(n))
