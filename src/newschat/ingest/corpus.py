"""The loaded, cleaned, de-duplicated article set plus corpus-level facts."""

import os
import re
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from newschat.ingest.dedupe import build_articles
from newschat.ingest.loader import load_records
from newschat.models import Article

DEFAULT_DATA_PATH = Path(__file__).resolve().parents[3] / "data" / "stock_news.json"

_MONTHS = {
    "january": "January",
    "february": "February",
    "march": "March",
    "april": "April",
    "may": "May",
    "june": "June",
    "july": "July",
    "august": "August",
    "september": "September",
    "october": "October",
    "november": "November",
    "december": "December",
    "jan": "January",
    "feb": "February",
    "mar": "March",
    "apr": "April",
    "jun": "June",
    "jul": "July",
    "aug": "August",
    "sep": "September",
    "sept": "September",
    "oct": "October",
    "nov": "November",
    "dec": "December",
}
# Full dates only ("February 18, 2025", "Feb. 19, 2025"). Bare "Month YYYY" mentions in this
# corpus are dominated by promotional and options-disclosure boilerplate and mislead.
_FULL_DATE_RE = re.compile(
    r"\b(" + "|".join(sorted(_MONTHS, key=len, reverse=True)) + r")\.?\s+\d{1,2},\s+(20\d\d)\b",
    re.IGNORECASE,
)


class Corpus:
    def __init__(self, articles: Iterable[Article]) -> None:
        self.articles: list[Article] = list(articles)
        self.by_id: dict[str, Article] = {a.id: a for a in self.articles}
        self.tickers: list[str] = sorted({t for a in self.articles for t in a.tickers})

    @classmethod
    def load(cls, path: Path | None = None) -> "Corpus":
        resolved = path or Path(os.environ.get("NEWSCHAT_DATA", str(DEFAULT_DATA_PATH)))
        return cls(build_articles(load_records(resolved)))

    def infer_period(self) -> str:
        counts: Counter[str] = Counter()
        for article in self.articles:
            for match in _FULL_DATE_RE.finditer(article.text):
                counts[f"{_MONTHS[match.group(1).lower()]} {match.group(2)}"] += 1
        if not counts:
            return "unknown period"
        return counts.most_common(1)[0][0]
