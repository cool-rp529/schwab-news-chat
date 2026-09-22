"""Collapse the 138 ticker-filed records into unique articles.

The same article is filed under up to four tickers. Titles agree across a duplicate group
but bodies do not always (a Yahoo video sidebar and drifting "N hours ago" stamps differ),
so "first occurrence" would be a dict-ordering coin flip; the longest raw body wins.
"""

import hashlib
from collections.abc import Iterable
from urllib.parse import urlparse

from newschat.ingest.cleaner import clean
from newschat.ingest.loader import CorpusError, RawRecord
from newschat.models import Article

# The UI renders every citation as an <a href>, so a `javascript:` or `data:` link in the
# corpus would become a clickable script. Pydantic's HttpUrl would validate this but also
# normalise the string, which would change every article id and break equality.
ALLOWED_SCHEMES = ("http://", "https://")


def article_id(url: str) -> str:
    return hashlib.sha1(url.encode("utf-8")).hexdigest()[:10]


def source_of(url: str) -> str:
    return urlparse(url).netloc.lower().removeprefix("www.")


def build_articles(records: Iterable[RawRecord]) -> list[Article]:
    groups: dict[str, list[RawRecord]] = {}
    for record in records:
        groups.setdefault(record.link, []).append(record)

    articles: list[Article] = []
    for link, group in groups.items():
        if not link.lower().startswith(ALLOWED_SCHEMES):
            raise CorpusError(f"{link!r}: unsupported URL scheme")
        first = group[0]
        body = max(group, key=lambda r: len(r.full_text)).full_text  # ties → first occurrence
        tickers = sorted({t for r in group for t in (r.ticker, r.key_ticker)})
        cleaned = clean(body, title=first.title)
        articles.append(
            Article(
                id=article_id(link),
                title=first.title.strip(),
                url=link,
                source=source_of(link),
                tickers=tickers,
                text=cleaned.text,
                is_truncated=cleaned.is_truncated,
                word_count=len(cleaned.text.split()),
            )
        )
    return articles
