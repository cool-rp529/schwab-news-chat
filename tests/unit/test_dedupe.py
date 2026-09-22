from pathlib import Path

import pytest

from newschat.ingest.cleaner import clean
from newschat.ingest.dedupe import article_id, build_articles, source_of
from newschat.ingest.loader import CorpusError, RawRecord, load_records


def rec(
    link: str, ticker: str, text: str, key: str | None = None, title: str = "Title here"
) -> RawRecord:
    return RawRecord(
        key_ticker=key or ticker, title=title, link=link, ticker=ticker, full_text=text
    )


def test_article_id_is_stable_short_sha1() -> None:
    assert article_id("https://a/b") == article_id("https://a/b")
    assert len(article_id("https://a/b")) == 10
    assert article_id("https://a/b") != article_id("https://a/c")


def test_source_strips_www() -> None:
    assert source_of("https://www.finance.yahoo.com/news/x.html") == "finance.yahoo.com"


def test_rejects_non_http_link_scheme() -> None:
    # Citation URLs are rendered as <a href> in the UI; a javascript: link would be a
    # clickable script sourced from the corpus.
    with pytest.raises(CorpusError, match="unsupported URL scheme"):
        build_articles([rec("javascript:alert(1)", "AAPL", "Body text.")])


def test_merges_duplicates_and_unions_tickers() -> None:
    long_body = "Long body. " * 60
    arts = build_articles(
        [
            rec("https://a/1", "NVDA", "Short body."),
            rec("https://a/1", "AAPL", long_body),
            rec("https://a/2", "IBM", "Other."),
        ]
    )
    assert [a.url for a in arts] == ["https://a/1", "https://a/2"]
    assert arts[0].tickers == ["AAPL", "NVDA"]
    assert arts[0].text.startswith("Long body.")


def test_key_ticker_and_record_ticker_both_kept() -> None:
    (art,) = build_articles([rec("https://a/1", "AAPL", "x", key="MSFT")])
    assert art.tickers == ["AAPL", "MSFT"]


def test_longest_body_wins_regardless_of_order() -> None:
    a, b = rec("https://a/1", "X", "short"), rec("https://a/1", "Y", "much longer body text")
    assert (
        build_articles([a, b])[0].text == build_articles([b, a])[0].text == "much longer body text"
    )


def test_real_corpus_dedupes_to_118_and_titles_agree(data_path: Path) -> None:
    records = load_records(data_path)
    arts = build_articles(records)
    assert len(arts) == 118
    by_link: dict[str, set[str]] = {}
    for r in records:
        by_link.setdefault(r.link, set()).add(r.title)
    assert all(len(titles) == 1 for titles in by_link.values())
    multi = [a for a in arts if len(a.tickers) > 1]
    assert len(multi) == 13
    assert max(len(a.tickers) for a in arts) == 4


def test_real_corpus_divergent_bodies_take_longest(data_path: Path) -> None:
    records = load_records(data_path)
    groups: dict[str, list[RawRecord]] = {}
    for r in records:
        groups.setdefault(r.link, []).append(r)
    divergent = {link: g for link, g in groups.items() if len({r.full_text for r in g}) > 1}
    assert len(divergent) == 4
    arts = {a.url: a for a in build_articles(records)}
    for link, g in divergent.items():
        longest = max(g, key=lambda r: len(r.full_text))
        assert arts[link].text == clean(longest.full_text, title=longest.title).text
