from pathlib import Path

import pytest

from newschat.ingest.corpus import Corpus
from newschat.ingest.loader import CorpusError
from newschat.models import Article


def art(text: str, i: int = 0) -> Article:
    return Article(
        id=f"id{i}",
        title="t",
        url=f"u{i}",
        source="s",
        tickers=["AAPL"],
        text=text,
        is_truncated=False,
        word_count=len(text.split()),
    )


def test_load_real_corpus(data_path: Path) -> None:
    corpus = Corpus.load(data_path)
    assert len(corpus.articles) == 118
    assert corpus.tickers == ["AAPL", "AMZN", "IBM", "INTC", "MSFT", "NFLX", "NVDA"]
    first = corpus.articles[0]
    assert corpus.by_id[first.id] is first


def test_load_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(CorpusError):
        Corpus.load(tmp_path / "missing.json")


def test_infer_period_counts_full_dates_only() -> None:
    corpus = Corpus(
        [
            art("Chicago, IL – February 18, 2025 – Zacks announces. Returned 275% since May 2014."),
            art("On Feb. 19, 2025 Apple will unveil. long January 2026 $395 calls.", 1),
            art("Nothing dated here at all. May 2014 again. January 2026 again.", 2),
        ]
    )
    assert corpus.infer_period() == "February 2025"


def test_infer_period_unknown_without_dates() -> None:
    assert Corpus([art("No dates.")]).infer_period() == "unknown period"


def test_real_corpus_period_is_february_2025(data_path: Path) -> None:
    # Guards the May-2014 / January-2026 boilerplate trap in the README's data-findings table.
    assert Corpus.load(data_path).infer_period() == "February 2025"
