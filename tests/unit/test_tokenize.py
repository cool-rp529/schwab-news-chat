import pytest

from newschat.text.tokenize import STOPWORDS, normalize_text, stem, tokenize


def test_lowercases_and_drops_stopwords() -> None:
    assert tokenize("What is the Intel deal?") == ["intel", "deal"]


def test_keeps_dollar_tickers_and_decimals() -> None:
    assert tokenize("$NVDA fell 1.5% to $120.50") == ["$nvda", "fell", "1.5", "$120.50"]


def test_possessives_are_stripped() -> None:
    assert tokenize("Apple's iPhone") == ["apple", "iphone"]
    assert tokenize("Apple’s iPhone") == ["apple", "iphone"]
    assert tokenize("We’ve seen the report") == ["seen", "report"]


@pytest.mark.parametrize(
    ("word", "expected"),
    [
        ("shares", "share"),
        ("companies", "company"),
        ("buying", "buy"),
        ("jumped", "jump"),
        ("news", "news"),
        ("business", "business"),
        ("class", "class"),
        ("ai", "ai"),
        ("2025", "2025"),
        ("$aapl", "$aapl"),
        ("earnings", "earning"),
    ],
)
def test_stem(word: str, expected: str) -> None:
    assert stem(word) == expected


def test_stopwords_include_question_words() -> None:
    assert {"what", "who", "how", "the", "is", "about"} <= STOPWORDS


def test_fullwidth_text_is_nfkc_normalised() -> None:
    assert tokenize("Ｉｎｔｅｌ") == ["intel"]
    assert normalize_text("Ｉｎｔｅｌ ｎｅｗｓ") == "Intel news"


def test_nfkc_leaves_curly_quotes_alone() -> None:
    assert normalize_text("’ “ ”") == "’ “ ”"
