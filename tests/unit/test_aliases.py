import pytest

from newschat.retrieval.aliases import ALIASES, alias_tokens, find_tickers, mentions


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("What is Apple launching?", ["AAPL"]),
        ("Is the iPhone SE 4 delayed?", ["AAPL"]),
        ("nvidia earnings and Intel breakup", ["INTC", "NVDA"]),
        ("$NVDA and $msft", ["MSFT", "NVDA"]),
        ("What did OpenAI announce?", ["MSFT"]),
        ("AWS in healthcare", ["AMZN"]),
        ("International Business Machines results", ["IBM"]),
        ("pineapple prices", []),
        ("Intel's rivals", ["INTC"]),
        ("IBM", ["IBM"]),
        ("the ibm deal", ["IBM"]),
        ("weather in Tokyo", []),
        ("Foxconn's factories in India", ["AAPL"]),
    ],
)
def test_find_tickers(text: str, expected: list[str]) -> None:
    assert find_tickers(text) == expected


def test_every_corpus_ticker_has_aliases() -> None:
    assert set(ALIASES) == {"AAPL", "AMZN", "IBM", "INTC", "MSFT", "NFLX", "NVDA"}
    assert all(names for names in ALIASES.values())


def test_mentions_is_whole_word() -> None:
    assert mentions("Intel stock rose", "INTC")
    assert mentions("Broadcom eyes INTC", "INTC")
    assert not mentions("intelligence agencies", "INTC")


def test_alias_tokens_are_stemmed_lowercase() -> None:
    toks = alias_tokens()
    assert {"apple", "iphone", "nvidia", "intel", "ibm", "netflix", "amazon", "aws"} <= toks


def test_fullwidth_names_are_found() -> None:
    assert find_tickers("Ｉｎｔｅｌ ｎｅｗｓ") == ["INTC"]
    assert mentions("Ｉｎｔｅｌ stock rose", "INTC")
