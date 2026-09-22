from newschat.text.sentences import is_prose, split_sentences


def test_splits_on_terminators() -> None:
    assert split_sentences("Intel rose. Nvidia fell! Did Apple move? Yes.") == [
        "Intel rose.",
        "Nvidia fell!",
        "Did Apple move?",
        "Yes.",
    ]


def test_abbreviations_do_not_split() -> None:
    text = "Apple Inc. said Mr. Cook will speak on Feb. 19. Shares rose in the U.S. on Tuesday."
    assert split_sentences(text) == [
        "Apple Inc. said Mr. Cook will speak on Feb. 19.",
        "Shares rose in the U.S. on Tuesday.",
    ]


def test_decimals_and_currency_do_not_split() -> None:
    assert split_sentences("Revenue hit $1.5 billion. Margins were 23.4% higher.") == [
        "Revenue hit $1.5 billion.",
        "Margins were 23.4% higher.",
    ]


def test_closing_quotes_stay_with_sentence() -> None:
    text = "He said “we are done.” Then he left."
    assert split_sentences(text) == ["He said “we are done.”", "Then he left."]


def test_empty_and_whitespace() -> None:
    assert split_sentences("") == []
    assert split_sentences("   \n ") == []


def test_is_prose_rejects_carousel_and_stamps() -> None:
    assert is_prose("Intel shares rose more than 10% on Tuesday after a report about Broadcom.")
    assert not is_prose("02:34 Growing bearish sentiments is actually bullish? Here's why.")
    assert not is_prose("Yahoo Finance Video • 26 minutes ago")
    assert not is_prose("Nvidia Stock News And Analysis Nvidia Soars As Stargate Sparks More AI")
    assert not is_prose("Too short here.")
    assert not is_prose("Key Takeaways:")
