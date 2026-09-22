import pytest

from newschat.ingest.corpus import Corpus
from newschat.summarize.claims import CLAIM_RULES, classify, is_advice, matching_rule


@pytest.mark.parametrize(
    ("passage", "kind"),
    [
        ("Intel shares rose more than 10% on Tuesday after a Wall Street Journal report.", "event"),
        ("Evercore ISI raised its price target on Intel to $27 from $22.", "opinion"),
        ("Nvidia carries a Zacks Rank #2 (Buy) and a Strong Buy consensus.", "opinion"),
        ("Analysts expect the company to beat estimates when it reports next week.", "opinion"),
        (
            "We recently compiled a list of the 10 High-Flying AI Stocks on News and Ratings.",
            "promo",
        ),
        ("Our quarterly newsletter's strategy has returned 275% since May 2014.", "promo"),
        ("The Motley Fool Stock Advisor analyst team just identified the 10 best stocks.", "promo"),
        ("Click to get this free report Apple Inc. (AAPL): Free Stock Analysis Report.", "promo"),
        (
            "Apple will unveil a product tomorrow, but Wall Street is already sure what it is.",
            "event",
        ),
    ],
)
def test_classify(passage: str, kind: str) -> None:
    assert classify(passage) == kind


def test_promo_beats_opinion_when_both_match() -> None:
    both = "Our newsletter says the stock is a Strong Buy with a $200 price target."
    assert classify(both) == "promo"
    rule = matching_rule(both)
    assert rule is not None and rule.kind == "promo"


@pytest.mark.parametrize(
    "passage",
    [
        "You can see the complete list of today's Zacks #1 Rank (Strong Buy) stocks here.",
        "You can see the complete list of today’s Zacks #1 Rank stocks here.",
        "See the Zacks #1 Rank (Strong Buy) stocks here.",
    ],
)
def test_zacks_list_cta_is_promo_not_opinion(passage: str) -> None:
    # "Zacks #1 Rank (Strong Buy)" also trips the opinion rating rule; promo must win.
    rule = matching_rule(passage)
    assert rule is not None and rule.name == "promo_zacks_list_cta" and rule.kind == "promo"


def test_title_participates() -> None:
    assert (
        classify(
            "Shares rose on the news this morning in early trading.",
            title="Should You Buy Apple Stock Hand Over Fist?",
        )
        == "opinion"
    )


def test_rules_are_named_and_ordered_promo_first() -> None:
    kinds = [r.kind for r in CLAIM_RULES]
    assert kinds.index("opinion") > kinds.index("promo")
    assert kinds[-1] == "opinion"  # no rule for event: it is the default
    assert len({r.name for r in CLAIM_RULES}) == len(CLAIM_RULES)


def test_default_is_event_and_matching_rule_none() -> None:
    assert classify("") == "event" and matching_rule("") is None


@pytest.mark.parametrize(
    ("text", "advice"),
    [
        ("Should you buy Nvidia stock hand over fist before Feb. 26?", True),
        ("It's a screaming buy right now.", True),
        ("We recommend investors load up on shares.", True),
        ("Intel shares rose 10% on the report.", False),
        ("Analysts at BofA said they expect a beat.", False),
    ],
)
def test_is_advice(text: str, advice: bool) -> None:
    assert is_advice(text) is advice


def test_real_corpus_has_all_three_kinds(corpus: Corpus) -> None:
    kinds = {classify(a.text[:400], a.title) for a in corpus.articles}
    assert kinds == {"event", "opinion", "promo"}
