import json
from pathlib import Path

import pytest

from newschat.ingest.cleaner import TRUNCATED_BELOW, apply_rule, clean, rule_named

PAYWALL = (
    "Apple (AAPL)'s launch of the fourth generation of the iPhone SE this week has most "
    "likely been taken PREMIUM Upgrade to read this MT Newswires article and get so much more. "
    "A Silver or Gold subscription plan is required to access premium news articles. Upgrade "
    "Already have a subscription? Sign in"
)
CONTINUE = "Intel stock rose Tuesday. Continue Reading View Comments"
INSIDER = (
    "DeepSeek has failed to respond. Why are we interested in the stocks that hedge funds pile "
    "into? The reason is simple: our research has shown that we can outperform the market by "
    "imitating the top stock picks of the best hedge funds. Our quarterly newsletter’s strategy "
    "selects 14 small-cap and large-cap stocks every quarter and has returned 275% since May "
    "2014, beating its benchmark by 150 percentage points (see more details here). Story "
    "Continues Apple is a technology company."
)
READ_ALSO = (
    "It also raised its price target for MSCI China to 85 from 75. READ ALSO: 10 AI Stocks "
    "Analysts Are Watching: Latest Ratings and News and Top 14 AI Stocks on Wall Street: "
    "News and Analyst Ratings The Chinese startup may be all the rage."
)
CAROUSEL = (
    "Cash levels fell. Related Videos 02:34 Growing bearish sentiments is actually bullish? "
    "Here's why. Yahoo Finance Video • 26 minutes ago 00:57 Conagra, Robinhood, Meta all "
    "slip: Market Minute Yahoo Finance Video • 30 minutes ago Beyond Occidental, Buffett "
    "has added Constellation Brands."
)
FOOL = (
    "I predict the company will announce a new product. Should you invest $1,000 in Apple "
    "right now? Before you buy stock in Apple, consider this: The Motley Fool Stock Advisor "
    "analyst team just identified what they believe are the 10 best stocks. Consider when "
    "Nvidia made this list on April 15, 2005... Learn more » *Stock Advisor returns as of "
    "February 7, 2025 Keith Speights has positions in Apple. The Motley Fool has positions in "
    "and recommends Apple. The Motley Fool recommends the following options: long January 2026 "
    "$395 calls on Microsoft. The Motley Fool has a disclosure policy. Should You Buy Apple "
    "Stock Hand Over Fist Before Feb. 19? was originally published by The Motley Fool View Comments"
)


def test_paywall_tail_removed_and_flagged_truncated() -> None:
    result = clean(PAYWALL)
    assert "PREMIUM" not in result.text
    assert result.text.startswith("Apple (AAPL)'s launch")
    assert result.is_truncated is True
    assert "mt_newswires_paywall" in result.applied


def test_continue_reading_tail_removed() -> None:
    assert clean(CONTINUE).text == "Intel stock rose Tuesday."


def test_insider_monkey_pitch_and_story_continues_removed() -> None:
    text = clean(INSIDER).text
    assert "hedge funds pile into" not in text
    assert "May 2014" not in text
    assert "Story Continues" not in text
    assert text == "DeepSeek has failed to respond. Apple is a technology company."


def test_read_also_removed_up_to_next_sentence() -> None:
    text = clean(READ_ALSO).text
    assert "READ ALSO" not in text
    assert text == (
        "It also raised its price target for MSCI China to 85 from 75. "
        "The Chinese startup may be all the rage."
    )


def test_related_videos_carousel_removed() -> None:
    text = clean(CAROUSEL).text
    assert "Related Videos" not in text
    assert "minutes ago" not in text
    assert text == "Cash levels fell. Beyond Occidental, Buffett has added Constellation Brands."


def test_motley_fool_promo_disclosure_and_syndication_removed() -> None:
    text = clean(FOOL, title="Should You Buy Apple Stock Hand Over Fist Before Feb. 19?").text
    assert text == "I predict the company will announce a new product."
    assert "January 2026" not in text
    assert "April 15, 2005" not in text


def test_single_rule_can_be_applied_in_isolation() -> None:
    rule = rule_named("view_comments_tail")
    assert apply_rule("Body. View Comments", rule).strip() == "Body."
    assert (
        apply_rule("Body. View Comments in the middle", rule) == "Body. View Comments in the middle"
    )


def test_nbsp_and_whitespace_normalised() -> None:
    assert clean("a b   c\n\nd").text == "a b c d"


def test_truncation_boundary() -> None:
    assert clean("x" * (TRUNCATED_BELOW - 1)).is_truncated is True
    assert clean("x" * TRUNCATED_BELOW).is_truncated is False


def test_idempotent() -> None:
    once = clean(FOOL).text
    assert clean(once).text == once


def test_unknown_rule_name() -> None:
    with pytest.raises(KeyError):
        rule_named("nope")


@pytest.fixture(scope="module")
def cleaned_bodies(data_path: Path) -> list[str]:
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    return [clean(a["full_text"], title=a["title"]).text for arts in payload.values() for a in arts]


@pytest.mark.parametrize(
    "marker",
    [
        "PREMIUM Upgrade",
        "View Comments",
        "Story Continues",
        "Related Videos",
        "disclosure policy",
        "hedge funds pile into",
        "Stock Advisor returns",
        "READ ALSO",
        "minutes ago",
        "hours ago",
        "originally published",
        "originally appeared",
        "Want the latest recommendations from Zacks",
        "Free Stock Analysis Report",
    ],
)
def test_real_corpus_has_no_boilerplate_marker_left(cleaned_bodies: list[str], marker: str) -> None:
    offenders = [b[:80] for b in cleaned_bodies if marker in b]
    assert offenders == [], f"{marker!r} survived in {len(offenders)} bodies"


def test_real_corpus_truncation_rate_is_plausible(cleaned_bodies: list[str]) -> None:
    truncated = sum(len(b) < TRUNCATED_BELOW for b in cleaned_bodies)
    # 22 bodies are under 400 chars pre-cleaning (CLAUDE.md); stubs get shorter, prose stays long.
    assert 20 <= truncated <= 45, truncated
