import json
import time
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


def test_zacks_free_report_tail_removed() -> None:
    rule = rule_named("zacks_free_report")
    body = (
        "Intel closed lower on Tuesday. Click to get this free report Intel Corporation "
        "(INTC) : Free Stock Analysis Report"
    )
    assert apply_rule(body, rule).strip() == "Intel closed lower on Tuesday."


def test_insider_monkey_footer_tail_removed() -> None:
    rule = rule_named("insider_monkey_footer")
    body = (
        "Nvidia led the group. READ NEXT: 20 Best AI Stocks To Buy Now. Disclosure: None. "
        "This article is originally published at Insider Monkey."
    )
    assert apply_rule(body, rule).strip() == "Nvidia led the group."


def test_simply_wall_st_footer_tail_removed() -> None:
    rule = rule_named("simply_wall_st_footer")
    opening = "IBM trades below our estimate of fair value."
    for footer in (
        " Have feedback on this article? Concerned about the content? Get in touch with us.",
        " This article by Simply Wall St is general in nature. We provide commentary based on "
        "historical data.",
    ):
        assert apply_rule(opening + footer, rule).strip() == opening


def test_yahoo_quote_widget_block_removed() -> None:
    rule = rule_named("yahoo_quote_widget")
    body = (
        "Shares slipped. Real Time Price • USD (INTC) 23.60 +0.42 (+1.81%) At close: February "
        "18 at 4:00:02 PM EST Advanced Chart Analysts remain divided."
    )
    # The rule leaves its replacement space behind; `whitespace_final` collapses it.
    assert apply_rule(body, rule) == "Shares slipped.  Analysts remain divided."


def test_bare_url_removed() -> None:
    rule = rule_named("bare_url")
    body = (
        "https://www.tipranks.com/news/the-fly/sunnova-energy Certain Apple TV (AAPL) 4K "
        "users are being prompted to link their accounts. See http://example.com/x?a=1&b=2 "
        "for more."
    )
    # In isolation the rule leaves its replacement space behind; `whitespace_final`
    # collapses it, so `clean()` gives the finished sentence.
    assert apply_rule(body, rule).strip() == (
        "Certain Apple TV (AAPL) 4K users are being prompted to link their accounts. See  for more."
    )
    assert clean(body).text == (
        "Certain Apple TV (AAPL) 4K users are being prompted to link their accounts. See for more."
    )
    # A bare domain is not a link and must survive: it is ordinary prose.
    assert (
        apply_rule("Zacks.com is a property of Zacks.", rule) == "Zacks.com is a property of Zacks."
    )


def test_motley_fool_promo_matches_curly_apostrophe() -> None:
    rule = rule_named("motley_fool_promo")
    for apostrophe in ("'", "’"):
        body = (
            f"Apple had a strong quarter. Don{apostrophe}t miss this second chance at a "
            "potentially lucrative opportunity. Stock Advisor returns as of February 7, 2025."
        )
        assert apply_rule(body, rule).strip() == "Apple had a strong quarter."


def test_nbsp_and_whitespace_normalised() -> None:
    assert clean("a b   c\n\nd").text == "a b c d"
    # The corpus carries NBSPs ("Vote Against\u00a0Unionizing"); \s is Unicode-aware.
    nbsp = "\u00a0"
    assert clean(f"a{nbsp}b{nbsp}{nbsp}c").text == "a b c"


def test_truncation_boundary() -> None:
    assert clean("x" * (TRUNCATED_BELOW - 1)).is_truncated is True
    assert clean("x" * TRUNCATED_BELOW).is_truncated is False


def test_idempotent() -> None:
    once = clean(FOOL).text
    assert clean(once).text == once


def test_unknown_rule_name() -> None:
    with pytest.raises(KeyError):
        rule_named("nope")


def test_syndication_footer_starts_at_sentence_boundary() -> None:
    rule = rule_named("syndication_footer")
    body = (
        "First sentence. Second one here. Some Title Words originally appeared on "
        "Benzinga.com View Comments"
    )
    assert apply_rule(body, rule).strip() == "First sentence. Second one here."
    # No preceding sentence terminator: the rule must not match starting mid-sentence.
    assert apply_rule("Report originally appeared on X", rule).strip() == ""


def test_real_corpus_cleans_fast(data_path: Path) -> None:
    # Pre-fix, syndication_footer's `\s*[^.!?]*` retried from every character position,
    # costing 3.36s of ~4.5s total to clean these 138 bodies. Anchoring it to sentence
    # boundaries made it linear; this guards against the quadratic behaviour regressing.
    payload = json.loads(data_path.read_text(encoding="utf-8"))
    records = [a for arts in payload.values() for a in arts]
    start = time.perf_counter()
    for record in records:
        clean(record["full_text"], title=record["title"])
    elapsed = time.perf_counter() - start
    assert elapsed < 1.5, f"cleaning {len(records)} bodies took {elapsed:.2f}s"


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
    # 22 bodies are under 400 chars pre-cleaning (see the README's data-findings table);
    # stubs get shorter still, while real prose stays well clear of the line.
    assert 20 <= truncated <= 45, truncated
