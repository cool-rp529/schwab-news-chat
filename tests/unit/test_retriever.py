import re

import pytest

from newschat.ingest.corpus import Corpus
from newschat.models import Article, Inheritance, ResolvedQuery
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever, RetrieverConfig
from newschat.text.sentences import is_prose, split_sentences


def art(i: int, text: str, tickers: list[str], truncated: bool = False, title: str = "") -> Article:
    return Article(
        id=f"a{i}",
        title=title or f"Title {i}",
        url=f"https://x/{i}",
        source="x",
        tickers=tickers,
        text=text,
        is_truncated=truncated,
        word_count=len(text.split()),
    )


PROSE = (
    "The company said the result reflected strong demand across all of its regions this quarter."
)
MINI = Corpus(
    [
        art(
            0,
            "Intel shares jumped after a report that Broadcom and TSMC are exploring deals. "
            f"{PROSE}",
            ["INTC"],
        ),
        art(
            1,
            "Intel gets a new price target from Evercore, which raised it to $27 from $22. "
            f"{PROSE}",
            ["INTC"],
            truncated=True,
        ),
        art(
            2,
            f"Nvidia is set to report earnings on February 26 and analysts expect a beat. {PROSE}",
            ["NVDA"],
        ),
        art(
            3,
            "Apple partners with Alibaba for AI features in China, sidestepping regulators. "
            f"{PROSE}",
            ["AAPL"],
        ),
        art(
            4,
            "A broadcom deal for parts of a chipmaker would reshape chip manufacturing "
            f"in the United States. {PROSE}",
            ["MSFT"],
        ),
    ]
)


def q(text: str) -> ResolvedQuery:
    return QueryAnalyzer().resolve(text, [])


@pytest.fixture
def mini() -> Retriever:
    # 0.4, not 0.5: the one-term rows in MINI land exactly on 0.50/0.50, so a 0.5 floor
    # decides them on float rounding. These tests are about ordering, not the gate.
    return Retriever(MINI, RetrieverConfig(min_strength=0.4, min_coverage=0.4))


def test_ticker_boost_orders_matching_article_first(mini: Retriever) -> None:
    hits = mini.search(q("Broadcom deal for Intel"))
    assert hits[0].article.id == "a0"  # INTC-tagged beats the MSFT-tagged mention
    assert {h.article.id for h in hits} >= {"a0", "a4"}


def test_truncated_article_is_penalised(mini: Retriever) -> None:
    ids = [h.article.id for h in mini.search(q("Intel"))]
    assert ids.index("a0") < ids.index("a1")


def test_restriction_then_top_up_keeps_ticker_matches_first(mini: Retriever) -> None:
    hits = mini.search(q("Intel deal"), k=3)
    ids = [h.article.id for h in hits]
    assert set(ids[:2]) == {"a0", "a1"}  # both INTC articles lead
    assert ids[2] == "a4"  # non-ticker article tops up the list
    assert max(h.score for h in hits) == 1.0


def test_absolute_gate_refuses_out_of_corpus_question(mini: Retriever) -> None:
    assert mini.search(q("What are Boeing's 737 MAX safety issues?")) == []


def test_no_keywords_returns_nothing(mini: Retriever) -> None:
    assert mini.search(ResolvedQuery(text="?", tickers=["INTC"])) == []


def test_context_keywords_carry_topic(mini: Retriever) -> None:
    follow_up = ResolvedQuery(
        text="Who wants to buy it?",
        tickers=["INTC"],
        keywords=["want", "buy"],
        context_keywords=["intel", "broadcom"],
        inherited_tickers=True,
    )
    hits = mini.search(follow_up)
    assert hits and hits[0].article.id == "a0"


def test_passages_are_prose_sentences_from_the_article(mini: Retriever) -> None:
    (hit, *_) = mini.search(q("Alibaba AI China Apple"))
    assert hit.article.id == "a3"
    assert hit.passages[0].startswith("Apple partners with Alibaba")
    assert all(p in hit.article.text for p in hit.passages)


def test_hit_fields(mini: Retriever) -> None:
    (hit, *_) = mini.search(q("Nvidia earnings"))
    assert hit.raw_score > 0 and hit.matched >= 2 and 0 < hit.score <= 1.0


# ---- real corpus: the gate must refuse plausible questions and pass in-corpus ones ----


@pytest.fixture(scope="module")
def real(corpus: Corpus) -> Retriever:
    return Retriever(corpus)


@pytest.mark.parametrize(
    "question",
    [
        "What are Boeing's 737 MAX safety issues?",
        "What's the latest on Tesla's Cybertruck recall?",
        "What is the weather in Tokyo tomorrow?",
        "What is the price of Bitcoin today?",
        "Who won the Super Bowl?",
        "How do I bake sourdough bread?",
        "What is the best pizza in Chicago?",
    ],
)
def test_real_gate_refuses(real: Retriever, question: str) -> None:
    hits = real.search(q(question))
    assert hits == [], [h.article.title for h in hits]


@pytest.mark.parametrize(
    ("question", "title_fragment"),
    [
        ("What's going on with Intel?", "Intel"),
        ("Who might buy parts of Intel?", "Broadcom"),
        ("What has Warren Buffett been buying?", "Buffett"),
        ("Why is Amazon shutting down Inspire?", "Inspire"),
        ("How is DeepSeek affecting AI stocks?", "AI"),
        ("Tell me about Netflix and live sports", "live sports"),
    ],
)
def test_real_gate_passes_in_corpus_questions(
    real: Retriever, question: str, title_fragment: str
) -> None:
    hits = real.search(q(question))
    assert hits, question
    assert any(title_fragment.lower() in h.article.title.lower() for h in hits[:3]), [
        h.article.title for h in hits
    ]


@pytest.mark.parametrize(
    "question",
    [
        "Tell me about DeepSeek",
        "DeepSeek",
        "What is DeepSeek?",
        # Not included: "Which companies are exposed to DeepSeek?". Its rare word
        # "exposed" never appears in the corpus, so it fails on coverage (0.35) and
        # strength (0.76) alike, and no floor admits it without also admitting the
        # "best pizza in Chicago" negative at 0.77. It lives in the thematic tier of
        # golden.yaml, where the false refusal is reported rather than asserted away.
    ],
)
def test_short_query_on_a_covered_topic_is_not_refused(real: Retriever, question: str) -> None:
    """A topic in 11 articles must not refuse merely because the query is short.

    BM25 sums over query terms, so a one-term query's raw score is bounded by about
    idf * (k1 + 1) -- under any floor calibrated on three-term queries. DeepSeek is the
    case that exposes it: no ticker alias, so the ticker_match bypass cannot rescue it
    the way it silently rescues "intel" (raw 4.07) and "nvidia" (raw 3.39).
    """
    hits = real.search(q(question))
    assert hits, question
    assert any("deepseek" in h.article.text.lower() for h in hits), [h.article.title for h in hits]


STAMP_RE = re.compile(r"\b\d+\s+(?:minutes?|hours?|days?)\s+ago\b|\b\d{1,2}:\d{2}\b|•")


def test_real_passages_never_contain_carousel_or_stamps(real: Retriever) -> None:
    for question in ["Wall Street bullish cash levels", "Buffett Occidental", "Meta stock streak"]:
        for hit in real.search(q(question)):
            for p in hit.passages:
                assert not STAMP_RE.search(p), p
    # Legitimate prose that merely contains "ago" must remain eligible as a passage.
    assert any(
        " ago" in s for a in real.corpus.articles for s in split_sentences(a.text) if is_prose(s)
    )


def test_known_terms_is_the_real_corpus_vocabulary(real: Retriever) -> None:
    # What the query analyzer checks a short follow-up's words against.
    assert "intel" in real.known_terms
    assert "bitcoin" not in real.known_terms


# ---- inheritance strength: soft tickers are a ranking preference, strong ones restrict ----


def inherited(inheritance: Inheritance, keywords: list[str]) -> ResolvedQuery:
    return ResolvedQuery(
        text="follow-up",
        tickers=["INTC"],
        keywords=keywords,
        inherited_tickers=True,
        inheritance=inheritance,
    )


def test_soft_inheritance_ranks_purely_by_score() -> None:
    # No gate, so only the ordering is under test. a0 (INTC) matches "broadcom" alone; a4
    # (MSFT) matches both terms and outscores a0 even after a0's ticker boost.
    ungated = Retriever(MINI, RetrieverConfig(min_strength=0.0, min_coverage=0.0))
    ids = [h.article.id for h in ungated.search(inherited("soft", ["broadcom", "chip"]))]
    assert ids[:2] == ["a4", "a0"]


def test_soft_inheritance_does_not_let_a_ticker_match_bypass_the_gate(mini: Retriever) -> None:
    # a0 is INTC-tagged but covers too little of the question to clear the gate on its own.
    ids = [h.article.id for h in mini.search(inherited("soft", ["broadcom", "chip"]))]
    assert ids == ["a4"]


def test_strong_inheritance_keeps_restriction_and_gate_bypass(mini: Retriever) -> None:
    ids = [h.article.id for h in mini.search(inherited("strong", ["broadcom", "chip"]))]
    assert ids == ["a0", "a4"]  # the ticker match leads, past the gate, then the top-up
