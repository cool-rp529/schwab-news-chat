import re
from pathlib import Path

import pytest

from newschat.evaluation import load_golden
from newschat.ingest.corpus import Corpus
from newschat.models import Article, Hit, ResolvedQuery
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever
from newschat.summarize.base import ADVISORY, CorpusInfo
from newschat.summarize.extractive import KIND_HEADINGS, ExtractiveSummarizer, jaccard

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.yaml"
INFO = CorpusInfo(period="February 2025", tickers=("INTC", "NVDA"))
S_EVENT = (
    "Intel shares rose more than 10% on Tuesday after a report that Broadcom and TSMC are "
    "weighing deals."
)
S_EVENT_DUP = (
    "Intel shares rose more than 10% Tuesday after a report that Broadcom and TSMC weigh deals."
)
S_OPINION = (
    "Evercore ISI raised its price target on Intel to $27 from $22 while keeping its rating."
)
S_PROMO = (
    "We recently compiled a list of the 10 best chip stocks according to our newsletter, "
    "including Intel."
)

DISTINCT = [
    "Intel shares climbed sharply after regulators cleared the long delayed Ohio fabrication "
    "plant.",
    "Intel appointed a veteran semiconductor executive to lead its struggling foundry division.",
    "Intel unveiled a low power laptop processor aimed squarely at premium ultrabook buyers.",
    "Intel suspended its quarterly dividend to preserve cash for capital spending commitments.",
    "Intel won a multiyear supply agreement with a major automotive manufacturer in Germany.",
    "Intel disclosed a cybersecurity vulnerability affecting several generations of server chips.",
    "Intel opened a research laboratory focused on photonic interconnects in Oregon last month.",
    "Intel settled a patent dispute with a rival over memory interface designs for a cash payment.",
]


def art(i: int, sentences: list[str], tickers: list[str] | None = None) -> Article:
    text = " ".join(sentences)
    return Article(
        id=f"a{i}",
        title=f"Title {i}",
        url=f"https://x/{i}",
        source="x",
        tickers=tickers or ["INTC"],
        text=text,
        is_truncated=False,
        word_count=len(text.split()),
    )


def hit(i: int, sentences: list[str], score: float = 1.0, tickers: list[str] | None = None) -> Hit:
    return Hit(
        article=art(i, sentences, tickers),
        score=score,
        raw_score=5.0,
        matched=2,
        passages=sentences,
    )


Q = ResolvedQuery(text="What's going on with Intel?", tickers=["INTC"], keywords=["intel"])


def test_not_covered_when_no_hits() -> None:
    ans = ExtractiveSummarizer(INFO).summarize(Q, [], [])
    assert ans.covered is False and ans.citations == [] and ans.backend == "extractive"
    assert ans.text.endswith(ADVISORY)


def test_not_covered_when_every_hit_has_no_eligible_passage() -> None:
    # Truncated MT Newswires stubs yield no prose sentence, so no candidate can be selected;
    # the alternative is rendering "From 0 articles on :" as if it were an answer.
    hits = [hit(0, []), hit(1, [], 0.8)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    assert ans.covered is False and ans.citations == []
    assert "From 0 articles" not in ans.text and "doesn't cover that" in ans.text


def test_every_bullet_is_verbatim_and_cited() -> None:
    hits = [hit(0, [S_EVENT]), hit(1, [S_OPINION], 0.8), hit(2, [S_PROMO], 0.5)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    bullets = [line[2:] for line in ans.text.splitlines() if line.startswith("- ")]
    assert bullets, ans.text
    for bullet in bullets:
        sentence, _, n = bullet.rpartition(" [")
        idx = int(n.rstrip("]")) - 1
        assert sentence in hits[idx].article.text


def test_grouped_under_headings_in_fixed_order() -> None:
    hits = [hit(0, [S_PROMO]), hit(1, [S_OPINION], 0.8), hit(2, [S_EVENT], 0.6)]
    text = ExtractiveSummarizer(INFO).summarize(Q, hits, []).text
    positions = [text.index(KIND_HEADINGS[k]) for k in ("event", "opinion", "promo")]
    assert positions == sorted(positions)


def test_empty_groups_are_omitted() -> None:
    text = ExtractiveSummarizer(INFO).summarize(Q, [hit(0, [S_EVENT])], []).text
    assert KIND_HEADINGS["event"] in text
    assert KIND_HEADINGS["opinion"] not in text and KIND_HEADINGS["promo"] not in text


def test_lead_names_tickers_of_cited_hits() -> None:
    # Hit 2 supplies no passage, so it is never cited — and its ticker must not appear in
    # the lead line either, which has to agree with the Sources list underneath it.
    hits = [hit(0, [S_EVENT], tickers=["INTC", "MSFT"]), hit(1, [], 0.8, tickers=["NVDA"])]
    text = ExtractiveSummarizer(INFO).summarize(Q, hits, []).text
    assert "From 1 article on INTC, MSFT:" in text
    assert "NVDA" not in text and "[2] Title 1" not in text


def test_near_duplicates_are_suppressed() -> None:
    hits = [hit(0, [S_EVENT]), hit(1, [S_EVENT_DUP], 0.9), hit(2, [S_OPINION], 0.8)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    bullets = [line for line in ans.text.splitlines() if line.startswith("- ")]
    assert len(bullets) == 2
    # Identity, not numbering: the duplicate article is the one that must be absent.
    # Numbering is renumbered densely, so asserting {1, 3} here would re-encode a gap.
    assert {c.article_id for c in ans.citations} == {"a0", "a2"}


def test_citation_numbers_are_dense_when_a_hit_contributes_no_sentence() -> None:
    """A retrieved-but-uncited hit must not leave a hole in the numbering.

    Hit 2 is a near-duplicate of hit 1, so MMR drops it. If citations keep their hit rank,
    the reader sees Sources jump [1] [3] and reasonably concludes a source is missing.
    """
    hits = [hit(0, [S_EVENT]), hit(1, [S_EVENT_DUP], 0.9), hit(2, [S_OPINION], 0.8)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    assert [c.n for c in ans.citations] == [1, 2]
    assert sorted({int(m) for m in re.findall(r"\[(\d+)\]", ans.text)}) == [1, 2]


def test_promo_is_down_ranked_below_event_with_equal_overlap() -> None:
    hits = [hit(0, [S_PROMO]), hit(1, [S_EVENT], 1.0)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    first_bullet = next(line for line in ans.text.splitlines() if line.startswith("- "))
    assert first_bullet.endswith("[2]")


def test_citations_only_for_cited_hits_with_kinds() -> None:
    hits = [hit(0, [S_EVENT]), hit(1, [S_OPINION], 0.8)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    assert [c.n for c in ans.citations] == [1, 2]
    assert [c.kind for c in ans.citations] == ["event", "opinion"]
    assert all(c.verified for c in ans.citations)
    assert ans.citations[0].passage == S_EVENT


def test_max_sentences_respected() -> None:
    hits = [hit(i, [DISTINCT[i]]) for i in range(8)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    assert sum(line.startswith("- ") for line in ans.text.splitlines()) == 5


def test_sources_section_lists_cited_titles() -> None:
    ans = ExtractiveSummarizer(INFO).summarize(Q, [hit(0, [S_EVENT])], [])
    assert "Sources" in ans.text and "[1] Title 0 (INTC)" in ans.text


def test_jaccard() -> None:
    assert jaccard({"a", "b"}, {"b", "c"}) == pytest.approx(1 / 3)
    assert jaccard(set(), set()) == 0.0


def test_deterministic_on_real_corpus(corpus: Corpus) -> None:
    retriever = Retriever(corpus)
    summarizer = ExtractiveSummarizer(CorpusInfo(corpus.infer_period(), tuple(corpus.tickers)))
    q = QueryAnalyzer().resolve("What's going on with Intel?", [])
    a = summarizer.summarize(q, retriever.search(q), [])
    b = summarizer.summarize(q, retriever.search(q), [])
    assert a == b and a.covered and a.citations
    for c in a.citations:
        assert c.passage in corpus.by_id[c.article_id].text


def test_no_golden_question_yields_a_truncated_bullet(corpus: Corpus) -> None:
    """Every bullet the real corpus can produce ends in a terminator.

    The MT Newswires paywall stubs are cut mid-word around 100 characters; before the
    terminal-punctuation guard in `is_prose` they surfaced as bullets like
    "…a benchmark of more than 1,400 free [1]".
    """
    terminators = (".", "!", "?", '"', "\u201d", "\u2019", ")")
    analyzer, retriever = QueryAnalyzer(), Retriever(corpus)
    summarizer = ExtractiveSummarizer(CorpusInfo(corpus.infer_period(), tuple(corpus.tickers)))
    offenders: list[str] = []
    for case in load_golden(GOLDEN).cases:
        q = analyzer.resolve(case.question, [])
        answer = summarizer.summarize(q, retriever.search(q), [])
        for line in answer.text.splitlines():
            if not line.startswith("- "):
                continue
            sentence = line[2:].rpartition(" [")[0]
            if not sentence.endswith(terminators):
                offenders.append(f"{case.id}: {sentence!r}")
    assert offenders == []


S_ADVICE = "Should you buy Intel stock hand over fist before Feb. 19 and not a day later?"


def test_advice_is_down_ranked_below_event_with_equal_overlap() -> None:
    # Both sentences mention Intel once, so overlap alone ties them; only the advice
    # penalty separates them. An answer that opens with "Should you buy…" reads as a
    # recommendation no matter what the advisory footer says.
    hits = [hit(0, [S_ADVICE]), hit(1, [S_EVENT], 1.0)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    bullets = [line for line in ans.text.splitlines() if line.startswith("- ")]
    assert bullets[0].endswith("[2]") and S_EVENT in bullets[0]
    assert S_ADVICE not in bullets[0]
