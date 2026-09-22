import pytest

from newschat.ingest.corpus import Corpus
from newschat.models import Article, Hit, ResolvedQuery
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever
from newschat.summarize.base import ADVISORY, CorpusInfo
from newschat.summarize.extractive import KIND_HEADINGS, ExtractiveSummarizer, jaccard

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
    "Intel settled a patent dispute with a rival over memory controller designs for a cash "
    "payment.",
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


def test_lead_names_returned_tickers_not_requested() -> None:
    hits = [hit(0, [S_EVENT], tickers=["INTC", "MSFT"])]
    text = ExtractiveSummarizer(INFO).summarize(Q, hits, []).text
    assert "From 1 article on INTC, MSFT:" in text


def test_near_duplicates_are_suppressed() -> None:
    hits = [hit(0, [S_EVENT]), hit(1, [S_EVENT_DUP], 0.9), hit(2, [S_OPINION], 0.8)]
    ans = ExtractiveSummarizer(INFO).summarize(Q, hits, [])
    bullets = [line for line in ans.text.splitlines() if line.startswith("- ")]
    assert len(bullets) == 2
    assert {c.n for c in ans.citations} == {1, 3}


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
