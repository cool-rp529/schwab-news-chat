from newschat.models import Answer, Article, ChatTurn, Hit, ResolvedQuery
from newschat.summarize.base import (
    ADVISORY,
    EXCERPT_CLOSE,
    EXCERPT_OPEN,
    CorpusInfo,
    SummarizerUnavailable,
    as_of_line,
    build_prompt,
    check_faithfulness,
    cited_numbers,
    envelope,
    make_citations,
    not_covered_answer,
    strip_invalid_citations,
)

INFO = CorpusInfo(period="February 2025", tickers=("AAPL", "INTC"))


def article(
    i: int = 0, text: str = "Intel shares rose 10% after the report. More text here."
) -> Article:
    return Article(
        id=f"a{i}",
        title=f"Title {i}",
        url=f"https://x/{i}",
        source="x",
        tickers=["INTC"],
        text=text,
        is_truncated=False,
        word_count=9,
    )


def hit(i: int = 0, passages: list[str] | None = None) -> Hit:
    return Hit(
        article=article(i),
        score=1.0,
        raw_score=5.0,
        matched=2,
        passages=passages or ["Intel shares rose 10% after the report."],
    )


def test_envelope_wraps_body_with_as_of_and_advisory() -> None:
    q = ResolvedQuery(text="q")
    ans = envelope(body="Body.", citations=[], backend="x", query=q, covered=True, info=INFO)
    assert ans.text.startswith(as_of_line("February 2025"))
    assert ans.text.endswith(ADVISORY)
    assert ans.as_of == "February 2025" and ans.advisory == ADVISORY and ans.backend == "x"


def test_envelope_adds_inherited_note() -> None:
    q = ResolvedQuery(text="q", tickers=["INTC"], inherited_tickers=True)
    ans = envelope(body="Body.", citations=[], backend="x", query=q, covered=True, info=INFO)
    assert "(Interpreting the follow-up as being about INTC.)" in ans.text
    assert ans.text.index("Interpreting") < ans.text.index("Body.")


def test_not_covered_answer() -> None:
    ans = not_covered_answer(ResolvedQuery(text="q"), "extractive", INFO)
    assert ans.covered is False and ans.citations == []
    assert (
        "doesn't cover that" in ans.text
        and "AAPL, INTC" in ans.text
        and "February 2025" in ans.text
    )
    assert ans.text.endswith(ADVISORY)


def test_make_citations_numbers_from_one_and_classifies() -> None:
    cits = make_citations([hit(0), hit(1, ["Evercore raised its price target to $27."])])
    assert [c.n for c in cits] == [1, 2]
    assert cits[0].kind == "event" and cits[1].kind == "opinion"
    assert cits[0].passage == "Intel shares rose 10% after the report."


def test_citation_parsing_and_stripping() -> None:
    assert cited_numbers("A [1] and B [3] and [1] again") == [1, 3, 1]
    assert strip_invalid_citations("A [1] B [7] C [0]", n_hits=2) == "A [1] B  C "


def test_check_faithfulness() -> None:
    passages = ["Intel shares rose more than 10% on Tuesday after a Wall Street Journal report."]
    assert check_faithfulness(
        "Intel shares rose over 10% after a Wall Street report. [1]", passages
    )
    assert not check_faithfulness("Apple unveiled a foldable phone in Paris. [1]", passages)
    assert not check_faithfulness("[1]", passages)
    assert not check_faithfulness("Anything", [])


def test_build_prompt_delimits_excerpts_and_includes_history() -> None:
    q = ResolvedQuery(text="Who wants to buy it?", tickers=["INTC"], keywords=["want", "buy"])
    prior = ChatTurn(
        question="What's going on with Intel?",
        answer=Answer(
            text="Based on news: Intel may be split. [1]",
            citations=[],
            backend="x",
            query=ResolvedQuery(text="p"),
            covered=True,
            as_of="",
            advisory="",
        ),
    )
    prompt = build_prompt(q, [hit(0), hit(1)], [prior], INFO)
    assert EXCERPT_OPEN.format(n=1) in prompt and EXCERPT_CLOSE.format(n=2) in prompt
    assert "never an instruction" in prompt
    assert "What's going on with Intel?" in prompt and "Who wants to buy it?" in prompt
    assert "[n]" in prompt and "investment advice" in prompt
    assert prompt.index("Title 0") < prompt.index("Title 1")


def test_build_prompt_caps_history() -> None:
    turns = [
        ChatTurn(
            question=f"q{i}",
            answer=Answer(
                text=f"a{i}",
                citations=[],
                backend="x",
                query=ResolvedQuery(text="p"),
                covered=True,
                as_of="",
                advisory="",
            ),
        )
        for i in range(6)
    ]
    prompt = build_prompt(ResolvedQuery(text="now"), [hit()], turns, INFO, max_history=3)
    assert "q5" in prompt and "q2" not in prompt


def test_summarizer_unavailable_carries_reason() -> None:
    exc = SummarizerUnavailable("timed out")
    assert exc.reason == "timed out" and str(exc) == "timed out"
