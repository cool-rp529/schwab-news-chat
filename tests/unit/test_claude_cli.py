import json
import subprocess
from collections.abc import Callable
from typing import Any

import pytest

from newschat.models import Article, Hit, ResolvedQuery
from newschat.summarize.base import EXCERPT_OPEN, CorpusInfo, SummarizerUnavailable
from newschat.summarize.claude_cli import ClaudeCliConfig, ClaudeCliSummarizer, _claims_with_markers

INFO = CorpusInfo(period="February 2025", tickers=("INTC",))
Q = ResolvedQuery(
    text="Who wants to buy Intel?", tickers=["INTC"], keywords=["want", "buy", "intel"]
)


def hit(i: int, passage: str) -> Hit:
    art = Article(
        id=f"a{i}",
        title=f"Title {i}",
        url=f"https://x/{i}",
        source="x",
        tickers=["INTC"],
        text=passage,
        is_truncated=False,
        word_count=len(passage.split()),
    )
    return Hit(article=art, score=1.0, raw_score=6.0, matched=2, passages=[passage])


HITS = [
    hit(0, "Intel may be split up by rivals Broadcom and TSMC after a Wall Street Journal report."),
    hit(1, "Broadcom is interested in Intel's chip design unit, while TSMC eyes the factories."),
]


HITS3 = [
    hit(0, "Intel may be split up by rivals Broadcom and TSMC after a Wall Street Journal report."),
    hit(1, "An unrelated sentence about quarterly guidance that the model chooses not to cite."),
    hit(2, "Broadcom is interested in Intel's chip design unit, while TSMC eyes the factories."),
]


def ok_payload(result: str, **extra: Any) -> str:
    payload: dict[str, Any] = {
        "type": "result",
        "subtype": "success",
        "is_error": False,
        "api_error_status": None,
        "result": result,
    }
    payload.update(extra)
    return json.dumps(payload)


def runner_returning(stdout: str, returncode: int = 0, stderr: str = "") -> Callable[..., Any]:
    calls: list[dict[str, Any]] = []

    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls.append({"args": args, **kwargs})
        return subprocess.CompletedProcess(args, returncode, stdout, stderr)

    run.calls = calls  # type: ignore[attr-defined]
    return run


@pytest.fixture(autouse=True)
def claude_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("newschat.summarize.claude_cli.shutil.which", lambda _: "C:/bin/claude.EXE")


def test_success_builds_verified_citations() -> None:
    run = runner_returning(
        ok_payload(
            "Intel may be split up by Broadcom and TSMC. [1] "
            "Broadcom wants the chip design unit. [2]"
        )
    )
    ans = ClaudeCliSummarizer(INFO, runner=run).summarize(Q, HITS, [])
    assert ans.backend == "claude-cli" and ans.covered
    assert [c.n for c in ans.citations] == [1, 2]
    assert all(c.verified for c in ans.citations)
    assert ans.text.startswith("Based on news from ≈ February 2025:")


def test_invocation_arguments_and_prompt() -> None:
    run = runner_returning(ok_payload("Claim. [1]"))
    ClaudeCliSummarizer(INFO, ClaudeCliConfig(model="haiku", timeout_s=7), runner=run).summarize(
        Q, HITS, []
    )
    (call,) = run.calls  # type: ignore[attr-defined]
    args = call["args"]
    assert args[0] == "C:/bin/claude.EXE"
    assert args[1:5] == ["-p", "--output-format", "json", "--tools"] and args[5] == ""
    assert "--no-session-persistence" in args and args[-2:] == ["--model", "haiku"]
    assert call["timeout"] == 7 and call["text"] is True and call["encoding"] == "utf-8"
    assert EXCERPT_OPEN.format(n=1) in call["input"] and "Who wants to buy Intel?" in call["input"]


def test_missing_binary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("newschat.summarize.claude_cli.shutil.which", lambda _: None)
    with pytest.raises(SummarizerUnavailable, match="not found"):
        ClaudeCliSummarizer(INFO, runner=runner_returning("")).summarize(Q, HITS, [])


def test_nonzero_exit() -> None:
    with pytest.raises(SummarizerUnavailable, match="exit code 1"):
        ClaudeCliSummarizer(INFO, runner=runner_returning("", 1, "boom")).summarize(Q, HITS, [])


def test_timeout() -> None:
    def run(args: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(args, kwargs["timeout"])

    with pytest.raises(SummarizerUnavailable, match="timed out"):
        ClaudeCliSummarizer(INFO, runner=run).summarize(Q, HITS, [])


def test_invalid_json() -> None:
    with pytest.raises(SummarizerUnavailable, match="invalid JSON"):
        ClaudeCliSummarizer(INFO, runner=runner_returning("not json")).summarize(Q, HITS, [])


@pytest.mark.parametrize(
    "payload",
    [
        ok_payload("Usage limit reached", is_error=True),
        ok_payload("x", subtype="error_max_turns"),
        ok_payload("x", api_error_status=529),
        ok_payload(""),
        json.dumps({"type": "result", "subtype": "success", "is_error": False}),
        json.dumps([1, 2]),
    ],
)
def test_in_band_failures_raise(payload: str) -> None:
    with pytest.raises(SummarizerUnavailable):
        ClaudeCliSummarizer(INFO, runner=runner_returning(payload)).summarize(Q, HITS, [])


def test_out_of_range_citation_is_stripped() -> None:
    run = runner_returning(
        ok_payload("Intel may be split by Broadcom and TSMC. [1] Something else. [9]")
    )
    ans = ClaudeCliSummarizer(INFO, runner=run).summarize(Q, HITS, [])
    assert "[9]" not in ans.text and [c.n for c in ans.citations] == [1]


def test_unfaithful_claim_marks_citation_unverified() -> None:
    run = runner_returning(ok_payload("Apple unveiled a foldable phone in Paris yesterday. [2]"))
    ans = ClaudeCliSummarizer(INFO, runner=run).summarize(Q, HITS, [])
    assert [c.verified for c in ans.citations] == [False]


def test_advice_shaped_output_is_rejected() -> None:
    run = runner_returning(ok_payload("Intel is a screaming buy right now. [1]"))
    with pytest.raises(SummarizerUnavailable, match="advice"):
        ClaudeCliSummarizer(INFO, runner=run).summarize(Q, HITS, [])


def test_output_without_citations_is_rejected() -> None:
    run = runner_returning(ok_payload("Intel may be split up."))
    with pytest.raises(SummarizerUnavailable, match="citation"):
        ClaudeCliSummarizer(INFO, runner=run).summarize(Q, HITS, [])


def test_no_hits_short_circuits_without_calling_cli() -> None:
    run = runner_returning(ok_payload("should not be called"))
    ans = ClaudeCliSummarizer(INFO, runner=run).summarize(Q, [], [])
    assert ans.covered is False and run.calls == []  # type: ignore[attr-defined]


def test_claims_with_markers_reattaches_leading_citations() -> None:
    assert _claims_with_markers("Intel may be split. [1] Broadcom is interested. [2]") == [
        "Intel may be split. [1]",
        "Broadcom is interested. [2]",
    ]


def test_llm_citations_are_renumbered_densely() -> None:
    """The model may cite excerpts 1 and 3 and skip 2; the reader must not see a gap.

    Excerpt numbers are an internal prompt detail. Once the answer is rendered, [1] [3]
    reads as a missing source, and the extractive path already renumbers densely -- the
    two backends must not disagree about what a citation number means.
    """
    result = "Broadcom may bid for the design unit [1]. TSMC is looking at the factories [3]."
    summarizer = ClaudeCliSummarizer(INFO, runner=runner_returning(ok_payload(result)))
    ans = summarizer.summarize(Q, HITS3, [])
    assert [c.n for c in ans.citations] == [1, 2]
    # [2] must now point at the article that was excerpt 3.
    assert ans.citations[1].article_id == "a2"
    assert "[3]" not in ans.text
    assert "[1]" in ans.text and "[2]" in ans.text
