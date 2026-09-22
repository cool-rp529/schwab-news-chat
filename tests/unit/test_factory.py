import pytest

from newschat.summarize.base import CorpusInfo
from newschat.summarize.claude_cli import ClaudeCliSummarizer
from newschat.summarize.extractive import ExtractiveSummarizer
from newschat.summarize.factory import BACKENDS, build_summarizer

INFO = CorpusInfo(period="February 2025", tickers=("INTC",))


def test_default_is_extractive_even_when_claude_is_on_path(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("shutil.which", lambda _: "C:/bin/claude.EXE")
    primary, fallback = build_summarizer(INFO, env={})
    assert isinstance(primary, ExtractiveSummarizer) and primary is fallback


def test_env_opts_in_to_cli() -> None:
    primary, fallback = build_summarizer(
        INFO, env={"NEWSCHAT_BACKEND": "claude-cli", "NEWSCHAT_CLAUDE_MODEL": "haiku"}
    )
    assert isinstance(primary, ClaudeCliSummarizer) and primary.config.model == "haiku"
    assert isinstance(fallback, ExtractiveSummarizer)


def test_explicit_argument_wins_over_env() -> None:
    primary, _ = build_summarizer(
        INFO, backend="extractive", env={"NEWSCHAT_BACKEND": "claude-cli"}
    )
    assert isinstance(primary, ExtractiveSummarizer)


def test_unknown_backend() -> None:
    with pytest.raises(ValueError, match="unknown backend 'gpt'"):
        build_summarizer(INFO, backend="gpt", env={})


def test_backends_constant() -> None:
    assert BACKENDS == ("extractive", "claude-cli")
