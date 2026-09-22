"""Backend selection. Explicit opt-in only: a reviewer who happens to have Claude Code
installed must not be routed onto a network-dependent path the test suite does not cover."""

import os
from collections.abc import Mapping

from newschat.summarize.base import CorpusInfo, Summarizer
from newschat.summarize.claude_cli import ClaudeCliConfig, ClaudeCliSummarizer
from newschat.summarize.extractive import ExtractiveSummarizer

BACKENDS = ("extractive", "claude-cli")


def build_summarizer(
    info: CorpusInfo, backend: str | None = None, env: Mapping[str, str] | None = None
) -> tuple[Summarizer, ExtractiveSummarizer]:
    env = os.environ if env is None else env
    name = (backend or env.get("NEWSCHAT_BACKEND") or "extractive").strip().lower()
    fallback = ExtractiveSummarizer(info)
    if name == "extractive":
        return fallback, fallback
    if name == "claude-cli":
        config = ClaudeCliConfig(model=env.get("NEWSCHAT_CLAUDE_MODEL") or None)
        return ClaudeCliSummarizer(info, config), fallback
    raise ValueError(f"unknown backend {name!r}; expected one of {BACKENDS}")
