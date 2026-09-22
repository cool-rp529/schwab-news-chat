import os
import shutil

import pytest

from newschat.ingest.corpus import Corpus
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever
from newschat.summarize.base import CorpusInfo
from newschat.summarize.claude_cli import ClaudeCliConfig, ClaudeCliSummarizer

pytestmark = pytest.mark.llm
_ENABLED = os.environ.get("NEWSCHAT_RUN_LLM_TESTS") == "1" and shutil.which("claude") is not None


@pytest.mark.skipif(not _ENABLED, reason="set NEWSCHAT_RUN_LLM_TESTS=1 with claude on PATH")
def test_live_answer_is_cited_and_enveloped(corpus: Corpus) -> None:
    q = QueryAnalyzer().resolve("Who might buy parts of Intel?", [])
    hits = Retriever(corpus).search(q)
    summarizer = ClaudeCliSummarizer(
        CorpusInfo(corpus.infer_period(), tuple(corpus.tickers)),
        ClaudeCliConfig(model=os.environ.get("NEWSCHAT_CLAUDE_MODEL", "haiku"), timeout_s=120),
    )
    ans = summarizer.summarize(q, hits, [])
    assert ans.covered and ans.citations
    assert all(1 <= c.n <= len(hits) for c in ans.citations)
    assert ans.text.startswith("Based on news from") and ans.text.endswith("not investment advice.")
