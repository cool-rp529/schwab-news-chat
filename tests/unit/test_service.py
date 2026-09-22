from collections.abc import Sequence

import pytest

from newschat.chat.service import MAX_QUESTION_CHARS, ChatService
from newschat.chat.session import SessionStore
from newschat.ingest.corpus import Corpus
from newschat.models import Answer, ChatTurn, Hit, ResolvedQuery
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever
from newschat.summarize.base import ADVISORY, CorpusInfo, SummarizerUnavailable
from newschat.summarize.extractive import ExtractiveSummarizer


class Broken:
    name = "broken"

    def summarize(
        self, query: ResolvedQuery, hits: Sequence[Hit], history: Sequence[ChatTurn]
    ) -> Answer:
        raise SummarizerUnavailable("kaput")


@pytest.fixture(scope="module")
def service(corpus: Corpus) -> ChatService:
    return ChatService.from_corpus(corpus)


def test_from_corpus_defaults_to_extractive(service: ChatService) -> None:
    assert service.backend == "extractive" and service.info.period == "February 2025"


def test_ask_returns_grounded_answer(service: ChatService) -> None:
    ans = service.ask(SessionStore.new_id(), "What's going on with Intel?")
    assert ans.covered and ans.citations and ans.text.endswith(ADVISORY)
    assert ans.query.tickers == ["INTC"]


def test_follow_up_uses_session_history(service: ChatService) -> None:
    sid = SessionStore.new_id()
    service.ask(sid, "What's going on with Intel?")
    ans = service.ask(sid, "Who wants to buy it?")
    assert ans.query.inherited_tickers and ans.query.tickers == ["INTC"]
    assert "(Interpreting the follow-up as being about INTC.)" in ans.text
    assert ans.query.context_keywords == ["intel"]


def test_separate_sessions_do_not_share_history(service: ChatService) -> None:
    service.ask(SessionStore.new_id(), "What's going on with Intel?")
    ans = service.ask(SessionStore.new_id(), "Who wants to buy it?")
    assert ans.query.inherited_tickers is False


@pytest.mark.parametrize("question", ["", "   ", "x" * (MAX_QUESTION_CHARS + 1)])
def test_rejects_blank_and_overlong(service: ChatService, question: str) -> None:
    with pytest.raises(ValueError):
        service.ask(SessionStore.new_id(), question)


def test_fallback_when_primary_unavailable(corpus: Corpus) -> None:
    info = CorpusInfo(corpus.infer_period(), tuple(corpus.tickers))
    svc = ChatService(
        QueryAnalyzer(),
        Retriever(corpus),
        Broken(),
        ExtractiveSummarizer(info),
        SessionStore(),
        info,
    )
    assert svc.backend == "broken"
    ans = svc.ask(SessionStore.new_id(), "What's going on with Intel?")
    assert ans.backend == "extractive (fallback: kaput)" and ans.covered


def test_not_covered_question_gets_refusal(service: ChatService) -> None:
    ans = service.ask(SessionStore.new_id(), "What is the price of Bitcoin today?")
    assert ans.covered is False and ans.citations == [] and "doesn't cover that" in ans.text


def test_advice_question_still_carries_advisory(service: ChatService) -> None:
    ans = service.ask(SessionStore.new_id(), "Should I buy Nvidia stock?")
    assert ans.text.endswith(ADVISORY)
    assert ans.advisory == ADVISORY
