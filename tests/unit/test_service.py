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


def test_out_of_corpus_question_mid_session_is_refused_not_inherited(
    service: ChatService,
) -> None:
    # Before the vocabulary check, this short question inherited INTC from the Intel turn
    # and came back with five confidently cited Intel articles.
    sid = SessionStore.new_id()
    service.ask(sid, "What's going on with Intel?")
    ans = service.ask(sid, "What is the price of Bitcoin today?")
    assert ans.covered is False and ans.citations == []
    assert ans.query.inherited_tickers is False


def test_inheritance_walks_back_past_a_refused_turn(service: ChatService) -> None:
    # The Bitcoin turn recorded no tickers, so a genuine follow-up after it still
    # resolves to the Intel turn rather than losing the thread.
    sid = SessionStore.new_id()
    service.ask(sid, "What's going on with Intel?")
    service.ask(sid, "What is the price of Bitcoin today?")
    ans = service.ask(sid, "Who wants to buy it?")
    assert ans.query.inherited_tickers is True and ans.query.tickers == ["INTC"]


# ---- follow-up chaining and inheritance strength, on the real corpus ----


def test_pronoun_less_new_topic_is_not_locked_to_the_previous_company(
    service: ChatService, corpus: Corpus
) -> None:
    # "DeepSeek" is one known word, so it soft-inherits INTC. Before soft inheritance it was
    # restricted to Intel articles and came back with the identical Intel answer.
    sid = SessionStore.new_id()
    intel = service.ask(sid, "What's going on with Intel?")
    ans = service.ask(sid, "DeepSeek")
    assert ans.covered and ans.citations
    assert ans.query.inheritance == "soft"
    assert "Reading this as a follow-up about INTC" in ans.text
    intel_titles = {c.title for c in intel.citations}
    titles = {c.title for c in ans.citations}
    assert any("deepseek" in t.lower() for t in titles) or not titles <= intel_titles
    bodies = {a.id: a.text.lower() for a in corpus.articles}
    assert all("deepseek" in bodies[c.article_id] for c in ans.citations)


def test_soft_follow_up_keeps_the_company_when_it_fits(service: ChatService) -> None:
    sid = SessionStore.new_id()
    service.ask(sid, "What do analysts expect from Nvidia's earnings?")
    ans = service.ask(sid, "And the price target?")
    assert ans.covered and ans.citations
    assert ans.query.tickers == ["NVDA"] and ans.query.inheritance == "soft"
    assert "nvidia" in ans.citations[0].title.lower()


def test_topic_survives_a_refused_turn_and_a_repeated_follow_up(service: ChatService) -> None:
    # The second "who wants to buy it?" used to inherit only [want, buy], lose "intel", and
    # surface "Buy rating" pieces on ASML and AMD.
    sid = SessionStore.new_id()
    service.ask(sid, "What's going on with Intel?")
    service.ask(sid, "Who wants to buy it?")
    assert service.ask(sid, "What is the price of Bitcoin today?").covered is False
    ans = service.ask(sid, "Who wants to buy it?")
    assert ans.covered and ans.citations
    assert ans.query.inheritance == "strong" and "intel" in ans.query.context_keywords
    assert all("INTC" in c.tickers for c in ans.citations), [c.title for c in ans.citations]
    assert not any("ASML" in c.title or "AMD" in c.title for c in ans.citations)


def test_out_of_corpus_question_does_not_inherit_at_all(service: ChatService) -> None:
    sid = SessionStore.new_id()
    service.ask(sid, "What's going on with Intel?")
    ans = service.ask(sid, "What is the price of Bitcoin today?")
    assert ans.covered is False and ans.query.inheritance == "none"
