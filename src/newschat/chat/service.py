"""Orchestration: resolve → retrieve → summarize (with fallback) → record the turn."""

import logging

from newschat.chat.session import SessionStore
from newschat.ingest.corpus import Corpus
from newschat.models import Answer, ChatTurn
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever
from newschat.summarize.base import CorpusInfo, Summarizer, SummarizerUnavailable
from newschat.summarize.extractive import ExtractiveSummarizer
from newschat.summarize.factory import build_summarizer

log = logging.getLogger(__name__)

MAX_QUESTION_CHARS = 1000


class ChatService:
    def __init__(
        self,
        analyzer: QueryAnalyzer,
        retriever: Retriever,
        primary: Summarizer,
        fallback: ExtractiveSummarizer,
        sessions: SessionStore,
        info: CorpusInfo,
        max_question_chars: int = MAX_QUESTION_CHARS,
    ) -> None:
        self.analyzer = analyzer
        self.retriever = retriever
        self.primary = primary
        self.fallback = fallback
        self.sessions = sessions
        self.info = info
        self.max_question_chars = max_question_chars

    @classmethod
    def from_corpus(cls, corpus: Corpus, backend: str | None = None) -> "ChatService":
        info = CorpusInfo(period=corpus.infer_period(), tickers=tuple(corpus.tickers))
        primary, fallback = build_summarizer(info, backend)
        retriever = Retriever(corpus)
        analyzer = QueryAnalyzer(known_terms=retriever.known_terms)
        return cls(analyzer, retriever, primary, fallback, SessionStore(), info)

    @property
    def backend(self) -> str:
        return self.primary.name

    def ask(self, session_id: str, question: str) -> Answer:
        text = question.strip()
        if not text:
            raise ValueError("question must not be blank")
        if len(text) > self.max_question_chars:
            raise ValueError(f"question must be at most {self.max_question_chars} characters")

        history = self.sessions.get(session_id)
        resolved = self.analyzer.resolve(text, history)
        hits = self.retriever.search(resolved)
        try:
            answer = self.primary.summarize(resolved, hits, list(history))
        except SummarizerUnavailable as exc:
            log.warning(
                "backend %s unavailable (%s); using extractive", self.primary.name, exc.reason
            )
            answer = self.fallback.summarize(resolved, hits, list(history))
            answer = answer.model_copy(update={"backend": f"extractive (fallback: {exc.reason})"})
        history.append(ChatTurn(question=text, answer=answer))
        return answer
