"""Turn a question (plus recent history) into tickers and keywords.

Alias tokens stay in `keywords`: "apple" is what separates Apple's 20 articles from the
rest. Follow-ups inherit both the previous turn's tickers and its keywords — inheriting the
ticker alone restores *which company* but loses *what the conversation was about*.

A question inherits only if it is anaphoric (a pronoun or "the company"-style reference) or
it is short *and* every keyword is in the corpus vocabulary: a short question containing a
word the corpus has never seen is about something else, and inheriting a ticker into it would
smuggle it past the retriever's refusal gate.
"""

import re
from collections.abc import Collection, Sequence

from newschat.models import ChatTurn, ResolvedQuery
from newschat.retrieval.aliases import find_tickers
from newschat.text.tokenize import stem, tokenize

# Question-shaped words that appear in most articles and carry no topical signal.
# Note: "want"/"wants" are deliberately excluded — they carry topical signal in
# questions like "who wants to buy it".
_QUERY_STOPWORDS_RAW = """
    tell know say says said give show explain summarize summarise summary overview describe
    news latest recent recently update updates happening happen happened going anything
    something everything stock stocks share shares company companies market markets today
    week month year currently right now please think thoughts mean means like
    need new best giant giants e team teaming teams
    """
QUERY_STOPWORDS: frozenset[str] = frozenset(stem(w) for w in _QUERY_STOPWORDS_RAW.split())

_ANAPHORA_RE = re.compile(
    r"\b(?:it|its|they|them|their|that|this|those|these|he|she|his|her"
    r"|the company|the stock|the deal|the firm|the shares)\b",
    re.IGNORECASE,
)


class QueryAnalyzer:
    def __init__(
        self, max_short_tokens: int = 4, known_terms: Collection[str] | None = None
    ) -> None:
        self.max_short_tokens = max_short_tokens
        # None means "no vocabulary check": short questions inherit on length alone.
        self.known_terms = known_terms

    def resolve(self, question: str, history: Sequence[ChatTurn]) -> ResolvedQuery:
        text = question.strip()
        tickers = find_tickers(text)
        tokens = tokenize(text)
        keywords = list(dict.fromkeys(t for t in tokens if t not in QUERY_STOPWORDS))
        context: list[str] = []
        inherited = False
        if not tickers and self._is_follow_up(text, tokens, keywords):
            prior = self._last_with_tickers(history)
            if prior is not None:
                tickers = list(prior.tickers)
                inherited = True
                context = [k for k in prior.keywords if k not in keywords]
        return ResolvedQuery(
            text=text,
            tickers=tickers,
            keywords=keywords,
            context_keywords=context,
            inherited_tickers=inherited,
        )

    def is_anaphoric(self, question: str, tokens: Sequence[str]) -> bool:
        """Whether the question *looks* like a follow-up: a reference word, or just short."""
        return bool(_ANAPHORA_RE.search(question)) or len(tokens) <= self.max_short_tokens

    def _is_follow_up(self, question: str, tokens: Sequence[str], keywords: Sequence[str]) -> bool:
        if _ANAPHORA_RE.search(question):
            return True
        short = len(tokens) <= self.max_short_tokens
        return short and self._is_known_short_question(keywords)

    def _is_known_short_question(self, keywords: Sequence[str]) -> bool:
        """Whether every word of a short question is one the corpus actually uses.

        With `known_terms=None` there is no vocabulary to check against, so every short
        question qualifies and the length rule stands on its own.
        """
        if self.known_terms is None:
            return True
        return all(k in self.known_terms for k in keywords)

    @staticmethod
    def _last_with_tickers(history: Sequence[ChatTurn]) -> ResolvedQuery | None:
        for turn in reversed(history):
            if turn.answer.query.tickers:
                return turn.answer.query
        return None
