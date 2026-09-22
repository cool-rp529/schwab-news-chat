"""Turn a question (plus recent history) into tickers and keywords.

Alias tokens stay in `keywords`: "apple" is what separates Apple's 20 articles from the
rest. Follow-ups inherit both the previous turn's tickers and its keywords — inheriting the
ticker alone restores *which company* but loses *what the conversation was about*.
"""

import re
from collections.abc import Sequence

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
    need new best
    """
QUERY_STOPWORDS: frozenset[str] = frozenset(stem(w) for w in _QUERY_STOPWORDS_RAW.split())

_ANAPHORA_RE = re.compile(
    r"\b(?:it|its|they|them|their|that|this|those|these|he|she|his|her"
    r"|the company|the stock|the deal|the firm|the shares)\b",
    re.IGNORECASE,
)


class QueryAnalyzer:
    def __init__(self, max_short_tokens: int = 4) -> None:
        self.max_short_tokens = max_short_tokens

    def resolve(self, question: str, history: Sequence[ChatTurn]) -> ResolvedQuery:
        text = question.strip()
        tickers = find_tickers(text)
        tokens = tokenize(text)
        keywords = list(dict.fromkeys(t for t in tokens if t not in QUERY_STOPWORDS))
        context: list[str] = []
        inherited = False
        if not tickers and self.is_anaphoric(text, tokens):
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
        return bool(_ANAPHORA_RE.search(question)) or len(tokens) <= self.max_short_tokens

    @staticmethod
    def _last_with_tickers(history: Sequence[ChatTurn]) -> ResolvedQuery | None:
        for turn in reversed(history):
            if turn.answer.query.tickers:
                return turn.answer.query
        return None
