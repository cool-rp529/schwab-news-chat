"""Turn a question (plus recent history) into tickers and keywords.

Alias tokens stay in `keywords`: "apple" is what separates Apple's 20 articles from the
rest. Follow-ups inherit both the previous turn's tickers and its topic — that turn's own
keywords plus the context it had itself inherited, so the topic chains down a thread instead
of being replaced by the last follow-up's words ("who wants to buy it?" twice in a row still
carries "intel"). Inheriting the ticker alone restores *which company* but loses *what the
conversation was about*.

Inheritance comes in two strengths:

* **strong** — the question is anaphoric (a pronoun or "the company"-style reference). The
  retriever restricts to the inherited tickers first and lets them bypass the refusal gate.
* **soft** — no reference word, but the question is short *and* every keyword is in the corpus
  vocabulary ("And the price target?", "DeepSeek"). The tickers are only a ranking preference,
  so a short new topic the corpus covers is not hard-locked to the previous company.

A short question containing a word the corpus has never seen is about something else and does
not inherit at all: inheriting a ticker into it would smuggle it past the refusal gate.

A reference word with nothing to refer to ("Who might buy it?" as the first question, or right
after "new topic") inherits nothing and is flagged `unresolved_reference`, so the chat service
asks which company is meant instead of searching for "buy" across the whole corpus.
"""

import re
from collections.abc import Collection, Sequence

from newschat.models import ChatTurn, Inheritance, ResolvedQuery
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

# The soft-inheritance note tells the reader to say "new topic" to drop the previous company.
# A turn containing it also ends the thread: later follow-ups never inherit from behind it.
_RESET_RE = re.compile(r"\bnew topic\b", re.IGNORECASE)
_RESET_TOKENS = frozenset(tokenize("new topic"))


def _strip_dollar(token: str) -> str:
    """Turn a cashtag like "$intc" into "intc", the index's plain token; "$1.5" stays a price."""
    rest = token[1:]
    return rest if token.startswith("$") and rest.isalpha() else token


def is_reset(question: str) -> bool:
    """Whether the question contains the "new topic" phrase that ends the current thread."""
    return _RESET_RE.search(question) is not None


class QueryAnalyzer:
    def __init__(
        self,
        max_short_tokens: int = 4,
        known_terms: Collection[str] | None = None,
        max_context_keywords: int = 8,
    ) -> None:
        self.max_short_tokens = max_short_tokens
        # Carried context is ordered most recent turn first; the cap drops the oldest words.
        self.max_context_keywords = max_context_keywords
        # None means "no vocabulary check": short questions inherit on length alone.
        self.known_terms = known_terms

    def resolve(self, question: str, history: Sequence[ChatTurn]) -> ResolvedQuery:
        text = question.strip()
        tickers = find_tickers(text)
        tokens = tokenize(text)
        reset = is_reset(text)
        keywords = list(
            dict.fromkeys(
                _strip_dollar(t)
                for t in tokens
                if t not in QUERY_STOPWORDS and not (reset and t in _RESET_TOKENS)
            )
        )
        context: list[str] = []
        inheritance: Inheritance = "none"
        unresolved = False
        strength: Inheritance = (
            "none" if tickers else self._follow_up_strength(text, tokens, keywords)
        )
        if strength != "none":
            prior = self._last_with_tickers(history)
            if prior is not None:
                tickers = list(prior.tickers)
                inheritance = strength
                topic = dict.fromkeys([*prior.keywords, *prior.context_keywords])
                context = [k for k in topic if k not in keywords][: self.max_context_keywords]
            elif strength == "strong":
                unresolved = True
        return ResolvedQuery(
            text=text,
            tickers=tickers,
            keywords=keywords,
            context_keywords=context,
            inherited_tickers=inheritance != "none",
            inheritance=inheritance,
            unresolved_reference=unresolved,
        )

    def is_anaphoric(self, question: str, tokens: Sequence[str]) -> bool:
        """Whether the question *looks* like a follow-up: a reference word, or just short."""
        return bool(_ANAPHORA_RE.search(question)) or len(tokens) <= self.max_short_tokens

    def _follow_up_strength(
        self, question: str, tokens: Sequence[str], keywords: Sequence[str]
    ) -> Inheritance:
        if _RESET_RE.search(question):
            return "none"
        if _ANAPHORA_RE.search(question):
            return "strong"
        short = len(tokens) <= self.max_short_tokens
        return "soft" if short and self._is_known_short_question(keywords) else "none"

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
            if is_reset(turn.question):
                return None
            if turn.answer.query.tickers:
                return turn.answer.query
        return None
