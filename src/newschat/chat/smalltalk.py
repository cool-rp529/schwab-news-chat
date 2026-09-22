"""Conversational turns that are not questions about the news: greetings, thanks, "who are
you", "what can you do", goodbyes.

Without this, retrieval treats "thanks" as a search term and answers it with cited articles.
Only a whole message matches, so "Hello, what's going on with Intel?" is still a news question.
"""

import re
from typing import Literal

from newschat.retrieval.aliases import find_tickers
from newschat.summarize.base import CorpusInfo
from newschat.text.tokenize import normalize_text

Intent = Literal["greeting", "thanks", "identity", "help", "goodbye"]

MAX_WORDS = 6

_PATTERNS: dict[Intent, str] = {
    "greeting": r"(?:hi|hello|hey|hiya|good (?:morning|afternoon|evening|day))"
    r"(?: there| all| everyone)?",
    "thanks": r"(?:ok(?:ay)? )?(?:thanks|thank you|thx|ty|cheers|many thanks)"
    r"(?: (?:a lot|so much|very much|again))?",
    "identity": r"who are you|what is your name|what's your name|whats your name|what are you",
    "help": r"help|what can you do|what can i ask(?: you)?|how does this work",
    "goodbye": r"(?:ok(?:ay)? )?(?:bye|goodbye|bye bye|see you(?: later)?|see ya)",
}
_COMPILED: dict[Intent, re.Pattern[str]] = {
    intent: re.compile(r"[\W_]*(?:" + pattern + r")[\W_]*", re.IGNORECASE)
    for intent, pattern in _PATTERNS.items()
}


def detect(question: str) -> Intent | None:
    """The small-talk intent of a whole message, or None for anything that may be a question."""
    text = normalize_text(question).replace("’", "'")
    text = " ".join(re.sub(r"[,;:]", " ", text).split())
    if not text or len(text.split()) > MAX_WORDS or find_tickers(text):
        return None
    for intent, pattern in _COMPILED.items():
        if pattern.fullmatch(text):
            return intent
    return None


def reply(intent: Intent, info: CorpusInfo) -> str:
    tickers = ", ".join(info.tickers)
    if intent in ("help", "identity"):
        return (
            f"I summarise the news set on {tickers} from around {info.period}. Ask about a "
            "company, an event, or a person in the news, and I'll answer only from those "
            "articles with citations."
        )
    if intent == "greeting":
        return f'Hello. Ask about {tickers} — e.g. "What\'s going on with Intel?"'
    if intent == "goodbye":
        return "Goodbye. Ask another question whenever you like."
    return "You're welcome. Ask another question whenever you like."
