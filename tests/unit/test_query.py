from newschat.models import Answer, ChatTurn, ResolvedQuery
from newschat.retrieval.query import QUERY_STOPWORDS, QueryAnalyzer


def turn(question: str, tickers: list[str], keywords: list[str]) -> ChatTurn:
    q = ResolvedQuery(text=question, tickers=tickers, keywords=keywords)
    return ChatTurn(
        question=question,
        answer=Answer(
            text="",
            citations=[],
            backend="extractive",
            query=q,
            covered=True,
            as_of="",
            advisory="",
        ),
    )


def test_keywords_keep_alias_tokens_and_drop_query_stopwords() -> None:
    q = QueryAnalyzer().resolve("What's the latest news on Apple's iPhone SE 4?", [])
    assert q.tickers == ["AAPL"]
    assert q.keywords == ["apple", "iphone", "se", "4"]
    assert q.context_keywords == [] and q.inherited_tickers is False


def test_query_stopwords_are_stemmed_forms() -> None:
    assert {"happen", "stock", "company", "news", "latest"} <= QUERY_STOPWORDS


def test_follow_up_inherits_tickers_and_keywords() -> None:
    history = [turn("What's going on with Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer().resolve("Who wants to buy it?", history)
    assert q.tickers == ["INTC"] and q.inherited_tickers is True
    assert q.keywords == ["want", "buy"]
    assert q.context_keywords == ["intel"] and q.inherited_keywords is True


def test_short_question_without_anaphora_still_inherits() -> None:
    history = [turn("Nvidia earnings?", ["NVDA"], ["nvidia", "earning"])]
    q = QueryAnalyzer().resolve("And the price target?", history)
    assert q.tickers == ["NVDA"] and q.context_keywords == ["nvidia", "earning"]


def test_naming_a_ticker_blocks_inheritance() -> None:
    history = [turn("Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer().resolve("What about Nvidia?", history)
    assert q.tickers == ["NVDA"] and q.inherited_tickers is False and q.context_keywords == []


def test_long_specific_question_does_not_inherit() -> None:
    history = [turn("Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer().resolve(
        "Which retailers are affected by the February 28 economic boycott?", history
    )
    assert q.tickers == [] and q.inherited_tickers is False


def test_inherits_from_most_recent_turn_with_tickers() -> None:
    history = [
        turn("Intel?", ["INTC"], ["intel"]),
        turn("Any Buffett news?", [], ["buffett"]),
    ]
    q = QueryAnalyzer().resolve("How did it react?", history)
    assert q.tickers == ["INTC"] and q.context_keywords == ["intel"]


def test_no_history_no_inheritance() -> None:
    q = QueryAnalyzer().resolve("Who wants to buy it?", [])
    assert q.tickers == [] and q.inherited_tickers is False


def test_context_excludes_repeated_keywords() -> None:
    history = [turn("Intel split?", ["INTC"], ["intel", "split"])]
    q = QueryAnalyzer().resolve("Is the split real?", history)
    assert q.keywords == ["split", "real"] and q.context_keywords == ["intel"]
