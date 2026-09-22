import pytest

from newschat.models import Answer, ChatTurn, ResolvedQuery
from newschat.retrieval.query import QUERY_STOPWORDS, QueryAnalyzer


def turn(
    question: str, tickers: list[str], keywords: list[str], context: list[str] | None = None
) -> ChatTurn:
    q = ResolvedQuery(
        text=question, tickers=tickers, keywords=keywords, context_keywords=context or []
    )
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


def test_best_is_a_stopword() -> None:
    # "best" is generic ("best pizza", "best price") and, unstemmed, carried an
    # out-of-corpus negative ("best pizza in Chicago") through the retriever's coverage gate.
    assert "best" in QUERY_STOPWORDS


def test_giant_bare_e_and_team_are_stopwords() -> None:
    # "giant" ("tech giant", "streaming giant") and "team"/"teaming" (generic organizational
    # nouns, as in "marketing team" or "teaming up") are generic business descriptors like the
    # existing "company"/"market" entries. "e" is a tokenizer split artifact of "e-commerce"
    # style hyphenated terms. All three diluted a paraphrase-tier golden case's ranking with
    # spurious matches on unrelated articles.
    assert {"giant", "e", "team"} <= QUERY_STOPWORDS


def test_follow_up_inherits_tickers_and_keywords() -> None:
    history = [turn("What's going on with Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer().resolve("Who wants to buy it?", history)
    assert q.tickers == ["INTC"] and q.inherited_tickers is True
    assert q.keywords == ["want", "buy"]
    assert q.context_keywords == ["intel"] and q.inherited_keywords is True
    assert q.inheritance == "strong"


def test_short_question_without_anaphora_still_inherits() -> None:
    history = [turn("Nvidia earnings?", ["NVDA"], ["nvidia", "earning"])]
    q = QueryAnalyzer().resolve("And the price target?", history)
    assert q.tickers == ["NVDA"] and q.context_keywords == ["nvidia", "earning"]
    assert q.inheritance == "soft" and q.inherited_tickers is True


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


# The corpus vocabulary a real analyzer is wired with, reduced to the terms these cases need.
KNOWN = {"intel", "price", "target", "want", "buy"}


def test_known_short_follow_up_still_inherits() -> None:
    history = [turn("What's going on with Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer(known_terms=KNOWN).resolve("And the price target?", history)
    assert q.tickers == ["INTC"] and q.inherited_tickers is True
    assert q.context_keywords == ["intel"]


def test_short_question_with_out_of_corpus_word_does_not_inherit() -> None:
    # "bitcoin" is nowhere in the corpus, so this is a new question that happens to be
    # short — not a follow-up. Inheriting INTC here would ticker-match it past the
    # retriever's refusal gate and answer it with confidently cited Intel articles.
    history = [turn("What's going on with Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer(known_terms=KNOWN).resolve("What is the price of Bitcoin today?", history)
    assert q.tickers == [] and q.inherited_tickers is False
    assert q.context_keywords == []


def test_pronoun_inherits_even_with_an_out_of_corpus_word() -> None:
    # The vocabulary check guards only the length rule. An explicit "it" is a direct
    # reference to the previous turn, so the regex case wins regardless.
    history = [turn("What's going on with Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer(known_terms=KNOWN).resolve("Did it beat Bitcoin?", history)
    assert q.tickers == ["INTC"] and q.inherited_tickers is True


def test_without_known_terms_short_questions_inherit_on_length_alone() -> None:
    # known_terms=None is the no-vocabulary mode: there is nothing to check a keyword
    # against, so length alone decides, as it did before the vocabulary check existed.
    history = [turn("What's going on with Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer().resolve("What is the price of Bitcoin today?", history)
    assert q.tickers == ["INTC"] and q.inherited_tickers is True


# ---- inheritance strength and topic chaining ----


def test_new_question_has_no_inheritance() -> None:
    q = QueryAnalyzer(known_terms=KNOWN).resolve("What's going on with Intel?", [])
    assert q.inheritance == "none" and q.inherited_tickers is False


@pytest.mark.parametrize(
    ("question", "inheritance"),
    [
        ("Who wants to buy it?", "strong"),  # pronoun
        ("Did the company buy anything?", "strong"),  # "the company"
        ("Did it beat Bitcoin?", "strong"),  # a pronoun wins over an unknown word
        ("And the price target?", "soft"),  # short, every word known, no reference word
        ("Price target", "soft"),
        ("What is the price of Bitcoin today?", "none"),  # short but "bitcoin" is unknown
        ("Which retailers are affected by the February 28 economic boycott?", "none"),  # long
        # The reset phrase the soft note offers; "topic" is a corpus word, so without the
        # reset this would soft-inherit on length alone.
        ("New topic: price target", "none"),
    ],
)
def test_inheritance_strength(question: str, inheritance: str) -> None:
    history = [turn("What's going on with Intel?", ["INTC"], ["intel"])]
    q = QueryAnalyzer(known_terms=KNOWN | {"topic"}).resolve(question, history)
    assert q.inheritance == inheritance
    assert q.inherited_tickers is (inheritance != "none")
    assert q.tickers == ([] if inheritance == "none" else ["INTC"])


def test_context_chains_through_an_earlier_follow_up() -> None:
    # The first follow-up's own words are [want, buy]; "intel" lives only in its context.
    # Inheriting keywords alone would lose "intel" on the second follow-up.
    history = [
        turn("What's going on with Intel?", ["INTC"], ["intel"]),
        turn("Who wants to buy it", ["INTC"], ["want", "buy"], context=["intel"]),
        turn("What is the price of Bitcoin today?", [], ["price", "bitcoin"]),
    ]
    q = QueryAnalyzer(known_terms=KNOWN).resolve("Who wants to buy it?", history)
    assert q.inheritance == "strong" and q.tickers == ["INTC"]
    assert q.keywords == ["want", "buy"] and q.context_keywords == ["intel"]


def test_chained_context_keeps_order_and_dedupes() -> None:
    history = [
        turn("Intel?", ["INTC"], ["intel"]),
        turn("Who wants to buy it", ["INTC"], ["want", "buy", "intel"], context=["intel"]),
    ]
    q = QueryAnalyzer(known_terms=KNOWN | {"deepseek"}).resolve("DeepSeek", history)
    assert q.inheritance == "soft"
    assert q.context_keywords == ["want", "buy", "intel"]
