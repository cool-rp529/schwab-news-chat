import pytest
from pydantic import ValidationError

from newschat.models import Answer, Article, Citation, Hit, ResolvedQuery


def make_article(**overrides: object) -> Article:
    base: dict[str, object] = {
        "id": "abc123def4",
        "title": "Intel Shares Jump",
        "url": "https://finance.yahoo.com/news/intel.html",
        "source": "finance.yahoo.com",
        "tickers": ["INTC"],
        "text": "Intel shares rose 10% on Tuesday.",
        "is_truncated": False,
        "word_count": 6,
    }
    base.update(overrides)
    return Article(**base)  # type: ignore[arg-type]


def test_article_round_trips_through_json() -> None:
    article = make_article()
    assert Article.model_validate_json(article.model_dump_json()) == article


def test_hit_requires_passages_list() -> None:
    hit = Hit(article=make_article(), score=1.0, raw_score=7.2, matched=2, passages=["a"])
    assert hit.passages == ["a"]


def test_citation_defaults_to_verified_event() -> None:
    cit = Citation(n=1, article_id="x", title="t", url="u", tickers=[], passage="p")
    assert cit.verified is True
    assert cit.kind == "event"


def test_citation_rejects_unknown_kind() -> None:
    with pytest.raises(ValidationError):
        Citation(n=1, article_id="x", title="t", url="u", tickers=[], passage="p", kind="rumour")  # type: ignore[arg-type]


def test_resolved_query_reports_inherited_keywords() -> None:
    plain = ResolvedQuery(text="q", keywords=["intel"])
    inherited = ResolvedQuery(text="q", keywords=["buy"], context_keywords=["intel", "split"])
    assert plain.inherited_keywords is False
    assert inherited.inherited_keywords is True
    assert "inherited_keywords" in inherited.model_dump()


def test_answer_carries_envelope_fields() -> None:
    answer = Answer(
        text="t",
        citations=[],
        backend="extractive",
        query=ResolvedQuery(text="q"),
        covered=False,
        as_of="February 2025",
        advisory="adv",
    )
    assert answer.covered is False
    assert answer.as_of == "February 2025"
