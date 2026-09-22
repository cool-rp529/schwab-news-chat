"""Data model shared by every layer. Nothing here imports from newschat.*"""

from typing import Literal

from pydantic import BaseModel, Field, computed_field

ClaimKind = Literal["event", "opinion", "promo"]


class Article(BaseModel):
    id: str
    title: str
    url: str
    source: str
    tickers: list[str]
    text: str
    is_truncated: bool
    word_count: int


class Hit(BaseModel):
    article: Article
    score: float = Field(ge=0.0, le=1.0)  # relative to the top hit, 0..1
    raw_score: float = Field(ge=0.0)  # BM25 before boosts, used by the refusal gate
    matched: int  # query keywords the article contains
    passages: list[str]


class Citation(BaseModel):
    n: int
    article_id: str
    title: str
    url: str
    tickers: list[str]
    passage: str
    verified: bool = True
    kind: ClaimKind = "event"


class ResolvedQuery(BaseModel):
    text: str
    tickers: list[str] = Field(default_factory=list)
    keywords: list[str] = Field(default_factory=list)
    # Keywords carried over from the previous turn; scored at half weight by the retriever.
    context_keywords: list[str] = Field(default_factory=list)
    inherited_tickers: bool = False

    @computed_field  # type: ignore[prop-decorator]
    @property
    def inherited_keywords(self) -> bool:
        return bool(self.context_keywords)


class Answer(BaseModel):
    text: str
    citations: list[Citation]
    backend: str
    query: ResolvedQuery
    covered: bool
    as_of: str
    advisory: str


class ChatTurn(BaseModel):
    question: str
    answer: Answer
