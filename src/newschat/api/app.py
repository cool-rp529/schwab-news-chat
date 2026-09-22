"""HTTP surface: one chat endpoint, article lookup for citation expansion, health, and the
static page. Validation errors from the service map to 422; nothing else is caught."""

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from newschat import __version__
from newschat.chat.service import MAX_QUESTION_CHARS, ChatService
from newschat.chat.session import SessionStore
from newschat.ingest.corpus import Corpus
from newschat.models import Answer, Article

STATIC_DIR = Path(__file__).parent / "static"


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=MAX_QUESTION_CHARS)
    session_id: str | None = Field(default=None, max_length=64)


class ChatResponse(BaseModel):
    session_id: str
    answer: Answer


class HealthResponse(BaseModel):
    articles: int
    unique_tickers: list[str]
    backend: str
    period: str
    sends_data_to_third_party: bool


def create_app(service: ChatService, corpus: Corpus) -> FastAPI:
    app = FastAPI(title="News Chat", version=__version__)

    @app.post("/api/chat", response_model=ChatResponse)
    def chat(request: ChatRequest) -> ChatResponse:
        session_id = request.session_id or SessionStore.new_id()
        try:
            answer = service.ask(session_id, request.question)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return ChatResponse(session_id=session_id, answer=answer)

    @app.get("/api/articles/{article_id}", response_model=Article)
    def article(article_id: str) -> Article:
        try:
            return corpus.by_id[article_id]
        except KeyError:
            raise HTTPException(status_code=404, detail=f"no article {article_id!r}") from None

    @app.get("/api/health", response_model=HealthResponse)
    def health() -> HealthResponse:
        return HealthResponse(
            articles=len(corpus.articles),
            unique_tickers=corpus.tickers,
            backend=service.backend,
            period=service.info.period,
            sends_data_to_third_party=service.backend != "extractive",
        )

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(STATIC_DIR / "index.html")

    return app


def default_app() -> FastAPI:
    corpus = Corpus.load()
    return create_app(ChatService.from_corpus(corpus), corpus)
