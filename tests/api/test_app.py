import pytest
from fastapi.testclient import TestClient

from newschat.api.app import create_app
from newschat.chat.service import ChatService
from newschat.ingest.corpus import Corpus


@pytest.fixture(scope="module")
def client(corpus: Corpus) -> TestClient:
    return TestClient(create_app(ChatService.from_corpus(corpus), corpus))


def test_index_serves_html(client: TestClient) -> None:
    res = client.get("/")
    assert res.status_code == 200 and "text/html" in res.headers["content-type"]
    assert 'aria-live="polite"' in res.text and "innerHTML" not in res.text
    # href is the one place article data becomes executable; it must be scheme-guarded.
    assert r"/^https?:\/\//i.test(c.url)" in res.text


def test_health(client: TestClient) -> None:
    body = client.get("/api/health").json()
    assert body == {
        "articles": 118,
        "unique_tickers": ["AAPL", "AMZN", "IBM", "INTC", "MSFT", "NFLX", "NVDA"],
        "backend": "extractive",
        "period": "February 2025",
        "sends_data_to_third_party": False,
    }


def test_chat_happy_path_issues_session_id(client: TestClient) -> None:
    res = client.post("/api/chat", json={"question": "What's going on with Intel?"})
    assert res.status_code == 200
    body = res.json()
    assert len(body["session_id"]) == 32
    answer = body["answer"]
    assert answer["covered"] is True and answer["citations"] and answer["backend"] == "extractive"
    assert answer["query"]["tickers"] == ["INTC"] and "inherited_keywords" in answer["query"]
    assert answer["as_of"] == "February 2025"


def test_follow_up_across_requests(client: TestClient) -> None:
    sid = client.post("/api/chat", json={"question": "What's going on with Intel?"}).json()[
        "session_id"
    ]
    body = client.post(
        "/api/chat", json={"question": "Who wants to buy it?", "session_id": sid}
    ).json()
    assert body["session_id"] == sid
    assert body["answer"]["query"]["inherited_tickers"] is True
    assert body["answer"]["query"]["tickers"] == ["INTC"]


@pytest.mark.parametrize(
    "payload", [{}, {"question": ""}, {"question": "   "}, {"question": "x" * 1001}]
)
def test_chat_validation_errors(client: TestClient, payload: dict[str, str]) -> None:
    assert client.post("/api/chat", json=payload).status_code == 422


def test_not_covered_is_200_not_error(client: TestClient) -> None:
    body = client.post("/api/chat", json={"question": "What is the price of Bitcoin today?"}).json()
    assert body["answer"]["covered"] is False and body["answer"]["citations"] == []


def test_article_lookup(client: TestClient, corpus: Corpus) -> None:
    article = corpus.articles[0]
    res = client.get(f"/api/articles/{article.id}")
    assert res.status_code == 200 and res.json()["url"] == article.url
    assert client.get("/api/articles/doesnotexist").status_code == 404


def test_citation_article_ids_resolve(client: TestClient) -> None:
    answer = client.post(
        "/api/chat", json={"question": "What has Warren Buffett been buying?"}
    ).json()["answer"]
    for c in answer["citations"]:
        assert client.get(f"/api/articles/{c['article_id']}").status_code == 200


def test_out_of_corpus_follow_up_is_refused_across_requests(client: TestClient) -> None:
    sid = client.post("/api/chat", json={"question": "What's going on with Intel?"}).json()[
        "session_id"
    ]
    body = client.post(
        "/api/chat", json={"question": "What is the price of Bitcoin today?", "session_id": sid}
    ).json()
    assert body["session_id"] == sid
    assert body["answer"]["covered"] is False
    assert body["answer"]["query"]["inherited_tickers"] is False
