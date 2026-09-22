from newschat.chat.session import SessionStore
from newschat.models import Answer, ChatTurn, ResolvedQuery


def turn(i: int) -> ChatTurn:
    return ChatTurn(
        question=f"q{i}",
        answer=Answer(
            text="",
            citations=[],
            backend="x",
            query=ResolvedQuery(text="q"),
            covered=True,
            as_of="",
            advisory="",
        ),
    )


def test_new_ids_are_unique_hex() -> None:
    a, b = SessionStore.new_id(), SessionStore.new_id()
    assert a != b and len(a) == 32 and int(a, 16) >= 0


def test_history_is_bounded() -> None:
    store = SessionStore(max_turns=2)
    history = store.get("s")
    for i in range(5):
        history.append(turn(i))
    assert [t.question for t in store.get("s")] == ["q3", "q4"]


def test_lru_evicts_oldest_session() -> None:
    store = SessionStore(max_sessions=2)
    store.get("a").append(turn(0))
    store.get("b")
    store.get("a")  # touch a → b is now oldest
    store.get("c")
    assert "b" not in store and "a" in store and "c" in store and len(store) == 2
    assert [t.question for t in store.get("a")] == ["q0"]


def test_unknown_id_is_a_fresh_session() -> None:
    assert list(SessionStore().get("never-seen")) == []
