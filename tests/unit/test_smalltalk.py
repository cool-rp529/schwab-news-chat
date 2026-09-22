import pytest

from newschat.chat.smalltalk import detect, reply
from newschat.summarize.base import CorpusInfo


@pytest.mark.parametrize(
    ("text", "intent"),
    [
        ("hello", "greeting"),
        ("Hi!", "greeting"),
        ("hey there", "greeting"),
        ("Good morning.", "greeting"),
        ("thanks", "thanks"),
        ("Thank you!", "thanks"),
        ("thx", "thanks"),
        ("Cheers", "thanks"),
        ("ok, thanks a lot", "thanks"),
        ("What is your name?", "identity"),
        ("who are you", "identity"),
        ("What’s your name?", "identity"),
        ("help", "help"),
        ("What can you do?", "help"),
        ("what can I ask", "help"),
        ("bye", "goodbye"),
        ("Goodbye!", "goodbye"),
        ("see you", "goodbye"),
        ("ＨＥＬＬＯ", "greeting"),
    ],
)
def test_detects_each_intent(text: str, intent: str) -> None:
    assert detect(text) == intent


@pytest.mark.parametrize(
    "text",
    [
        "Hello, what's going on with Intel?",
        "thanks for the Intel summary",
        "hi $NVDA",
        "help me understand the Nvidia earnings",
        "What are you saying about Apple?",
        "Who might buy it?",
        "What is the price of Bitcoin today?",
        "hello hello hello hello hello hello hello",
        "",
    ],
)
def test_news_questions_and_long_messages_are_not_small_talk(text: str) -> None:
    assert detect(text) is None


def test_replies_name_the_corpus() -> None:
    info = CorpusInfo(period="February 2025", tickers=("INTC", "NVDA"))
    assert "INTC, NVDA from around February 2025" in reply("help", info)
    assert reply("identity", info) == reply("help", info)
    assert reply("greeting", info).startswith("Hello. Ask about INTC, NVDA")
    assert reply("thanks", info) == "You're welcome. Ask another question whenever you like."
    assert reply("goodbye", info).startswith("Goodbye.")
