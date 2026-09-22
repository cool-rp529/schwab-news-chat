from pathlib import Path
from typing import Any

import pytest

from newschat.cli import build_parser, main


def test_parser_subcommands() -> None:
    parser = build_parser()
    assert parser.parse_args(["ask", "hi"]).command == "ask"
    assert parser.parse_args(["eval"]).command == "eval"
    assert parser.parse_args(["serve", "--port", "9000"]).port == 9000


def test_ask_prints_answer_and_sources(capsys: pytest.CaptureFixture[str], data_path: Path) -> None:
    code = main(["ask", "What's going on with Intel?", "--data", str(data_path)])
    out = capsys.readouterr().out
    assert code == 0
    assert "Based on news from" in out and "Sources" in out and "not investment advice" in out
    assert "backend: extractive" in out


def test_ask_not_covered_exits_zero(capsys: pytest.CaptureFixture[str], data_path: Path) -> None:
    assert main(["ask", "price of Bitcoin today?", "--data", str(data_path)]) == 0
    assert "doesn't cover that" in capsys.readouterr().out


def test_ask_rejects_unknown_backend(capsys: pytest.CaptureFixture[str], data_path: Path) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main(["ask", "hi", "--backend", "gpt", "--data", str(data_path)])
    assert excinfo.value.code == 2
    assert "invalid choice" in capsys.readouterr().err


def test_ask_rejects_unknown_env_backend(
    capsys: pytest.CaptureFixture[str], data_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("NEWSCHAT_BACKEND", "gpt")
    assert main(["ask", "hi", "--data", str(data_path)]) == 2
    assert "unknown backend" in capsys.readouterr().err


def test_serve_calls_uvicorn(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[dict[str, Any]] = []
    monkeypatch.setattr("uvicorn.run", lambda app, **kw: calls.append({"app": app, **kw}))
    assert main(["serve", "--port", "9123", "--host", "0.0.0.0"]) == 0
    assert calls == [
        {"app": "newschat.api.app:default_app", "factory": True, "host": "0.0.0.0", "port": 9123}
    ]


def test_no_command_prints_help(capsys: pytest.CaptureFixture[str]) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().out
