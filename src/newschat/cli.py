"""Command line: `ask` for one-shot demos and README transcripts, `eval` for the golden-set
report, `serve` for the web app."""

import argparse
import sys
from collections.abc import Sequence
from pathlib import Path

from newschat.chat.service import ChatService
from newschat.chat.session import SessionStore
from newschat.ingest.corpus import Corpus
from newschat.summarize.factory import BACKENDS

DEFAULT_GOLDEN = Path("tests/eval/golden.yaml")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="newschat", description="Chat over the news corpus.")
    sub = parser.add_subparsers(dest="command")

    ask = sub.add_parser("ask", help="answer one question and exit")
    ask.add_argument("question")
    ask.add_argument("--backend", choices=BACKENDS, default=None)
    ask.add_argument("--data", type=Path, default=None, help="path to stock_news.json")

    ev = sub.add_parser("eval", help="run the golden-set evaluation and print the report")
    ev.add_argument("--golden", type=Path, default=DEFAULT_GOLDEN)
    ev.add_argument("--data", type=Path, default=None)

    serve = sub.add_parser("serve", help="run the web app")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "ask":
        return _ask(args.question, args.backend, args.data)
    if args.command == "eval":
        return _eval(args.golden, args.data)
    if args.command == "serve":
        return _serve(args.host, args.port)
    parser.print_help()
    return 2


def _ask(question: str, backend: str | None, data: Path | None) -> int:
    try:
        service = ChatService.from_corpus(Corpus.load(data), backend)
    except ValueError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    answer = service.ask(SessionStore.new_id(), question)
    print(answer.text)
    print(f"\nbackend: {answer.backend}")
    for c in answer.citations:
        marker = "" if c.verified else " (unverified)"
        print(f"[{c.n}] {c.kind:<7} {c.url}{marker}")
    return 0


def _eval(golden: Path, data: Path | None) -> int:
    from newschat.evaluation import format_report, run_evaluation  # type: ignore[import-untyped]

    report = run_evaluation(golden, Corpus.load(data))
    print(format_report(report))
    return 0 if report.passed else 1


def _serve(host: str, port: int) -> int:
    import uvicorn

    uvicorn.run("newschat.api.app:default_app", factory=True, host=host, port=port)
    return 0


def _utf8_stdout() -> None:
    # Windows consoles default to a legacy code page; the answers contain "≈" and "—".
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8", errors="replace")
