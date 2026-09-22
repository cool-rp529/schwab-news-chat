"""Opt-in LLM backend that shells out to the Claude Code CLI (`claude -p`).

The CLI exits 0 with parseable JSON even when a run fails (usage limit, overload, max
turns); `is_error`, `subtype`, `api_error_status` and an empty `result` are the in-band
signals. Every failure mode raises SummarizerUnavailable so ChatService can fall back.
"""

import json
import re
import shutil
import subprocess
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from newschat.models import Answer, ChatTurn, Citation, Hit, ResolvedQuery
from newschat.summarize.base import (
    CorpusInfo,
    SummarizerUnavailable,
    build_prompt,
    check_faithfulness,
    cited_numbers,
    envelope,
    make_citations,
    not_covered_answer,
    strip_invalid_citations,
)
from newschat.summarize.claims import is_advice
from newschat.text.sentences import split_sentences

Runner = Callable[..., "subprocess.CompletedProcess[str]"]

_LEADING_MARKERS_RE = re.compile(r"^(?:\s*\[\d+\])+")


def _claims_with_markers(text: str) -> list[str]:
    """Sentences with their trailing [n] markers re-attached (the splitter moves them forward)."""
    merged: list[str] = []
    for sentence in split_sentences(text):
        match = _LEADING_MARKERS_RE.match(sentence)
        if match and merged:
            merged[-1] = f"{merged[-1]} {match.group(0).strip()}"
            rest = sentence[match.end() :].strip()
            if rest:
                merged.append(rest)
        else:
            merged.append(sentence)
    return merged


@dataclass(frozen=True)
class ClaudeCliConfig:
    executable: str = "claude"
    model: str | None = None
    timeout_s: float = 60.0
    faithfulness_threshold: float = 0.5


class ClaudeCliSummarizer:
    name = "claude-cli"

    def __init__(
        self,
        info: CorpusInfo,
        config: ClaudeCliConfig | None = None,
        runner: Runner = subprocess.run,
    ) -> None:
        self.info = info
        self.config = config or ClaudeCliConfig()
        self._run = runner

    def summarize(
        self, query: ResolvedQuery, hits: Sequence[Hit], history: Sequence[ChatTurn]
    ) -> Answer:
        if not hits:
            return not_covered_answer(query, self.name, self.info)
        raw = self._invoke(build_prompt(query, hits, history, self.info))
        text = strip_invalid_citations(raw, len(hits)).strip()
        if is_advice(text):
            raise SummarizerUnavailable("advice-shaped output rejected")
        if not cited_numbers(text):
            raise SummarizerUnavailable("output carried no citations")
        return envelope(
            body=text,
            citations=self._citations(text, hits),
            backend=self.name,
            query=query,
            covered=True,
            info=self.info,
        )

    def _invoke(self, prompt: str) -> str:
        executable = shutil.which(self.config.executable)
        if executable is None:
            raise SummarizerUnavailable(f"{self.config.executable!r} not found on PATH")
        # Argument list, never a shell string; tools disabled so the CLI cannot act, only answer.
        args = [
            executable,
            "-p",
            "--output-format",
            "json",
            "--tools",
            "",
            "--no-session-persistence",
        ]
        if self.config.model:
            args += ["--model", self.config.model]
        try:
            proc = self._run(
                args,
                input=prompt,
                capture_output=True,
                text=True,
                encoding="utf-8",
                timeout=self.config.timeout_s,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise SummarizerUnavailable(f"timed out after {self.config.timeout_s}s") from exc
        except OSError as exc:
            raise SummarizerUnavailable(f"could not start CLI: {exc}") from exc
        if proc.returncode != 0:
            raise SummarizerUnavailable(f"exit code {proc.returncode}: {proc.stderr.strip()[:200]}")
        return self._extract_result(proc.stdout)

    @staticmethod
    def _extract_result(stdout: str) -> str:
        try:
            payload: Any = json.loads(stdout)
        except json.JSONDecodeError as exc:
            raise SummarizerUnavailable(f"invalid JSON from CLI: {exc}") from exc
        if not isinstance(payload, dict):
            raise SummarizerUnavailable("unexpected JSON shape from CLI")
        if (
            payload.get("is_error")
            or payload.get("subtype") != "success"
            or payload.get("api_error_status") is not None
        ):
            raise SummarizerUnavailable(
                f"CLI reported failure: subtype={payload.get('subtype')!r} "
                f"api_error_status={payload.get('api_error_status')!r}"
            )
        result = payload.get("result")
        if not isinstance(result, str) or not result.strip():
            raise SummarizerUnavailable("CLI returned an empty result")
        return result

    def _citations(self, text: str, hits: Sequence[Hit]) -> list[Citation]:
        sentences = _claims_with_markers(text)
        all_citations = make_citations(hits)
        out: list[Citation] = []
        for n in sorted(set(cited_numbers(text))):
            claims = [s for s in sentences if f"[{n}]" in s]
            verified = bool(claims) and all(
                check_faithfulness(c, hits[n - 1].passages, self.config.faithfulness_threshold)
                for c in claims
            )
            out.append(all_citations[n - 1].model_copy(update={"verified": verified}))
        return out
