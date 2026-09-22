"""Golden-set evaluation shared by `tests/eval` and `newschat eval`.

Metrics per tier: Recall@k and MRR over expected articles, ticker-detection accuracy,
claim-kind accuracy (expected kinds ⊆ kinds of cited passages), refusal rate on negatives,
and ticker coverage for thematic questions. The seeded tier is asserted; the others are
reported so their numbers are visible rather than hidden.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from newschat.ingest.corpus import Corpus
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever
from newschat.summarize.base import CorpusInfo
from newschat.summarize.extractive import ExtractiveSummarizer

THRESHOLDS: dict[str, float] = {
    "recall_at_k": 0.80,
    "mrr": 0.70,
    "ticker_accuracy": 0.90,
    "kind_accuracy": 0.80,
    "refusal_rate": 1.00,
}
ASSERTED_TIER = "seeded"


@dataclass(frozen=True)
class GoldenCase:
    id: str
    tier: str
    question: str
    expected_articles: list[str] = field(default_factory=list)
    expected_tickers: list[str] | None = None  # None → not checked; [] → assert none
    expected_kinds: list[str] = field(default_factory=list)
    expect_covered: bool = True
    expected_ticker_coverage: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class CaseResult:
    case: GoldenCase
    hit_titles: list[str]
    recall: float | None
    reciprocal_rank: float | None
    ticker_ok: bool | None
    kinds_ok: bool | None
    refusal_ok: bool | None
    coverage: float | None


@dataclass(frozen=True)
class TierReport:
    tier: str
    n: int
    recall_at_k: float
    mrr: float
    ticker_accuracy: float
    kind_accuracy: float
    refusal_rate: float
    ticker_coverage: float | None
    failures: list[str]


@dataclass(frozen=True)
class EvalReport:
    tiers: dict[str, TierReport]
    thresholds: dict[str, float]
    passed: bool
    k: int


def load_golden(path: Path) -> list[GoldenCase]:
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    cases: list[GoldenCase] = []
    for raw in payload["cases"]:
        raw_case: dict[str, Any] = raw
        for key in ("id", "tier", "question"):
            if key not in raw_case:
                raise ValueError(f"golden case {raw_case.get('id', '?')!r} is missing '{key}'")
        cases.append(GoldenCase(**raw_case))
    return cases


def _golden_k(path: Path) -> int:
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    return int(payload.get("k", 5))


def run_case(
    case: GoldenCase,
    analyzer: QueryAnalyzer,
    retriever: Retriever,
    summarizer: ExtractiveSummarizer,
) -> CaseResult:
    resolved = analyzer.resolve(case.question, [])
    hits = retriever.search(resolved)
    answer = summarizer.summarize(resolved, hits, [])
    titles = [h.article.title for h in hits]

    if not case.expect_covered:
        return CaseResult(case, titles, None, None, None, None, not answer.covered, None)

    recall: float | None = None
    rr: float | None = None
    if case.expected_articles:
        found = [e for e in case.expected_articles if any(e.lower() in t.lower() for t in titles)]
        recall = len(found) / len(case.expected_articles)
        rr = 0.0
        for rank, title in enumerate(titles, start=1):
            if any(e.lower() in title.lower() for e in case.expected_articles):
                rr = 1.0 / rank
                break

    ticker_ok = None
    if case.expected_tickers is not None:
        ticker_ok = set(resolved.tickers) == set(case.expected_tickers)

    kinds_ok = None
    if case.expected_kinds:
        kinds_ok = set(case.expected_kinds) <= {c.kind for c in answer.citations}

    coverage = None
    if case.expected_ticker_coverage:
        seen = {t for h in hits for t in h.article.tickers}
        coverage = len(seen & set(case.expected_ticker_coverage)) / len(
            case.expected_ticker_coverage
        )

    return CaseResult(case, titles, recall, rr, ticker_ok, kinds_ok, None, coverage)


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 1.0


def _tier_report(tier: str, results: Sequence[CaseResult]) -> TierReport:
    recalls = [r.recall for r in results if r.recall is not None]
    rrs = [r.reciprocal_rank for r in results if r.reciprocal_rank is not None]
    tickers = [float(r.ticker_ok) for r in results if r.ticker_ok is not None]
    kinds = [float(r.kinds_ok) for r in results if r.kinds_ok is not None]
    refusals = [float(r.refusal_ok) for r in results if r.refusal_ok is not None]
    coverages = [r.coverage for r in results if r.coverage is not None]
    failures: list[str] = []
    for r in results:
        if r.recall is not None and r.recall < 1.0:
            failures.append(f"{r.case.id}: recall {r.recall:.2f}; got {r.hit_titles[:3]}")
        if r.ticker_ok is False:
            failures.append(f"{r.case.id}: tickers expected {r.case.expected_tickers}")
        if r.kinds_ok is False:
            failures.append(f"{r.case.id}: kinds expected {r.case.expected_kinds}")
        if r.refusal_ok is False:
            failures.append(f"{r.case.id}: expected refusal; got {r.hit_titles[:3]}")
    return TierReport(
        tier=tier,
        n=len(results),
        recall_at_k=_mean(recalls),
        mrr=_mean(rrs),
        ticker_accuracy=_mean(tickers),
        kind_accuracy=_mean(kinds),
        refusal_rate=_mean(refusals),
        ticker_coverage=_mean(coverages) if coverages else None,
        failures=failures,
    )


def evaluate(cases: Sequence[GoldenCase], corpus: Corpus, k: int = 5) -> EvalReport:
    analyzer = QueryAnalyzer()
    retriever = Retriever(corpus)
    summarizer = ExtractiveSummarizer(CorpusInfo(corpus.infer_period(), tuple(corpus.tickers)))
    by_tier: dict[str, list[CaseResult]] = {}
    for case in cases:
        by_tier.setdefault(case.tier, []).append(run_case(case, analyzer, retriever, summarizer))
    tiers = {tier: _tier_report(tier, results) for tier, results in by_tier.items()}
    seeded = tiers.get(ASSERTED_TIER)
    passed = seeded is not None and all(
        getattr(seeded, metric) >= floor for metric, floor in THRESHOLDS.items()
    )
    return EvalReport(tiers=tiers, thresholds=dict(THRESHOLDS), passed=passed, k=k)


def run_evaluation(golden_path: Path, corpus: Corpus) -> EvalReport:
    return evaluate(load_golden(golden_path), corpus, k=_golden_k(golden_path))


def format_report(report: EvalReport) -> str:
    k = report.k
    header = (
        f"{'tier':<11}{'n':>4}{f'recall@{k}':>11}{'mrr':>8}{'tickers':>9}"
        f"{'kinds':>8}{'refused':>9}{'coverage':>10}"
    )
    lines = [header, "-" * len(header)]
    for tier, t in report.tiers.items():
        cov = f"{t.ticker_coverage:.2f}" if t.ticker_coverage is not None else "-"
        lines.append(
            f"{tier:<11}{t.n:>4}{t.recall_at_k:>11.2f}{t.mrr:>8.2f}{t.ticker_accuracy:>9.2f}"
            f"{t.kind_accuracy:>8.2f}{t.refusal_rate:>9.2f}{cov:>10}"
        )
    th = report.thresholds
    lines.append(
        f"{'threshold':<11}{'':>4}{th['recall_at_k']:>11.2f}{th['mrr']:>8.2f}"
        f"{th['ticker_accuracy']:>9.2f}{th['kind_accuracy']:>8.2f}{th['refusal_rate']:>9.2f}"
        f"{'-':>10}   (asserted on {ASSERTED_TIER})"
    )
    lines.append(f"passed: {report.passed}")
    for tier, t in report.tiers.items():
        for failure in t.failures:
            lines.append(f"  [{tier}] {failure}")
    return "\n".join(lines)
