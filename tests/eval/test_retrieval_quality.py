"""Golden-set evaluation. Seeded tier is asserted; paraphrase and thematic tiers are printed
so the numbers are visible in CI output and can be pasted into the README."""

from pathlib import Path

import pytest

from newschat.evaluation import THRESHOLDS, EvalReport, format_report, run_evaluation
from newschat.ingest.corpus import Corpus

GOLDEN = Path(__file__).with_name("golden.yaml")


@pytest.fixture(scope="module")
def report(corpus: Corpus) -> EvalReport:
    rep = run_evaluation(GOLDEN, corpus)
    print("\n" + format_report(rep))
    return rep


def test_seeded_recall(report: EvalReport) -> None:
    seeded = report.tiers["seeded"]
    assert seeded.recall_at_k is not None and seeded.recall_at_k >= THRESHOLDS["recall_at_k"], (
        seeded.failures
    )


def test_seeded_mrr(report: EvalReport) -> None:
    seeded = report.tiers["seeded"]
    assert seeded.mrr is not None and seeded.mrr >= THRESHOLDS["mrr"], seeded.failures


def test_seeded_ticker_accuracy(report: EvalReport) -> None:
    seeded = report.tiers["seeded"]
    assert (
        seeded.ticker_accuracy is not None
        and seeded.ticker_accuracy >= THRESHOLDS["ticker_accuracy"]
    ), seeded.failures


def test_seeded_kind_accuracy(report: EvalReport) -> None:
    seeded = report.tiers["seeded"]
    assert (
        seeded.kind_accuracy is not None and seeded.kind_accuracy >= THRESHOLDS["kind_accuracy"]
    ), seeded.failures


def test_negatives_all_refused(report: EvalReport) -> None:
    assert report.tiers["seeded"].refusal_rate == 1.0, report.tiers["seeded"].failures


def test_paraphrase_tier_is_reported(report: EvalReport) -> None:
    para = report.tiers["paraphrase"]
    assert para.n >= 8
    # deliberately lower bar; the gap is the cost of no embeddings
    assert para.recall_at_k is not None and para.recall_at_k >= 0.5, para.failures


def test_thematic_tier_is_reported_not_asserted(report: EvalReport) -> None:
    theme = report.tiers["thematic"]
    assert theme.n >= 5 and theme.ticker_coverage is not None


def test_overall_passed_flag(report: EvalReport) -> None:
    assert report.passed
