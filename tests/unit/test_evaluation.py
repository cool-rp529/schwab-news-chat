from pathlib import Path

import pytest
import yaml

from newschat.evaluation import THRESHOLDS, GoldenCase, format_report, load_golden, run_evaluation
from newschat.ingest.corpus import Corpus

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.yaml"


def test_load_golden_parses_all_tiers() -> None:
    golden = load_golden(GOLDEN)
    cases = golden.cases
    assert golden.k == 5
    tiers = {c.tier for c in cases}
    assert tiers == {"seeded", "paraphrase", "thematic"}
    assert all(isinstance(c, GoldenCase) for c in cases)
    assert len({c.id for c in cases}) == len(cases)
    neg = [c for c in cases if not c.expect_covered]
    assert len(neg) >= 6 and all(c.tier == "seeded" for c in neg)


def test_load_golden_rejects_bad_case(tmp_path: Path) -> None:
    p = tmp_path / "g.yaml"
    p.write_text(
        yaml.safe_dump({"k": 5, "cases": [{"id": "x", "tier": "seeded"}]}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="question"):
        load_golden(p)


def test_report_structure_and_thresholds(corpus: Corpus) -> None:
    report = run_evaluation(GOLDEN, corpus)
    assert set(report.tiers) == {"seeded", "paraphrase", "thematic"}
    seeded = report.tiers["seeded"]
    assert 0.0 <= seeded.recall_at_k <= 1.0 and 0.0 <= seeded.mrr <= 1.0
    assert report.thresholds == THRESHOLDS and report.k == load_golden(GOLDEN).k
    assert "false_refusal_rate" not in THRESHOLDS  # reported, deliberately not asserted
    assert seeded.false_refusal_rate is not None
    text = format_report(report)
    assert "seeded" in text and "paraphrase" in text and "thematic" in text
    assert "recall@5" in text and "threshold" in text and "false_ref" in text


def test_metrics_on_synthetic_cases(corpus: Corpus, tmp_path: Path) -> None:
    golden = {
        "k": 5,
        "cases": [
            {
                "id": "hit",
                "tier": "seeded",
                "question": "What's going on with Intel?",
                "expected_tickers": ["INTC"],
                "expected_articles": ["Intel"],
                "expected_kinds": ["event"],
            },
            {
                "id": "miss",
                "tier": "seeded",
                "question": "What's going on with Intel?",
                "expected_tickers": ["NVDA"],
                "expected_articles": ["Zebra Unicorn Article"],
            },
            {
                "id": "neg",
                "tier": "seeded",
                "question": "What is the price of Bitcoin today?",
                "expect_covered": False,
            },
            {
                # A positive the absolute gate refuses. In its own tier so the share is
                # unambiguous: every other metric reports 0.00 for this case, which reads
                # like a ranking miss rather than the refusal it actually is.
                "id": "false-refusal",
                "tier": "paraphrase",
                "question": "What is the price of Bitcoin today?",
                "expected_articles": ["Bitcoin"],
            },
        ],
    }
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(golden), encoding="utf-8")
    report = run_evaluation(p, corpus)
    seeded = report.tiers["seeded"]
    assert seeded.recall_at_k == pytest.approx(0.5)  # 1.0 and 0.0 averaged
    assert seeded.mrr == pytest.approx(0.5)
    assert seeded.ticker_accuracy == pytest.approx(0.5)
    assert seeded.kind_accuracy == pytest.approx(1.0)
    assert seeded.refusal_rate == pytest.approx(1.0)  # still only over the negatives
    assert seeded.false_refusal_rate == pytest.approx(0.0)  # neither positive was refused
    assert any("miss" in f for f in seeded.failures)

    refused_positive = report.tiers["paraphrase"]
    assert refused_positive.false_refusal_rate == pytest.approx(1.0)
    assert refused_positive.refusal_rate is None  # the tier holds no negatives
    assert "false-refusal: refused a covered question" in refused_positive.failures
    assert report.passed is False


def test_unmeasured_metrics_render_as_dash(corpus: Corpus, tmp_path: Path) -> None:
    golden = {
        "k": 5,
        "cases": [
            {
                "id": "theme-only",
                "tier": "thematic",
                "question": "Which companies are exposed to DeepSeek?",
                "expected_ticker_coverage": ["AAPL", "MSFT"],
            },
        ],
    }
    p = tmp_path / "g.yaml"
    p.write_text(yaml.safe_dump(golden), encoding="utf-8")
    report = run_evaluation(p, corpus)
    thematic = report.tiers["thematic"]
    assert thematic.recall_at_k is None
    assert thematic.mrr is None
    assert thematic.ticker_accuracy is None
    assert thematic.kind_accuracy is None
    assert thematic.refusal_rate is None
    assert thematic.ticker_coverage is not None

    text = format_report(report)
    rows = [line for line in text.splitlines() if line.startswith("thematic")]
    assert len(rows) == 1
    row = rows[0]
    assert row.count("-") >= 5  # recall, mrr, tickers, kinds, refused all render as "-"
    assert "0." in row  # the coverage column is a real number

    # No seeded tier at all -> passed must be False, not vacuously True.
    assert report.passed is False
