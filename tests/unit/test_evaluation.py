from pathlib import Path

import pytest
import yaml

from newschat.evaluation import THRESHOLDS, GoldenCase, format_report, load_golden, run_evaluation
from newschat.ingest.corpus import Corpus

GOLDEN = Path(__file__).resolve().parents[1] / "eval" / "golden.yaml"


def test_load_golden_parses_all_tiers() -> None:
    cases = load_golden(GOLDEN)
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
    assert report.thresholds == THRESHOLDS and report.k == 5
    text = format_report(report)
    assert "seeded" in text and "paraphrase" in text and "thematic" in text
    assert "recall@5" in text and "threshold" in text


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
    assert seeded.refusal_rate == pytest.approx(1.0)
    assert any("miss" in f for f in seeded.failures)
    assert report.passed is False
