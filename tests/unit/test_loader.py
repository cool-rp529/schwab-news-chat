import json
from pathlib import Path

import pytest

from newschat.ingest.loader import CorpusError, RawRecord, load_records


def write(tmp_path: Path, payload: object) -> Path:
    p = tmp_path / "news.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


GOOD = {"title": "T", "link": "https://x/a", "ticker": "AAPL", "full_text": "Body."}


def test_loads_real_corpus(data_path: Path) -> None:
    records = load_records(data_path)
    assert len(records) == 138
    assert all(isinstance(r, RawRecord) for r in records)
    assert {r.key_ticker for r in records} == {
        "AAPL",
        "MSFT",
        "AMZN",
        "NFLX",
        "NVDA",
        "INTC",
        "IBM",
    }


def test_keeps_key_ticker_and_record_ticker(tmp_path: Path) -> None:
    p = write(tmp_path, {"MSFT": [dict(GOOD, ticker="AAPL")]})
    (rec,) = load_records(p)
    assert (rec.key_ticker, rec.ticker) == ("MSFT", "AAPL")


def test_top_level_must_be_dict(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match="top level"):
        load_records(write(tmp_path, [GOOD]))


def test_ticker_value_must_be_a_list(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match=r"AAPL.*expected a list"):
        load_records(write(tmp_path, {"AAPL": GOOD}))


def test_list_item_must_be_an_object(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match=r"IBM\[1\].*expected an object"):
        load_records(write(tmp_path, {"IBM": [GOOD, "not an article"]}))


def test_missing_field_reports_location(tmp_path: Path) -> None:
    bad = {k: v for k, v in GOOD.items() if k != "full_text"}
    with pytest.raises(CorpusError, match=r"AAPL\[1\].*full_text"):
        load_records(write(tmp_path, {"AAPL": [GOOD, bad]}))


def test_non_string_field_rejected(tmp_path: Path) -> None:
    with pytest.raises(CorpusError, match=r"NVDA\[0\].*title"):
        load_records(write(tmp_path, {"NVDA": [dict(GOOD, title=42)]}))


def test_invalid_json_and_missing_file(tmp_path: Path) -> None:
    p = tmp_path / "broken.json"
    p.write_text("{not json", encoding="utf-8")
    with pytest.raises(CorpusError, match="invalid JSON"):
        load_records(p)
    with pytest.raises(CorpusError, match="not found"):
        load_records(tmp_path / "nope.json")
