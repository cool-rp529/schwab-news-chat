from pathlib import Path

import pytest

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "stock_news.json"


@pytest.fixture(scope="session")
def data_path() -> Path:
    assert DATA_PATH.exists(), f"corpus missing at {DATA_PATH}"
    return DATA_PATH
