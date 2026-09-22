"""Read the ticker-keyed news JSON into validated flat records.

Validation is strict on purpose: a malformed corpus should stop the process at startup
with a message that names the offending record, not surface later as a KeyError.
"""

import json
from pathlib import Path

from pydantic import BaseModel

FIELDS = ("title", "link", "ticker", "full_text")


class CorpusError(ValueError):
    """The corpus file is missing, unreadable, or not in the expected shape."""


class RawRecord(BaseModel):
    key_ticker: str  # the dict key the record was filed under
    title: str
    link: str
    ticker: str
    full_text: str


def load_records(path: Path) -> list[RawRecord]:
    if not path.exists():
        raise CorpusError(f"{path}: not found")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise CorpusError(f"{path}: invalid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise CorpusError(f"{path}: top level must be an object keyed by ticker")

    records: list[RawRecord] = []
    for key, items in payload.items():
        if not isinstance(items, list):
            raise CorpusError(f"{path}: {key}: expected a list of articles")
        for i, item in enumerate(items):
            records.append(_record(path, str(key), i, item))
    return records


def _record(path: Path, key: str, index: int, item: object) -> RawRecord:
    where = f"{path}: {key}[{index}]"
    if not isinstance(item, dict):
        raise CorpusError(f"{where}: expected an object")
    for field in FIELDS:
        if field not in item:
            raise CorpusError(f"{where}: missing field '{field}'")
        if not isinstance(item[field], str):
            raise CorpusError(f"{where}: field '{field}' must be a string")
    return RawRecord(key_ticker=key, **{f: item[f] for f in FIELDS})
