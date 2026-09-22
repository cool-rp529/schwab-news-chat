import math

import pytest

from newschat.retrieval.bm25 import BM25Index

DOCS = [
    ["intel", "share", "jump", "broadcom", "tsmc"],
    ["intel", "intel", "intel", "price", "target"],
    ["nvidia", "earning", "beat", "china"],
    ["apple", "iphone", "launch", "intel"],
]


@pytest.fixture
def index() -> BM25Index:
    return BM25Index(DOCS)


def test_df_and_idf_monotone(index: BM25Index) -> None:
    assert index.df("intel") == 3
    assert index.df("nvidia") == 1
    assert index.df("missing") == 0
    assert index.idf("nvidia") > index.idf("intel") > 0
    assert index.idf("missing") == pytest.approx(math.log((4 + 0.5) / 0.5 + 1))


def test_unknown_term_scores_zero_everywhere(index: BM25Index) -> None:
    assert index.score(["missing"]) == [0.0] * 4


def test_rare_term_ranks_its_document_first(index: BM25Index) -> None:
    scores = index.score(["nvidia"])
    assert scores.index(max(scores)) == 2
    assert scores[0] == scores[1] == scores[3] == 0.0


def test_term_frequency_saturates(index: BM25Index) -> None:
    once, thrice = index.score(["intel"])[0], index.score(["intel"])[1]
    assert thrice > once
    assert thrice < 3 * once


def test_length_normalisation_prefers_shorter_doc() -> None:
    idx = BM25Index([["apple", "launch"], ["apple", "launch"] + ["filler"] * 20])
    short, long = idx.score(["apple"])
    assert short > long


def test_weights_scale_contribution(index: BM25Index) -> None:
    full = index.score(["broadcom"])[0]
    half = index.score(["broadcom"], weights={"broadcom": 0.5})[0]
    assert half == pytest.approx(full / 2)


def test_repeated_query_terms_count_once(index: BM25Index) -> None:
    assert index.score(["intel", "intel"]) == index.score(["intel"])


def test_contains(index: BM25Index) -> None:
    assert index.contains(0, "tsmc") and not index.contains(2, "tsmc")


def test_empty_index_scores_nothing() -> None:
    assert BM25Index([]).score(["x"]) == []
