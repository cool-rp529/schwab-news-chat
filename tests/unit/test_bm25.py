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


def test_terms_is_the_indexed_vocabulary(index: BM25Index) -> None:
    assert index.terms() == {t for doc in DOCS for t in doc}
    assert "missing" not in index.terms()
    assert BM25Index([]).terms() == frozenset()


def test_terms_with_df_at_most(index: BM25Index) -> None:
    assert index.terms_with_df_at_most(1) == index.terms() - {"intel"}
    assert index.terms_with_df_at_most(3) == index.terms()
    assert index.terms_with_df_at_most(0) == frozenset()


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


def test_precomputed_norms_match_the_inline_formula(index: BM25Index) -> None:
    # Captured before the per-document length norm was hoisted out of `score`'s inner loop;
    # the optimisation must be exactly score-preserving, not merely close.
    assert index.score(["intel", "price", "nvidia"]) == [
        0.33969042279879275,
        1.7250325670064293,
        1.2673397940273012,
        0.375447309409192,
    ]
    assert index.score(["intel"]) == [
        0.33969042279879275,
        0.5783918009817282,
        0.0,
        0.375447309409192,
    ]
