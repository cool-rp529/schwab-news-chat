"""Rank articles for a resolved query and decide when to refuse.

Scoring: BM25 over (title x2 + body) -> x ticker boost -> x truncation penalty.
Refusal: relative filtering alone can never refuse (the top hit is 1.0 by construction), so an
absolute gate runs first: idf-normalised match strength >= `min_strength` and idf-weighted keyword
coverage >= `min_coverage`, unless the article ticker-matches the query.

A soft-inherited follow-up (see `retrieval/query.py`) treats its tickers as a ranking preference
only: ticker matches still get the boost, but they must clear the gate like any other article and
are not restricted to first — so "DeepSeek" after an Intel question is answered about DeepSeek.

Strength is `raw / total_idf`, not raw BM25. Raw BM25 sums over query terms, so a fixed raw floor
measures how many rare words a question happens to contain rather than how well the corpus answers
it: a one-term query is bounded by about `idf * (k1 + 1)` and can never clear a floor calibrated on
three-term queries. Dividing by the query's total idf yields the idf-weighted mean term saturation,
bounded by `k1 + 1` for every query length, so the same floor means the same thing whether the
question is "DeepSeek" or "How is DeepSeek affecting AI stocks?".
"""

from dataclasses import dataclass

from newschat.ingest.corpus import Corpus
from newschat.models import Hit, ResolvedQuery
from newschat.retrieval.aliases import mentions
from newschat.retrieval.bm25 import BM25Index
from newschat.text.sentences import is_prose, split_sentences
from newschat.text.tokenize import tokenize


@dataclass(frozen=True)
class RetrieverConfig:
    k: int = 5
    ticker_boost: float = 1.5
    truncated_penalty: float = 0.5
    context_weight: float = 0.5
    # measured 2026-09-22 by scripts/measure_gate.py. Negatives' best row peaked at
    # strength=0.77 cov=0.59 ("best pizza in Chicago", after stemming "best" out as a query
    # stopword); every other negative sat at strength <= 0.64. Positives' worst row was
    # strength=1.20 cov=0.55 ("How is DeepSeek affecting AI stocks?"), and the bare-topic
    # queries ("DeepSeek", "Tell me about DeepSeek") sit at strength=2.17 cov=1.00 -- those
    # are the cases a raw floor silently refused, because one term cannot sum past ~5.1.
    # 1.0/0.45 clears every negative on at least one axis and every positive on both, with
    # margin >= 0.20 on strength.
    min_strength: float = 1.0
    min_coverage: float = 0.45
    min_relative_score: float = 0.2
    passages_per_hit: int = 3


@dataclass(frozen=True)
class _Candidate:
    score: float
    raw: float
    matched: int
    ticker_match: bool
    index: int


class Retriever:
    def __init__(self, corpus: Corpus, config: RetrieverConfig | None = None) -> None:
        self.corpus = corpus
        self.config = config or RetrieverConfig()
        self.index = BM25Index([tokenize(a.title) * 2 + tokenize(a.text) for a in corpus.articles])
        self._sentences = [
            [s for s in split_sentences(a.text) if is_prose(s)] for a in corpus.articles
        ]

    @property
    def known_terms(self) -> frozenset[str]:
        """The corpus vocabulary, for callers that need to spot out-of-corpus words."""
        return self.index.terms()

    def search(self, q: ResolvedQuery, k: int | None = None) -> list[Hit]:
        k = k or self.config.k
        own = list(dict.fromkeys(q.keywords))
        ctx = [t for t in dict.fromkeys(q.context_keywords) if t not in own]
        if not own and not ctx:
            return []
        weights = {t: 1.0 for t in own} | {t: self.config.context_weight for t in ctx}
        raw_scores = self.index.score(own + ctx, weights)
        total_idf = sum(self.index.idf(t) for t in own)
        wanted = set(q.tickers)
        # Soft inheritance: tickers boost the ranking but neither bypass the gate nor restrict.
        hard = q.inheritance != "soft"

        candidates: list[_Candidate] = []
        for i, article in enumerate(self.corpus.articles):
            raw = raw_scores[i]
            if raw <= 0:
                continue
            matched_own = [t for t in own if self.index.contains(i, t)]
            matched = len(matched_own) + sum(self.index.contains(i, t) for t in ctx)
            if matched == 0:
                continue
            ticker_match = bool(wanted) and (
                bool(wanted & set(article.tickers))
                or any(mentions(article.text, t) for t in wanted)
            )
            if not (ticker_match and hard):
                coverage = (
                    sum(self.index.idf(t) for t in matched_own) / total_idf if total_idf else 0.0
                )
                strength = raw / total_idf if total_idf else 0.0
                if strength < self.config.min_strength or coverage < self.config.min_coverage:
                    continue
            score = raw
            score *= self.config.ticker_boost if ticker_match else 1.0
            score *= self.config.truncated_penalty if article.is_truncated else 1.0
            candidates.append(_Candidate(score, raw, matched, ticker_match, i))

        candidates.sort(key=lambda c: (-c.score, c.index))
        if wanted and hard:
            restricted = [c for c in candidates if c.ticker_match]
            top_up = [c for c in candidates if not c.ticker_match]
            chosen = (restricted + top_up)[:k]
        else:
            chosen = candidates[:k]
        if not chosen:
            return []

        top = max(c.score for c in chosen)
        hits: list[Hit] = []
        for c in chosen:
            relative = c.score / top
            if relative < self.config.min_relative_score:
                continue
            hits.append(
                Hit(
                    article=self.corpus.articles[c.index],
                    score=round(relative, 4),
                    raw_score=round(c.raw, 4),
                    matched=c.matched,
                    passages=self.select_passages(c.index, q),
                )
            )
        return hits

    def select_passages(self, article_index: int, q: ResolvedQuery) -> list[str]:
        weights = {t: 1.0 for t in q.keywords} | {
            t: self.config.context_weight for t in q.context_keywords if t not in q.keywords
        }
        sentences = self._sentences[article_index]
        scored: list[tuple[float, int, str]] = []
        for position, sentence in enumerate(sentences):
            tokens = set(tokenize(sentence))
            overlap = sum(w for t, w in weights.items() if t in tokens)
            if overlap > 0:
                scored.append((-overlap, position, sentence))
        scored.sort()
        chosen = [s for _, _, s in scored[: self.config.passages_per_hit]]
        if not chosen and sentences:
            chosen = [sentences[0]]
        return chosen
