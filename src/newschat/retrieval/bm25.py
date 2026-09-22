"""Okapi BM25 over pre-tokenised documents. ~60 lines, no dependencies.

idf uses the +1 smoothing variant so a term present in every document still contributes a
small positive weight rather than a negative one.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence


class BM25Index:
    def __init__(self, docs: Sequence[Sequence[str]], k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._tfs: list[Counter[str]] = [Counter(d) for d in docs]
        self._lengths = [len(d) for d in docs]
        self.n_docs = len(docs)
        self._avg_len = (sum(self._lengths) / self.n_docs) if self.n_docs else 0.0
        self._df: Counter[str] = Counter(term for tf in self._tfs for term in tf)

    def df(self, term: str) -> int:
        return self._df.get(term, 0)

    def idf(self, term: str) -> float:
        df = self.df(term)
        return math.log((self.n_docs - df + 0.5) / (df + 0.5) + 1.0)

    def contains(self, doc_index: int, term: str) -> bool:
        return term in self._tfs[doc_index]

    def score(
        self, query: Sequence[str], weights: Mapping[str, float] | None = None
    ) -> list[float]:
        scores = [0.0] * self.n_docs
        for term in dict.fromkeys(query):  # unique, order preserved
            if self.df(term) == 0:
                continue
            idf = self.idf(term)
            weight = weights.get(term, 1.0) if weights else 1.0
            for i, tf in enumerate(self._tfs):
                freq = tf.get(term, 0)
                if not freq:
                    continue
                norm = self.k1 * (1 - self.b + self.b * self._lengths[i] / self._avg_len)
                scores[i] += weight * idf * (freq * (self.k1 + 1)) / (freq + norm)
        return scores
