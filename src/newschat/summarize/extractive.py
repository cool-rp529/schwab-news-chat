"""Deterministic summarizer: pick sentences from the retrieved passages with MMR, group them
by claim type, cite each verbatim. This is the guaranteed offline path."""

from collections.abc import Sequence
from dataclasses import dataclass

from newschat.models import Answer, ChatTurn, Citation, ClaimKind, Hit, ResolvedQuery
from newschat.summarize.base import CorpusInfo, envelope, make_citations, not_covered_answer
from newschat.summarize.claims import classify, is_advice
from newschat.text.tokenize import tokenize

KIND_HEADINGS: dict[ClaimKind, str] = {
    "event": "What happened",
    "opinion": "Analyst views",
    "promo": "Promotional",
}
_KIND_ORDER: tuple[ClaimKind, ...] = ("event", "opinion", "promo")


@dataclass(frozen=True)
class ExtractiveConfig:
    max_sentences: int = 5
    mmr_lambda: float = 0.7
    duplicate_jaccard: float = 0.6
    rank_bonus: float = 0.15
    lead_bonus: float = 0.05
    promo_penalty: float = 0.5
    # 0.2, not 0.7: at 0.7 an advice-shaped headline sentence ("Should you buy X stock hand
    # over fist before Feb. 19?") still out-scored the event sentences and led the answer.
    advice_penalty: float = 0.2


@dataclass(frozen=True)
class _Candidate:
    order: int  # flattened (hit rank, position) — the deterministic tie-breaker
    n: int  # citation number == hit rank, 1-based
    position: int
    sentence: str
    tokens: frozenset[str]
    relevance: float
    kind: ClaimKind


def jaccard(a: set[str] | frozenset[str], b: set[str] | frozenset[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / len(a | b)


def dense_numbering(cited: Sequence[int]) -> dict[int, int]:
    """Map hit ranks to contiguous citation numbers.

    A hit can be retrieved and still contribute no sentence -- MMR drops a near-duplicate, or
    max_sentences runs out before its turn. Keeping its rank renders Sources as [1] [2] [4],
    which reads as a source that went missing rather than one that was never cited.
    """
    return {n: i for i, n in enumerate(cited, start=1)}


class ExtractiveSummarizer:
    name = "extractive"

    def __init__(self, info: CorpusInfo, config: ExtractiveConfig | None = None) -> None:
        self.info = info
        self.config = config or ExtractiveConfig()

    def summarize(
        self, query: ResolvedQuery, hits: Sequence[Hit], history: Sequence[ChatTurn]
    ) -> Answer:
        if not hits:
            return not_covered_answer(query, self.name, self.info)
        selected = self._select(self._candidates(query, hits))
        # Every hit can be a truncated stub with no eligible passage; rendering "From 0
        # articles on :" would claim coverage that no sentence backs.
        if not selected:
            return not_covered_answer(query, self.name, self.info)
        return envelope(
            body=self._render(hits, selected),
            citations=self._citations(hits, selected),
            backend=self.name,
            query=query,
            covered=True,
            info=self.info,
        )

    # -- selection -------------------------------------------------------------------------

    def _candidates(self, query: ResolvedQuery, hits: Sequence[Hit]) -> list[_Candidate]:
        terms = set(query.keywords) | set(query.context_keywords)
        out: list[_Candidate] = []
        order = 0
        for n, hit in enumerate(hits, start=1):
            for position, sentence in enumerate(hit.passages):
                tokens = frozenset(tokenize(sentence))
                relevance = jaccard(terms, tokens) + self.config.rank_bonus / n
                if hit.article.text.startswith(sentence):
                    relevance += self.config.lead_bonus
                kind = classify(sentence, hit.article.title)
                if kind == "promo":
                    relevance *= self.config.promo_penalty
                if is_advice(sentence):
                    relevance *= self.config.advice_penalty
                out.append(_Candidate(order, n, position, sentence, tokens, relevance, kind))
                order += 1
        return out

    def _select(self, candidates: list[_Candidate]) -> list[_Candidate]:
        remaining = list(candidates)
        selected: list[_Candidate] = []
        lam = self.config.mmr_lambda
        while remaining and len(selected) < self.config.max_sentences:
            best: tuple[float, _Candidate] | None = None
            for c in remaining:
                max_sim = max((jaccard(c.tokens, s.tokens) for s in selected), default=0.0)
                if max_sim > self.config.duplicate_jaccard:
                    continue
                mmr = lam * c.relevance - (1 - lam) * max_sim
                if best is None or mmr > best[0] or (mmr == best[0] and c.order < best[1].order):
                    best = (mmr, c)
            if best is None:
                break
            selected.append(best[1])
            remaining.remove(best[1])
        return selected

    # -- rendering -------------------------------------------------------------------------

    def _render(self, hits: Sequence[Hit], selected: Sequence[_Candidate]) -> str:
        cited = sorted({c.n for c in selected})
        dense = dense_numbering(cited)
        tickers = sorted({t for n in cited for t in hits[n - 1].article.tickers})
        noun = "article" if len(cited) == 1 else "articles"
        lines = [f"From {len(cited)} {noun} on {', '.join(tickers)}:", ""]
        for kind in _KIND_ORDER:
            group = sorted((c for c in selected if c.kind == kind), key=lambda c: (c.n, c.position))
            if not group:
                continue
            lines.append(KIND_HEADINGS[kind])
            lines.extend(f"- {c.sentence} [{dense[c.n]}]" for c in group)
            lines.append("")
        lines.append("Sources")
        for n in cited:
            article = hits[n - 1].article
            lines.append(f"[{dense[n]}] {article.title} ({', '.join(article.tickers)})")
        return "\n".join(lines)

    def _citations(self, hits: Sequence[Hit], selected: Sequence[_Candidate]) -> list[Citation]:
        first_per_hit: dict[int, _Candidate] = {}
        for c in sorted(selected, key=lambda c: c.order):
            first_per_hit.setdefault(c.n, c)
        all_citations = make_citations(hits)
        dense = dense_numbering(sorted(first_per_hit))
        return [
            all_citations[n - 1].model_copy(
                update={"n": dense[n], "passage": c.sentence, "kind": c.kind}
            )
            for n, c in sorted(first_per_hit.items())
        ]
