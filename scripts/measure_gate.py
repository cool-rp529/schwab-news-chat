"""Print raw BM25 / coverage for the gate's positive and negative probes so the constants in
RetrieverConfig are frozen from a measurement, not a guess.

Run: PYTHONIOENCODING=utf-8 uv run python scripts/measure_gate.py
"""

from newschat.ingest.corpus import Corpus
from newschat.retrieval.query import QueryAnalyzer
from newschat.retrieval.retriever import Retriever, RetrieverConfig

NEGATIVES = [
    "What are Boeing's 737 MAX safety issues?",
    "What's the latest on Tesla's Cybertruck recall?",
    "What is the weather in Tokyo tomorrow?",
    "What is the price of Bitcoin today?",
    "Who won the Super Bowl?",
    "How do I bake sourdough bread?",
    "What is the best pizza in Chicago?",
]
POSITIVES = [
    "What has Warren Buffett been buying?",
    "How is DeepSeek affecting AI stocks?",
    "Why are BofA strategists bullish?",
    "Which retailers face the February 28 boycott?",
    "What happened with the OpenAI nonprofit board?",
    "Is Meta's winning streak at risk?",
]


def main() -> None:
    corpus = Corpus.load()
    # Gate disabled so we can see what *would* pass.
    retriever = Retriever(corpus, RetrieverConfig(min_absolute_score=0.0, min_coverage=0.0))
    analyzer = QueryAnalyzer()
    for label, questions in (("NEGATIVE", NEGATIVES), ("POSITIVE", POSITIVES)):
        print(f"\n== {label} ==")
        for question in questions:
            q = analyzer.resolve(question, [])
            total_idf = sum(retriever.index.idf(t) for t in q.keywords) or 1.0
            raw_scores = retriever.index.score(q.keywords)
            rows = []
            for i, raw in enumerate(raw_scores):
                if raw <= 0:
                    continue
                matched = [t for t in q.keywords if retriever.index.contains(i, t)]
                cov = sum(retriever.index.idf(t) for t in matched) / total_idf
                rows.append((raw, cov, corpus.articles[i].title[:60]))
            rows.sort(reverse=True)
            print(f"\n{question}  keywords={q.keywords}")
            for raw, cov, title in rows[:3]:
                print(f"  raw={raw:5.2f} cov={cov:4.2f}  {title}")


if __name__ == "__main__":
    main()
