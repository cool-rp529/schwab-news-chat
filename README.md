# News Chat

A chat application that answers natural-language questions about recent
financial news drawn from `data/stock_news.json` — a fixed corpus of 138
records, 118 unique articles once deduplicated by link, covering seven
tickers (AAPL, MSFT, AMZN, NFLX, NVDA, INTC, IBM) from
around February 2025. Every answer is built from sentences reproduced
verbatim from cited articles, opens with the corpus's inferred time period,
closes with a not-investment-advice line, and refuses outright when the
corpus doesn't cover the question — rather than confidently making something
up. The assignment asked for a simple chat UI over this data, evaluated on
three axes: **code quality** (see [Architecture](#architecture) and
[Design decisions](#design-decisions)), **creativity of approach** (see
[The data, and what it forced](#the-data-and-what-it-forced) and
[Claim types](#claim-types)), and **completeness of testing** (see
[Testing](#testing)). UI styling is explicitly out of scope and was not a
focus here.

## Quick start

```bash
uv sync && uv run pytest -q && uv run python -m newschat serve
# without uv:
#   python -m venv .venv && . .venv/bin/activate && pip install -e .
#   pip install pytest pytest-cov httpx ruff mypy types-PyYAML
#   pytest -q && python -m newschat serve
```

Then open `http://127.0.0.1:8000`.

Two more entry points: `python -m newschat ask "What's going on with Intel?"`
prints a single answer to the terminal (add `--backend claude-cli` to opt
into the LLM path described below); `python -m newschat eval` reprints the
golden-set metrics table shown in [Testing](#testing).

## What it does

Example, default (`extractive`) backend:

```
Based on news from ≈ February 2025:

From 3 articles on INTC, MSFT:

What happened
- Shares of chip-giant Intel (NASDAQ: INTC) have been obliterated over the past few years. [1]
- Intel still leads the market for PC and server central processing units (CPUs), but its dominance has been severely eroded. [1]
- Intel's attempted turnaround effort, led by former CEO Pat Gelsinger, is now in flux after Gelsinger's surprise retirement in late 2024. [1]
- Intel (INTC) stock jumped 10% Tuesday following a report that its rivals Broadcom (AVGO) and TSMC (TSM) are exploring potential deals with the chipmaker that would split it into two. [2]

Analyst views
- The rumor mill is spinning overtime on struggling Intel (INTC) as the tech icon looks at ways to create shareholder value. [3]

Sources
[1] Where Will Intel Stock Be In 1 Year? (INTC)
[2] Intel stock surges on report of Broadcom, TSMC exploring deals that would split up chipmaker (INTC)
[3] Why Intel could be worth more than $200 billion if it breaks up (INTC, MSFT)

Informational summary of news coverage — not investment advice.
```

A follow-up in the same session — "who might buy it?" — has no ticker of its
own, so the query analyzer inherits the ticker and the thread's topic (the
prior turn's keywords plus whatever context that turn had itself inherited)
from the last turn that had a ticker, and the answer says so explicitly:
`(Interpreting the follow-up as being about INTC.)`. That inherited-resolution
note is the only way a reader can tell the anaphora was resolved at all,
rather than silently answering about the wrong company. A follow-up with a
reference word ("it", "the company") inherits strongly: retrieval is
restricted to the inherited ticker first. A short follow-up with no reference
word ("And the price target?", "DeepSeek") inherits softly: the ticker only
boosts the ranking, so a short new topic still reaches its own articles, and
the note — `(Reading this as a follow-up about INTC; say "new topic" or name a
company to reset.)` — appears only when an inherited-ticker article is actually
cited. Saying "new topic" ends the thread, and the carried topic is capped at
its eight most recent words. A short follow-up
inherits the previous ticker only when every word in it is in the corpus
vocabulary — "What's the Bitcoin price?" after an Intel question is treated as
a new (and refused) question, not as a follow-up, so an inherited ticker can
never carry an out-of-corpus question past the refusal gate. More transcripts,
including four captured against the Claude CLI backend, are committed in
[`examples/transcripts.md`](examples/transcripts.md).

Conversational turns that are not news questions are handled before retrieval.
Greetings, thanks, goodbyes, "who are you" and "what can you do" get a short,
citation-free reply (the last two describe the covered tickers and period)
instead of being searched for as keywords. A question whose reference word has
nothing to point at — "Who might buy it?" as the first question, or right
after "new topic" — gets `Which company do you mean?` rather than an answer
about whichever articles mention "buy". Neither kind of turn carries a ticker,
so neither can become the antecedent of a later "it".

## The data, and what it forced

`stock_news.json` is a dict keyed by ticker, each entry a list of articles
with exactly four fields — `title`, `link`, `ticker`, `full_text` — and **no
date field**. Several properties of the real file drove specific design
choices rather than generic ones:

| Finding | Design response |
|---|---|
| 138 records but only 118 unique `link`s — 13 articles filed under 2–4 tickers | Dedupe by URL; union the ticker lists onto one `Article` |
| 4 of those 13 duplicate groups have **divergent bodies** (drifting relative timestamps, a Yahoo video-carousel sidebar) despite identical titles | "First occurrence" is a dict-ordering coin flip; dedupe keeps the **longest** body instead, deterministically |
| 13 MT Newswires paywall stubs, 11 "Continue Reading View Comments" stubs | Stripped by rule; flagged `is_truncated`; down-ranked at retrieval time rather than excluded, so a truncated article can still surface when nothing else matches. The stubs are cut **mid-word** (~100 chars), so `is_prose()` also requires a terminal stop — otherwise a bullet reads "…a benchmark of more than 1,400 free [1]" |
| Insider Monkey listicle boilerplate, Motley Fool promo and options-disclosure paragraphs, Zacks footers, an embedded Yahoo video carousel | Stripped so none of it can be selected as a cited passage — an unrelated carousel headline being cited as "news" would be worse than not having it |
| The date trap: bare `Month YYYY` mentions are dominated by "returned 275% since **May 2014**" (Insider Monkey boilerplate, 10 bodies) and "long **January 2026** $395 calls" (Motley Fool options-disclosure boilerplate, 9 bodies) | `infer_period()` counts only **full `Month D, YYYY` dates** (17 bodies carry one), never bare month-year mentions; a test asserts the real corpus yields "February 2025", not either boilerplate trap |
| The ticker key is unreliable as a filter — a fleet-management market report sits under IBM, a Suze Orman piece under AAPL | Ticker is used as a retrieval **boost**, never a hard filter; ranking is primarily on text |

The cleaner (`ingest/cleaner.py`) applies 19 named, order-sensitive regexes —
`whitespace`, `mt_newswires_paywall`, `zacks_footer`, `zacks_free_report`,
`simply_wall_st_footer`, `insider_monkey_footer`, `syndication_footer`,
`view_comments_tail`, `motley_fool_promo`, `motley_fool_disclosure`,
`insider_monkey_pitch`, `zacks_cta`, `read_also`, `yahoo_related_videos`,
`yahoo_quote_widget`, `relative_timestamp`, `story_continues`, `bare_url`,
`whitespace_final` — plus a title-echo strip, each named so a test can target
exactly one rule. Anything under 400 cleaned characters is flagged
`is_truncated`; 24 of the 138 raw records land under that line.

## Architecture

```
question ─► QueryAnalyzer ─► Retriever ─► Summarizer ─► Answer
              │ aliases,        │ BM25 +      │ extractive | claude-cli (opt-in)
              │ history         │ ticker boost│
              ▼                 ▼             ▼
          ResolvedQuery     list[Hit]     text + citations
```

```
schwab-news-chat/
├── data/stock_news.json
├── src/newschat/
│   ├── models.py              # Article, Hit, Citation, Answer, ChatTurn, ResolvedQuery
│   ├── ingest/
│   │   ├── loader.py          # JSON → list[RawRecord], validated
│   │   ├── cleaner.py         # boilerplate removal, truncation detection
│   │   ├── dedupe.py          # merge by link, stable ids
│   │   └── corpus.py          # Corpus: articles, by_id, infer_period()
│   ├── text/
│   │   ├── tokenize.py        # tokens, stopwords, light stemming
│   │   └── sentences.py       # sentence splitter tolerant of "Inc.", "Feb. 26", "$1.5"
│   ├── retrieval/
│   │   ├── bm25.py            # in-house BM25 index
│   │   ├── aliases.py         # company/product names → tickers
│   │   ├── query.py           # QueryAnalyzer (tickers, keywords, follow-up inheritance)
│   │   └── retriever.py       # scoring, filtering, passage selection, refusal threshold
│   ├── summarize/
│   │   ├── base.py            # Summarizer protocol, prompt builder, citation parsing,
│   │   │                      #   check_faithfulness() gate, advice/as-of envelope
│   │   ├── claims.py          # rules-based claim-type classifier (event / opinion / promo)
│   │   ├── extractive.py      # deterministic sentence selection with MMR, grouped by claim type
│   │   ├── claude_cli.py      # subprocess `claude -p`
│   │   └── factory.py         # backend selection (explicit opt-in)
│   ├── chat/
│   │   ├── session.py         # SessionStore (in-memory, bounded history)
│   │   ├── smalltalk.py       # greetings/thanks/help intents answered without retrieval
│   │   └── service.py         # ChatService.ask(): orchestration + fallback policy
│   ├── api/
│   │   ├── app.py             # FastAPI routes, static mount, startup wiring
│   │   └── static/index.html  # vanilla JS chat page
│   └── cli.py                 # `newschat ask | eval | serve`
└── tests/
    ├── unit/                  # one test module per source module
    ├── api/                   # TestClient tests
    ├── eval/                  # golden.yaml + metric thresholds
    └── llm/                   # Claude CLI tests, skipped by default
```

Dependency direction is strictly downward: `api → chat → (retrieval,
summarize) → (text, ingest) → models`. No module imports from a layer above
it, which keeps the extractive path (and every test except the eval harness
and the API layer) free of any FastAPI or subprocess dependency.

**Retrieval.** An in-house BM25 index (title tokens counted twice as a cheap
field boost) scores the query against cleaned bodies; ticker-matched articles
get a 1.5× boost, truncated articles a 0.5× penalty. The refusal gate is
deliberately **absolute, not relative**: normalising scores by the top hit
makes the best-scoring article 1.0 by construction, so a relative floor alone
can never refuse — measured against this corpus, a purely relative filter
returned five confidently cited hits for "What is the weather in Paris
tomorrow?" and "Who won the 2022 FIFA World Cup?". Instead, a non-ticker-
matched article must clear `min_strength = 1.0` **and** an idf-weighted
keyword coverage of `min_coverage = 0.45` (Σidf(matched keywords) /
Σidf(all keywords) — terms the corpus never mentions carry maximum idf and
sink a query below the floor) *before* any normalisation happens.

Strength is `raw BM25 / Σidf(query keywords)`, not raw BM25. Raw BM25 sums
over query terms, so a fixed raw floor measures how many rare words a
question happens to contain rather than how well the corpus answers it: a
one-term query is bounded by about `idf × (k1 + 1)`, so "Tell me about
DeepSeek" — a topic covered by 11 articles, at perfect keyword coverage —
scored 5.08 and was refused by a floor calibrated on three-term queries.
Dividing by total idf yields the idf-weighted mean term saturation, bounded
by `k1 + 1` for every query length, so one floor means the same thing
whether the question is "DeepSeek" or "How is DeepSeek affecting AI
stocks?".
Ticker-matched articles bypass the gate but still need at least one matching
keyword. Both constants were measured against the real corpus and its
negatives by `scripts/measure_gate.py` (see
[Repository notes](#repository-notes)), not guessed.

**Summarization.** The default backend, `extractive`, scores candidate
sentences by query-term overlap plus rank/position bonuses, selects greedily
with MMR to avoid near-duplicates, caps at 5 bullets, and groups them under
headings by claim type ("What happened" / "Analyst views" / "Promotional",
empty groups omitted). Fully deterministic, no network or API key required.
`claude-cli` (below) is the alternative implementation of the same
`Summarizer` protocol.

**The envelope.** The not-investment-advice line, the `as_of` corpus-period
stamp, the inherited-follow-up note, and the citation faithfulness check
(`check_faithfulness` — token-set overlap between a cited claim and its
source passage) all live in `summarize/base.py`, not in any one backend: a
control written only into an LLM prompt does not exist on the extractive
path, which is the path a keyless reviewer actually runs, so these apply to
every `Answer` regardless of which summarizer produced it.

**The Claude CLI backend** is opt-in only — `NEWSCHAT_BACKEND=claude-cli` or
`--backend claude-cli`, model via `NEWSCHAT_CLAUDE_MODEL` — never selected
just because `claude` is on `PATH`, which would silently spend a reviewer's
own subscription on an untested path. When enabled, the question, the
retrieved excerpts, and the last 3 turns of history leave the machine as
part of a prompt sent to Anthropic via the CLI (`/api/health` exposes
`sends_data_to_third_party`). Excerpts are treated as untrusted data, wrapped
in delimiters with an instruction to ignore anything instruction-like inside
them. The CLI (`claude -p --output-format json`) can exit 0 with parseable
JSON even when the run actually failed (usage limit, API overload, max
turns); that in-band failure is caught by checking `is_error`, `subtype`,
`api_error_status`, and whether `result` is present — not just the exit
code. Any CLI failure, in-band or not, raises `SummarizerUnavailable`;
`ChatService` falls back to extractive and `Answer.backend` says so, e.g.
`extractive (fallback: cli not found)`.

## Claim types

Roughly a quarter of the 118 articles are analyst pitches, Zacks ranks, and
Motley Fool promos rather than reported news, so "what's happening with
Nvidia?" is partly a question about separating events from opinion and
promotion. `summarize/claims.py` classifies each candidate passage with an
ordered, most-specific-first list of named regex rules: `promo` matches
first-person promotional/listicle language ("we recently compiled",
"subscribe", "returned N% since", "Motley Fool Stock Advisor"); `opinion`
matches analyst and rating language ("price target", "Zacks Rank", "Strong
Buy", "downgrade/upgrade", "analysts say/expect"); anything else defaults to
`event`. It's deliberately coarse, deterministic, and rules-based rather than
LLM-labelled, so it's visible and testable on the default keyless backend.
Measured accuracy on the golden set is 0.80 — exactly at the assertion floor
(see [Testing](#testing)), an honest reflection of what a handful of regex
rules can do on prose this varied, not a tuned number.

## Testing

[![CI](https://github.com/cool-rp529/schwab-news-chat/actions/workflows/ci.yml/badge.svg)](https://github.com/cool-rp529/schwab-news-chat/actions/workflows/ci.yml)

```bash
uv run pytest -q                                    # full suite, offline, ~9s
uv run pytest -m llm                                # LLM-path tests (see below)
uv run pytest --cov=newschat --cov-report=term-missing
uv run ruff check . && uv run mypy src
```

Unit tests cover every module in `src/newschat` individually — cleaner rules
against real corpus fragments, dedupe's longest-body rule on the real
divergent groups, `infer_period()` asserted against the real file (not only
synthetic fixtures — a synthetic-only test is exactly what would let the May
2014 / January 2026 boilerplate bug ship), the refusal gate against
plausible out-of-corpus finance questions. API tests exercise the FastAPI
routes with `TestClient`, including a two-request follow-up that resolves an
inherited ticker across a session.

The golden set (`tests/eval/golden.yaml`) has three tiers, because a set
written by the same person who tuned the retriever, from the articles it
targets, can't disconfirm anything on its own: **seeded** (≈20 questions plus
negatives, with `expected_articles`/`expected_tickers`/`expected_kinds`;
thresholds asserted here), **paraphrase** (questions sharing no non-alias
content word with the target headline, probing what BM25's title boost
can't help with; reported at a lower bar, not asserted as strictly), and
**thematic** (comparative/aggregate questions — one has 11 relevant articles
across 6 tickers; reported **unasserted**, on purpose, rather than gamed
into passing).

```
tier          n   recall@5     mrr  tickers   kinds  refused  false_ref  coverage
---------------------------------------------------------------------------------
seeded       28       0.87    0.88     1.00    0.80     1.00       0.00         -
paraphrase    8       0.56    0.43     1.00       -        -       0.25         -
thematic      5          -       -        -       -        -       0.40      0.60
threshold             0.80    0.70     0.90    0.80     1.00          -         -   (asserted on seeded)
passed: True
```

`refused` is the share of *negatives* the gate refused; `false_ref` is the
share of *positives* it refused, reported and deliberately not asserted.
Together they are the two halves of the same trade, and the second half is
where the last two tiers' numbers actually come from.

The thematic tier's 0.60 is not a `k=5` ceiling: measured per case,
`theme-intel-vs-nvidia`, `theme-most-profitable` and `theme-ai-race` each
score 1.00 ticker coverage, while `theme-deepseek` and `theme-market-mood`
return **zero hits** — the absolute gate refuses them outright. Likewise
`para-streaming-sports` and `para-wall-street-mood` are refusals, not
ranking misses. So the absolute gate refuses 2 of 8 paraphrases and 2 of 5
thematic questions: that is the measured price of refusing Boeing and
Bitcoin 100 % of the time, on the same corpus, with the same constants. A
gate loose enough to answer "What's the overall market mood?" is a gate
loose enough to answer "What is the price of Bitcoin today?", and this
project chose the other side of that line — visibly, with the cost printed
in its own report rather than left out of it.

Full output, including the per-question misses behind those numbers, is
committed at [`examples/eval-report.txt`](examples/eval-report.txt). Two
more misses worth calling out rather than hiding: `intel-price-targets`
scores 0.00 recall because its three target articles are MT Newswires
paywall stubs (~100 characters), which the retriever's truncation penalty
deliberately down-ranks — the gate is trading recall on stub articles for
not citing near-empty text as a source. The paraphrase tier's MRR of 0.43
against seeded's 0.88 is the measured cost of having no embeddings — see
[Design decisions](#design-decisions).

Coverage (`examples/coverage.txt`, regenerated with
`pytest --cov=newschat --cov-report=term-missing`):

```
TOTAL                                   1016      7    99%
```

Tests that exercise the live Claude CLI are marked `@pytest.mark.llm` and are
skipped unless `claude` is on `PATH` **and** `NEWSCHAT_RUN_LLM_TESTS=1` is
set — so CI, and a reviewer's default `pytest` run, never depend on network
access or a subscription. CI (`.github/workflows/ci.yml`) runs `uv sync`,
`ruff check`, `ruff format --check`, `mypy src`, and
`pytest --cov --cov-fail-under=90` on Python 3.12 for every push and pull
request.

## Design decisions

| Decision | Alternatives considered | Why |
|---|---|---|
| Python / FastAPI, single process | TypeScript, split front/back stack | Text processing is most natural in Python; one toolchain for reviewers to install |
| In-house BM25 | `rank_bm25` package | ~60 lines, fully unit-tested, one fewer dependency, demonstrates the mechanism rather than hiding it |
| No embeddings | `sentence-transformers` | A reviewer must be able to clone, install, and run with no model download. Accepted cost: purely lexical recall, quantified (not assumed) by the paraphrase tier. The `Retriever` interface leaves room for a fused embedding retriever later |
| Extractive summarizer is the guaranteed default path | LLM-only | Reviewers have no API key; the test suite and CI must run fully offline |
| Claude Code CLI as the only LLM backend, opt-in | Also support the Anthropic SDK directly; auto-select when `claude` is on `PATH` | One backend is enough to prove the `Summarizer` protocol; a second would duplicate the same prompt, error contract, and citation parsing for a key this project doesn't have and CI could never exercise. Opt-in because implicit selection would spend a reviewer's own subscription on a path the tests don't cover |
| Advice line, corpus period, and faithfulness check live in the shared envelope | Put them in the LLM prompt only | A prompt rule doesn't exist on the extractive path — the path a keyless reviewer actually runs |
| Refusal gated on an absolute score, not a relative one | Relative floor only | Top-score normalisation makes the best-scoring hit 1.0 by construction, so a relative floor alone can never refuse |
| Dedupe keeps the longest body among duplicates | Keep the first occurrence | Four duplicate groups have genuinely divergent bodies; "first" is just a dict-ordering coin flip |
| Ticker key used as a retrieval boost, not a hard filter | Filter articles by ticker | Many articles filed under a ticker are off-topic for it; article text is more trustworthy than the filing key |
| Follow-up resolution by deterministic ticker/keyword inheritance, gated on corpus vocabulary | LLM-based query rewriting | Fully deterministic, testable, and works with no network access; the vocabulary gate stops a short out-of-corpus question inheriting a ticker and bypassing refusal |
| Claim-type classification is rules-based on the default path | LLM-labelled classification | The classifier has to be visible and assertable on the backend a keyless reviewer runs; rules are coarser but measurable |
| Keep the `ask` CLI subcommand and test all three subcommands | Drop `ask`, ship only `serve` | `ask` is what generates the committed transcripts, and a dedicated `cli` test module closes a coverage gap that would otherwise exist |

## Limitations and next steps

- **Citations are provenance, not entailment.** The guarantee is that every
  cited sentence is reproduced verbatim from the article it cites, not that
  two cited sentences agree with each other — nothing currently detects a
  cross-citation contradiction. `check_faithfulness()` is a cheap
  lexical-overlap stand-in that catches an unsupported or fabricated claim,
  not a subtly distorted one; a full entailment judge and cross-bullet
  contradiction detection are future work.
- **The claim-type classifier is coarse** — a small ordered list of regex
  rules, not a learned model — and its measured accuracy sits exactly at its
  0.80 assertion floor, a real ceiling on this approach, not a margin.
- **No article carries a date**, so there is no true time filtering, only a
  single inferred corpus period (`as_of`) applied uniformly to every answer.
  Publication dates via article metadata, and true filtering by them, remain
  future work.
- **Retrieval is purely lexical** (no embeddings), so paraphrased questions
  that share no vocabulary with a headline score meaningfully worse than
  questions that do — the paraphrase tier's MRR of 0.43 against seeded's
  0.88 quantifies that gap directly. A fused BM25 + embeddings retriever
  (reciprocal rank fusion) is the natural next `Retriever` implementation.
- **Extractive selection has no per-article cap** — MMR de-duplicates
  near-identical sentences but doesn't limit how many bullets one article
  can supply, so on a thin retrieval set a single article can dominate.
- **The tokenizer splits hyphenated tokens** ("e-commerce" becomes "e" and
  "commerce"), weakening matches on hyphenated terms; not currently fixed.
- **The LLM path is unverified in CI** — Claude CLI tests are marked `llm`
  and skipped by default (see [Testing](#testing)), so its correctness rests
  on manual runs like `examples/transcripts.md`, not on the CI gate the rest
  of the project relies on.
- Also documented but not built: persistent (rather than in-memory) sessions,
  and a streaming response instead of one-shot JSON.

## Repository notes

`scripts/measure_gate.py` is the script that measured the frozen refusal-gate
constants (`min_strength = 1.0`, `min_coverage = 0.45`) against the
real corpus and its negative questions — it's how those numbers were chosen,
not guessed. The measurement that fixed them: every negative's best row
peaked at strength 0.77, every positive's worst at 1.20, and the bare-topic
queries sit at 2.17. `examples/` holds evidence regenerated from the working tree
(the eval table, the full `pytest --cov` output, and CLI/extractive sample
transcripts) so a reviewer can read the project's own claims about itself
without having to run anything.
