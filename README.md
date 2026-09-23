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

**Contents:** [Quick start](#quick-start) · [What it does](#what-it-does) ·
[The data](#the-data-and-what-it-forced) · [Architecture](#architecture) ·
[API reference](#api-reference) · [Configuration](#configuration) ·
[Claim types](#claim-types) · [Testing](#testing) · [Design decisions](#design-decisions) ·
[Limitations](#limitations-and-next-steps) · [Repository notes](#repository-notes)

## Quick start

**Requirements:** Python ≥ 3.12. [`uv`](https://docs.astral.sh/uv/) is
recommended but optional. No API key and no network access are needed for the
default path (after dependencies are installed).

```bash
uv sync
uv run pytest -q                  # full suite, offline
uv run python -m newschat serve   # then open http://127.0.0.1:8000
```

Without `uv`, use a virtual environment and install the package plus dev tools:

```bash
# macOS / Linux (bash, zsh)
python -m venv .venv && . .venv/bin/activate
pip install -e . pytest pytest-cov httpx ruff mypy types-PyYAML
pytest -q && python -m newschat serve
```

```powershell
# Windows PowerShell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
pip install -e . pytest pytest-cov httpx ruff mypy types-PyYAML
pytest -q; python -m newschat serve
```

The CLI reconfigures its own stdout to UTF-8, so answers (which contain `≈`
and `—`) print correctly in any console; only `pytest -s` output on a legacy
Windows code page may need `PYTHONIOENCODING=utf-8`.

Other entry points: `python -m newschat ask "What's going on with Intel?"`
prints one answer (add `--backend claude-cli` for the opt-in LLM path), and
`python -m newschat eval` reprints the metrics table in [Testing](#testing).

The page renders claim-type groups as headings with bullets and each `[n]` as
a footnote button that expands the matching source (title link, tickers, claim
kind, grounding passage, word count). "New topic" resets the follow-up context,
"New chat" starts a fresh session, and each answer has a Copy button.

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

More transcripts, including four captured against the Claude CLI backend, are
in [`examples/transcripts.md`](examples/transcripts.md).

### How a question is answered

1. **Conversational turns first.** Greetings, thanks, goodbyes, "who are you"
   and "what can you do" get a short citation-free reply instead of a search
   (the last two list the covered tickers and period).
2. **Resolve the query.** Input is NFKC-normalised (fullwidth "Ｉｎｔｅｌ" reads
   as "Intel"; a cashtag such as `$INTC` is searched as `intc`); company names
   and aliases map to tickers; the remaining words become keywords; follow-up
   rules (below) may inherit a ticker and topic from earlier turns.
3. **Retrieve.** BM25 over cleaned bodies, with a ticker boost, a truncation
   penalty, and carried-over topic words at half weight.
4. **Refusal gate.** An absolute evidence floor (see
   [Architecture](#architecture)); if nothing clears it, the answer says the
   corpus doesn't cover the question.
5. **Summarize.** `extractive` (default, deterministic) or `claude-cli`
   (opt-in), both behind one `Summarizer` protocol.
6. **Envelope.** As-of period, follow-up note, per-citation faithfulness check
   and advice line, applied whichever backend produced the answer.

### Follow-up rules

A follow-up inherits from the latest turn that had a ticker: that ticker, plus
the turn's keywords and whatever it had itself inherited (capped at the eight
most recent words). Small talk and clarifications carry no ticker, so they
never become the antecedent of a later "it".

| Kind | When | Effect |
|---|---|---|
| **strong** | A reference word: "it", "they", "that", "the company", "the stock", … ("Who might buy it?") | Inherits ticker and topic; retrieval is restricted to the inherited ticker first. Note: `(Interpreting the follow-up as being about INTC.)` |
| **soft** | No reference word, at most four words, and every keyword is in the corpus vocabulary ("And the price target?", "DeepSeek") | Ticker is only a ranking boost, so a short new topic still reaches its own articles. Note `(Reading this as a follow-up about INTC; say "new topic" or name a company to reset.)` appears only if an inherited-ticker article is actually cited |
| **none** | The question names a ticker; is longer than four words with no reference word; or is short but contains a word the corpus never uses ("What's the Bitcoin price?") | Treated as a new question, so an inherited ticker can never carry an out-of-corpus question past the refusal gate |
| **reset** | The question contains "new topic" | Ends the thread; later turns never inherit from behind it |
| **clarification** | A reference word with no earlier ticker in the thread, and no specific (rare) topic word ("Who might buy it?" as the first question) | Reply is `Which company do you mean?`; "What is this DeepSeek thing?" is searched as usual |

## The data, and what it forced

`stock_news.json` is a dict keyed by ticker, each entry a list of articles
with exactly four fields — `title`, `link`, `ticker`, `full_text` — and **no
date field**. Properties of the real file drove specific design choices:

| Finding | Design response |
|---|---|
| 138 records but only 118 unique `link`s — 13 articles filed under 2–4 tickers | Dedupe by URL; union the ticker lists onto one `Article` |
| 4 of those 13 duplicate groups have **divergent bodies** (drifting relative timestamps, a Yahoo video-carousel sidebar) despite identical titles | "First occurrence" is a dict-ordering coin flip; dedupe keeps the **longest** body instead, deterministically |
| 13 MT Newswires paywall stubs, 11 "Continue Reading View Comments" stubs | Stripped by rule; flagged `is_truncated`; down-ranked at retrieval time rather than excluded, so a truncated article can still surface when nothing else matches. The stubs are cut **mid-word** (~100 chars), so `is_prose()` also requires a terminal stop — otherwise a bullet reads "…a benchmark of more than 1,400 free [1]" |
| Insider Monkey listicle boilerplate, Motley Fool promo and options-disclosure paragraphs, Zacks footers, an embedded Yahoo video carousel | Stripped so none of it can be selected as a cited passage — an unrelated carousel headline being cited as "news" would be worse than not having it |
| The date trap: bare `Month YYYY` mentions are dominated by "returned 275% since **May 2014**" (Insider Monkey boilerplate, 10 articles) and "long **January 2026** $395 calls" (Motley Fool options-disclosure boilerplate, 9 mentions in 5 articles) | `infer_period()` counts only **full `Month D, YYYY` dates** (25 of the 118 cleaned articles carry one), never bare month-year mentions; a test asserts the real corpus yields "February 2025", not either boilerplate trap |
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
├── .github/workflows/ci.yml   # lint, format, types, tests + coverage floor on push/PR
├── data/stock_news.json
├── examples/                  # committed evidence: eval report, coverage, transcripts
├── scripts/measure_gate.py    # measures the refusal-gate constants against the corpus
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
    └── llm/                   # live Claude CLI test, skipped by default
```

Dependencies point strictly downward: `api → chat → (retrieval, summarize) →
(text, ingest) → models`, which keeps the extractive path (and every test
except the eval harness and API layer) free of FastAPI and subprocesses.

**Retrieval.** An in-house BM25 index (title tokens counted twice as a cheap
field boost) scores cleaned bodies; ticker-matched articles get a 1.5× boost,
truncated articles a 0.5× penalty. The refusal gate is **absolute, not
relative**: normalising by the top hit makes the best article 1.0 by
construction, so a relative floor alone never refuses — measured here, it
returned five confidently cited hits for "What is the weather in Paris
tomorrow?" and "Who won the 2022 FIFA World Cup?". Instead, a non-ticker-matched
article must clear `min_strength = 1.0` **and** idf-weighted keyword coverage
`min_coverage = 0.45` (Σidf(matched) / Σidf(all keywords); words the corpus
never uses carry maximum idf) *before* any normalisation.

Strength is `raw BM25 / Σidf(query keywords)`, not raw BM25. Raw BM25 sums
over query terms, so a raw floor measures how many rare words a question
contains rather than how well the corpus answers it: "Tell me about DeepSeek"
(11 articles, perfect coverage) is bounded by about `idf × (k1 + 1)`, scores
about 5.1 raw, and was refused by a floor calibrated on three-term queries.
Dividing by total idf gives the idf-weighted mean term saturation, bounded by
`k1 + 1` at every query length, so one floor means the same thing for
"DeepSeek" and "How is DeepSeek affecting AI stocks?". Ticker-matched articles
bypass the gate but still need one matching keyword. Both constants were
measured by `scripts/measure_gate.py` (see
[Repository notes](#repository-notes)), not guessed.

**Summarization.** `extractive` scores sentences by query-term overlap plus
rank/position bonuses, selects greedily with MMR to avoid near-duplicates,
caps at 5 bullets, and groups them by claim type ("What happened" / "Analyst
views" / "Promotional"). It is deterministic and needs no network or key;
`claude-cli` is the alternative implementation of the same protocol.

**The envelope.** The advice line, the `as_of` stamp, the follow-up note,
and the citation faithfulness check (`check_faithfulness` — token-set overlap
between a cited claim and its source passage) live in `summarize/base.py`,
not in a backend: a control written only into an LLM prompt does not exist on
the extractive path a keyless reviewer runs.

**The Claude CLI backend** is opt-in only (see [Configuration](#configuration)),
never selected just because `claude` is on `PATH`, which would silently spend a
reviewer's own subscription on an untested path. When enabled, the question,
the retrieved excerpts, and the last 3 turns of history are sent to Anthropic
via the CLI (`/api/health` reports `sends_data_to_third_party`). Excerpts are
wrapped in delimiters as untrusted data. `claude -p --output-format json` can
exit 0 with parseable JSON when the run failed (usage limit, overload, max
turns), so `is_error`, `subtype`, `api_error_status` and a missing `result`
are all checked. Any failure raises `SummarizerUnavailable`; `ChatService`
falls back to extractive and says so in `Answer.backend`, e.g.
`extractive (fallback: 'claude' not found on PATH)`.

## API reference

The server (`python -m newschat serve`) exposes three JSON endpoints plus the
page at `/`. FastAPI also serves interactive docs at `/docs`.

**`POST /api/chat`** — ask one question.

| Request field | Type | Notes |
|---|---|---|
| `question` | string, 1–1000 chars | Required. Blank-after-trim or longer than 1000 → **422** |
| `session_id` | string ≤ 64 chars, optional | Omit to start a session; send the returned id back for follow-ups |

A missing, empty or over-long field returns FastAPI's validation error; a
whitespace-only question returns `{"detail": "question must not be blank"}`.
An unknown `session_id` starts a new history under that id. The response is
`{"session_id": str, "answer": Answer}`:

| `Answer` field | Meaning |
|---|---|
| `text` | The full rendered answer, as printed by `ask` |
| `citations[]` | `n` (the `[n]` marker), `article_id`, `title`, `url`, `tickers`, `passage` (the grounding sentence), `verified` (passed `check_faithfulness`), `kind` (`event` \| `opinion` \| `promo`) |
| `backend` | `extractive`, `claude-cli`, or `extractive (fallback: <reason>)` |
| `covered` | `false` for refusals, small talk and clarifications |
| `as_of` | Inferred corpus period, e.g. `February 2025` |
| `advisory` | The not-investment-advice line |
| `query` | How the question was read: `text`, `tickers`, `keywords`, `context_keywords` (topic carried from earlier turns), `inherited_tickers`, `inheritance` (`none` \| `strong` \| `soft`), `unresolved_reference`, `inherited_keywords` |

**`GET /api/articles/{article_id}`** — 200 with the full `Article` (`id`,
`title`, `url`, `source`, `tickers`, `text` (cleaned body), `is_truncated`,
`word_count`), or 404 for an unknown id. The page uses it to expand a
citation.

**`GET /api/health`** — `articles` (118), `unique_tickers`, `backend`,
`period`, and `sends_data_to_third_party` (`true` only when the backend is not
`extractive`).

```bash
curl -s -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "What is going on with Intel?"}'
# follow-up: reuse the session_id from the first response
curl -s -X POST http://127.0.0.1:8000/api/chat -H "Content-Type: application/json" \
  -d '{"question": "Who might buy it?", "session_id": "<session_id>"}'
```

In Windows PowerShell (where `curl` is an alias for `Invoke-WebRequest`):

```powershell
$u = "http://127.0.0.1:8000/api/chat"
$r = Invoke-RestMethod $u -Method Post -ContentType application/json -Body '{"question": "What is going on with Intel?"}'
Invoke-RestMethod $u -Method Post -ContentType application/json -Body (@{question = "Who might buy it?"; session_id = $r.session_id} | ConvertTo-Json)
```

## Configuration

| Setting | Default | Effect |
|---|---|---|
| `NEWSCHAT_BACKEND` | `extractive` | `extractive` or `claude-cli` (opt-in LLM path; needs the Claude Code CLI on `PATH`) |
| `NEWSCHAT_CLAUDE_MODEL` | CLI's default | Model passed to `claude -p` when the backend is `claude-cli` |
| `NEWSCHAT_DATA` | `data/stock_news.json` | Path to another JSON file of the same shape (dict of ticker → list of `title`/`link`/`ticker`/`full_text`) |
| `NEWSCHAT_RUN_LLM_TESTS` | unset | `1` enables the live Claude CLI test (also needs `claude` on `PATH`) |

CLI flags: `ask QUESTION [--backend extractive|claude-cli] [--data PATH]`;
`eval [--golden PATH] [--data PATH]` (exit 1 if an asserted threshold fails);
`serve [--host 127.0.0.1] [--port 8000]`. Flags override the environment.
Sessions are in memory (6 turns each, at most 1000) and lost on restart.

## Claim types

Roughly a quarter of the 118 articles are analyst pitches, Zacks ranks, and
Motley Fool promos rather than reported news, so answers separate events from
opinion and promotion. `summarize/claims.py` applies an ordered,
most-specific-first list of named regex rules: `promo` matches promotional or
listicle language ("we recently compiled", "subscribe", "returned N% since",
"Motley Fool Stock Advisor"); `opinion` matches analyst and rating language
("price target", "Zacks Rank", "Strong Buy", "downgrade/upgrade", "analysts
say/expect"); anything else is `event`. It is coarse, deterministic and
testable on the keyless backend. Measured accuracy on the golden set is 0.80 —
exactly at the assertion floor (see [Testing](#testing)), not a tuned number.

## Testing

[![CI](https://github.com/cool-rp529/schwab-news-chat/actions/workflows/ci.yml/badge.svg)](https://github.com/cool-rp529/schwab-news-chat/actions/workflows/ci.yml)

```bash
uv run pytest -q                                    # 369 tests, offline, ~10 s (1 skipped: live CLI)
uv run pytest -m llm                                # LLM-path tests (see below)
uv run pytest --cov=newschat --cov-report=term-missing
uv run ruff check . && uv run ruff format --check . && uv run mypy src
```

Unit tests cover every module in `src/newschat` — cleaner rules against real
corpus fragments, dedupe's longest-body rule on the real divergent groups,
`infer_period()` against the real file (a synthetic-only test is exactly what
would let the May 2014 / January 2026 boilerplate bug ship), and the refusal
gate against plausible out-of-corpus finance questions. API tests drive the
routes with `TestClient`, including a two-request follow-up across a session.

The golden set (`tests/eval/golden.yaml`) has three tiers, because a set
written by whoever tuned the retriever can't disconfirm anything on its own:
**seeded** (23 questions plus 6 negatives, with `expected_articles` /
`expected_tickers` / `expected_kinds`; thresholds asserted), **paraphrase**
(no non-alias content word shared with the target headline, probing what
BM25's title boost can't help; reported at a lower bar), and **thematic**
(comparative/aggregate — one has 11 relevant articles across 6 tickers;
reported **unasserted**, on purpose, rather than gamed into passing).

```
tier          n   recall@5     mrr  tickers   kinds  refused  false_ref  coverage
---------------------------------------------------------------------------------
seeded       29       0.88    0.88     1.00    0.80     1.00       0.00         -
paraphrase    8       0.56    0.43     1.00       -        -       0.25         -
thematic      5          -       -        -       -        -       0.40      0.60
threshold             0.80    0.70     0.90    0.80     1.00          -         -   (asserted on seeded)
passed: True
```

`refused` is the share of *negatives* the gate refused; `false_ref` the share
of *positives* it refused (reported, not asserted) — two halves of one trade,
and the second half is where the last two tiers' numbers come from.

The thematic tier's 0.60 is not a `k=5` ceiling: `theme-intel-vs-nvidia`,
`theme-most-profitable` and `theme-ai-race` each score 1.00 ticker coverage,
while `theme-deepseek` and `theme-market-mood` return **zero hits** — the
absolute gate refuses them, as it does `para-streaming-sports` and
`para-wall-street-mood`. Refusing 2 of 8 paraphrases and 2 of 5 thematic
questions is the measured price of refusing Boeing and Bitcoin 100 % of the
time with the same constants: a gate loose enough to answer "What's the
overall market mood?" is loose enough to answer "What is the price of Bitcoin
today?", and this project chose the other side of that line, with the cost
printed in its own report.

Per-question misses are in
[`examples/eval-report.txt`](examples/eval-report.txt). One worth calling out:
`intel-price-targets` scores 0.00 recall because its three targets are MT
Newswires paywall stubs (~100 characters), which the truncation penalty
deliberately down-ranks rather than cite near-empty text. The paraphrase
tier's MRR of 0.43 against seeded's 0.88 is the measured cost of having no
embeddings (see [Design decisions](#design-decisions)).

Coverage ([`examples/coverage.txt`](examples/coverage.txt), regenerated with
`pytest --cov=newschat --cov-report=term-missing`):

```
TOTAL                                   1143      8    99%
```

The live Claude CLI test (`@pytest.mark.llm`) is skipped unless `claude` is
on `PATH` **and** `NEWSCHAT_RUN_LLM_TESTS=1`, so CI and a default run never need
network or a subscription. CI (`.github/workflows/ci.yml`) runs `uv sync`,
`ruff check`, `ruff format --check`, `mypy src`, and `pytest
--cov=newschat --cov-fail-under=90` on Python 3.12 for every push and pull
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

- **Citations are provenance, not entailment.** Every cited sentence is
  verbatim from its article, but nothing checks that two cited sentences agree.
  `check_faithfulness()` is a lexical-overlap stand-in that catches a fabricated
  claim, not a subtly distorted one; an entailment judge is future work.
- **The claim-type classifier is coarse** — ordered regex rules — and its
  accuracy sits exactly at its 0.80 assertion floor: a ceiling, not a margin.
- **No article carries a date**, so there is no time filtering, only one
  inferred corpus period (`as_of`) applied to every answer.
- **Retrieval is purely lexical.** Paraphrases that share no vocabulary with a
  headline score worse (paraphrase MRR 0.43 vs seeded 0.88); a fused BM25 +
  embeddings retriever (reciprocal rank fusion) is the natural next step.
- **No typo tolerance.** "What's going on with intell?" is refused: "intell"
  is neither an alias nor a corpus word, and there is no fuzzy matching.
- **The gate measures evidence, not relevance.** Any question whose words
  genuinely occur in the corpus gets answered: "Best restaurants in Chicago"
  cites a Zacks preview datelined Chicago and a McDonald's article.
- **Comparisons are not balanced per ticker.** "Compare Apple and Netflix"
  ranks both together, so the higher-scoring side can take every bullet.
- **No per-article cap in extractive selection** — MMR removes near-duplicate
  sentences, but one article can still supply most bullets on a thin result.
- **The tokenizer splits hyphenated tokens** ("e-commerce" → "e", "commerce").
- **The LLM path is not exercised in CI**; its evidence is manual runs such as
  [`examples/transcripts.md`](examples/transcripts.md).
- Not built: persistent sessions, streaming responses.

## Repository notes

`scripts/measure_gate.py` (`uv run python scripts/measure_gate.py`) measured
the gate constants (`min_strength = 1.0`, `min_coverage = 0.45`) on the real
corpus: every negative's best row peaked at strength 0.77, every positive's
worst at 1.20, and bare-topic queries sit at 2.17. `examples/` holds evidence
regenerated from the working tree (eval report, full `pytest --cov` output,
sample transcripts), so the project's claims can be checked without running it.
