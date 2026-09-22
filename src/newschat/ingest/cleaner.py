"""Boilerplate removal for scraped article bodies.

Every rule is a named, compiled regex so a test can target exactly one rule and the README
can list what is removed. Order matters: whitespace is normalised first so later patterns
can rely on single spaces; tail rules (marker → end of body) run before inner-block rules;
spans of inner blocks are bounded (`.{0,N}?`) so a rule can never eat half an article.
"""

import re
from dataclasses import dataclass

TRUNCATED_BELOW = 400

_MONTH = (
    r"(?:January|February|March|April|May|June|July|August|September|October|November|December)"
)
_AGO = r"\d+\s+(?:minutes?|hours?|days?)\s+ago"
_SENTENCE_START = r"(?:The|In|This|It|A|An|As|But|However|Meanwhile|While)"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern[str]
    replacement: str = " "


def _rule(name: str, pattern: str, replacement: str = " ") -> Rule:
    return Rule(name, re.compile(pattern, re.DOTALL), replacement)


RULES: tuple[Rule, ...] = (
    # Unicode-mode `\s` already covers the corpus NBSPs (e.g. "Vote Against\u00a0Unionizing"),
    # so the literal space the class used to carry alongside it was redundant.
    _rule("whitespace", r"\s+"),
    # ---- tails: from the marker to the end of the body ----
    _rule("mt_newswires_paywall", r"\s*PREMIUM Upgrade to read this MT Newswires article.*$"),
    _rule(
        "zacks_footer",
        r"\s*(?:Want the latest recommendations from Zacks Investment Research\?"
        r"|Media Contact Zacks Investment Research"
        r"|About Zacks Zacks\.com is a property"
        r"|Why Haven't You Looked at Zacks' Top Stocks\?).*$",
    ),
    _rule("zacks_free_report", r"\s*Click to get this free report.*$"),
    _rule(
        "simply_wall_st_footer",
        r"\s*(?:Have feedback on this article\?"
        r"|This article by Simply Wall St is general in nature).*$",
    ),
    _rule(
        "insider_monkey_footer",
        r"\s*(?:READ NEXT:.{0,400}?)?Disclosure: None\. This article is originally published at"
        r"\s*Insider Monkey\.?.*$",
    ),
    _rule(
        "syndication_footer",
        # `(?<![^.!?])` anchors the match to start only at a sentence boundary (start-of-string
        # or right after a `.`, `!`, or `?`) instead of retrying `[^.!?]*` from every character,
        # which made this rule quadratic on long bodies (see test_real_corpus_cleans_fast).
        r"(?<![^.!?])\s*[^.!?]*\b(?:was originally published by"
        r"|originally appeared (?:on|in))\b.*$",
    ),
    _rule("view_comments_tail", r"\s*(?:Continue Reading\s*)?View Comments\s*$"),
    # ---- inner blocks, span-bounded ----
    _rule(
        "motley_fool_promo",
        r"\s*(?:Should you invest \$1,000 in [^?]{1,80}\?\s*)?"
        r"(?:Before you buy stock in .{0,1500}?"
        r"|Don['’]t miss this second chance.{0,1500}?"
        r"|Ever feel like you missed the boat.{0,1500}?)"
        r"\*?Stock Advisor returns as of " + _MONTH + r" \d{1,2}, \d{4}\.?",
    ),
    _rule(
        "motley_fool_disclosure",
        r"\s*\b[A-Z][\w'.-]+(?: [A-Z][\w'.-]+){0,3} has (?:no )?positions? in .{0,1200}?"
        r"The Motley Fool has a disclosure policy\.",
    ),
    _rule(
        "insider_monkey_pitch",
        r"\s*(?:Why are we interested in|At Insider Monkey we are obsessed with)"
        r" the stocks that hedge funds pile into\??.{0,600}?"
        r"\(see more details here\)\.?",
    ),
    _rule(
        "read_also",
        r"\s*READ ALSO:.{0,400}?(?:\.(?=\s|$)|(?=\s" + _SENTENCE_START + r"\s+\w+\s+[a-z]))",
    ),
    _rule(
        "yahoo_related_videos",
        r"\s*Related Videos(?:\s+\d{1,2}:\d{2}\s.{0,300}?•\s*" + _AGO + r")+",
    ),
    _rule("yahoo_quote_widget", r"\s*Real Time Price • USD \([A-Z.]+\).{0,300}?Advanced Chart"),
    _rule("relative_timestamp", r"\s*\b" + _AGO + r"\b"),
    _rule("story_continues", r"\s*\bStory Continues\b"),
    _rule("whitespace_final", r"\s+"),
)

_BY_NAME = {r.name: r for r in RULES}


@dataclass(frozen=True)
class CleanResult:
    text: str
    is_truncated: bool
    applied: tuple[str, ...]


def rule_named(name: str) -> Rule:
    return _BY_NAME[name]


def apply_rule(text: str, rule: Rule) -> str:
    return rule.pattern.sub(rule.replacement, text)


def clean(text: str, title: str | None = None) -> CleanResult:
    applied: list[str] = []
    # Syndication footers repeat the headline ("<title> was originally published by …");
    # dropping the echo keeps a headline from being selected as a passage.
    if title and len(title) >= 20 and title in text:
        text = text.replace(title, " ")
        applied.append("title_echo")
    for rule in RULES:
        replaced = apply_rule(text, rule)
        if replaced != text and not rule.name.startswith("whitespace"):
            applied.append(rule.name)
        text = replaced
    text = text.strip()
    return CleanResult(text=text, is_truncated=len(text) < TRUNCATED_BELOW, applied=tuple(applied))
