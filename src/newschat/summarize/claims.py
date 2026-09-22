"""Rules-based claim typing: reported event vs analyst opinion vs promotional copy.

Roughly a quarter of the corpus is analyst pitches, Zacks ranks and Motley Fool promos, so
"what's happening with X?" is partly a question about separating those from news. The rules
are deliberately coarse and deterministic; their precision is measured by the golden set.
Order matters: promo is checked before opinion because promotional copy borrows rating words.
"""

import re
from dataclasses import dataclass

from newschat.models import ClaimKind


@dataclass(frozen=True)
class ClaimRule:
    name: str
    kind: ClaimKind
    pattern: re.Pattern[str]


def _rule(name: str, kind: ClaimKind, pattern: str) -> ClaimRule:
    return ClaimRule(name, kind, re.compile(pattern, re.IGNORECASE))


CLAIM_RULES: tuple[ClaimRule, ...] = (
    _rule(
        "promo_first_person_listicle",
        "promo",
        r"\bwe recently compiled\b|\bour (?:quarterly )?newsletter\b",
    ),
    _rule(
        "promo_track_record",
        "promo",
        r"\breturned \d+% since\b|\bmarket-crushing\b|\boutperform(?:ed)? the market by\b",
    ),
    _rule(
        "promo_publisher",
        "promo",
        r"\bstock advisor\b|\bzacks investment research\b|\binsider monkey\b|\bmotley fool\b"
        r"|\bsimply wall st\b",
    ),
    _rule(
        "promo_call_to_action",
        "promo",
        r"\bfree (?:stock analysis )?report\b|\bclick (?:here|to get)\b|\bsign up\b|\bsubscribe\b"
        r"|\bsee (?:more details|stocks free)\b|\blearn more\b",
    ),
    _rule(
        "promo_hedge_fund_hook",
        "promo",
        r"\bhedge fund (?:investor letters|holders)\b|\bnumber of hedge fund\b",
    ),
    _rule(
        "opinion_rating",
        "opinion",
        r"\bprice target\b|\bzacks rank\b|\bstrong (?:buy|sell)\b"
        r"|\b(?:buy|sell|hold|neutral|outperform|underperform) rating\b|\brating\b",
    ),
    _rule(
        "opinion_rating_change", "opinion", r"\bdowngrad|\bupgrad|\bmaintain(?:s|ed)?\b|\breiterat"
    ),
    _rule(
        "opinion_analyst_voice",
        "opinion",
        r"\banalysts? (?:say|said|expect|believe|think|estimate|forecast|see|predict)",
    ),
    _rule(
        "opinion_valuation",
        "opinion",
        r"\bcould be worth\b|\bintrinsic value\b|\bundervalued\b|\bovervalued\b|\bbargain\b"
        r"|\bmulti-bagger\b|\bbuy point\b|\bfair value\b",
    ),
    _rule(
        "opinion_stance",
        "opinion",
        r"\bshould you buy\b|\bis .{1,40} a buy\b|\bbullish\b|\bbearish\b"
        r"|\bwe (?:believe|think|expect|estimate)\b|\bin (?:my|our) (?:view|opinion)\b"
        r"|\bi (?:predict|think|believe)\b",
    ),
)

_ADVICE_RE = re.compile(
    r"\b(?:should (?:you|i) (?:buy|sell)|buy (?:now|the dip|hand over fist)|hand over fist"
    r"|strong (?:buy|sell)|screaming buy|(?:time|reason) to (?:buy|sell)|(?:i|we) recommend"
    r"|you should (?:buy|sell)|load up on|must-buy|must buy|no-brainer buy)\b",
    re.IGNORECASE,
)


def matching_rule(passage: str, title: str = "") -> ClaimRule | None:
    haystack = f"{title}\n{passage}"
    for rule in CLAIM_RULES:
        if rule.pattern.search(haystack):
            return rule
    return None


def classify(passage: str, title: str = "") -> ClaimKind:
    rule = matching_rule(passage, title)
    return rule.kind if rule else "event"


def is_advice(text: str) -> bool:
    return _ADVICE_RE.search(text) is not None
