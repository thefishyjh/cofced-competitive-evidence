from __future__ import annotations

import re

from .retrieval import tokenize
from .schema import StanceResult


REFUTE_CUES = {
    "false",
    "fake",
    "hoax",
    "incorrect",
    "wrong",
    "misleading",
    "debunk",
    "debunked",
    "deny",
    "denied",
    "no evidence",
    "not true",
    "did not",
    "does not",
    "cannot",
}

SUPPORT_CUES = {
    "confirmed",
    "according",
    "reported",
    "shows",
    "found",
    "announced",
    "said",
    "evidence",
}

NEGATION_RE = re.compile(r"\b(no|not|never|none|without|cannot|can't|didn't|doesn't|isn't|wasn't)\b", re.I)


def _contains_phrase(text: str, phrases: set[str]) -> bool:
    lowered = text.lower()
    return any(phrase in lowered for phrase in phrases)


def classify_stance(claim: str, evidence_sentence: str) -> StanceResult:
    """Deterministic stance fallback; can be replaced by an NLI model later."""
    claim_tokens = set(tokenize(claim))
    evidence_tokens = set(tokenize(evidence_sentence))
    overlap = len(claim_tokens & evidence_tokens) / max(1, len(claim_tokens))
    claim_negated = bool(NEGATION_RE.search(claim or ""))
    evidence_negated = bool(NEGATION_RE.search(evidence_sentence or ""))
    has_refute_cue = _contains_phrase(evidence_sentence, REFUTE_CUES)
    has_support_cue = _contains_phrase(evidence_sentence, SUPPORT_CUES)

    if overlap < 0.08:
        return StanceResult("neutral", 0.72, 0.14, 0.14, 0.72)

    if has_refute_cue or claim_negated != evidence_negated:
        score = min(0.88, 0.58 + overlap)
        return StanceResult("refute", score, 1.0 - score, score, max(0.05, 1.0 - score))

    if has_support_cue or overlap >= 0.20:
        score = min(0.84, 0.50 + overlap)
        return StanceResult("support", score, score, max(0.05, 1.0 - score), max(0.05, 1.0 - score))

    return StanceResult("neutral", 0.62, 0.19, 0.19, 0.62)
