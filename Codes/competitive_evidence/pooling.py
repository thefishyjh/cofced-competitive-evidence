from __future__ import annotations

from collections import defaultdict

from .schema import EvidenceCandidate, EvidencePools
from .stance import classify_stance


def compute_evidence_weight(
    relevance_score: float,
    source_score: float,
    stance_score: float,
    diversity_factor: float = 1.0,
) -> float:
    value = relevance_score * source_score * stance_score * diversity_factor
    return max(0.0, min(1.0, value))


def _assign_stance_and_weight(candidates: list[EvidenceCandidate]) -> None:
    domain_counts: defaultdict[str, int] = defaultdict(int)
    for candidate in candidates:
        stance = classify_stance(candidate.claim, candidate.sentence)
        candidate.stance = stance.stance
        candidate.stance_score = stance.stance_score
        domain_counts[candidate.source_domain] += 1
        diversity_factor = 1.0 / (1.0 + 0.25 * max(0, domain_counts[candidate.source_domain] - 1))
        candidate.final_weight = compute_evidence_weight(
            candidate.relevance_score,
            candidate.source_score,
            candidate.stance_score,
            diversity_factor,
        )


def _top_with_domain_cap(candidates: list[EvidenceCandidate], top_k: int, per_domain_cap: int = 2) -> list[EvidenceCandidate]:
    selected: list[EvidenceCandidate] = []
    domain_counts: defaultdict[str, int] = defaultdict(int)
    for candidate in sorted(candidates, key=lambda item: item.final_weight, reverse=True):
        if domain_counts[candidate.source_domain] >= per_domain_cap:
            continue
        selected.append(candidate)
        domain_counts[candidate.source_domain] += 1
        if len(selected) >= top_k:
            break
    return selected


def build_competitive_pools(candidates: list[EvidenceCandidate], top_k_per_pool: int = 5) -> EvidencePools:
    """Split evidence into support/refute/neutral pools and compute aggregate features."""
    _assign_stance_and_weight(candidates)
    support_all = [candidate for candidate in candidates if candidate.stance == "support"]
    refute_all = [candidate for candidate in candidates if candidate.stance == "refute"]
    neutral_all = [candidate for candidate in candidates if candidate.stance == "neutral"]

    support = _top_with_domain_cap(support_all, top_k_per_pool)
    refute = _top_with_domain_cap(refute_all, top_k_per_pool)
    neutral = _top_with_domain_cap(neutral_all, top_k_per_pool)

    support_strength = sum(candidate.final_weight for candidate in support)
    refute_strength = sum(candidate.final_weight for candidate in refute)
    conflict_score = min(support_strength, refute_strength)
    margin_score = abs(support_strength - refute_strength)

    return EvidencePools(
        support=support,
        refute=refute,
        neutral=neutral,
        support_strength=support_strength,
        refute_strength=refute_strength,
        conflict_score=conflict_score,
        margin_score=margin_score,
    )
