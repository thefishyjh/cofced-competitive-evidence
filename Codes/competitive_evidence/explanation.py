from __future__ import annotations

from .schema import EvidenceCandidate, EvidencePools


def _domains(candidates: list[EvidenceCandidate]) -> str:
    values = []
    for candidate in candidates:
        if candidate.source_domain and candidate.source_domain not in values:
            values.append(candidate.source_domain)
    return ", ".join(values[:3]) if values else "no selected sources"


def _strongest_sentence(candidates: list[EvidenceCandidate]) -> str:
    if not candidates:
        return "No decisive evidence was selected."
    strongest = max(candidates, key=lambda item: item.final_weight)
    return strongest.sentence


def generate_explanation(claim: str, label: str | None, pools: EvidencePools) -> str:
    """Generate a template explanation grounded in selected evidence only."""
    if pools.refute_strength > pools.support_strength:
        direction = "refuting"
        domains = _domains(pools.refute)
        sentence = _strongest_sentence(pools.refute)
    elif pools.support_strength > pools.refute_strength:
        direction = "supporting"
        domains = _domains(pools.support)
        sentence = _strongest_sentence(pools.support)
    else:
        direction = "neutral or balanced"
        domains = _domains(pools.neutral)
        sentence = _strongest_sentence(pools.neutral)

    label_text = f"Prediction: {label}. " if label else ""
    return (
        f"{label_text}The selected evidence is {direction}: "
        f"support strength={pools.support_strength:.3f}, refute strength={pools.refute_strength:.3f}. "
        f"Main selected sources: {domains}. Strongest selected sentence: {sentence}"
    )
