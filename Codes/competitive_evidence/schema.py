from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional


@dataclass
class StanceResult:
    stance: str
    stance_score: float
    support_prob: float
    refute_prob: float
    neutral_prob: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceCandidate:
    event_id: str
    claim: str
    sentence: str
    source_domain: str
    source_url: Optional[str]
    report_id: Optional[str]
    sentence_id: int
    relevance_score: float
    source_score: float = 0.55
    stance: str = "neutral"
    stance_score: float = 1.0
    final_weight: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        return data


@dataclass
class EvidencePools:
    support: list[EvidenceCandidate]
    refute: list[EvidenceCandidate]
    neutral: list[EvidenceCandidate]
    support_strength: float
    refute_strength: float
    conflict_score: float
    margin_score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "support": [candidate.to_dict() for candidate in self.support],
            "refute": [candidate.to_dict() for candidate in self.refute],
            "neutral": [candidate.to_dict() for candidate in self.neutral],
            "support_strength": self.support_strength,
            "refute_strength": self.refute_strength,
            "conflict_score": self.conflict_score,
            "margin_score": self.margin_score,
        }
