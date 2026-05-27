from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from statistics import mean
from typing import Any

if __package__ is None or __package__ == "":
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from competitive_evidence.explanation import generate_explanation
from competitive_evidence.pooling import build_competitive_pools
from competitive_evidence.retrieval import retrieve_candidates
from competitive_evidence.schema import EvidenceCandidate, EvidencePools


def load_instances(input_path: str) -> list[dict[str, Any]]:
    path = Path(input_path)
    if path.is_file():
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return data if isinstance(data, list) else [data]

    if path.is_dir():
        instances = []
        for file_path in sorted(path.glob("*.json")):
            with file_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle)
            if isinstance(data, list):
                instances.extend(data)
            else:
                instances.append(data)
        return instances

    raise FileNotFoundError(f"Input path does not exist: {input_path}")


def _avg_source_score(candidates: list[EvidenceCandidate]) -> float:
    return mean([candidate.source_score for candidate in candidates]) if candidates else 0.0


def _max_weight(candidates: list[EvidenceCandidate]) -> float:
    return max([candidate.final_weight for candidate in candidates], default=0.0)


FEATURE_SCHEMA_VERSION = 1


def _pool_features(instance: dict[str, Any], pools: EvidencePools) -> dict[str, Any]:
    label = instance.get("label", "")
    claim = instance.get("claim", "")
    return {
        "event_id": str(instance.get("event_id", "")),
        "claim": claim,
        "gold_label": label,
        "support_strength": pools.support_strength,
        "refute_strength": pools.refute_strength,
        "conflict_score": pools.conflict_score,
        "margin_score": pools.margin_score,
        "num_support": len(pools.support),
        "num_refute": len(pools.refute),
        "num_neutral": len(pools.neutral),
        "avg_support_source_score": _avg_source_score(pools.support),
        "avg_refute_source_score": _avg_source_score(pools.refute),
        "max_support_weight": _max_weight(pools.support),
        "max_refute_weight": _max_weight(pools.refute),
        "support_evidence": [candidate.to_dict() for candidate in pools.support],
        "refute_evidence": [candidate.to_dict() for candidate in pools.refute],
        "neutral_evidence": [candidate.to_dict() for candidate in pools.neutral],
        "explanation": generate_explanation(claim, None, pools),
    }


def build_features(
    instances: list[dict[str, Any]],
    top_k_candidates: int = 50,
    top_k_per_pool: int = 5,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    if limit is not None:
        instances = instances[:limit]

    rows = []
    for index, instance in enumerate(instances, start=1):
        candidates = retrieve_candidates(instance, top_k=top_k_candidates)
        pools = build_competitive_pools(candidates, top_k_per_pool=top_k_per_pool)
        row = _pool_features(instance, pools)
        row["candidate_count"] = len(candidates)
        row["instance_index"] = index
        rows.append(row)
    return rows


def build_feature_payload(
    rows: list[dict[str, Any]],
    input_path: str | None = None,
    top_k_candidates: int = 50,
    top_k_per_pool: int = 5,
) -> dict[str, Any]:
    resolved_input_path = str(Path(input_path).resolve()) if input_path else None
    return {
        "metadata": {
            "schema_version": FEATURE_SCHEMA_VERSION,
            "input_path": resolved_input_path,
            "row_count": len(rows),
            "top_k_candidates": top_k_candidates,
            "top_k_per_pool": top_k_per_pool,
        },
        "rows": rows,
    }


def dump_json(data: list[dict[str, Any]] | dict[str, Any], output_path: str) -> None:
    path = Path(output_path)
    if path.parent:
        os.makedirs(path.parent, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(data, handle, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build source-weighted competitive evidence features for CofCED datasets."
    )
    parser.add_argument("--input", required=True, help="Path to a LIAR-RAW JSON file or RAWFC split directory.")
    parser.add_argument("--output", required=True, help="Path to write feature JSON.")
    parser.add_argument("--top-k-candidates", type=int, default=50)
    parser.add_argument("--top-k-per-pool", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None, help="Optional smoke-test limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    instances = load_instances(args.input)
    features = build_features(
        instances,
        top_k_candidates=args.top_k_candidates,
        top_k_per_pool=args.top_k_per_pool,
        limit=args.limit,
    )
    payload = build_feature_payload(
        features,
        input_path=args.input,
        top_k_candidates=args.top_k_candidates,
        top_k_per_pool=args.top_k_per_pool,
    )
    dump_json(payload, args.output)
    print(f"Wrote {len(features)} competitive evidence feature rows to {args.output}")


if __name__ == "__main__":
    main()
