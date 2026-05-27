from __future__ import annotations

import math
import re
from collections import Counter
from typing import Iterable
from urllib.parse import urlparse

from .schema import EvidenceCandidate
from .source_reliability import normalize_domain, score_source


TOKEN_RE = re.compile(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?")
SENTENCE_RE = re.compile(r"(?<=[.!?])\s+")


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text or "")]


def split_sentences(text: str) -> list[str]:
    return [sent.strip() for sent in SENTENCE_RE.split(text or "") if sent.strip()]


def _iter_report_sentences(report: dict) -> Iterable[tuple[str, int, dict]]:
    tokenized = report.get("tokenized") or []
    if tokenized:
        for index, item in enumerate(tokenized):
            if isinstance(item, dict):
                sentence = item.get("sent") or item.get("sentence") or ""
                metadata = {key: value for key, value in item.items() if key not in {"sent", "sentence"}}
            else:
                sentence = str(item)
                metadata = {}
            if sentence.strip():
                yield sentence.strip(), index, metadata
        return

    for index, sentence in enumerate(split_sentences(report.get("content", ""))):
        yield sentence, index, {}


def _cosine_sparse(left: Counter, right: Counter) -> float:
    if not left or not right:
        return 0.0
    dot = sum(left[key] * right.get(key, 0.0) for key in left)
    left_norm = math.sqrt(sum(value * value for value in left.values()))
    right_norm = math.sqrt(sum(value * value for value in right.values()))
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return dot / (left_norm * right_norm)


def _tfidf_vectors(texts: list[str]) -> list[Counter]:
    tokenized_texts = [tokenize(text) for text in texts]
    doc_count = len(tokenized_texts)
    df = Counter()
    for tokens in tokenized_texts:
        df.update(set(tokens))

    vectors = []
    for tokens in tokenized_texts:
        tf = Counter(tokens)
        vector = Counter()
        for token, count in tf.items():
            idf = math.log((1.0 + doc_count) / (1.0 + df[token])) + 1.0
            vector[token] = (1.0 + math.log(count)) * idf
        vectors.append(vector)
    return vectors


def retrieve_candidates(instance: dict, top_k: int = 50) -> list[EvidenceCandidate]:
    """Return top evidence sentences from an instance's raw reports."""
    claim = instance.get("claim", "")
    event_id = str(instance.get("event_id", ""))
    raw_candidates: list[EvidenceCandidate] = []
    seen_sentences: set[str] = set()

    for report_index, report in enumerate(instance.get("reports") or []):
        url = report.get("link") or report.get("url")
        domain = normalize_domain(report.get("domain"), url)
        if not domain and url:
            domain = normalize_domain(urlparse(url).netloc)
        report_id = str(report.get("report_id", report_index))

        for sentence, sentence_id, metadata in _iter_report_sentences(report):
            source_score = score_source(domain, url, metadata)
            compact = " ".join(sentence.lower().split())
            if len(compact) < 8 or compact in seen_sentences:
                continue
            seen_sentences.add(compact)
            raw_candidates.append(
                EvidenceCandidate(
                    event_id=event_id,
                    claim=claim,
                    sentence=sentence,
                    source_domain=domain,
                    source_url=url,
                    report_id=report_id,
                    sentence_id=sentence_id,
                    relevance_score=0.0,
                    source_score=source_score,
                    metadata=metadata,
                )
            )

    if not raw_candidates:
        return []

    texts = [claim] + [candidate.sentence for candidate in raw_candidates]
    vectors = _tfidf_vectors(texts)
    claim_vector = vectors[0]
    scores = [_cosine_sparse(claim_vector, vector) for vector in vectors[1:]]
    max_score = max(scores) if scores else 0.0

    for candidate, score in zip(raw_candidates, scores):
        candidate.relevance_score = score / max_score if max_score > 0.0 else 0.0

    raw_candidates.sort(key=lambda item: (item.relevance_score, item.source_score), reverse=True)
    return raw_candidates[:top_k]
