from __future__ import annotations

from urllib.parse import urlparse


HIGH_RELIABILITY_DOMAINS = {
    "who.int",
    "cdc.gov",
    "nih.gov",
    "fda.gov",
    "nasa.gov",
    "noaa.gov",
    "sec.gov",
    "irs.gov",
    "census.gov",
    "snopes.com",
    "politifact.com",
    "factcheck.org",
    "apnews.com",
    "reuters.com",
    "bbc.com",
    "bbc.co.uk",
}

RELIABLE_NEWS_DOMAINS = {
    "nytimes.com",
    "washingtonpost.com",
    "wsj.com",
    "theguardian.com",
    "npr.org",
    "pbs.org",
    "abcnews.go.com",
    "cbsnews.com",
    "nbcnews.com",
}

LOW_RELIABILITY_HINTS = {
    "blogspot.",
    "wordpress.",
    "question-it.",
    "quora.",
    "answers.",
    "rumor",
    "viral",
    "beforeitsnews",
    "naturalnews",
}


def normalize_domain(domain: str | None, url: str | None = None) -> str:
    if domain:
        value = domain.strip().lower()
    elif url:
        value = urlparse(url).netloc.lower()
    else:
        return ""

    if value.startswith("http://") or value.startswith("https://"):
        value = urlparse(value).netloc.lower()
    if value.startswith("www."):
        value = value[4:]
    return value.strip("/")


def score_source(domain: str | None, url: str | None = None, metadata: dict | None = None) -> float:
    """Return a transparent heuristic reliability score in [0, 1]."""
    normalized = normalize_domain(domain, url)
    if not normalized:
        return 0.50

    if any(normalized == domain or normalized.endswith("." + domain) for domain in HIGH_RELIABILITY_DOMAINS):
        return 0.95
    if normalized.endswith(".gov") or normalized.endswith(".edu"):
        return 0.95
    if any(normalized == domain or normalized.endswith("." + domain) for domain in RELIABLE_NEWS_DOMAINS):
        return 0.82
    if normalized.endswith(".org"):
        return 0.75
    if any(hint in normalized for hint in LOW_RELIABILITY_HINTS):
        return 0.30

    score = 0.58
    if metadata:
        if metadata.get("has_author"):
            score += 0.05
        if metadata.get("has_date"):
            score += 0.04
        if metadata.get("has_references"):
            score += 0.06

    return max(0.0, min(1.0, score))
