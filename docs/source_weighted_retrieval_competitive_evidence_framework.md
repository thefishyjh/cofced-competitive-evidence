# Source-Weighted Retrieval-Augmented Competitive Evidence Framework

## 1. Goal

This document describes a concrete upgrade path for the existing CofCED project.

The original CofCED pipeline uses raw reports to extract evidence and predict fake news labels:

```text
claim -> raw reports -> evidence extraction -> veracity prediction
```

The proposed method extends this into a source-weighted, retrieval-augmented, stance-aware competitive evidence framework:

```text
claim
  -> claim decomposition and query rewriting
  -> raw report retrieval + optional external retrieval
  -> source reliability scoring
  -> evidence sentence reranking
  -> stance/NLI classification
  -> support/refute competitive evidence pooling
  -> veracity prediction + evidence-grounded explanation
```

The main research idea is:

> Upgrade CofCED from "extracting relevant evidence" to "reasoning over trusted competing evidence".

This directly targets three limitations of raw-report-based fake news detection:

- Raw reports have uneven source quality.
- Relevant sentences may support, refute, or merely mention the claim.
- Final explanations should expose evidence conflict rather than only output selected sentences.

## 2. Expected Inputs and Outputs

### 2.1 Input Instance

Each instance follows the existing CofCED data style:

```json
{
  "event_id": "example_id",
  "claim": "The claim to be checked.",
  "label": "false",
  "explain": "Gold explanation if available.",
  "reports": [
    {
      "link": "https://example.com/article",
      "domain": "example.com",
      "content": "Full report text.",
      "tokenized": [
        {
          "sent": "Sentence text.",
          "is_evidence": 1
        }
      ]
    }
  ]
}
```

### 2.2 Output Instance

The upgraded system should output both prediction and interpretable evidence:

```json
{
  "event_id": "example_id",
  "claim": "The claim to be checked.",
  "predicted_label": "false",
  "confidence": 0.83,
  "support_strength": 0.21,
  "refute_strength": 0.74,
  "supporting_evidence": [
    {
      "sentence": "Evidence sentence.",
      "source": "example.com",
      "relevance_score": 0.78,
      "source_score": 0.62,
      "stance": "support",
      "stance_score": 0.81,
      "final_weight": 0.39
    }
  ],
  "refuting_evidence": [
    {
      "sentence": "Evidence sentence.",
      "source": "cdc.gov",
      "relevance_score": 0.91,
      "source_score": 0.95,
      "stance": "refute",
      "stance_score": 0.88,
      "final_weight": 0.76
    }
  ],
  "explanation": "The claim is predicted false because high-reliability refuting evidence outweighs low-reliability supporting evidence."
}
```

## 3. Module-Level Design

### 3.1 Claim Decomposition and Query Rewriting

Purpose:

- Convert a complex claim into atomic subclaims or search queries.
- Improve evidence recall.

Minimal implementation:

- Use the original claim as the first query.
- Extract named entities, dates, numbers, and noun phrases using simple rules or an NLP library.
- Generate additional queries by combining important entities with the main predicate.

Advanced implementation:

- Use a small sequence-to-sequence model or LLM prompt to decompose the claim.
- Generate multiple subclaims:

```text
Original claim:
"A vaccine increased heart disease risk by 300%."

Subclaims:
1. Which vaccine is being discussed?
2. What heart disease outcome is claimed?
3. What study or statistic supports the 300% number?
4. Do authoritative sources confirm or reject this statistic?
```

Recommended interface:

```python
def decompose_claim(claim: str) -> list[str]:
    """Return claim-focused retrieval queries or atomic subclaims."""
```

### 3.2 Retrieval-Augmented Evidence Collection

Purpose:

- Retrieve candidate evidence from existing raw reports and optionally external sources.

Minimal implementation for the current CofCED repo:

- Use existing `reports` in LIAR-RAW or RAWFC.
- Split each report into sentences.
- Score each sentence against the claim.

Recommended retrieval stages:

```text
Stage 1: Candidate recall
  - BM25 top-N sentences or documents
  - Dense retrieval top-N sentences or documents

Stage 2: Reranking
  - Cross-encoder relevance scoring for claim-sentence pairs

Stage 3: Diversity filtering
  - Remove near-duplicate evidence
  - Keep evidence from diverse domains
```

Recommended technologies:

- BM25: `rank_bm25`, Elasticsearch, OpenSearch, or Lucene.
- Dense retrieval: Sentence-BERT, E5, BGE.
- Vector index: FAISS, Milvus, or local NumPy index for a small project.
- Reranker: BGE reranker, MiniLM cross-encoder, or DeBERTa pair classifier.
- Diversity: MMR or cosine-similarity deduplication.

Recommended interface:

```python
@dataclass
class EvidenceCandidate:
    event_id: str
    claim: str
    sentence: str
    source_domain: str
    source_url: str | None
    report_id: str | None
    sentence_id: int
    relevance_score: float

def retrieve_candidates(instance: dict, top_k: int = 50) -> list[EvidenceCandidate]:
    """Return top candidate evidence sentences from raw reports and optional external sources."""
```

### 3.3 Source Reliability Scoring

Purpose:

- Assign higher weights to evidence from more reliable sources.
- Reduce the effect of noisy, repetitive, or low-quality raw reports.

Reliability score range:

```text
source_score in [0.0, 1.0]
```

Suggested feature groups:

Static source features:

- Domain suffix: `.gov`, `.edu`, `.org`, media site, personal blog.
- Known high-reliability whitelist: government, academic, official statistics, established fact-checking sites.
- Known low-reliability blacklist if available.
- Whether the report has author, date, references, and outgoing links.

Dynamic source features:

- Agreement with other high-reliability sources.
- Number of independent sources making the same assertion.
- Near-duplicate or syndication pattern.
- Whether publication time is plausible for the event.

Minimal scoring rule:

```python
def heuristic_source_score(domain: str) -> float:
    domain = domain.lower()
    if domain.endswith(".gov") or domain.endswith(".edu"):
        return 0.95
    if domain in {"who.int", "cdc.gov", "snopes.com", "politifact.com", "factcheck.org"}:
        return 0.95
    if domain.endswith(".org"):
        return 0.75
    if domain in known_reliable_media:
        return 0.80
    if domain in known_unreliable_media:
        return 0.30
    return 0.55
```

Advanced scoring:

- Build a source graph where nodes are domains and edges are citation, hyperlink, or content-copy relations.
- Use PageRank/HITS to estimate source authority.
- Train an MLP or learning-to-rank model from source metadata and historical correctness.

Recommended interface:

```python
def score_source(domain: str, url: str | None = None, metadata: dict | None = None) -> float:
    """Return a calibrated source reliability score in [0, 1]."""
```

### 3.4 Stance/NLI Classification

Purpose:

- Determine whether an evidence sentence supports, refutes, or is neutral toward the claim.

Labels:

```text
support
refute
neutral
insufficient
```

Natural language inference mapping:

```text
entailment -> support
contradiction -> refute
neutral -> neutral/insufficient
```

Recommended technologies:

- `roberta-large-mnli`
- `deberta-v3-large-mnli`
- A fine-tuned stance classifier if labels are available.

Minimal implementation:

- Use a pretrained NLI model to produce pseudo labels and confidence.
- Cache all claim-evidence stance predictions to avoid repeated inference.

Recommended interface:

```python
@dataclass
class StanceResult:
    stance: str
    stance_score: float
    support_prob: float
    refute_prob: float
    neutral_prob: float

def classify_stance(claim: str, evidence_sentence: str) -> StanceResult:
    """Classify evidence stance toward the claim."""
```

### 3.5 Evidence Weighting

Each candidate evidence receives a final weight:

```text
final_weight_i =
    relevance_score_i
  * source_score_i
  * stance_confidence_i
  * diversity_factor_i
```

Recommended clipping:

```text
final_weight_i = clamp(final_weight_i, 0.0, 1.0)
```

Diversity factor:

- Start with `1.0`.
- Reduce it if the evidence is near-duplicate with an already selected sentence.
- Reduce it if too many selected sentences come from the same domain.

Recommended interface:

```python
def compute_evidence_weight(
    relevance_score: float,
    source_score: float,
    stance_score: float,
    diversity_factor: float = 1.0,
) -> float:
    """Compute final evidence weight."""
```

### 3.6 Competitive Evidence Pooling

Purpose:

- Separate evidence into support and refute pools.
- Compare the two sides explicitly.

Pool construction:

```text
Support Pool = top weighted evidence with stance == support
Refute Pool  = top weighted evidence with stance == refute
Neutral Pool = related but non-decisive evidence
```

Aggregate strength:

```text
support_strength = sum(final_weight_i for i in Support Pool)
refute_strength  = sum(final_weight_i for i in Refute Pool)
conflict_score   = min(support_strength, refute_strength)
margin_score     = abs(support_strength - refute_strength)
```

Neural aggregation option:

```text
support_repr = weighted_attention_pool(support_evidence_embeddings)
refute_repr  = weighted_attention_pool(refute_evidence_embeddings)

veracity_input = [
  claim_repr,
  support_repr,
  refute_repr,
  support_strength,
  refute_strength,
  conflict_score,
  margin_score
]
```

Recommended interface:

```python
@dataclass
class EvidencePools:
    support: list[EvidenceCandidate]
    refute: list[EvidenceCandidate]
    neutral: list[EvidenceCandidate]
    support_strength: float
    refute_strength: float
    conflict_score: float
    margin_score: float

def build_competitive_pools(candidates: list[EvidenceCandidate], top_k_per_pool: int = 5) -> EvidencePools:
    """Split evidence into support/refute/neutral pools and compute aggregate features."""
```

### 3.7 Veracity Prediction

The final classifier should combine:

- Claim representation.
- Support evidence representation.
- Refute evidence representation.
- Source reliability features.
- Stance distribution.
- Competitive strength features.

Minimal feature-based classifier:

```text
features = [
  support_strength,
  refute_strength,
  conflict_score,
  margin_score,
  num_support,
  num_refute,
  avg_support_source_score,
  avg_refute_source_score,
  max_support_weight,
  max_refute_weight
]

classifier = LogisticRegression / MLP
```

Neural classifier:

```text
Encoder: DistilBERT / RoBERTa / DeBERTa
Aggregator: weighted attention or graph attention
Classifier: MLP
```

Recommended loss:

```text
L = L_veracity
  + alpha * L_evidence
  + beta  * L_stance
  + gamma * L_source
  + delta * L_contrastive
```

For a first implementation, use:

```text
L = L_veracity + alpha * L_evidence
```

Then add stance and source losses after the pipeline is stable.

### 3.8 Evidence-Grounded Explanation

Purpose:

- Produce explanations grounded only in selected evidence.
- Avoid free-form hallucinated reasoning.

Template-based minimal explanation:

```text
The claim is predicted as {label}. The refuting evidence is stronger than the supporting evidence.
High-reliability refuting sources include {domains}. The strongest refuting sentence is: "{sentence}".
Supporting evidence mainly comes from {support_domains}, with lower aggregate reliability.
```

LLM-based advanced explanation:

- Provide only selected support/refute evidence to the LLM.
- Require citations to evidence IDs.
- Run an NLI check between explanation and selected evidence.

Recommended interface:

```python
def generate_explanation(claim: str, label: str, pools: EvidencePools) -> str:
    """Generate a faithful explanation from selected evidence only."""
```

## 4. Implementation Plan for the Existing CofCED Project

### Phase 1: Non-Invasive Evidence Pipeline

Goal:

- Add new modules without modifying the original CofCED model heavily.

Tasks:

1. Create a new package under `Codes/competitive_evidence/`.
2. Implement data classes for evidence candidates, stance results, and evidence pools.
3. Implement sentence candidate extraction from existing `reports`.
4. Implement heuristic source reliability scoring.
5. Implement simple lexical or embedding relevance scoring.
6. Implement competitive support/refute pooling.
7. Save generated evidence features to JSON files.

Suggested files:

```text
Codes/competitive_evidence/
  __init__.py
  schema.py
  source_reliability.py
  retrieval.py
  stance.py
  pooling.py
  explanation.py
  build_features.py
```

Suggested output:

```text
Codes/dataset/features/
  liar_raw_competitive_train.json
  liar_raw_competitive_val.json
  liar_raw_competitive_test.json
```

### Phase 2: Feature-Based Baseline

Goal:

- Verify whether source-weighted competitive evidence features improve classification.

Tasks:

1. Build numeric features from evidence pools.
2. Train a simple classifier using scikit-learn or a small PyTorch MLP.
3. Compare with CofCED or a claim-only baseline.

Feature examples:

```text
support_strength
refute_strength
margin_score
conflict_score
max_support_weight
max_refute_weight
avg_support_source_score
avg_refute_source_score
num_unique_support_domains
num_unique_refute_domains
```

### Phase 3: Integrate With CofCED Neural Model

Goal:

- Add competitive evidence features to CofCED's final classifier.

Tasks:

1. Extend the dataset reader to load precomputed competitive evidence features.
2. Add an MLP to encode numeric competitive features.
3. Concatenate this feature representation with CofCED's existing final representation.
4. Train with the same train/val/test split.
5. Keep a config flag to enable/disable the new features.

Recommended config:

```python
use_competitive_features = True
competitive_feature_dim = 10
competitive_hidden_dim = 64
```

### Phase 4: Add Stance/NLI Model

Goal:

- Replace rule-based stance with a pretrained NLI model.

Tasks:

1. Add a stance inference script.
2. Cache predictions to JSON.
3. Add batching for GPU inference if available.
4. Use stance confidence in evidence weighting.

Fallback:

- If pretrained model download is not available, keep rule-based or lexical stance for local experiments and document the limitation.

### Phase 5: External Retrieval

Goal:

- Add dynamic retrieval from web or local indexed documents.

Tasks:

1. Implement a retrieval interface independent of data source.
2. Start with local raw reports.
3. Add optional web/news/fact-check retrieval later.
4. Store all external evidence with URL, domain, retrieval date, and snippet.

Important:

- Keep external retrieval optional so experiments remain reproducible.

## 5. Evaluation Plan

### 5.1 Veracity Metrics

Use the same metrics as CofCED:

- Accuracy
- Macro-F1
- Per-class F1

### 5.2 Evidence Metrics

If gold evidence labels are available:

- Evidence precision
- Evidence recall
- Evidence F1
- Top-K evidence hit rate

### 5.3 Explanation Metrics

If gold explanations are available:

- ROUGE
- BERTScore
- Evidence faithfulness by NLI
- Human readability evaluation if possible

### 5.4 Source-Aware Metrics

Recommended additional metrics:

- Source-weighted evidence precision.
- Average source reliability of selected evidence.
- Unique domain count in selected evidence.
- Duplicate evidence ratio.

### 5.5 Ablation Study

Run the following variants:

```text
Baseline: Original CofCED
+ Retrieval reranking
+ Source weighting
+ Stance classification
+ Competitive evidence pooling
+ All components
```

Expected result pattern:

- Source weighting should improve robustness and selected evidence quality.
- Stance classification should improve explanation clarity.
- Competitive pooling should improve false/true distinction and reduce ambiguous predictions.
- Retrieval augmentation should improve recall when raw reports miss key evidence.

## 6. Minimal Viable Version

If time is limited, implement only this version:

1. Use existing raw reports only.
2. Split reports into candidate sentences.
3. Compute sentence relevance with BM25 or sentence embeddings.
4. Compute source reliability with heuristic domain rules.
5. Classify stance using a pretrained NLI model if available; otherwise use a simple placeholder.
6. Build support/refute pools.
7. Generate numeric competitive features.
8. Train a small MLP or append features to CofCED's final classifier.
9. Report ablation:

```text
CofCED
CofCED + source-weighted evidence features
CofCED + source-weighted competitive evidence features
```

This version is enough for a course project because it demonstrates the central idea without requiring full web retrieval or LLM explanation generation.

## 7. Risk and Mitigation

### Risk 1: Source reliability labels are unavailable

Mitigation:

- Start with heuristic domain rules.
- Report this as a limitation.
- Later replace with a learned model or external reliability database.

### Risk 2: NLI model is noisy on short evidence sentences

Mitigation:

- Use sentence plus neighboring context.
- Keep neutral evidence instead of forcing support/refute.
- Use confidence thresholding.

### Risk 3: External retrieval harms reproducibility

Mitigation:

- Cache retrieved evidence.
- Store retrieval date.
- Make external retrieval optional.
- Run main experiments on fixed raw reports.

### Risk 4: Evidence pools become dominated by one domain

Mitigation:

- Apply same-domain penalty.
- Use MMR or diversity filtering.
- Cap the number of evidence sentences per domain.

## 8. Suggested Coding Agent Prompt

Copy the following prompt into a programming agent working inside the existing CofCED repository.

```text
You are working in the existing CofCED project. Implement a source-weighted retrieval-augmented competitive evidence pipeline as an incremental extension, without breaking the original training scripts.

Main goal:
Add a new evidence feature pipeline that turns each claim's raw reports into source-weighted support/refute evidence pools, then exports numeric features that can later be used by CofCED's classifier.

Repository context:
- Existing dataset readers parse LIAR-RAW and RAWFC instances with fields such as event_id, claim, label, explain, reports.
- Each report contains link, domain, content, and tokenized sentences with optional is_evidence labels.
- Existing model code lives under Codes/model.
- Existing helper/data reader code lives under Codes/helpers.

Implementation requirements:

1. Create a new package:
   Codes/competitive_evidence/

2. Add these files:
   - __init__.py
   - schema.py
   - source_reliability.py
   - retrieval.py
   - stance.py
   - pooling.py
   - explanation.py
   - build_features.py

3. In schema.py, define dataclasses:
   - EvidenceCandidate
   - StanceResult
   - EvidencePools

4. In source_reliability.py:
   - Implement score_source(domain, url=None, metadata=None) -> float.
   - Use deterministic heuristic rules first.
   - Give high scores to .gov, .edu, official health/science/government domains, and known fact-checking domains.
   - Give medium scores to normal news/org domains.
   - Give low scores to unknown low-quality or suspicious domains if listed.
   - Keep all rules transparent and easy to edit.

5. In retrieval.py:
   - Implement extraction of candidate sentences from the existing reports field.
   - Implement a lightweight relevance scorer.
   - Prefer a dependency-light baseline first: token overlap or TF-IDF/BM25.
   - If sklearn is already available, use TfidfVectorizer cosine similarity.
   - Return top-K EvidenceCandidate objects with relevance_score.

6. In stance.py:
   - Implement classify_stance(claim, evidence_sentence) -> StanceResult.
   - Start with a deterministic fallback so the code runs without downloading models.
   - The fallback can assign neutral by default, with simple lexical contradiction/support cues if useful.
   - Design the function so a pretrained NLI model can be plugged in later.

7. In pooling.py:
   - Implement compute_evidence_weight(relevance_score, source_score, stance_score, diversity_factor=1.0).
   - Implement build_competitive_pools(candidates, top_k_per_pool=5).
   - Split evidence into support, refute, neutral pools.
   - Compute support_strength, refute_strength, conflict_score, and margin_score.
   - Apply a simple same-domain cap or diversity penalty.

8. In explanation.py:
   - Implement generate_explanation(claim, label, pools) using a deterministic template.
   - The explanation must only mention selected evidence from the pools.

9. In build_features.py:
   - Provide a CLI that accepts:
     --input path/to/train.json or path/to/RAWFC/split_dir
     --output path/to/features.json
     --top-k-candidates
     --top-k-per-pool
   - It should read LIAR-RAW single JSON files and RAWFC directories of JSON files.
   - For each instance, produce a JSON object with:
     event_id, claim, label, support_strength, refute_strength, conflict_score, margin_score,
     num_support, num_refute, num_neutral,
     avg_support_source_score, avg_refute_source_score,
     max_support_weight, max_refute_weight,
     support_evidence, refute_evidence, neutral_evidence,
     explanation.

10. Keep the implementation deterministic and CPU-friendly.
    Do not require network access.
    Do not require downloading pretrained models for the first version.
    Avoid large refactors of the original CofCED code.

11. Add a short README section or docstring showing example commands:
    python Codes/competitive_evidence/build_features.py --input Codes/dataset/LIAR-RAW/train.json --output Codes/dataset/features/liar_raw_train_competitive.json

12. Add basic smoke tests or a small demo command that runs on one or a few instances.

Validation:
- Run the feature builder on a tiny sample or the first few instances if full data is large.
- Confirm the output JSON contains all required fields.
- Confirm the original CofCED files are not broken.

Coding style:
- Keep changes localized.
- Use clear type hints.
- Add comments only where the logic is not obvious.
- Handle missing fields gracefully.
- Preserve compatibility with Windows paths.
```

## 9. Recommended Next Step

For a real implementation, start with Phase 1 and Phase 2. This gives a runnable and testable version quickly:

```text
raw reports -> source-weighted support/refute pools -> JSON features -> feature-based classifier
```

After this baseline works, integrate the features into CofCED's neural model and then add a stronger NLI stance model.
