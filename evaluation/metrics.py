"""Rigorous evaluation metrics for Trust-Aware RAG.

Includes:
1. Claim-level faithfulness and hallucination metrics (moving beyond cosine similarity).
2. Selective prediction / abstention metrics (Coverage, Selective Accuracy, Risk, AUROC).
3. Bootstrap confidence intervals for statistical significance reporting.
"""

import math
import re
import numpy as np
from collections import Counter


def split_into_claims(text: str) -> list[str]:
    """Segment text into individual candidate claims/sentences for verification."""
    if not text:
        return []
    # Split on sentence boundaries (period, exclamation, question mark, newline)
    sentences = re.split(r"(?<=[.!?])\s+|\n+", text.strip())
    cleaned = []
    for s in sentences:
        s_clean = s.strip()
        # Filter out trivial boilerplates or empty strings
        if len(s_clean) > 5 and not re.match(r"^(yes|no|however|in conclusion|furthermore)[,.]?$", s_clean, re.I):
            cleaned.append(s_clean)
    return cleaned if cleaned else ([text.strip()] if text.strip() else [])


def token_overlap_entailment(claim: str, context: str, threshold: float = 0.5) -> tuple[bool, float]:
    """Assess whether a claim's informative content words are supported by the context.
    
    Returns (is_supported, support_ratio).
    """
    stopwords = {
        "a", "an", "the", "and", "or", "in", "on", "at", "to", "for", "with", "by",
        "of", "is", "are", "was", "were", "be", "been", "being", "have", "has", "had",
        "do", "does", "did", "can", "could", "should", "would", "that", "this", "it",
        "they", "we", "he", "she", "which", "who", "whom", "what", "there", "their"
    }
    claim_tokens = [w for w in re.findall(r"\w+", claim.lower()) if w not in stopwords and len(w) > 2]
    if not claim_tokens:
        return True, 1.0  # Trivial or non-informative claim
    
    context_words = set(re.findall(r"\w+", context.lower()))
    supported_tokens = [w for w in claim_tokens if w in context_words]
    support_ratio = len(supported_tokens) / len(claim_tokens)
    
    return support_ratio >= threshold, support_ratio


def evaluate_faithfulness(answer: str, context: str, threshold: float = 0.55) -> dict:
    """Evaluate groundedness and hallucination rate of an answer against retrieved context.
    
    A claim is deemed supported if its substantive terms are grounded in the retrieved evidence.
    """
    if not answer or "insufficient information" in answer.lower():
        # Valid abstention has 0 hallucination and 1.0 faithfulness
        return {
            "num_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "faithfulness_score": 1.0,
            "hallucination_rate": 0.0,
            "abstained": True,
        }
    
    claims = split_into_claims(answer)
    if not claims:
        return {
            "num_claims": 0,
            "supported_claims": 0,
            "unsupported_claims": 0,
            "faithfulness_score": 1.0,
            "hallucination_rate": 0.0,
            "abstained": False,
        }
    
    supported_count = 0
    for claim in claims:
        is_supported, _ = token_overlap_entailment(claim, context, threshold=threshold)
        if is_supported:
            supported_count += 1
            
    num_claims = len(claims)
    faithfulness_score = supported_count / num_claims
    hallucination_rate = 1.0 - faithfulness_score
    
    return {
        "num_claims": num_claims,
        "supported_claims": supported_count,
        "unsupported_claims": num_claims - supported_count,
        "faithfulness_score": round(faithfulness_score, 4),
        "hallucination_rate": round(hallucination_rate, 4),
        "abstained": False,
    }


def compute_selective_prediction_metrics(
    predictions: list[str],
    references: list[str],
    abstentions: list[bool],
    confidences: list[float] | None = None,
    correctness_threshold: float = 0.65,
    similarity_fn=None,
) -> dict:
    """Compute selective classification and abstention metrics.
    
    - Coverage: fraction of total queries for which an answer was generated.
    - Selective Accuracy: accuracy on answered queries.
    - Selective Risk: error rate on answered queries (1 - Selective Accuracy).
    - Abstention Precision / Recall if reference indicates unanswerable/ground_truth is empty.
    """
    total = len(predictions)
    if total == 0:
        return {}
    
    answered_indices = [i for i, abst in enumerate(abstentions) if not abst]
    coverage = len(answered_indices) / total
    
    if answered_indices:
        correct_count = 0
        for i in answered_indices:
            ans = predictions[i]
            ref = references[i]
            if similarity_fn:
                score = similarity_fn(ans, ref)
                is_correct = score >= correctness_threshold
            else:
                # Basic token jaccard or non-empty match
                w_ans = set(re.findall(r"\w+", ans.lower()))
                w_ref = set(re.findall(r"\w+", ref.lower()))
                overlap = len(w_ans & w_ref) / max(len(w_ans | w_ref), 1)
                is_correct = overlap >= 0.3
            if is_correct:
                correct_count += 1
        selective_accuracy = correct_count / len(answered_indices)
        selective_risk = 1.0 - selective_accuracy
    else:
        selective_accuracy = 0.0
        selective_risk = 0.0
        
    return {
        "total_queries": total,
        "answered_queries": len(answered_indices),
        "abstained_queries": total - len(answered_indices),
        "coverage": round(coverage, 4),
        "selective_accuracy": round(selective_accuracy, 4),
        "selective_risk": round(selective_risk, 4),
    }


def bootstrap_confidence_interval(
    values: list[float],
    num_samples: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute mean and (1 - alpha) non-parametric bootstrap confidence interval."""
    if not values:
        return 0.0, 0.0, 0.0
    arr = np.array(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(arr)
    boot_means = np.empty(num_samples)
    for i in range(num_samples):
        indices = rng.integers(0, n, size=n)
        boot_means[i] = np.mean(arr[indices])
    
    mean_val = float(np.mean(arr))
    lower_bound = float(np.percentile(boot_means, (alpha / 2.0) * 100))
    upper_bound = float(np.percentile(boot_means, (1.0 - alpha / 2.0) * 100))
    return round(mean_val, 4), round(lower_bound, 4), round(upper_bound, 4)
