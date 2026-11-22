from dataclasses import dataclass


_VERDICT_SCORES = {
    "BAD": 0.0,
    "PARTIAL": 0.5,
    "GOOD": 1.0,
}
DEFAULT_THRESHOLD = 0.5
DEFAULT_RELEVANCE_WEIGHT = 0.5


@dataclass(frozen=True)
class TrustAssessment:
    retrieval_relevance: float
    evidence_score: float
    trust_score: float
    verdict: str
    abstain: bool


def assess_retrieval(
    similarities: list[float],
    verdict: str,
    threshold: float = DEFAULT_THRESHOLD,
    relevance_weight: float = DEFAULT_RELEVANCE_WEIGHT,
) -> TrustAssessment:
    """Combine dense retrieval relevance and the retrieval critic's verdict.

    The resulting score is a decision index, not a calibrated probability.
    """
    normalized_verdict = verdict.strip().upper()
    if normalized_verdict not in _VERDICT_SCORES:
        raise ValueError(f"Unknown retrieval verdict: {verdict!r}")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")
    if not 0.0 <= relevance_weight <= 1.0:
        raise ValueError("relevance_weight must be between 0 and 1")

    finite_similarities = [score for score in similarities if score == score]
    relevance = max(0.0, min(1.0, max(finite_similarities, default=0.0)))
    evidence_score = _VERDICT_SCORES[normalized_verdict]
    trust_score = (
        relevance_weight * relevance
        + (1.0 - relevance_weight) * evidence_score
    )
    abstain = normalized_verdict == "BAD" or trust_score < threshold

    return TrustAssessment(
        retrieval_relevance=relevance,
        evidence_score=evidence_score,
        trust_score=trust_score,
        verdict=normalized_verdict,
        abstain=abstain,
    )
