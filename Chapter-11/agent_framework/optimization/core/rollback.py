"""Rollback helpers shared by local and library optimizers."""


def should_accept_candidate(
    candidate_dev_score: float,
    best_dev_score: float,
    *,
    rollback: bool = True,
) -> bool:
    """Return True when a candidate prompt should replace the current best."""
    if not rollback:
        return True
    return candidate_dev_score >= best_dev_score
