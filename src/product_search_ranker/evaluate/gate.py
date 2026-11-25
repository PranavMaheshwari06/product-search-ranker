"""`psr gate`: block publishing a model that scores worse than the published one.

ponytail: each publish becomes the next baseline, so repeated dips inside the margin can add up
(SPEC 11 wording). Compare with a fixed baseline if the DAG ever retrains on changing data.
"""

import math

SYSTEM = "ltr"


def ndcg_all(report: dict) -> float | None:
    v = report.get("systems", {}).get(SYSTEM, {}).get("all", {}).get("ndcg@10")
    return v if isinstance(v, (int, float)) and math.isfinite(v) else None


def decide(candidate: dict, published: dict | None, margin: float) -> tuple[bool, str]:
    if not (math.isfinite(margin) and margin >= 0):
        return False, f"margin must be a finite number >= 0, got {margin}"
    new = ndcg_all(candidate)
    if new is None:
        return False, f"candidate has no {SYSTEM} NDCG@10 (all): run `psr eval --candidate`"
    if published is None:
        return True, f"first run: nothing published yet, candidate {new}"
    old = ndcg_all(published)
    if old is None:
        return False, f"published metrics have no {SYSTEM} NDCG@10 (all); fix the target first"
    if new < old - margin:
        return False, f"candidate {new} < published {old} - margin {margin}"
    return True, f"candidate {new} >= published {old} - margin {margin}"
