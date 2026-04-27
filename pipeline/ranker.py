"""
Step 5: Multi-Objective Ranking
Scores each trial across 5 dimensions, normalizes to 0-1, applies user weights.
1. Eligibility Score (from LLM filter)
2. Geographic Distance (patient to trial site)
3. Trial Phase (Phase 3 > Phase 2 > Phase 1)
4. Recruitment Urgency (closer to completion = more urgent)
5. Evidence Strength (Semantic Scholar + PubMed)
"""

from datetime import datetime
from geopy.distance import geodesic
import config


PHASE_SCORES = {
    "PHASE3": 1.0,
    "PHASE2_PHASE3": 0.85,
    "PHASE2": 0.7,
    "PHASE1_PHASE2": 0.55,
    "PHASE1": 0.4,
    "EARLY_PHASE1": 0.25,
    "NA": 0.1,
    "N/A": 0.1,
}


def _normalize(value: float, min_val: float, max_val: float) -> float:
    """Normalize a value to 0-1 range."""
    if max_val == min_val:
        return 0.5
    return max(0.0, min(1.0, (value - min_val) / (max_val - min_val)))


def score_distance(patient_location: tuple, trial: dict) -> float:
    """Score geographic proximity (closer = higher score). Returns 0-1."""
    if not patient_location or not patient_location[0]:
        return 0.5  # neutral if no location

    min_distance = float("inf")
    for loc in trial.get("locations", []):
        lat, lon = loc.get("lat"), loc.get("lon")
        if lat and lon:
            try:
                dist = geodesic(patient_location, (lat, lon)).miles
                min_distance = min(min_distance, dist)
            except Exception:
                continue

    if min_distance == float("inf"):
        return 0.3  # no location data available

    # Score: 0 miles = 1.0, 500+ miles = 0.0 (linear decay)
    return max(0.0, 1.0 - (min_distance / 500.0))


def score_phase(trial: dict) -> float:
    """Score trial phase (higher phase = more established)."""
    phase = trial.get("phase", "N/A").upper().replace(" ", "")
    return PHASE_SCORES.get(phase, 0.1)


def score_urgency(trial: dict) -> float:
    """Score recruitment urgency (closer to completion date = more urgent = higher)."""
    date_str = trial.get("completion_date", "")
    if not date_str:
        return 0.5

    try:
        # CT.gov dates are like "2025-12" or "December 2025"
        for fmt in ("%Y-%m-%d", "%Y-%m", "%B %Y", "%B %d, %Y"):
            try:
                comp_date = datetime.strptime(date_str, fmt)
                break
            except ValueError:
                continue
        else:
            return 0.5

        days_left = (comp_date - datetime.now()).days
        if days_left < 0:
            return 0.2  # past completion date
        # Closer to closing = more urgent = higher score
        # 0 days = 1.0, 730 days (2 years) = 0.0
        return max(0.0, min(1.0, 1.0 - (days_left / 730.0)))
    except Exception:
        return 0.5


def rank_trials(trials: list[dict], weights: dict = None, patient_coords: tuple = None) -> list[dict]:
    """
    Rank trials using multi-objective weighted scoring.
    Each trial should already have 'eligibility' and 'evidence' dicts attached.
    """
    weights = weights or config.DEFAULT_WEIGHTS

    for trial in trials:
        # Skip filtered-out trials
        if trial.get("eligibility", {}).get("_filtered_out"):
            trial["final_score"] = 0.0
            trial["scores"] = {}
            continue

        scores = {
            "eligibility": trial.get("eligibility", {}).get("overall_score", 0.0),
            "distance": score_distance(patient_coords, trial),
            "phase": score_phase(trial),
            "urgency": score_urgency(trial),
            "evidence": trial.get("evidence", {}).get("evidence_score", 0.0),
        }

        # Weighted sum
        final = sum(scores[k] * weights.get(k, 0.0) for k in scores)
        trial["scores"] = scores
        trial["final_score"] = round(final, 4)

    # Sort descending by final score
    trials.sort(key=lambda t: t.get("final_score", 0), reverse=True)
    return trials
