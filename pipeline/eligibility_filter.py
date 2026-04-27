"""
Step 3: Eligibility Filtering via LLM with Topic-Wise RAG Chunking
- Chunks eligibility criteria by medical topic (demographics, disease, treatment, labs, etc.)
- Evaluates each chunk against relevant parts of the patient profile
- Aggregates chunk-level results into an overall eligibility decision
"""

import re
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)

# ── Topic definitions for chunking ───────────────────────────────────
TOPIC_PATTERNS = {
    "demographics": [
        r"(?i)\bage\b", r"(?i)\bgender\b", r"(?i)\bsex\b", r"(?i)\bpregnant",
        r"(?i)\blife expectancy\b", r"(?i)\becog\b", r"(?i)\bperformance status\b",
        r"(?i)\bconsent\b", r"(?i)\b(?:18|21)\s*years?\b",
    ],
    "disease": [
        r"(?i)\bdiagnos", r"(?i)\bhistolog", r"(?i)\bstage\b", r"(?i)\btumor\b",
        r"(?i)\bneoplasm\b", r"(?i)\bcancer\b", r"(?i)\bcarcinoma\b",
        r"(?i)\bmalignant\b", r"(?i)\bmetasta", r"(?i)\brecurrent\b",
        r"(?i)\bmeasurable disease\b", r"(?i)\bRECIST\b",
    ],
    "prior_treatment": [
        r"(?i)\bprior\b.*(?:therap|treatment|regimen)", r"(?i)\bprevious\b.*(?:therap|treatment)",
        r"(?i)\bchemotherap", r"(?i)\bimmunotherap", r"(?i)\bradiation\b",
        r"(?i)\bsurger", r"(?i)\brefractory\b", r"(?i)\bprogress(?:ed|ion)\b",
        r"(?i)\bline.?of.?therap", r"(?i)\bwashout\b", r"(?i)\bnaive\b",
    ],
    "lab_values": [
        r"(?i)\begfr\b", r"(?i)\bcreatinine\b", r"(?i)\bhemoglobin\b",
        r"(?i)\bplatelet\b", r"(?i)\bwbc\b", r"(?i)\bneutrophil\b",
        r"(?i)\bbilirubin\b", r"(?i)\balt\b", r"(?i)\bast\b",
        r"(?i)\bliver function\b", r"(?i)\brenal function\b",
        r"(?i)\badequate.*function\b", r"(?i)\blab\b.*\bvalue",
    ],
    "biomarkers": [
        r"(?i)\bher2\b", r"(?i)\ber\b.?positive", r"(?i)\bpr\b.?positive",
        r"(?i)\bpd.?l1\b", r"(?i)\begfr\b.*mutation", r"(?i)\balk\b",
        r"(?i)\bbraf\b", r"(?i)\bkras\b", r"(?i)\bbiomarker\b",
        r"(?i)\bmolecular\b", r"(?i)\bgenomic\b",
    ],
    "comorbidities": [
        r"(?i)\bcomorbid", r"(?i)\bhiv\b", r"(?i)\bhepatitis\b",
        r"(?i)\bcardiac\b", r"(?i)\bautoimmune\b", r"(?i)\bdiabetes\b",
        r"(?i)\bhypertension\b", r"(?i)\bseizure\b", r"(?i)\bbrain\s*metasta",
        r"(?i)\borgan\s*(?:dysfunction|failure)\b", r"(?i)\binfection\b",
    ],
    "medications": [
        r"(?i)\bconcomitant\b.*(?:medication|drug|therap)",
        r"(?i)\bcorticosteroid\b", r"(?i)\banticoagulant\b",
        r"(?i)\bimmunosuppress", r"(?i)\bcyp\d", r"(?i)\bprohibited\b.*(?:medication|drug)",
        r"(?i)\bcurrent.*(?:medication|treatment)\b", r"(?i)\bprednisone\b",
    ],
    "other_exclusions": [
        r"(?i)\btransplant\b", r"(?i)\bpregnant\b", r"(?i)\bbreastfeed",
        r"(?i)\ballerg", r"(?i)\bhypersensitiv", r"(?i)\bcontraindic",
        r"(?i)\bsecond(?:ary)?\s*malignanc",
    ],
}


def chunk_criteria_by_topic(criteria_text: str) -> dict[str, list[str]]:
    """
    Split eligibility criteria into topic-wise chunks.
    Each criterion line is assigned to the most relevant medical topic.
    Lines that don't match any topic go into 'general'.
    """
    if not criteria_text:
        return {"general": ["No criteria available"]}

    # Split into individual criteria lines
    lines = re.split(r'\n(?=\s*[-*•]|\s*\d+[.)]\s|\s*(?:Inclusion|Exclusion))', criteria_text)
    # Further split on numbered lists and bullet points within lines
    criteria = []
    for line in lines:
        sub = re.split(r'\n\s*[-*•]\s*|\n\s*\d+[.)]\s*', line)
        criteria.extend(s.strip() for s in sub if s.strip())

    # Classify each criterion into a topic
    chunks: dict[str, list[str]] = {}
    for criterion in criteria:
        assigned = False
        for topic, patterns in TOPIC_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, criterion):
                    chunks.setdefault(topic, []).append(criterion)
                    assigned = True
                    break
            if assigned:
                break
        if not assigned:
            chunks.setdefault("general", []).append(criterion)

    return chunks


CHUNK_EVAL_PROMPT = """You are evaluating clinical trial eligibility criteria for a specific medical topic.

## Patient Profile
- Age: {age}
- Gender: {gender}
- Diagnosis: {diagnosis}
- Stage: {stage}
- Prior Treatments: {prior_treatments}
- Current Medications: {medications}
- Comorbidities: {comorbidities}
- Lab Values: {lab_values}
- Additional Notes: {notes}

## Topic: {topic}
## Criteria in this topic:
{criteria_text}

## Instructions
Evaluate EACH criterion. If patient info for a criterion is missing, mark as "unknown" not "fail".
Understand medical synonyms: "high BP" = "hypertension", "tumor" = "neoplasm", "sugar problem" = "diabetes mellitus", "heart attack" = "MI", "chemo didn't work" = "refractory to prior therapy".

Return ONLY valid JSON:
{{
  "topic": "{topic}",
  "score": <float 0.0 to 1.0 — how well patient matches THIS topic's criteria>,
  "results": [
    {{"criterion": "<text>", "met": <true/false/null>, "is_exclusion": <true/false>, "reason": "<brief>"}}
  ],
  "has_exclusion_hit": <true if any exclusion criterion disqualifies the patient>
}}"""


def evaluate_chunk(patient: dict, topic: str, criteria: list[str]) -> dict:
    """Evaluate a single topic chunk of eligibility criteria."""
    criteria_text = "\n".join(f"- {c}" for c in criteria)

    prompt = CHUNK_EVAL_PROMPT.format(
        age=patient.get("age", "Unknown"),
        gender=patient.get("gender", "Unknown"),
        diagnosis=patient.get("diagnosis", "Unknown"),
        stage=patient.get("stage", "Unknown"),
        prior_treatments=patient.get("prior_treatments", "None"),
        medications=patient.get("medications", "None"),
        comorbidities=patient.get("comorbidities", "None"),
        lab_values=patient.get("lab_values", "None"),
        notes=patient.get("notes", "None"),
        topic=topic,
        criteria_text=criteria_text,
    )

    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )

    result = json.loads(resp.choices[0].message.content)
    result.setdefault("topic", topic)
    result.setdefault("score", 0.0)
    result.setdefault("results", [])
    result.setdefault("has_exclusion_hit", False)
    return result


def evaluate_eligibility(patient: dict, trial: dict) -> dict:
    """
    Full eligibility evaluation with topic-wise RAG chunking.
    1. Chunk criteria by medical topic
    2. Evaluate each chunk independently
    3. Aggregate into overall score and recommendation
    """
    criteria_text = trial.get("eligibility_criteria", "")
    chunks = chunk_criteria_by_topic(criteria_text)

    # Evaluate each topic chunk IN PARALLEL — these are independent LLM calls,
    # so we can fan out without affecting accuracy.
    chunk_results = []
    workers = max(1, min(getattr(config, "CHUNK_WORKERS", 8), len(chunks)))
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = {
            ex.submit(evaluate_chunk, patient, topic, criteria): topic
            for topic, criteria in chunks.items()
        }
        for fut in as_completed(futures):
            try:
                chunk_results.append(fut.result())
            except Exception as e:
                topic = futures[fut]
                # Fail-soft: surface a neutral chunk rather than crashing the trial
                chunk_results.append({
                    "topic": topic,
                    "score": 0.5,
                    "results": [],
                    "has_exclusion_hit": False,
                    "_error": str(e),
                })

    # Aggregate results
    inclusion_results = []
    exclusion_results = []
    topic_scores = {}
    has_exclusion = False

    for chunk in chunk_results:
        topic_scores[chunk["topic"]] = chunk["score"]
        if chunk.get("has_exclusion_hit"):
            has_exclusion = True

        for r in chunk.get("results", []):
            entry = {
                "criterion": r.get("criterion", ""),
                "reason": r.get("reason", ""),
                "topic": chunk["topic"],
            }
            if r.get("is_exclusion"):
                entry["excluded"] = r.get("met", None)  # True = excluded
                exclusion_results.append(entry)
            else:
                entry["met"] = r.get("met", None)
                inclusion_results.append(entry)

    # Overall score: weighted average of topic scores
    if topic_scores:
        overall_score = sum(topic_scores.values()) / len(topic_scores)
    else:
        overall_score = 0.0

    # Count how many exclusion criteria genuinely fired (excluded == True)
    exclusion_hits = sum(1 for r in exclusion_results if r.get("excluded") is True)
    total_exclusions = len(exclusion_results) or 1
    exclusion_ratio = exclusion_hits / total_exclusions

    # Soft penalty: each genuine exclusion drops the score, but does not zero it out.
    # Only mark a trial as fully EXCLUDED if there are MULTIPLE strong exclusion hits
    # OR a single hit AND a low overall topic score. A single soft exclusion
    # (e.g. one borderline criterion in a complex profile) becomes a POSSIBLE_MATCH
    # with a warning, not a hard zero — the clinician decides.
    if exclusion_hits > 0:
        overall_score = max(0.0, overall_score - 0.15 * exclusion_hits)

    # Determine recommendation
    hard_exclude = (exclusion_hits >= 2) or (exclusion_hits >= 1 and overall_score < 0.30)

    if hard_exclude:
        recommendation = "EXCLUDED"
    elif overall_score >= 0.7:
        recommendation = "STRONG_MATCH"
    elif overall_score >= 0.4:
        recommendation = "POSSIBLE_MATCH"
    else:
        recommendation = "UNLIKELY_MATCH"

    # Keep has_exclusion flag for callers that want it, but it no longer auto-excludes
    has_exclusion = hard_exclude

    met_count = sum(1 for r in inclusion_results if r.get("met") is True)
    total_inc = len(inclusion_results) or 1

    if hard_exclude:
        excl_msg = f"{exclusion_hits} exclusion criteria fired — patient disqualified."
    elif exclusion_hits > 0:
        excl_msg = f"{exclusion_hits} soft exclusion warning(s) — review with clinician."
    else:
        excl_msg = "No exclusion criteria triggered."

    return {
        "overall_score": round(overall_score, 3),
        "eligible": not hard_exclude and overall_score >= 0.4,
        "recommendation": recommendation,
        "summary": (
            f"Evaluated {len(chunks)} topic chunks ({', '.join(chunks.keys())}). "
            f"Patient meets {met_count}/{total_inc} inclusion criteria. {excl_msg}"
        ),
        "topic_scores": topic_scores,
        "inclusion_results": inclusion_results,
        "exclusion_results": exclusion_results,
        "chunks_evaluated": list(chunks.keys()),
    }


def _evaluate_one(patient: dict, trial: dict) -> dict:
    """Evaluate a single trial; safe to run in a worker thread."""
    try:
        elig = evaluate_eligibility(patient, trial)
    except Exception as e:
        elig = {
            "overall_score": 0.0,
            "eligible": False,
            "recommendation": "UNLIKELY_MATCH",
            "summary": f"Eligibility evaluation failed: {e}",
            "topic_scores": {},
            "inclusion_results": [],
            "exclusion_results": [],
            "chunks_evaluated": [],
        }
    trial["eligibility"] = elig
    trial["eligibility"]["_filtered_out"] = elig.get("recommendation") == "EXCLUDED"
    return trial


def filter_trials(patient: dict, trials: list[dict]) -> list[dict]:
    """Evaluate eligibility for all trials IN PARALLEL.

    Each trial is independent, so we fan out across a thread pool. This brings
    a typical 25-trial run from ~10–15 minutes down to ~30–60 seconds.
    """
    workers = max(1, min(getattr(config, "TRIAL_WORKERS", 10), len(trials) or 1))
    if not trials:
        return trials

    with ThreadPoolExecutor(max_workers=workers) as ex:
        futures = [ex.submit(_evaluate_one, patient, t) for t in trials]
        for fut in as_completed(futures):
            # Results are already attached to each trial dict in-place,
            # but we still call .result() to surface any unexpected exceptions.
            fut.result()
    return trials
