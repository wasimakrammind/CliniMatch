"""
Step 2: Disease Extraction + Trial Retrieval
- Uses OpenAI to extract structured disease info from patient profile
- Queries ClinicalTrials.gov API v2 for recruiting trials
"""

import requests
from openai import OpenAI
import config

client = OpenAI(api_key=config.OPENAI_API_KEY)


def extract_disease_info(patient: dict) -> dict:
    """Send patient profile to LLM, get back structured disease info for querying."""
    prompt = f"""You are a medical information extractor. Given a patient profile, extract:

1. primary_condition: The BROAD disease category in standard medical terminology.
   GOOD: "breast cancer", "non-small cell lung cancer", "renal cell carcinoma".
   BAD: "invasive ductal carcinoma stage IIIA HER2-positive" (too specific — would return zero trials).
   Use the broadest term that still describes the disease.

2. search_query: A SHORT 1-3 word string for ClinicalTrials.gov search.
   This goes directly into the API. It MUST be broad enough to return 20+ recruiting trials.
   Examples: "breast cancer", "lung cancer", "melanoma".
   Do NOT include stage, mutations, biomarkers, or histology subtype.

3. stage: Disease stage if applicable (free text).
4. key_factors: List of important medical factors (prior treatments, mutations, biomarkers) — used later for eligibility, NOT for search.

Patient Profile:
- Age: {patient.get('age', 'Unknown')}
- Gender: {patient.get('gender', 'Unknown')}
- Diagnosis: {patient.get('diagnosis', 'Unknown')}
- Stage: {patient.get('stage', 'Unknown')}
- Prior Treatments: {patient.get('prior_treatments', 'None')}
- Medications: {patient.get('medications', 'None')}
- Comorbidities: {patient.get('comorbidities', 'None')}
- Lab Values: {patient.get('lab_values', 'None')}
- Notes: {patient.get('notes', 'None')}

Return ONLY valid JSON with keys: primary_condition, search_query, stage, key_factors.
condition_keywords (alias for search_query) is also accepted."""

    resp = client.chat.completions.create(
        model=config.OPENAI_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        response_format={"type": "json_object"},
    )
    import json
    return json.loads(resp.choices[0].message.content)


def search_trials(condition_keywords: str, max_results: int = None) -> list[dict]:
    """Query ClinicalTrials.gov API v2 for actively recruiting trials."""
    max_results = max_results or config.CTGOV_MAX_RESULTS
    params = {
        "query.cond": condition_keywords,
        "filter.overallStatus": "RECRUITING",
        "pageSize": max_results,
        "format": "json",
        "fields": (
            "NCTId,BriefTitle,OfficialTitle,OverallStatus,Phase,"
            "BriefSummary,EligibilityCriteria,EnrollmentCount,"
            "StartDate,PrimaryCompletionDate,LeadSponsorName,"
            "LocationCity,LocationState,LocationCountry,"
            "LocationFacility,PointOfContactEMail,"
            "InterventionName,InterventionType,"
            "LocationGeoPoint"
        ),
    }

    resp = requests.get(config.CTGOV_BASE_URL, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    trials = []
    for study in data.get("studies", []):
        proto = study.get("protocolSection", {})
        ident = proto.get("identificationModule", {})
        status_mod = proto.get("statusModule", {})
        design = proto.get("designModule", {})
        elig = proto.get("eligibilityModule", {})
        desc = proto.get("descriptionModule", {})
        sponsor = proto.get("sponsorCollaboratorsModule", {})
        contacts = proto.get("contactsLocationsModule", {})
        arms = proto.get("armsInterventionsModule", {})

        # Extract locations with coordinates
        locations = []
        for loc in contacts.get("locations", []):
            geo = loc.get("geoPoint", {})
            locations.append({
                "facility": loc.get("facility", ""),
                "city": loc.get("city", ""),
                "state": loc.get("state", ""),
                "country": loc.get("country", ""),
                "lat": geo.get("lat"),
                "lon": geo.get("lon"),
            })

        # Extract interventions
        interventions = []
        for arm in arms.get("interventions", []):
            interventions.append({
                "name": arm.get("name", ""),
                "type": arm.get("type", ""),
            })

        phases = design.get("phases", [])

        trials.append({
            "nct_id": ident.get("nctId", ""),
            "title": ident.get("briefTitle", ""),
            "official_title": ident.get("officialTitle", ""),
            "status": status_mod.get("overallStatus", ""),
            "phase": phases[0] if phases else "N/A",
            "summary": desc.get("briefSummary", ""),
            "eligibility_criteria": elig.get("eligibilityCriteria", ""),
            "enrollment": design.get("enrollmentInfo", {}).get("count"),
            "start_date": status_mod.get("startDateStruct", {}).get("date", ""),
            "completion_date": status_mod.get("primaryCompletionDateStruct", {}).get("date", ""),
            "sponsor": sponsor.get("leadSponsor", {}).get("name", ""),
            "locations": locations,
            "interventions": interventions,
        })

    return trials


def retrieve_trials(patient: dict) -> tuple[list[dict], dict]:
    """Full pipeline: extract disease info, then search for trials.

    Search strategy: prefer the BROAD primary_condition; fall back to search_query / condition_keywords;
    final fallback is the patient's raw diagnosis text. If the broad search returns < 5 trials, retry
    with a shorter form (first 2 words) so we never end up with an empty result page.
    """
    disease_info = extract_disease_info(patient)

    # Pick the broadest available term, in priority order.
    keywords = (
        disease_info.get("primary_condition")
        or disease_info.get("search_query")
        or disease_info.get("condition_keywords")
        or patient.get("diagnosis", "")
    )
    if isinstance(keywords, list):
        keywords = " ".join(keywords)
    keywords = (keywords or "").strip()

    trials = search_trials(keywords) if keywords else []

    # If the search came back too narrow, retry with the first 2 words only ("breast cancer").
    if len(trials) < 5 and keywords:
        broad = " ".join(keywords.split()[:2])
        if broad and broad.lower() != keywords.lower():
            trials = search_trials(broad)
            disease_info["search_query_used"] = broad
        else:
            disease_info["search_query_used"] = keywords
    else:
        disease_info["search_query_used"] = keywords

    # Surface the actual query string used so the UI can show it
    disease_info["condition_keywords"] = disease_info.get("search_query_used", keywords)
    return trials, disease_info
