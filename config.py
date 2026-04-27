import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

PUBMED_EMAIL = os.getenv("PUBMED_EMAIL", "clinimatch@example.com")

OPENAI_MODEL = "gpt-4.1-mini"

CTGOV_BASE_URL = "https://clinicaltrials.gov/api/v2/studies"
CTGOV_MAX_RESULTS = 25

TRIAL_WORKERS = 10    # how many trials to evaluate in parallel
CHUNK_WORKERS = 8     # how many topic chunks per trial in parallel
EVIDENCE_WORKERS = 8  # parallel evidence (Semantic Scholar + PubMed) lookups
EVIDENCE_TOP_K = 15   # only run evidence on top-K eligible trials by score

DEFAULT_WEIGHTS = {
    "eligibility": 0.35,
    "distance": 0.20,
    "phase": 0.15,
    "urgency": 0.15,
    "evidence": 0.15,
}
