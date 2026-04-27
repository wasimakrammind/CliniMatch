# CliniMatch

**AI-powered clinical trial matching — patient profile in, ranked active trials out, in ~1–2 minutes.**

Built for AI for Engineers · Milestone 2 · Group 13.


---

## What it does

CliniMatch takes a clinical patient profile (diagnosis, stage, prior treatments, comorbidities, labs, location) and returns a **ranked list of actively-recruiting clinical trials** from [ClinicalTrials.gov](https://clinicaltrials.gov), with **explainable per-criterion reasoning** and **drug-evidence verification** from Semantic Scholar and PubMed.

It's a 5-step AI pipeline running on a single OpenAI API key — total cost per query: about **5 cents**.

```
Patient Profile  ─▶  Trial Retrieval  ─▶  Topic-wise RAG Eligibility
   (11 fields)        (CT.gov v2 API)        (8 medical topics, GPT-4.1 mini)
                                                       │
                                                       ▼
                            Multi-Objective Ranker  ◀─  Evidence Verification
                            (5 weighted dimensions)     (Semantic Scholar + PubMed)
```

## Highlights

- **Topic-wise RAG** — eligibility criteria are split into 8 medical topics (demographics, disease, prior treatment, lab values, biomarkers, comorbidities, medications, other exclusions) using regex classification, then evaluated **independently per topic** so you get **explainable per-topic scores**, not one black-box number.
- **Multi-objective ranking** — weighted sum of eligibility, geographic distance, trial phase, recruitment urgency, and drug-evidence strength. User-tunable sliders.
- **Live data** — ClinicalTrials.gov v2 API, Semantic Scholar, PubMed via Biopython. No staleness, no caching.
- **Parallel orchestration** — `ThreadPoolExecutor` runs 10 trials × 8 topic chunks concurrently. Cut wall-clock from ~30 min to ~1–2 min.
- **Free everywhere we could** — only paid dependency is the OpenAI API. Total demo cost: ~$5.

## Demo screenshots

The dashboard at `http://localhost:8501`:

- Red-themed hero header with the CliniMatch heart-with-ECG logo
- Sidebar: load a sample patient or enter manually, plus 5 ranking-weight sliders
- Pipeline step indicator (Search → Eligibility → Evidence → Rank → Done)
- Trial cards with circular match-score rings, colored chips (NCT, phase, sponsor), recommendation pills (STRONG / POSSIBLE / UNLIKELY MATCH)
- Per-trial details: 4 tabs for Scores, Eligibility (with topic-wise progress bars + per-criterion PASS/FAIL pills), Evidence (citations), Locations (with map)

## Quick start

### Prerequisites
- Python 3.10+
- An [OpenAI API key](https://platform.openai.com/api-keys) (~$5 covers a full demo)

### Setup

```bash
git clone https://github.com/wasimakrammind/CliniMatch.git
cd CliniMatch

# Install dependencies
py -m pip install -r requirements.txt
# (macOS/Linux: python3 -m pip install -r requirements.txt)

# Add your API key
cp .env.example .env
# Open .env and replace `sk-your-openai-key-here` with your real key
```

### Run

```bash
py -m streamlit run app.py
# (macOS/Linux: python3 -m streamlit run app.py)
```

Browser opens at [http://localhost:8501](http://localhost:8501). To stop, press **Ctrl+C** in the terminal.

### First demo
1. Sidebar → **Load Sample Patient** → pick *Sarah Martinez (breast cancer)*
2. Click **Find Matching Trials**
3. Wait ~1–2 min — pipeline indicator advances 5 steps
4. Top trials appear with score rings; expand any one for the per-criterion breakdown

## Project structure

```
CliniMatch/
├── app.py                          # Streamlit dashboard (UI + 5-step orchestration)
├── config.py                       # Env vars, model name, weights, concurrency knobs
├── requirements.txt                # Python dependencies
├── README.md                       # This file
├── PROJECT_GUIDE.md                # Full project explanation (read this for the deep dive)
├── .env.example                    # Template for your .env file
├── .gitignore                      # Don't push .env, .venv, __pycache__, etc.
├── .streamlit/
│   └── config.toml                 # Theme — primary red, sans-serif
├── support_files/
│   ├── clinimatch_logo.svg         # Red heart with W-wave ECG (brand mark)
│   ├── HOW_TO_RUN.md               # Step-by-step run guide
│   └── architecture_diagram.html   # Standalone HTML pipeline diagram
├── pipeline/
│   ├── trial_retrieval.py          # Step 1+2: GPT disease extraction + CT.gov search
│   ├── eligibility_filter.py       # Step 3: topic-wise RAG (the brain)
│   ├── evidence.py                 # Step 4: Semantic Scholar + PubMed lookups
│   └── ranker.py                   # Step 5: 5-objective weighted ranking
└── data/
    ├── patient_1_breast_cancer.json
    ├── patient_2_renal_cell.json
    └── patient_3_lung_cancer.json
```

## Tech stack

| Layer | Tool | License/cost |
|---|---|---|
| LLM | OpenAI GPT-4.1 mini | Paid (~$0.05/query) |
| Trial database | ClinicalTrials.gov v2 API | Free |
| Drug evidence | Semantic Scholar API | Free |
| Medical literature | PubMed via Biopython | Free |
| Geocoding | Geopy + OpenStreetMap | Free |
| UI | Streamlit | Free |
| Concurrency | `concurrent.futures.ThreadPoolExecutor` | stdlib |
| Config | python-dotenv | Free |

A full **9-layer Implemented vs Ideal** breakdown lives in `PROJECT_GUIDE.md` and on Slide 4 of the presentation deck.

## Sample patients

Three synthetic profiles in `data/`:

- **Sarah Martinez** — 52F, Stage IIIA invasive ductal carcinoma (breast), on tamoxifen, history of trastuzumab, comorbidities: type 2 diabetes + hypertension
- **James Chen** — male, renal cell carcinoma
- **Maria Johnson** — female, non-small cell lung cancer

You can also enter a patient manually in the sidebar.

## Configuration

Tune in `config.py`:

```python
OPENAI_MODEL       = "gpt-4.1-mini"  # bigger = more accurate, slower, more expensive
CTGOV_MAX_RESULTS  = 25              # trials per query (more = slower)
TRIAL_WORKERS      = 10              # parallel trials (raise if rate-limit OK)
CHUNK_WORKERS      = 8               # parallel topic chunks per trial
EVIDENCE_WORKERS   = 8               # parallel evidence calls
EVIDENCE_TOP_K     = 15              # only fetch evidence for top-K eligible trials

DEFAULT_WEIGHTS = {                  # users override these via sliders at runtime
    "eligibility": 0.35,
    "distance":    0.20,
    "phase":       0.15,
    "urgency":     0.15,
    "evidence":    0.15,
}
```

## Performance

| Metric | Value |
|---|---|
| Trials retrieved per query | 25 |
| End-to-end wall clock | ~1–2 min (first run can be ~3 min for cold connections) |
| Parallel trials | 10× concurrent |
| Parallel topic chunks per trial | 8× concurrent |
| OpenAI cost per query | ~$0.05 |
| Free APIs | ClinicalTrials.gov · Semantic Scholar · PubMed · OSM |

Before parallelization (sequential calls): ~30 min wall clock. After: 1–2 min. **Single biggest engineering win of the project.**

## Known limitations

- **English-only** — eligibility text and patient input are assumed to be in English.
- **No offline mode** — every run hits live APIs. CT.gov occasional 5xx errors are handled with try/except + neutral fallback.
- **3 sample patients** — built around oncology workflows (breast, renal, lung). Other domains will work but haven't been thoroughly validated.
- **Regex chunking** — production version would use OpenAI embeddings + FAISS for synonym recall (e.g. "high BP" → "hypertension"). Roadmap.
- **No clinical validation** — output is a ranked recommendation, not medical advice. Always defer to a qualified oncologist before enrolling a patient.

## Roadmap (the "Ideal" stack)

- Replace regex chunking with OpenAI `text-embedding-3-large` + FAISS
- Replace weighted-sum ranker with learning-to-rank (XGBoost / LambdaMART) trained on real outcome data
- Add LLM-as-judge evaluation + clinician panel
- Migrate from Streamlit to React + Next.js + Vercel
- FastAPI + Celery + Cloud Run for distributed backend
- Add Embase + ClinicalKey for clinical-grade evidence

## License

MIT. See `LICENSE` if present.

## Acknowledgments

- **ClinicalTrials.gov** for the open trial registry
- **Semantic Scholar** for the open academic API
- **NCBI / PubMed** for biomedical literature access
- **OpenAI** for GPT-4.1 mini
- **Streamlit** for making Python web apps trivial

## Authors

**Wasim Akram Shaik** and Group 13 · *AI for Engineers · Spring 2026*

---

For the deep technical walkthrough — what each file does, how the pipeline flows, every challenge and fix — see [`PROJECT_GUIDE.md`](./PROJECT_GUIDE.md).
