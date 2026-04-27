"""
CliniMatch — AI-Powered Clinical Trial Matching Dashboard
Main Streamlit application tying the full pipeline together.
"""

import streamlit as st
import json
import os
import sys
import math
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(__file__))

import config
from pipeline.trial_retrieval import retrieve_trials
from pipeline.eligibility_filter import filter_trials
from pipeline.evidence import get_evidence
from pipeline.ranker import rank_trials


ASSET_DIR = os.path.join(os.path.dirname(__file__), "support_files")
LOGO_PATH = os.path.join(ASSET_DIR, "clinimatch_logo.svg")


def _load_logo_data_uri() -> str:
    try:
        with open(LOGO_PATH, "rb") as fh:
            encoded = base64.b64encode(fh.read()).decode("ascii")
        return f"data:image/svg+xml;base64,{encoded}"
    except Exception:
        return ""


LOGO_URI = _load_logo_data_uri()


def logo_img(size_px: int = 20, margin_right: int = 8) -> str:
    if not LOGO_URI:
        return ""
    return (
        f'<img src="{LOGO_URI}" width="{size_px}" height="{size_px}" '
        f'style="vertical-align:middle; margin-right:{margin_right}px;"/>'
    )


def logo_img_white(size_px: int = 20, margin_right: int = 8) -> str:
    """Logo with a white-tinted CSS filter, for dark backgrounds."""
    if not LOGO_URI:
        return ""
    return (
        f'<img src="{LOGO_URI}" width="{size_px}" height="{size_px}" '
        f'style="vertical-align:middle; margin-right:{margin_right}px; '
        f'filter: brightness(0) invert(1);"/>'
    )



st.set_page_config(
    page_title="CliniMatch",
    page_icon=LOGO_PATH if os.path.exists(LOGO_PATH) else None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    /* Use a clean modern font stack */
    html, body, [class*="css"] {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter",
                     Roboto, "Helvetica Neue", Arial, sans-serif;
    }

    /* Hide the default Streamlit header padding so the hero hugs the top */
    .block-container { padding-top: 1.5rem; padding-bottom: 4rem; }

    /* ── Hero ───────────────────────────────────────────────── */
    .hero {
        background: linear-gradient(135deg, #7F1D1D 0%, #DC2626 55%, #EF4444 110%);
        border-radius: 18px;
        padding: 28px 36px;
        color: #ffffff;
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 24px;
        box-shadow: 0 10px 30px rgba(220, 38, 38, 0.22);
    }
    .hero-logo {
        background: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.5);
        border-radius: 16px;
        padding: 14px;
        display: flex;
        align-items: center;
        justify-content: center;
        box-shadow: 0 4px 14px rgba(0, 0, 0, 0.15);
    }
    .hero-text h1 {
        margin: 0;
        font-size: 2.0rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #ffffff;
    }
    .hero-text p {
        margin: 4px 0 0 0;
        font-size: 0.95rem;
        color: rgba(255, 255, 255, 0.88);
    }
    .hero-tag {
        margin-left: auto;
        background: rgba(255, 255, 255, 0.18);
        border: 1px solid rgba(255, 255, 255, 0.35);
        padding: 6px 14px;
        border-radius: 999px;
        font-size: 0.78rem;
        font-weight: 600;
        color: #ffffff;
        letter-spacing: 0.4px;
    }

    /* ── Sidebar ────────────────────────────────────────────── */
    section[data-testid="stSidebar"] { background: #FDF7F7; }
    .sidebar-section-title {
        display: flex; align-items: center;
        font-weight: 700; font-size: 1.0rem;
        color: #991B1B;
        margin: 4px 0 8px 0;
    }

    .patient-card {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 8px;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .patient-card .pname {
        font-weight: 700; color: #0B1D33; font-size: 1.05rem; margin-bottom: 2px;
    }
    .patient-card .pdiag {
        color: #B91C1C; font-weight: 600; font-size: 0.85rem; margin-bottom: 10px;
    }
    .patient-card .prow {
        display: flex; justify-content: space-between;
        font-size: 0.82rem; padding: 4px 0; border-top: 1px dashed #E2E8F0;
    }
    .patient-card .prow:first-of-type { border-top: none; }
    .patient-card .pkey { color: #64748B; font-weight: 500; }
    .patient-card .pval { color: #0B1D33; font-weight: 600; text-align: right; max-width: 60%; }

    /* ── Trial card ─────────────────────────────────────────── */
    .trial-card {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 18px 22px 18px 26px;
        margin-bottom: 14px;
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .trial-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 8px 24px rgba(15, 23, 42, 0.08);
    }
    .trial-card .accent {
        position: absolute; top: 0; left: 0; bottom: 0; width: 6px;
    }
    .accent-strong   { background: linear-gradient(180deg, #02C39A, #028090); }
    .accent-possible { background: linear-gradient(180deg, #F59E0B, #F97316); }
    .accent-unlikely { background: linear-gradient(180deg, #DC2626, #991B1B); }

    .trial-row {
        display: flex; align-items: center; justify-content: space-between; gap: 16px;
    }
    .trial-main { flex: 1; min-width: 0; }
    .trial-title {
        font-size: 1.08rem; font-weight: 700; color: #0B1D33;
        margin-bottom: 8px; line-height: 1.35;
    }
    .chip-row { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; }
    .chip {
        display: inline-flex; align-items: center;
        background: #FEF2F2; color: #991B1B;
        border: 1px solid #FECACA;
        padding: 3px 10px; border-radius: 999px;
        font-size: 0.74rem; font-weight: 600;
        white-space: nowrap;
    }
    .chip-nct { background: #EEF2FF; color: #4338CA; border-color: #C7D2FE; }
    .chip-phase { background: #ECFDF5; color: #047857; border-color: #BBF7D0; }
    .chip-sponsor { background: #FEF3C7; color: #92400E; border-color: #FDE68A; max-width: 320px; overflow: hidden; text-overflow: ellipsis; }

    .rec-pill {
        display: inline-block; padding: 3px 10px; border-radius: 999px;
        font-size: 0.72rem; font-weight: 700; letter-spacing: 0.3px;
    }
    .rec-strong-pill   { background: #DCFCE7; color: #047857; }
    .rec-possible-pill { background: #FEF3C7; color: #92400E; }
    .rec-unlikely-pill { background: #FEE2E2; color: #991B1B; }

    /* Score ring */
    .ring-wrap {
        flex: 0 0 auto; display: flex; flex-direction: column;
        align-items: center; justify-content: center;
        min-width: 96px;
    }
    .ring-label {
        font-size: 0.68rem; color: #64748B; font-weight: 600;
        margin-top: 4px; text-transform: uppercase; letter-spacing: 0.5px;
    }

    /* ── Status pills (PASS / FAIL) ──────────────────────────── */
    .pill {
        display: inline-block; padding: 2px 9px; border-radius: 999px;
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.4px;
        margin-right: 8px; vertical-align: middle;
    }
    .pill-pass     { background: #DCFCE7; color: #047857; }
    .pill-fail     { background: #FEE2E2; color: #991B1B; }
    .pill-unknown  { background: #E2E8F0; color: #475569; }
    .pill-excludes { background: #FEE2E2; color: #991B1B; }
    .pill-ok       { background: #DCFCE7; color: #047857; }

    /* ── Topic bar ───────────────────────────────────────────── */
    .topic-row {
        display: flex; align-items: center; gap: 10px;
        margin: 6px 0;
    }
    .topic-name {
        flex: 0 0 140px; font-size: 0.85rem; font-weight: 600; color: #334155;
    }
    .topic-bar-bg {
        flex: 1; height: 10px; background: #EEF2F6;
        border-radius: 999px; overflow: hidden;
    }
    .topic-bar-fill {
        height: 100%; border-radius: 999px;
        background: linear-gradient(90deg, #DC2626, #F87171);
    }
    .topic-pct {
        flex: 0 0 50px; text-align: right;
        font-variant-numeric: tabular-nums;
        font-size: 0.85rem; font-weight: 700; color: #0B1D33;
    }

    /* ── Pipeline step indicator ────────────────────────────── */
    .step-row { display: flex; align-items: center; gap: 0; margin: 6px 0 12px 0; }
    .step {
        flex: 1; display: flex; align-items: center; gap: 8px;
    }
    .step-circle {
        width: 28px; height: 28px; border-radius: 50%;
        display: flex; align-items: center; justify-content: center;
        font-weight: 700; font-size: 0.85rem;
        background: #E2E8F0; color: #64748B;
        border: 2px solid #E2E8F0;
        flex-shrink: 0;
    }
    .step.done .step-circle { background: #02C39A; color: white; border-color: #02C39A; }
    .step.active .step-circle { background: #ffffff; color: #DC2626; border-color: #DC2626; }
    .step-label { font-size: 0.78rem; color: #475569; font-weight: 600; }
    .step.done .step-label { color: #047857; }
    .step.active .step-label { color: #DC2626; }
    .step-line { flex: 1; height: 2px; background: #E2E8F0; margin: 0 6px; }
    .step.done + .step-line, .step-line.done { background: #02C39A; }

    /* ── Landing page hero cards ─────────────────────────────── */
    .landing-wrap { display: flex; flex-direction: column; align-items: center; padding: 30px 0; }
    .landing-logo { margin-bottom: 16px; }
    .landing-title { font-size: 1.6rem; font-weight: 800; color: #991B1B; margin-bottom: 4px; }
    .landing-sub   { color: #64748B; font-size: 0.95rem; margin-bottom: 26px; }
    .step-card {
        background: #ffffff;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 20px;
        height: 100%;
        box-shadow: 0 1px 3px rgba(15, 23, 42, 0.04);
    }
    .step-card .num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 32px; height: 32px; border-radius: 50%;
        background: linear-gradient(135deg, #DC2626, #EF4444);
        color: white; font-weight: 700; margin-bottom: 10px;
    }
    .step-card h4 { margin: 4px 0 6px 0; color: #0B1D33; }
    .step-card p  { margin: 0; color: #64748B; font-size: 0.88rem; }

    /* Hide Streamlit default footer/menu for cleanliness */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

def score_ring_html(score: float, size: int = 76) -> str:
    """Return an SVG circular progress ring HTML for a 0..1 score."""
    pct = max(0.0, min(1.0, score))
    if pct >= 0.6:
        color = "#02C39A"
    elif pct >= 0.35:
        color = "#F59E0B"
    else:
        color = "#DC2626"

    r = (size - 10) / 2
    cx = cy = size / 2
    circ = 2 * math.pi * r
    offset = circ * (1 - pct)

    return (
        f'<svg width="{size}" height="{size}" viewBox="0 0 {size} {size}">'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="#EEF2F6" stroke-width="8" fill="none"/>'
        f'<circle cx="{cx}" cy="{cy}" r="{r}" stroke="{color}" stroke-width="8" fill="none" '
        f'stroke-linecap="round" stroke-dasharray="{circ:.2f}" '
        f'stroke-dashoffset="{offset:.2f}" transform="rotate(-90 {cx} {cy})"/>'
        f'<text x="{cx}" y="{cy + 5}" text-anchor="middle" '
        f'font-family="-apple-system, Segoe UI, sans-serif" '
        f'font-size="{int(size * 0.28)}" font-weight="800" fill="#0B1D33">'
        f'{round(pct * 100)}%</text>'
        f'</svg>'
    )


# ── Sidebar logo ─────────────────────────────────────────────────────
if LOGO_URI:
    try:
        st.logo(LOGO_PATH, size="large")
    except Exception:
        pass


# ── Hero header ──────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div class="hero-logo">{logo_img(56, 0)}</div>
  <div class="hero-text">
    <h1>CliniMatch</h1>
    <p>AI-powered clinical trial matching · Topic-wise RAG · Multi-objective ranking</p>
  </div>
  <span class="hero-tag">MILESTONE 2 · GROUP 13</span>
</div>
""", unsafe_allow_html=True)


# ── Sidebar: Patient Input ───────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<div class="sidebar-section-title">{logo_img(20)}Patient Profile</div>',
        unsafe_allow_html=True,
    )

    input_method = st.radio(
        "Input method:",
        ["Load Sample Patient", "Enter Manually"],
        horizontal=True,
        label_visibility="collapsed",
    )

    if input_method == "Load Sample Patient":
        data_dir = os.path.join(os.path.dirname(__file__), "data")
        profiles = {}
        if os.path.exists(data_dir):
            for f in sorted(os.listdir(data_dir)):
                if f.endswith(".json"):
                    with open(os.path.join(data_dir, f)) as fh:
                        p = json.load(fh)
                        label = f"{p.get('name', f)} — {p.get('diagnosis', 'Unknown')}"
                        profiles[label] = p

        if profiles:
            selected = st.selectbox("Select patient:", list(profiles.keys()))
            patient = profiles[selected]

            # Premium patient card (replaces raw st.json dump)
            def _row(k, v):
                if v in (None, "", [], {}):
                    return ""
                if isinstance(v, list):
                    v = ", ".join(str(x) for x in v)
                return (
                    f'<div class="prow"><span class="pkey">{k}</span>'
                    f'<span class="pval">{v}</span></div>'
                )

            rows = "".join([
                _row("Age", patient.get("age")),
                _row("Gender", patient.get("gender")),
                _row("Stage", patient.get("stage")),
                _row("Prior Tx", patient.get("prior_treatments")),
                _row("Medications", patient.get("medications")),
                _row("Comorbidities", patient.get("comorbidities")),
                _row("Lab Values", patient.get("lab_values")),
                _row("Location", patient.get("location")),
            ])
            st.markdown(f"""
            <div class="patient-card">
              <div class="pname">{patient.get('name', 'Unnamed Patient')}</div>
              <div class="pdiag">{patient.get('diagnosis', 'Unknown diagnosis')}</div>
              {rows}
            </div>
            """, unsafe_allow_html=True)

            with st.expander("View raw JSON"):
                st.json(patient)
        else:
            st.warning("No patient profiles found in data/ folder.")
            patient = {}
    else:
        patient = {
            "name": st.text_input("Name", ""),
            "age": st.number_input("Age", 18, 100, 50),
            "gender": st.selectbox("Gender", ["Female", "Male", "Other"]),
            "diagnosis": st.text_input("Diagnosis", placeholder="e.g., Non-small cell lung cancer"),
            "stage": st.text_input("Stage", placeholder="e.g., Stage IIIB"),
            "prior_treatments": st.text_area("Prior Treatments", placeholder="e.g., Carboplatin + Pemetrexed (4 cycles)"),
            "medications": st.text_input("Current Medications", placeholder="e.g., Tamoxifen 20mg daily"),
            "comorbidities": st.text_input("Comorbidities", placeholder="e.g., Type 2 diabetes, Hypertension"),
            "lab_values": st.text_input("Lab Values", placeholder="e.g., eGFR: 72, HER2: positive"),
            "location": st.text_input("Location", placeholder="e.g., Houston, TX"),
            "notes": st.text_area("Additional Notes", placeholder="Any other relevant clinical information"),
        }
        patient["location_coords"] = [None, None]

    st.divider()
    st.markdown(
        f'<div class="sidebar-section-title">{logo_img(20)}Ranking Weights</div>',
        unsafe_allow_html=True,
    )
    st.caption("Adjust priorities — values auto-normalize to 100%")

    w_elig     = st.slider("Eligibility Match",      0.0, 1.0, 0.35, 0.05)
    w_dist     = st.slider("Geographic Proximity",   0.0, 1.0, 0.20, 0.05)
    w_phase    = st.slider("Trial Phase",            0.0, 1.0, 0.15, 0.05)
    w_urgency  = st.slider("Recruitment Urgency",    0.0, 1.0, 0.15, 0.05)
    w_evidence = st.slider("Evidence Strength",      0.0, 1.0, 0.15, 0.05)

    weights = {
        "eligibility": w_elig, "distance": w_dist, "phase": w_phase,
        "urgency": w_urgency,  "evidence": w_evidence,
    }
    total_w = sum(weights.values())
    if total_w > 0:
        weights = {k: v / total_w for k, v in weights.items()}

    st.divider()
    run_btn = st.button("Find Matching Trials", type="primary", use_container_width=True)


# ── Pipeline step indicator helper ───────────────────────────────────
def render_steps(current: int, total: int = 5):
    """Render a 5-step pipeline progress strip."""
    labels = ["Search", "Eligibility", "Evidence", "Rank", "Done"]
    parts = []
    for i, label in enumerate(labels[:total]):
        cls = "done" if i < current else ("active" if i == current else "")
        parts.append(
            f'<div class="step {cls}">'
            f'<div class="step-circle">{i+1}</div>'
            f'<div class="step-label">{label}</div>'
            f'</div>'
        )
        if i < total - 1:
            line_cls = "done" if i < current else ""
            parts.append(f'<div class="step-line {line_cls}"></div>')
    return f'<div class="step-row">{"".join(parts)}</div>'


# ── Main Area ────────────────────────────────────────────────────────
if run_btn:
    if not patient.get("diagnosis"):
        st.error("Please enter a diagnosis or select a sample patient.")
        st.stop()

    if not config.OPENAI_API_KEY:
        st.error("OPENAI_API_KEY not set. Add it to your .env file or environment.")
        st.stop()

    patient_coords = tuple(patient.get("location_coords", [None, None]))
    if patient_coords == (None, None):
        patient_coords = None

    step_holder = st.empty()

    # Step 1: Retrieve
    step_holder.markdown(render_steps(0), unsafe_allow_html=True)
    with st.status("Searching ClinicalTrials.gov...", expanded=True) as status:
        st.write("Extracting disease information from patient profile...")
        trials, disease_info = retrieve_trials(patient)
        st.write(f"Found **{len(trials)}** recruiting trials for: `{disease_info.get('condition_keywords', '')}`")

        if not trials:
            status.update(label="No trials found", state="error")
            step_holder.markdown(render_steps(0), unsafe_allow_html=True)
            st.warning("No recruiting trials found for this condition. Try broadening the diagnosis.")
            st.stop()

        # Step 2: Eligibility
        step_holder.markdown(render_steps(1), unsafe_allow_html=True)
        st.write("Evaluating eligibility criteria with AI (topic-wise RAG chunking)...")
        trials = filter_trials(patient, trials)
        eligible_count = sum(1 for t in trials if not t.get("eligibility", {}).get("_filtered_out"))
        st.write(f"**{eligible_count}** trials passed eligibility screening")

        # Step 3: Evidence — only for top-K eligible trials by score, in parallel.
        # This avoids burning Semantic Scholar + PubMed + GPT calls on UNLIKELY matches.
        step_holder.markdown(render_steps(2), unsafe_allow_html=True)
        condition = disease_info.get("primary_condition", patient.get("diagnosis", ""))

        eligible_trials = [t for t in trials if not t.get("eligibility", {}).get("_filtered_out")]
        eligible_trials.sort(
            key=lambda t: t.get("eligibility", {}).get("overall_score", 0.0),
            reverse=True,
        )
        top_k = getattr(config, "EVIDENCE_TOP_K", 15)
        evidence_targets = eligible_trials[:top_k]
        st.write(
            f"Checking drug evidence (Semantic Scholar + PubMed) "
            f"on top **{len(evidence_targets)}** eligible trials in parallel..."
        )

        ev_workers = max(1, min(getattr(config, "EVIDENCE_WORKERS", 8), len(evidence_targets) or 1))
        if evidence_targets:
            with ThreadPoolExecutor(max_workers=ev_workers) as ex:
                fut_map = {ex.submit(get_evidence, t, condition): t for t in evidence_targets}
                for fut in as_completed(fut_map):
                    trial = fut_map[fut]
                    try:
                        trial["evidence"] = fut.result()
                    except Exception as e:
                        trial["evidence"] = {
                            "drug_name": "Unknown",
                            "evidence_score": 0.0,
                            "summary": f"Evidence lookup failed: {e}",
                            "semantic_scholar": {"available": False, "reason": str(e)},
                            "pubmed": {"available": False, "reason": str(e)},
                        }
        # Trials beyond top-K get a neutral placeholder so ranking still works
        for t in eligible_trials[top_k:]:
            t.setdefault("evidence", {"evidence_score": 0.5, "drug_name": "—",
                                      "summary": "Skipped — outside top-K for evidence lookup.",
                                      "semantic_scholar": {"available": False, "reason": "skipped"},
                                      "pubmed": {"available": False, "reason": "skipped"}})
        st.write("Evidence retrieved.")

        # Step 4: Rank
        step_holder.markdown(render_steps(3), unsafe_allow_html=True)
        st.write("Ranking trials across 5 objectives...")
        trials = rank_trials(trials, weights, patient_coords)
        st.write("Ranking complete.")

        # Step 5: Done
        step_holder.markdown(render_steps(5), unsafe_allow_html=True)
        status.update(label=f"Found {eligible_count} matching trials", state="complete")

    # ── Results Display ──────────────────────────────────────────────
    st.divider()
    st.markdown(
        f'<h3 style="margin-bottom:6px;">{logo_img(26)}Top Matching Trials '
        f'<span style="color:#64748B; font-weight:400; font-size:0.9rem;">'
        f'({eligible_count} results)</span></h3>',
        unsafe_allow_html=True,
    )

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Trials Searched", len(trials))
    col2.metric("Eligible", eligible_count)
    col3.metric("Excluded", len(trials) - eligible_count)
    best_score = trials[0]["final_score"] if trials else 0
    col4.metric("Best Match Score", f"{best_score:.0%}")

    st.divider()

    for trial in trials:
        if trial.get("eligibility", {}).get("_filtered_out"):
            continue

        score = trial.get("final_score", 0)
        rec = trial.get("eligibility", {}).get("recommendation", "UNKNOWN")

        if rec == "STRONG_MATCH":
            accent_cls, rec_cls, rec_label = "accent-strong", "rec-strong-pill", "STRONG MATCH"
        elif rec == "POSSIBLE_MATCH":
            accent_cls, rec_cls, rec_label = "accent-possible", "rec-possible-pill", "POSSIBLE MATCH"
        else:
            accent_cls, rec_cls, rec_label = "accent-unlikely", "rec-unlikely-pill", rec.replace("_", " ")

        nct_id  = trial.get("nct_id", "N/A")
        phase   = trial.get("phase", "N/A")
        sponsor = trial.get("sponsor", "Unknown sponsor")
        title   = trial.get("title", "Untitled")

        card_html = (
            f'<div class="trial-card">'
            f'<div class="accent {accent_cls}"></div>'
            f'<div class="trial-row">'
            f'<div class="trial-main">'
            f'<div class="chip-row">'
            f'<span class="chip chip-nct">{nct_id}</span>'
            f'<span class="chip chip-phase">{phase}</span>'
            f'<span class="chip chip-sponsor">{sponsor}</span>'
            f'<span class="rec-pill {rec_cls}">{rec_label}</span>'
            f'</div>'
            f'<div class="trial-title">{title}</div>'
            f'</div>'
            f'<div class="ring-wrap">'
            f'{score_ring_html(score, 76)}'
            f'<div class="ring-label">Match Score</div>'
            f'</div>'
            f'</div>'
            f'</div>'
        )
        st.markdown(card_html, unsafe_allow_html=True)

        with st.expander(f"Details — {nct_id}"):
            tab1, tab2, tab3, tab4 = st.tabs(["Scores", "Eligibility", "Evidence", "Locations"])

            with tab1:
                scores = trial.get("scores", {})
                score_cols = st.columns(5)
                labels = ["Eligibility", "Distance", "Phase", "Urgency", "Evidence"]
                keys = ["eligibility", "distance", "phase", "urgency", "evidence"]
                for col, label, key in zip(score_cols, labels, keys):
                    val = scores.get(key, 0)
                    col.metric(label, f"{val:.0%}", delta=f"weight {weights.get(key, 0):.0%}")

            with tab2:
                elig = trial.get("eligibility", {})
                st.markdown(
                    f"**Summary:** {elig.get('summary', 'N/A')}  \n"
                    f"**Overall Score:** {elig.get('overall_score', 0):.0%}"
                )

                topic_scores = elig.get("topic_scores", {})
                if topic_scores:
                    st.markdown("**Topic-Wise Scores** _(criteria chunked by medical topic via RAG)_:")
                    bars_html = ""
                    for topic, tscore in topic_scores.items():
                        pct = max(0.0, min(1.0, tscore))
                        bars_html += f"""
                        <div class="topic-row">
                          <div class="topic-name">{topic.replace('_', ' ').title()}</div>
                          <div class="topic-bar-bg">
                            <div class="topic-bar-fill" style="width:{pct*100:.0f}%;"></div>
                          </div>
                          <div class="topic-pct">{pct*100:.0f}%</div>
                        </div>
                        """
                    st.markdown(bars_html, unsafe_allow_html=True)

                st.divider()

                inc = elig.get("inclusion_results", [])
                if inc:
                    st.markdown("**Inclusion Criteria:**")
                    for cr in inc:
                        if cr.get("met") is True:
                            pill = '<span class="pill pill-pass">PASS</span>'
                        elif cr.get("met") is False:
                            pill = '<span class="pill pill-fail">FAIL</span>'
                        else:
                            pill = '<span class="pill pill-unknown">UNKNOWN</span>'
                        topic_tag = (
                            f' <span class="chip" style="font-size:0.68rem;">{cr.get("topic","")}</span>'
                            if cr.get("topic") else ""
                        )
                        st.markdown(
                            f'{pill} {cr.get("criterion","")} '
                            f'<span style="color:#64748B; font-size:0.85rem;">— {cr.get("reason","")}</span>'
                            f'{topic_tag}',
                            unsafe_allow_html=True,
                        )

                exc = elig.get("exclusion_results", [])
                if exc:
                    st.markdown("**Exclusion Criteria:**")
                    for cr in exc:
                        if cr.get("excluded") is True:
                            pill = '<span class="pill pill-excludes">EXCLUDES</span>'
                        elif cr.get("excluded") is False:
                            pill = '<span class="pill pill-ok">OK</span>'
                        else:
                            pill = '<span class="pill pill-unknown">UNKNOWN</span>'
                        topic_tag = (
                            f' <span class="chip" style="font-size:0.68rem;">{cr.get("topic","")}</span>'
                            if cr.get("topic") else ""
                        )
                        st.markdown(
                            f'{pill} {cr.get("criterion","")} '
                            f'<span style="color:#64748B; font-size:0.85rem;">— {cr.get("reason","")}</span>'
                            f'{topic_tag}',
                            unsafe_allow_html=True,
                        )

            with tab3:
                ev = trial.get("evidence", {})
                c1, c2 = st.columns([2, 1])
                c1.markdown(
                    f"**Drug:** {ev.get('drug_name', 'Unknown')}  \n"
                    f"**Evidence Score:** {ev.get('evidence_score', 0):.0%}"
                )
                c2.markdown(score_ring_html(ev.get("evidence_score", 0), 70), unsafe_allow_html=True)

                summary = ev.get("summary", "")
                if summary:
                    st.info(summary)

                ss = ev.get("semantic_scholar", {})
                if ss.get("available"):
                    st.markdown(f"**Semantic Scholar Papers ({ss.get('count', 0)}):**")
                    for paper in ss.get("papers", []):
                        cite_badge = (
                            f' <span class="chip" style="background:#FEF3C7;color:#92400E;border-color:#FDE68A;">'
                            f'{paper.get("citations", 0)} citations</span>'
                            if paper.get("citations") else ""
                        )
                        year = f" ({paper.get('year')})" if paper.get("year") else ""
                        st.markdown(
                            f'- [{paper.get("title","")}]({paper.get("url","")}){year}{cite_badge}',
                            unsafe_allow_html=True,
                        )
                        if paper.get("authors"):
                            st.caption(f"Authors: {paper['authors']}")
                else:
                    st.warning(f"Semantic Scholar: {ss.get('reason', 'Not available')}")

                pub = ev.get("pubmed", {})
                if pub.get("available"):
                    st.markdown(f"**PubMed Papers ({pub.get('count', 0)}):**")
                    for paper in pub.get("papers", []):
                        st.markdown(f"- [{paper.get('title','')}]({paper.get('url','')})")
                        if paper.get("abstract"):
                            st.caption(paper["abstract"][:300] + "...")
                else:
                    st.warning(f"PubMed: {pub.get('reason', 'Not available')}")

            with tab4:
                locs = trial.get("locations", [])
                if locs:
                    for loc in locs[:10]:
                        parts = [loc.get("facility"), loc.get("city"), loc.get("state"), loc.get("country")]
                        addr = ", ".join(p for p in parts if p)
                        st.markdown(f"- {addr}")

                    map_data = [
                        {"lat": loc["lat"], "lon": loc["lon"]}
                        for loc in locs
                        if loc.get("lat") and loc.get("lon")
                    ]
                    if map_data:
                        import pandas as pd
                        st.map(pd.DataFrame(map_data))
                else:
                    st.info("No location data available for this trial.")

else:
    # Premium landing page
    st.markdown(f"""
    <div class="landing-wrap">
      <div class="landing-logo">{logo_img(96, 0)}</div>
      <div class="landing-title">Match patients to clinical trials in seconds</div>
      <div class="landing-sub">Powered by GPT-4.1 mini · Semantic Scholar · PubMed · ClinicalTrials.gov</div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    c1.markdown("""
    <div class="step-card">
      <div class="num">1</div>
      <h4>Enter a patient</h4>
      <p>Load a sample profile or enter clinical data manually in the sidebar — diagnosis, stage, prior treatments, comorbidities, labs, and location.</p>
    </div>
    """, unsafe_allow_html=True)
    c2.markdown("""
    <div class="step-card">
      <div class="num">2</div>
      <h4>Tune priorities</h4>
      <p>Adjust the five ranking weights — eligibility, distance, phase, urgency, and evidence — to reflect what matters for this patient.</p>
    </div>
    """, unsafe_allow_html=True)
    c3.markdown("""
    <div class="step-card">
      <div class="num">3</div>
      <h4>Get ranked matches</h4>
      <p>The pipeline searches ClinicalTrials.gov, applies topic-wise RAG eligibility, verifies drug evidence, and returns ranked trials with explainable scores.</p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.info("Start by selecting a patient profile in the sidebar, then click **Find Matching Trials**.")
