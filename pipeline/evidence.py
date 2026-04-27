"""
Step 4: Evidence Verification (ALL FREE — no paid APIs)
- Primary: Semantic Scholar API — free, no API key, academic paper search with citations
- Secondary: PubMed via Biopython — free, 30M+ peer-reviewed papers
- Summary: OpenAI (already used for eligibility) generates an evidence summary
"""

import requests
from Bio import Entrez
from openai import OpenAI
import config

Entrez.email = config.PUBMED_EMAIL
client = OpenAI(api_key=config.OPENAI_API_KEY)


def get_semantic_scholar_evidence(drug_name: str, condition: str, max_results: int = 5) -> dict:
    """
    Query Semantic Scholar API for academic papers on drug + condition.
    100% FREE — no API key needed.
    Returns titles, abstracts, citation counts, URLs.
    """
    if not drug_name:
        return {"available": False, "papers": [], "reason": "No drug name"}

    query = f"{drug_name} {condition} clinical trial"
    params = {
        "query": query,
        "limit": max_results,
        "fields": "title,abstract,citationCount,year,url,externalIds,authors",
    }

    try:
        resp = requests.get(
            "https://api.semanticscholar.org/graph/v1/paper/search",
            params=params,
            timeout=15,
        )
        resp.raise_for_status()
        data = resp.json()

        papers = []
        total_citations = 0
        for paper in data.get("data", []):
            abstract = paper.get("abstract", "") or ""
            cite_count = paper.get("citationCount", 0) or 0
            total_citations += cite_count

            # Build URL — prefer DOI link
            ext_ids = paper.get("externalIds", {}) or {}
            doi = ext_ids.get("DOI", "")
            url = f"https://doi.org/{doi}" if doi else paper.get("url", "")

            authors = paper.get("authors", []) or []
            author_str = ", ".join(a.get("name", "") for a in authors[:3])
            if len(authors) > 3:
                author_str += " et al."

            papers.append({
                "title": paper.get("title", ""),
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                "citations": cite_count,
                "year": paper.get("year"),
                "authors": author_str,
                "url": url,
            })

        return {
            "available": bool(papers),
            "papers": papers,
            "count": len(papers),
            "total_citations": total_citations,
            "source": "Semantic Scholar",
        }
    except Exception as e:
        return {"available": False, "papers": [], "reason": str(e)}


def get_pubmed_evidence(drug_name: str, condition: str, max_results: int = 5) -> dict:
    """Search PubMed for recent papers on the drug + condition. FREE."""
    if not drug_name:
        return {"available": False, "papers": [], "reason": "No drug name"}

    query = f"{drug_name} {condition} clinical trial"

    try:
        handle = Entrez.esearch(db="pubmed", term=query, retmax=max_results, sort="date")
        record = Entrez.read(handle)
        handle.close()

        ids = record.get("IdList", [])
        if not ids:
            return {"available": False, "papers": [], "reason": "No papers found"}

        handle = Entrez.efetch(db="pubmed", id=ids, rettype="xml", retmode="xml")
        articles_xml = Entrez.read(handle)
        handle.close()

        papers = []
        for article in articles_xml.get("PubmedArticle", []):
            medline = article.get("MedlineCitation", {})
            art = medline.get("Article", {})
            title = art.get("ArticleTitle", "")
            abstract_parts = art.get("Abstract", {}).get("AbstractText", [])
            abstract = " ".join(str(p) for p in abstract_parts) if abstract_parts else ""
            pmid = str(medline.get("PMID", ""))

            papers.append({
                "title": title,
                "abstract": abstract[:500] + "..." if len(abstract) > 500 else abstract,
                "pmid": pmid,
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            })

        return {
            "available": True,
            "papers": papers,
            "count": len(papers),
            "source": "PubMed",
        }
    except Exception as e:
        return {"available": False, "papers": [], "reason": str(e)}


def generate_evidence_summary(drug_name: str, condition: str, ss_papers: list, pm_papers: list) -> str:
    """Use OpenAI to generate a concise evidence summary from the papers found."""
    if not ss_papers and not pm_papers:
        return "No published evidence found for this drug-condition combination."

    paper_texts = []
    for p in (ss_papers + pm_papers)[:6]:
        entry = f"- {p.get('title', 'Untitled')}"
        if p.get("citations"):
            entry += f" ({p['citations']} citations)"
        if p.get("abstract"):
            entry += f"\n  Abstract: {p['abstract'][:300]}"
        paper_texts.append(entry)

    prompt = f"""Based on these research papers, write a 3-4 sentence evidence summary for {drug_name} in treating {condition}. Focus on: efficacy, safety, and current status. Be factual.

Papers:
{chr(10).join(paper_texts)}

Return ONLY the summary paragraph, nothing else."""

    try:
        resp = client.chat.completions.create(
            model=config.OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=200,
        )
        return resp.choices[0].message.content.strip()
    except Exception:
        return "Evidence summary could not be generated."


def get_evidence(trial: dict, condition: str) -> dict:
    """Get combined evidence from Semantic Scholar + PubMed for a trial's intervention."""
    # Extract the primary drug/intervention name
    interventions = trial.get("interventions", [])
    drug_name = ""
    for interv in interventions:
        if interv.get("type") in ("DRUG", "BIOLOGICAL", "COMBINATION_PRODUCT"):
            drug_name = interv.get("name", "")
            break
    if not drug_name and interventions:
        drug_name = interventions[0].get("name", "")

    semantic_scholar = get_semantic_scholar_evidence(drug_name, condition)
    pubmed = get_pubmed_evidence(drug_name, condition)

    # Generate AI summary from found papers
    all_papers = semantic_scholar.get("papers", []) + pubmed.get("papers", [])
    summary = generate_evidence_summary(drug_name, condition, semantic_scholar.get("papers", []), pubmed.get("papers", []))

    # Compute evidence strength score
    score = 0.0
    if semantic_scholar.get("available"):
        score += 0.3
        # More citations = stronger evidence
        total_cites = semantic_scholar.get("total_citations", 0)
        score += min(total_cites / 500, 0.3)  # up to 0.3 for high-citation papers
    if pubmed.get("available"):
        score += 0.2 + min(pubmed.get("count", 0) * 0.04, 0.2)

    return {
        "drug_name": drug_name,
        "summary": summary,
        "semantic_scholar": semantic_scholar,
        "pubmed": pubmed,
        "evidence_score": round(min(score, 1.0), 3),
    }
