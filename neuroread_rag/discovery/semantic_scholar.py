"""
Semantic Scholar Scientific Paper Discovery Client.
Query academic publications using Semantic Scholar Graph API.
"""

from typing import List, Optional
import requests
from neuroread_rag.discovery.models import PaperMetadata


class SemanticScholarClient:
    """Discovers peer-reviewed literature via Semantic Scholar Graph API."""

    BASE_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.headers = {
            "User-Agent": "NeuroRead-RAG/1.0 (Research Pipeline)"
        }

    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        params = {
            "query": query,
            "limit": limit,
            "fields": "title,abstract,authors,year,venue,externalIds,citationCount,isOpenAccess,openAccessPdf"
        }
        papers: List[PaperMetadata] = []
        try:
            resp = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
            for item in data.get("data", []):
                external_ids = item.get("externalIds") or {}
                doi = external_ids.get("DOI") or ""
                paper_id = item.get("paperId") or ""
                title = item.get("title") or "Untitled"
                year = int(item.get("year") or 2020)
                venue = item.get("venue") or "Academic Source"
                abstract = item.get("abstract") or ""

                authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]

                oa_pdf = item.get("openAccessPdf") or {}
                pdf_url = oa_pdf.get("url")
                is_oa = bool(item.get("isOpenAccess", False))
                citations = int(item.get("citationCount") or 0)
                url = f"https://www.semanticscholar.org/paper/{paper_id}" if paper_id else (f"https://doi.org/{doi}" if doi else "")

                if not doi and not abstract:
                    continue

                papers.append(PaperMetadata(
                    paper_id=f"s2_{paper_id[:16]}",
                    title=title,
                    authors=authors[:5],
                    year=year,
                    venue=venue,
                    doi=doi,
                    abstract=abstract,
                    url=url,
                    pdf_url=pdf_url,
                    citation_count=citations,
                    sources=["semantic_scholar"],
                    is_open_access=is_oa
                ))
        except Exception:
            pass
        return papers
