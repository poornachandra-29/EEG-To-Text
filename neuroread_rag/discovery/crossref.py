"""
Crossref API Client for DOI and Publication Metadata Verification.
Verifies citation accuracy, publisher metadata, and official DOIs.
"""

from typing import Optional, Dict, Any, List
import requests
from neuroread_rag.discovery.models import PaperMetadata


class CrossrefClient:
    """Verifies and fetches official publication metadata via Crossref REST API."""

    BASE_URL = "https://api.crossref.org/works"

    def __init__(self, timeout: int = 8, mailto: str = "neuroread.research@splab.edu"):
        self.timeout = timeout
        self.headers = {
            "User-Agent": f"NeuroRead-RAG/1.0 (mailto:{mailto})"
        }

    def verify_doi(self, doi: str) -> Optional[Dict[str, Any]]:
        """Verify a DOI and fetch its official metadata record."""
        clean_doi = doi.replace("https://doi.org/", "").replace("http://doi.org/", "").strip()
        url = f"{self.BASE_URL}/{clean_doi}"
        try:
            resp = requests.get(url, headers=self.headers, timeout=self.timeout)
            if resp.status_code == 200:
                message = resp.json().get("message", {})
                title = message.get("title", [""])[0] if message.get("title") else ""
                container = message.get("container-title", [""])[0] if message.get("container-title") else ""
                published = message.get("published-print") or message.get("published-online") or {}
                date_parts = published.get("date-parts", [[2020]])[0]
                year = date_parts[0] if date_parts else 2020

                authors = []
                for a in message.get("author", []):
                    given = a.get("given", "")
                    family = a.get("family", "")
                    authors.append(f"{given} {family}".strip())

                return {
                    "is_valid": True,
                    "doi": clean_doi,
                    "title": title,
                    "venue": container,
                    "year": year,
                    "authors": authors[:5],
                    "citation_count": message.get("is-referenced-by-count", 0),
                    "url": f"https://doi.org/{clean_doi}"
                }
        except Exception:
            pass
        return None

    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        """Search Crossref works by keywords."""
        params = {"query": query, "rows": limit}
        papers: List[PaperMetadata] = []
        try:
            resp = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            items = resp.json().get("message", {}).get("items", [])
            for item in items:
                doi = str(item.get("DOI") or "").strip()
                title_list = item.get("title") or []
                title = title_list[0] if title_list else "Untitled"
                container_list = item.get("container-title") or []
                venue = container_list[0] if container_list else "Crossref Publication"

                published = item.get("published-print") or item.get("published-online") or {}
                date_parts = published.get("date-parts", [[2020]])[0]
                year = date_parts[0] if date_parts else 2020

                authors = []
                for a in item.get("author", []):
                    authors.append(f"{a.get('given', '')} {a.get('family', '')}".strip())

                abstract = item.get("abstract") or ""

                papers.append(PaperMetadata(
                    paper_id=f"crossref_{doi.replace('/', '_')}",
                    title=title,
                    authors=authors[:5],
                    year=year,
                    venue=venue,
                    doi=doi,
                    abstract=abstract,
                    url=f"https://doi.org/{doi}" if doi else "",
                    citation_count=item.get("is-referenced-by-count", 0),
                    sources=["crossref"],
                    is_open_access=True
                ))
        except Exception:
            pass
        return papers
