"""
OpenAlex Scientific Paper Discovery Client.
Query academic literature using the free OpenAlex REST API.
"""

from typing import List, Dict, Any, Optional
import requests
from neuroread_rag.discovery.models import PaperMetadata


def reconstruct_openalex_abstract(inverted_index: Optional[Dict[str, List[int]]]) -> str:
    """Reconstruct continuous text from OpenAlex's abstract_inverted_index format."""
    if not inverted_index or not isinstance(inverted_index, dict):
        return ""
    words_by_pos = {}
    for word, positions in inverted_index.items():
        for pos in positions:
            words_by_pos[pos] = word
    sorted_positions = sorted(words_by_pos.keys())
    return " ".join(words_by_pos[p] for p in sorted_positions)


class OpenAlexClient:
    """Discovers peer-reviewed literature using OpenAlex API."""

    BASE_URL = "https://api.openalex.org/works"

    def __init__(self, timeout: int = 8, user_email: str = "neuroread.research@splab.edu"):
        self.timeout = timeout
        self.headers = {
            "User-Agent": f"NeuroRead-RAG/1.0 (mailto:{user_email})"
        }

    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        """Search OpenAlex works for papers relevant to a query."""
        params = {
            "search": query,
            "per-page": limit,
            "filter": "has_doi:true,type:article|book-chapter"
        }
        papers: List[PaperMetadata] = []
        try:
            resp = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
            for item in data.get("results", []):
                doi = str(item.get("doi") or "").replace("https://doi.org/", "").strip()
                if not doi:
                    continue

                title = item.get("display_name") or item.get("title") or "Untitled"
                year = int(item.get("publication_year") or 2020)

                authors = []
                for auth in item.get("authorships", []):
                    name = auth.get("author", {}).get("display_name")
                    if name:
                        authors.append(name)

                venue = item.get("primary_location", {}).get("source", {}).get("display_name") or "Academic Source"
                abstract = reconstruct_openalex_abstract(item.get("abstract_inverted_index"))
                url = item.get("doi") or item.get("id") or ""
                pdf_url = item.get("open_access", {}).get("oa_url")
                cited_by = int(item.get("cited_by_count") or 0)
                is_oa = bool(item.get("open_access", {}).get("is_oa", False))

                papers.append(PaperMetadata(
                    paper_id=f"openalex_{item.get('id', '').split('/')[-1]}",
                    title=title,
                    authors=authors[:5],
                    year=year,
                    venue=venue,
                    doi=doi,
                    abstract=abstract,
                    url=url,
                    pdf_url=pdf_url,
                    citation_count=cited_by,
                    sources=["openalex"],
                    is_open_access=is_oa
                ))
        except Exception:
            # Resilient against network timeouts or sandboxed network boundaries
            pass
        return papers
