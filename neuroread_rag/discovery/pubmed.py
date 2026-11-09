"""
PubMed and Europe PMC Scientific Discovery Clients.
Queries biomedical and neuroscience literature via NCBI E-Utilities and Europe PMC REST APIs.
"""

from typing import List, Dict, Any, Optional
import requests
import xml.etree.ElementTree as ET
from neuroread_rag.discovery.models import PaperMetadata


class EuropePMCClient:
    """Discovers biomedical literature using the Europe PMC REST API."""

    BASE_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.headers = {"User-Agent": "NeuroRead-RAG/1.0"}

    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        params = {
            "query": query,
            "format": "json",
            "pageSize": limit,
            "resultType": "core"
        }
        papers: List[PaperMetadata] = []
        try:
            resp = requests.get(self.BASE_URL, params=params, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
            result_list = data.get("resultList", {}).get("result", [])
            for item in result_list:
                doi = str(item.get("doi") or "").strip()
                title = item.get("title") or "Untitled"
                year = int(item.get("pubYear") or 2020)
                venue = item.get("journalTitle") or "Biomedical Literature"
                abstract = item.get("abstractText") or ""
                authors_str = item.get("authorString") or ""
                authors = [a.strip() for a in authors_str.split(",") if a.strip()]
                pmid = item.get("pmid") or ""
                citations = int(item.get("citedByCount") or 0)
                is_oa = item.get("isOpenAccess") == "Y"

                url = f"https://doi.org/{doi}" if doi else (f"https://europepmc.org/article/MED/{pmid}" if pmid else "")

                if not doi and not abstract:
                    continue

                papers.append(PaperMetadata(
                    paper_id=f"epmc_{pmid or doi.replace('/', '_')}",
                    title=title,
                    authors=authors[:5],
                    year=year,
                    venue=venue,
                    doi=doi,
                    abstract=abstract,
                    url=url,
                    citation_count=citations,
                    sources=["europe_pmc"],
                    is_open_access=is_oa
                ))
        except Exception:
            pass
        return papers


class PubMedClient:
    """Discovers biomedical literature using NCBI E-utilities."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

    def __init__(self, timeout: int = 8):
        self.timeout = timeout
        self.headers = {"User-Agent": "NeuroRead-RAG/1.0"}

    def search(self, query: str, limit: int = 5) -> List[PaperMetadata]:
        papers: List[PaperMetadata] = []
        try:
            # 1. Search for PMIDs
            search_params = {
                "db": "pubmed",
                "term": query,
                "retmax": limit,
                "retmode": "json"
            }
            resp = requests.get(self.ESEARCH_URL, params=search_params, headers=self.headers, timeout=self.timeout)
            if resp.status_code != 200:
                return []
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # 2. Fetch summaries
            summary_params = {
                "db": "pubmed",
                "id": ",".join(id_list),
                "retmode": "json"
            }
            s_resp = requests.get(self.ESUMMARY_URL, params=summary_params, headers=self.headers, timeout=self.timeout)
            if s_resp.status_code != 200:
                return []
            s_data = s_resp.json()
            result_dict = s_data.get("result", {})

            for pmid in id_list:
                item = result_dict.get(pmid)
                if not item:
                    continue
                title = item.get("title") or "Untitled"
                pubdate = str(item.get("pubdate") or "2020")
                year_part = pubdate.split()[0]
                year = int(year_part) if year_part.isdigit() else 2020
                venue = item.get("fulljournalname") or item.get("source") or "PubMed"

                authors = [a.get("name") for a in item.get("authors", []) if a.get("name")]

                # Extract DOI from articleids
                doi = ""
                for aid in item.get("articleids", []):
                    if aid.get("idtype") == "doi":
                        doi = aid.get("value") or ""
                        break

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                papers.append(PaperMetadata(
                    paper_id=f"pmid_{pmid}",
                    title=title,
                    authors=authors[:5],
                    year=year,
                    venue=venue,
                    doi=doi,
                    abstract="",  # PubMed summary does not include full abstract; filled by EuropePMC or Crossref
                    url=url,
                    citation_count=0,
                    sources=["pubmed"],
                    is_open_access=True
                ))
        except Exception:
            pass
        return papers
