"""
Vector Store and Hybrid Index for Scientific Literature.
Implements dual BM25 keyword search and dense semantic similarity with SQLite metadata persistence.
"""

import math
import sqlite3
import json
import re
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import numpy as np

from neuroread_rag.discovery.models import ResearchChunk, PaperMetadata
from neuroread_rag.config import OUTPUT_DIR


class HybridVectorStore:
    """Persistent vector store combining BM25 keyword matching and dense semantic similarity."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or (OUTPUT_DIR / "scientific_literature.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

        # In-memory index structures
        self.chunks: List[ResearchChunk] = []
        self.doc_ids: List[str] = []
        self.corpus_terms: List[List[str]] = []
        self.idf: Dict[str, float] = {}
        self.doc_lens: List[int] = []
        self.avg_doc_len: float = 0.0
        self.dense_vectors: Optional[np.ndarray] = None
        self.vocab: Dict[str, int] = {}

        self.load_from_db()

    def _init_db(self) -> None:
        """Initialize SQLite tables for papers, chunks, and metadata."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS papers (
                    paper_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    authors TEXT,
                    year INTEGER,
                    venue TEXT,
                    doi TEXT,
                    abstract TEXT,
                    url TEXT,
                    citation_count INTEGER,
                    sources TEXT,
                    metrics_covered TEXT,
                    is_open_access INTEGER
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    chunk_id TEXT PRIMARY KEY,
                    paper_id TEXT,
                    paper_title TEXT,
                    doi TEXT,
                    year INTEGER,
                    authors_short TEXT,
                    section_name TEXT,
                    content TEXT,
                    metrics_tagged TEXT,
                    evidence_type TEXT,
                    FOREIGN KEY(paper_id) REFERENCES papers(paper_id)
                )
            """)
            conn.commit()

    @staticmethod
    def tokenize(text: str) -> List[str]:
        """Tokenize text into lowercase alpha-numeric terms."""
        return re.findall(r"\b[a-z0-9_]{2,}\b", text.lower())

    def index_papers(self, papers: List[PaperMetadata], chunks: List[ResearchChunk]) -> None:
        """Save papers and chunks to SQLite and build in-memory BM25 + dense index."""
        with sqlite3.connect(self.db_path) as conn:
            for p in papers:
                conn.execute("""
                    INSERT OR REPLACE INTO papers 
                    (paper_id, title, authors, year, venue, doi, abstract, url, citation_count, sources, metrics_covered, is_open_access)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    p.paper_id, p.title, json.dumps(p.authors), p.year, p.venue, p.doi,
                    p.abstract, p.url, p.citation_count, json.dumps(p.sources),
                    json.dumps(p.metrics_covered), 1 if p.is_open_access else 0
                ))

            for c in chunks:
                conn.execute("""
                    INSERT OR REPLACE INTO chunks
                    (chunk_id, paper_id, paper_title, doi, year, authors_short, section_name, content, metrics_tagged, evidence_type)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    c.chunk_id, c.paper_id, c.paper_title, c.doi, c.year, c.authors_short,
                    c.section_name, c.content, json.dumps(c.metrics_tagged), c.evidence_type
                ))
            conn.commit()

        self.load_from_db()

    def load_from_db(self) -> None:
        """Load stored chunks and construct the BM25 and vector indices."""
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM chunks")
            rows = cursor.fetchall()

        self.chunks = []
        self.corpus_terms = []
        self.doc_ids = []
        self.doc_lens = []

        for r in rows:
            chunk = ResearchChunk(
                chunk_id=r["chunk_id"],
                paper_id=r["paper_id"],
                paper_title=r["paper_title"],
                doi=r["doi"],
                year=r["year"],
                authors_short=r["authors_short"],
                section_name=r["section_name"],
                content=r["content"],
                metrics_tagged=json.loads(r["metrics_tagged"]) if r["metrics_tagged"] else [],
                evidence_type=r["evidence_type"]
            )
            self.chunks.append(chunk)
            self.doc_ids.append(chunk.chunk_id)

            tokens = self.tokenize(chunk.content + " " + " ".join(chunk.metrics_tagged) + " " + chunk.paper_title)
            self.corpus_terms.append(tokens)
            self.doc_lens.append(len(tokens))

        if not self.corpus_terms:
            return

        self.avg_doc_len = sum(self.doc_lens) / len(self.doc_lens)

        # Build BM25 IDF
        df_counts: Dict[str, int] = {}
        n_docs = len(self.corpus_terms)
        for doc in self.corpus_terms:
            unique_terms = set(doc)
            for t in unique_terms:
                df_counts[t] = df_counts.get(t, 0) + 1

        self.idf = {}
        for t, count in df_counts.items():
            self.idf[t] = math.log(1.0 + (n_docs - count + 0.5) / (count + 0.5))

        # Build Dense Semantic Vectors (Sublinear TF-IDF Normalized Embeddings)
        self.vocab = {term: idx for idx, term in enumerate(sorted(df_counts.keys()))}
        vocab_size = len(self.vocab)
        vectors = np.zeros((n_docs, vocab_size), dtype=np.float32)

        for i, tokens in enumerate(self.corpus_terms):
            tf_dict: Dict[int, float] = {}
            for t in tokens:
                if t in self.vocab:
                    col = self.vocab[t]
                    tf_dict[col] = tf_dict.get(col, 0.0) + 1.0
            for col, count in tf_dict.items():
                vectors[i, col] = (1.0 + math.log(count)) * self.idf.get(tokens[0], 1.0)
            # L2 normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 1e-9:
                vectors[i] /= norm

        self.dense_vectors = vectors

    def search_bm25(self, query: str, top_k: int = 5, metric_filter: Optional[str] = None) -> List[Tuple[ResearchChunk, float]]:
        """BM25 Okapi retrieval with optional metric filtering."""
        if not self.chunks:
            return []

        q_tokens = self.tokenize(query)
        if not q_tokens:
            return []

        k1 = 1.5
        b = 0.75
        scores = np.zeros(len(self.chunks), dtype=np.float32)

        for i, doc in enumerate(self.corpus_terms):
            # Apply metric filter if specified
            if metric_filter and metric_filter not in self.chunks[i].metrics_tagged:
                continue

            doc_len = self.doc_lens[i]
            doc_counts: Dict[str, int] = {}
            for t in doc:
                doc_counts[t] = doc_counts.get(t, 0) + 1

            for qt in q_tokens:
                if qt in doc_counts:
                    freq = doc_counts[qt]
                    term_idf = self.idf.get(qt, 0.0)
                    numerator = freq * (k1 + 1.0)
                    denominator = freq + k1 * (1.0 - b + b * (doc_len / (self.avg_doc_len + 1e-9)))
                    scores[i] += term_idf * (numerator / denominator)

        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if scores[idx] > 0.0:
                results.append((self.chunks[idx], float(scores[idx])))
        return results

    def search_dense(self, query: str, top_k: int = 5, metric_filter: Optional[str] = None) -> List[Tuple[ResearchChunk, float]]:
        """Dense semantic cosine similarity retrieval."""
        if self.dense_vectors is None or len(self.chunks) == 0:
            return []

        q_tokens = self.tokenize(query)
        q_vec = np.zeros((1, len(self.vocab)), dtype=np.float32)
        for t in q_tokens:
            if t in self.vocab:
                col = self.vocab[t]
                q_vec[0, col] += 1.0 * self.idf.get(t, 1.0)

        norm = np.linalg.norm(q_vec)
        if norm > 1e-9:
            q_vec /= norm
        else:
            return []

        sims = np.dot(self.dense_vectors, q_vec.T).flatten()

        # Filter if requested
        if metric_filter:
            for i, c in enumerate(self.chunks):
                if metric_filter not in c.metrics_tagged:
                    sims[i] = -1.0

        top_indices = np.argsort(sims)[::-1][:top_k]
        results = []
        for idx in top_indices:
            if sims[idx] > 0.0:
                results.append((self.chunks[idx], float(sims[idx])))
        return results

    def hybrid_search(self, query: str, top_k: int = 5, metric_filter: Optional[str] = None, alpha: float = 0.5) -> List[Tuple[ResearchChunk, float]]:
        """
        Hybrid retrieval combining BM25 and dense semantic search with reciprocal rank fusion (RRF).
        alpha: weight balance between dense (alpha) and BM25 (1 - alpha).
        """
        bm25_res = self.search_bm25(query, top_k=top_k * 2, metric_filter=metric_filter)
        dense_res = self.search_dense(query, top_k=top_k * 2, metric_filter=metric_filter)

        rrf_scores: Dict[str, float] = {}
        chunk_map: Dict[str, ResearchChunk] = {}
        c_k = 60.0  # standard RRF constant

        for rank, (chunk, _) in enumerate(bm25_res):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + (1.0 - alpha) / (c_k + rank + 1)

        for rank, (chunk, _) in enumerate(dense_res):
            cid = chunk.chunk_id
            chunk_map[cid] = chunk
            rrf_scores[cid] = rrf_scores.get(cid, 0.0) + alpha / (c_k + rank + 1)

        sorted_items = sorted(rrf_scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return [(chunk_map[cid], score) for cid, score in sorted_items]
