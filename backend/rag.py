"""
Everything RAG in one module: file loaders, markdown-aware chunking, the
MongoDB Atlas Vector Search backend, the hybrid (BM25 + dense) retriever,
and the ingest / add-doc CLI utilities.

Formerly split across rag/loaders.py, rag/chunking.py, rag/mongo_store.py,
rag/retriever.py, rag/ingest.py, rag/add_doc.py, rag/__init__.py - merged
into one file. Sections are kept clearly separated below.

Public interface used elsewhere in the codebase: `TaxRAGRetriever` (see
graph.py). Everything else here is internal plumbing or CLI-only.

CLI usage (unchanged behavior, now under one file):
    python -m rag ingest                     # build dense index, config.RAG_DENSE_BACKEND
    python -m rag ingest --backend mongodb   # push chunks + embeddings into Atlas
    python -m rag ingest --backend faiss     # build/save a local FAISS index
    python -m rag add-doc /path/to/file.pdf [--country US] [--name my_notes]
"""

import glob
import logging
import os
import re
from typing import Dict, List, Optional

import config

logger = logging.getLogger(__name__)


# ===========================================================================
# SECTION 1: File loaders (formerly rag/loaders.py)
# Extracts plain text from a source file of any supported format, so it
# can be written as a .md doc into data/tax_docs/ - from that point on
# it's indistinguishable from the built-in docs to the retriever.
# Supported: .md, .txt, .pdf, .docx
# ===========================================================================

def extract_text(file_path: str) -> str:
    ext = os.path.splitext(file_path)[1].lower()

    if ext in (".md", ".txt"):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()

    if ext == ".pdf":
        return _extract_pdf(file_path)

    if ext == ".docx":
        return _extract_docx(file_path)

    raise ValueError(f"Unsupported file type: {ext}. Supported: .md, .txt, .pdf, .docx")


def _extract_pdf(file_path: str) -> str:
    from pypdf import PdfReader

    reader = PdfReader(file_path)
    pages = [page.extract_text() or "" for page in reader.pages]
    return "\n\n".join(pages)


def _extract_docx(file_path: str) -> str:
    import docx

    doc = docx.Document(file_path)
    parts = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if not text:
            continue
        # preserve Word heading styles as markdown headers, so the chunker
        # below still gets section structure
        style = (para.style.name or "").lower()
        if style.startswith("heading 1") or style == "title":
            parts.append(f"# {text}")
        elif style.startswith("heading 2"):
            parts.append(f"## {text}")
        elif style.startswith("heading 3"):
            parts.append(f"### {text}")
        else:
            parts.append(text)
    return "\n\n".join(parts)


# ===========================================================================
# SECTION 2: Markdown-aware chunking (formerly rag/chunking.py)
# Splits on markdown headers first (keeps sections intact), prepends the
# header path to each chunk so it's self-contained, then sub-splits long
# sections on sentence boundaries with overlap.
# ===========================================================================

def _split_sentences(text: str) -> List[str]:
    # good enough sentence splitter for our doc style (no exotic abbreviations)
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z(])", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _split_by_headers(text: str):
    """Returns list of (header_path, body_text) preserving header hierarchy."""
    lines = text.split("\n")
    sections = []
    header_stack = []  # list of (level, title)
    current_body = []

    def flush():
        body = "\n".join(current_body).strip()
        if body:
            path = " > ".join(h[1] for h in header_stack)
            sections.append((path, body))

    for line in lines:
        m = re.match(r"^(#{1,3})\s+(.*)", line)
        if m:
            flush()
            current_body.clear()
            level = len(m.group(1))
            title = m.group(2).strip()
            header_stack = [h for h in header_stack if h[0] < level]
            header_stack.append((level, title))
        else:
            current_body.append(line)
    flush()
    return sections


def chunk_markdown(text: str, source: str, max_chars: int = 600, overlap_sentences: int = 1) -> List[Dict]:
    """
    Returns list of {"text": <self-contained chunk incl. header path>,
    "source": source, "header_path": ...}
    """
    chunks = []
    for header_path, body in _split_by_headers(text):
        sentences = _split_sentences(body)
        if not sentences:
            continue

        buf, buf_len = [], 0
        for sent in sentences:
            if buf_len + len(sent) > max_chars and buf:
                chunks.append(_make_chunk(header_path, buf, source))
                buf = buf[-overlap_sentences:] if overlap_sentences else []
                buf_len = sum(len(s) for s in buf)
            buf.append(sent)
            buf_len += len(sent)
        if buf:
            chunks.append(_make_chunk(header_path, buf, source))

    return chunks


def _make_chunk(header_path: str, sentences: List[str], source: str) -> Dict:
    body = " ".join(sentences)
    prefixed = f"{header_path}: {body}" if header_path else body
    return {"text": prefixed, "raw_text": body, "source": source, "header_path": header_path}


def _load_and_chunk_docs() -> List[Dict]:
    """Shared by the retriever and the ingest CLI - one scan of data/tax_docs/*.md."""
    all_chunks = []
    for path in sorted(glob.glob(os.path.join(config.TAX_DOCS_DIR, "**", "*.md"), recursive=True)):
        with open(path, "r", encoding="utf-8") as f:
            text = f.read()
        source = os.path.basename(path)
        for chunk in chunk_markdown(text, source):
            chunk["country"] = "US" if source.startswith("us_") else None
            all_chunks.append(chunk)
    return all_chunks


# ===========================================================================
# SECTION 3: MongoDB Atlas Vector Search backend (formerly rag/mongo_store.py)
#
# Atlas setup: cluster agenticvectordb / db auth_app (same DB the app's
# backend already uses) / collection tax_rag_chunks / index vector_index.
# The index has no metadata filter fields, so country-filtering is done
# as a post-filter in Python (see vector_search() below), not inside the
# $vectorSearch stage.
#
# ONE-TIME SETUP IN ATLAS (before running ingest): Atlas UI -> cluster ->
# "Search" tab -> "Create Search Index" -> "JSON Editor" -> database
# `auth_app`, collection `tax_rag_chunks` -> paste:
#   {"fields": [{"type": "vector", "path": "embedding",
#                "numDimensions": 384, "similarity": "cosine"}]}
# Name it exactly "vector_index" (must match config.MONGODB_VECTOR_INDEX).
# ===========================================================================

class MongoVectorStore:
    def __init__(self):
        from pymongo import MongoClient

        if not config.MONGODB_URI:
            raise RuntimeError("MONGODB_URI not set - put it in your .env, see .env.example")

        # short server selection timeout so a bad URI/network fails fast
        # instead of hanging the whole graph
        self._client = MongoClient(config.MONGODB_URI, serverSelectionTimeoutMS=5000)
        self._client.admin.command("ping")  # fail fast if unreachable
        self._collection = self._client[config.MONGODB_DB_NAME][config.MONGODB_COLLECTION]

    def upsert_chunks(self, chunks: List[Dict], embeddings: List[List[float]]) -> int:
        """Upserts chunk+embedding docs, keyed so re-running ingest doesn't duplicate."""
        from pymongo import UpdateOne

        ops = []
        for chunk, emb in zip(chunks, embeddings):
            doc_id = f"{chunk['source']}::{chunk.get('header_path', '')}::{hash(chunk['text'])}"
            ops.append(UpdateOne(
                {"_id": doc_id},
                {"$set": {
                    "text": chunk["text"],
                    "raw_text": chunk.get("raw_text", chunk["text"]),
                    "source": chunk["source"],
                    "header_path": chunk.get("header_path", ""),
                    "country": chunk.get("country"),
                    "embedding": emb,
                }},
                upsert=True,
            ))
        if not ops:
            return 0
        result = self._collection.bulk_write(ops)
        return result.upserted_count + result.modified_count

    def vector_search(self, query_embedding: List[float], k: int = 5, country: Optional[str] = None) -> List[Dict]:
        """
        Runs $vectorSearch against `vector_index`. Since the index has no
        metadata/filter fields, country filtering is done as a
        post-filter after retrieval - over-fetch (limit=k*4) so filtering
        still leaves enough results.
        """
        fetch_limit = k * 4 if country else k
        pipeline = [
            {
                "$vectorSearch": {
                    "index": config.MONGODB_VECTOR_INDEX,
                    "path": "embedding",
                    "queryVector": query_embedding,
                    "numCandidates": max(fetch_limit * 10, 50),
                    "limit": fetch_limit,
                }
            },
            {
                "$project": {
                    "text": 1, "raw_text": 1, "source": 1, "header_path": 1,
                    "country": 1, "score": {"$meta": "vectorSearchScore"},
                }
            },
        ]
        try:
            results = list(self._collection.aggregate(pipeline))
        except Exception as e:
            logger.warning("MongoDB vector search failed: %s", e)
            return []

        if country:
            results = [r for r in results if r.get("country") in (None, country)]
        return results[:k]


# ===========================================================================
# SECTION 4: Hybrid retriever (formerly rag/retriever.py)
#
# Backends (config.RAG_DENSE_BACKEND):
#   - "mongodb" (default) - MongoDB Atlas Vector Search, see Section 3.
#   - "faiss" - local FAISS index built by `python -m rag ingest --backend faiss`.
#   - "none" / "" - BM25 only, no dense retrieval.
#
# Fusion is done over chunk **content**, not list indices, because dense
# results may come from MongoDB and not exist in the locally-loaded
# data/tax_docs chunks at all. BM25 always runs locally against
# data/tax_docs.
# ===========================================================================

class TaxRAGRetriever:
    def __init__(self):
        self._chunks: List[Dict] = []
        self._bm25 = None
        self._embedder = None          # shared embedding model, lazy-loaded
        self._mongo_store = None
        self._faiss_store = None

        self._load_and_chunk_docs()
        self._build_bm25()
        self._init_dense_backend()

    # ------------------------------------------------------------------
    # Setup
    # ------------------------------------------------------------------

    def _load_and_chunk_docs(self):
        self._chunks = _load_and_chunk_docs()

    def _build_bm25(self):
        from rank_bm25 import BM25Okapi
        tokenized = [self._tokenize(c["text"]) for c in self._chunks]
        self._bm25 = BM25Okapi(tokenized) if tokenized else None

    def _init_dense_backend(self):
        backend = (config.RAG_DENSE_BACKEND or "none").lower()
        if backend == "mongodb":
            try:
                self._mongo_store = MongoVectorStore()
                logger.info("Connected to MongoDB Atlas Vector Search (%s.%s)",
                            config.MONGODB_DB_NAME, config.MONGODB_COLLECTION)
            except Exception as e:
                logger.warning("MongoDB dense backend unavailable, falling back to BM25-only: %s", e)
                self._mongo_store = None
        elif backend == "faiss":
            try:
                from langchain_community.vectorstores import FAISS
                if not os.path.isdir(config.VECTOR_STORE_PATH):
                    raise FileNotFoundError("FAISS index not built - run: python -m rag ingest --backend faiss")
                self._faiss_store = FAISS.load_local(
                    config.VECTOR_STORE_PATH, self._get_embedder(), allow_dangerous_deserialization=True
                )
            except Exception as e:
                logger.warning("FAISS dense backend unavailable, falling back to BM25-only: %s", e)
                self._faiss_store = None
        # "none" / anything else -> BM25 only, nothing to init

    def _get_embedder(self):
        if self._embedder is None:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            self._embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
        return self._embedder

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"[a-z0-9]+", text.lower())

    # ------------------------------------------------------------------
    # Query
    # ------------------------------------------------------------------

    def query(self, text: str, k: int = None, country: Optional[str] = None) -> List[Dict]:
        k = k or config.RAG_TOP_K

        bm25_hits = self._bm25_search(text, country)
        dense_hits = self._dense_search(text, k=max(k, 10), country=country)

        return self._reciprocal_rank_fusion([bm25_hits, dense_hits], k=k)

    def _bm25_search(self, query: str, country: Optional[str], top_n: int = 10) -> List[Dict]:
        if not self._bm25:
            return []
        candidate_idx = [
            i for i, c in enumerate(self._chunks) if not country or c["country"] in (None, country)
        ]
        if not candidate_idx:
            return []
        scores = self._bm25.get_scores(self._tokenize(query))
        ranked = sorted(candidate_idx, key=lambda i: scores[i], reverse=True)
        return [self._chunks[i] for i in ranked if scores[i] > 0][:top_n]

    def _dense_search(self, query: str, k: int, country: Optional[str]) -> List[Dict]:
        if self._mongo_store is not None:
            try:
                query_embedding = self._get_embedder().embed_query(query)
                return self._mongo_store.vector_search(query_embedding, k=k, country=country)
            except Exception as e:
                logger.warning("MongoDB vector search failed, continuing with BM25 only: %s", e)
                return []

        if self._faiss_store is not None:
            try:
                results = self._faiss_store.similarity_search(query, k=k)
                return [
                    {
                        "text": r.page_content,
                        "source": r.metadata.get("source", ""),
                        "header_path": r.metadata.get("header_path", ""),
                    }
                    for r in results
                    if not country or r.metadata.get("country") in (None, country)
                ]
            except Exception as e:
                logger.warning("FAISS search failed, continuing with BM25 only: %s", e)
                return []

        return []

    @staticmethod
    def _reciprocal_rank_fusion(rankings: List[List[Dict]], k: int, rrf_k: int = 60) -> List[Dict]:
        """
        RRF over chunk content (source + text), not list position, so it
        works whether a hit came from local BM25 chunks or Mongo/FAISS
        results that may not overlap 1:1 with the local chunk list.
        """
        scores: Dict[str, float] = {}
        chunk_by_key: Dict[str, Dict] = {}

        for ranking in rankings:
            for rank, chunk in enumerate(ranking):
                key = f"{chunk.get('source', '')}::{chunk['text'][:120]}"
                scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
                chunk_by_key.setdefault(key, chunk)

        ranked_keys = sorted(scores.keys(), key=lambda key: scores[key], reverse=True)
        return [chunk_by_key[key] for key in ranked_keys[:k]]


# ===========================================================================
# SECTION 5: Ingest + add-doc CLI utilities
# (formerly rag/ingest.py + rag/add_doc.py)
#
# Drop new .md files (or run `python -m rag add-doc your_file.pdf` first)
# into data/tax_docs and re-run `python -m rag ingest` - no other code
# changes needed. Both build_* functions use the SAME chunker as the
# retriever's BM25 side, so chunk boundaries match everywhere.
# ===========================================================================

def build_mongo_index():
    from langchain_community.embeddings import HuggingFaceEmbeddings

    chunks = _load_and_chunk_docs()
    if not chunks:
        raise RuntimeError(f"No .md docs found in {config.TAX_DOCS_DIR}")

    print(f"Embedding {len(chunks)} chunks with {config.EMBEDDING_MODEL}...")
    embedder = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    embeddings = embedder.embed_documents([c["text"] for c in chunks])

    print(f"Upserting into MongoDB {config.MONGODB_DB_NAME}.{config.MONGODB_COLLECTION} "
          f"(index: {config.MONGODB_VECTOR_INDEX})...")
    store = MongoVectorStore()
    count = store.upsert_chunks(chunks, embeddings)
    print(f"Upserted {count} documents.")
    print("Reminder: the Atlas Search index must exist before queries work - "
          "see the setup notes in Section 3 of this file.")


def build_faiss_index():
    from langchain_community.vectorstores import FAISS
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain_core.documents import Document

    chunks = _load_and_chunk_docs()
    if not chunks:
        raise RuntimeError(f"No .md docs found in {config.TAX_DOCS_DIR}")

    documents = [
        Document(page_content=c["text"], metadata={"source": c["source"], "header_path": c["header_path"]})
        for c in chunks
    ]
    embeddings = HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)
    store = FAISS.from_documents(documents, embeddings)
    os.makedirs(config.VECTOR_STORE_PATH, exist_ok=True)
    store.save_local(config.VECTOR_STORE_PATH)
    print(f"Indexed {len(documents)} chunks -> {config.VECTOR_STORE_PATH}")


def _slugify(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return slug or "custom_doc"


def add_doc(source_path: str, country: str = None, name: str = None):
    """
    Pre-ingest a single custom file (notes, a tax-rule doc, whatever) into
    the RAG corpus: extracts text, writes it as .md into data/tax_docs/,
    and (if country="US") prefixes the filename with "us_" - the existing
    country-inference convention used by the retriever above.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(source_path)

    text = extract_text(source_path)
    if not text.strip():
        raise ValueError(f"No extractable text found in {source_path}")

    base_name = name or os.path.splitext(os.path.basename(source_path))[0]
    slug = _slugify(base_name)
    prefix = "us_" if country == "US" else ""
    dest_filename = f"{prefix}{slug}.md"
    dest_path = os.path.join(config.TAX_DOCS_DIR, dest_filename)

    os.makedirs(config.TAX_DOCS_DIR, exist_ok=True)
    if os.path.exists(dest_path):
        raise FileExistsError(
            f"{dest_path} already exists - pass --name to use a different filename, "
            f"or delete the existing file first if you're intentionally replacing it."
        )

    with open(dest_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(f"Ingested {source_path}")
    print(f"  -> {dest_path}  ({len(text)} chars, country={country or 'any'})")
    print("BM25 will pick this up automatically on the next TaxRAGRetriever() init.")
    print("If you're also using the FAISS dense index, re-run: python -m rag ingest --backend faiss")
    return dest_path


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAG corpus utilities: build the dense index, or add a doc.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    ingest_parser = subparsers.add_parser("ingest", help="Build the dense RAG index.")
    ingest_parser.add_argument("--backend", choices=["mongodb", "faiss"], default=None,
                                help="Defaults to config.RAG_DENSE_BACKEND (currently: %s)" % config.RAG_DENSE_BACKEND)

    add_doc_parser = subparsers.add_parser("add-doc", help="Pre-ingest a custom file into the tax RAG corpus.")
    add_doc_parser.add_argument("source_path", help="Path to the file to ingest (.pdf, .docx, .txt, .md)")
    add_doc_parser.add_argument("--country", choices=["IN", "US"], default=None,
                                 help="Tag this doc as country-specific (omit for generic/applies-to-any)")
    add_doc_parser.add_argument("--name", default=None, help="Override the output filename (without extension)")

    args = parser.parse_args()

    if args.command == "ingest":
        backend = args.backend or config.RAG_DENSE_BACKEND
        if backend == "mongodb":
            build_mongo_index()
        elif backend == "faiss":
            build_faiss_index()
        else:
            print(f"Unknown or unset backend '{backend}'. Pass --backend mongodb or --backend faiss.")
    elif args.command == "add-doc":
        add_doc(args.source_path, country=args.country, name=args.name)
