# Indexes data/local_corpus/ into Chroma so the researcher can pull from
# local docs, not just web search. Run with --index whenever the corpus
# changes, then query it via search_local_docs().
#
# TODO: this wipes and rebuilds the whole collection every time, which is
# fine at portfolio scale but wasteful once the corpus gets big - should
# diff and upsert instead.

from __future__ import annotations

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import os
os.environ["ANONYMIZED_TELEMETRY"] = "False"

import argparse

import chromadb
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

from src import config

_embedder: SentenceTransformer | None = None
_collection = None


def _get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
    return _embedder


def _get_collection():
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(config.CHROMA_PERSIST_DIR))
        _collection = client.get_or_create_collection("local_corpus")
    return _collection


def build_index() -> int:
    """Chunks every file in data/local_corpus/ and (re)builds the index."""
    config.LOCAL_CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    files = [
        p for p in config.LOCAL_CORPUS_DIR.rglob("*")
        if p.is_file() and p.suffix.lower() in {".txt", ".md"}
    ]
    if not files:
        print(f"No .txt/.md files found in {config.LOCAL_CORPUS_DIR}")
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE, chunk_overlap=config.CHUNK_OVERLAP
    )
    embedder = _get_embedder()
    collection = _get_collection()

    # just nuke and rebuild for now, see TODO above
    existing = collection.get()["ids"]
    if existing:
        collection.delete(ids=existing)

    ids, docs, metadatas = [], [], []
    for path in files:
        text = path.read_text(errors="ignore")
        for i, chunk in enumerate(splitter.split_text(text)):
            ids.append(f"{path.stem}-{i}")
            docs.append(chunk)
            metadatas.append({"source": str(path.relative_to(config.LOCAL_CORPUS_DIR))})

    embeddings = embedder.encode(docs).tolist()
    collection.add(ids=ids, documents=docs, metadatas=metadatas, embeddings=embeddings)
    print(f"Indexed {len(docs)} chunks from {len(files)} files.")
    return len(docs)


def search_local_docs(query: str, top_k: int = None) -> list[dict]:
    """Returns a list of {"content", "source"} dicts from the local corpus."""
    top_k = top_k or config.LOCAL_DOCS_TOP_K
    collection = _get_collection()
    if collection.count() == 0:
        return []

    query_embedding = _get_embedder().encode([query]).tolist()
    results = collection.query(query_embeddings=query_embedding, n_results=top_k)

    return [
        {"content": doc, "source": meta.get("source", "unknown")}
        for doc, meta in zip(results["documents"][0], results["metadatas"][0])
    ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", action="store_true", help="Build/refresh the local index")
    args = parser.parse_args()
    if args.index:
        build_index()
