import argparse
import collections
import json
import os
from pathlib import Path
from src.data.make_dataset import load_and_inspect_qasper

# Absolute path to project root — works whether this script is run as
# "python pipeline_ingest.py" from src/ or "python -m src.pipeline_ingest"
# from the project root.
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
from src.retrieval.chunking import QasperChunker
from src.retrieval.vector_store import DenseIndexer
from src.retrieval.sparse_store import SparseIndexer

_DEFAULT_TABLE_BODIES = _PROJECT_ROOT / "data" / "table_bodies.json"


def run_ingestion_pipeline(indices_dir: Path = _PROJECT_ROOT / "data" / "indices",
                           table_bodies_path: Path = _DEFAULT_TABLE_BODIES,
                           table_format: str = "markdown"):
    print("=== Data Ingestion & Indexing ===")
    # 1. Load Data
    raw_data = load_and_inspect_qasper()

    # We load the whole train  len(raw_data) = 1585 papers
    subset_data = [raw_data[i] for i in range(len(raw_data))]
    print(f"Processing subset of {len(subset_data)} papers.")

    # Table bodies extracted from arXiv LaTeX (python -m src.data.arxiv_tables).
    # Optional: without them, tables are indexed by caption only.
    table_bodies = {}
    if table_bodies_path and Path(table_bodies_path).exists():
        with open(table_bodies_path) as fh:
            table_bodies = json.load(fh)
        n_tables = sum(len(v) for v in table_bodies.values())
        print(f"Loaded {n_tables} table bodies for {len(table_bodies)} papers "
              f"from {table_bodies_path}.")
    else:
        print(f"No table bodies at {table_bodies_path} — tables indexed by caption only. "
              "Run `python -m src.data.arxiv_tables` to extract them.")

    # 2. Chunking
    print("\n--- Chunking ---")
    chunker = QasperChunker(table_format=table_format)
    print(f"Table format: {table_format}")
    all_chunks = []

    for paper in subset_data:
        paper_chunks = chunker.process_paper(paper, table_bodies=table_bodies.get(paper['id']))
        all_chunks.extend(paper_chunks)

    types = collections.Counter(c["chunk_type"] for c in all_chunks)
    print(f"Result: Generated {len(all_chunks)} chunks ({dict(types)}).")

    # 3. Dense Indexing (Vectors)
    # Dense and sparse indices are always rebuilt together: they share the
    # chunk ordering, and the retriever relies on that alignment.
    indices_dir = Path(indices_dir)
    print("\n--- Dense Indexing (FAISS) ---")
    dense_idx = DenseIndexer(index_path=str(indices_dir / "dense.index"))
    dense_idx.build_index(all_chunks)
    dense_idx.save()

    # 4. Sparse Indexing (BM25)
    print("\n--- Sparse Indexing (BM25) ---")
    sparse_idx = SparseIndexer(index_path=str(indices_dir / "sparse.pkl"))
    sparse_idx.build_index(all_chunks)
    sparse_idx.save()

    print("\n=== Ingestion Complete ===")
    print(f"Files created in {indices_dir}:")
    print(os.listdir(str(indices_dir)))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Build the dense + sparse indices.")
    parser.add_argument("--indices-dir", default=str(_PROJECT_ROOT / "data" / "indices"))
    parser.add_argument("--table-bodies", default=str(_DEFAULT_TABLE_BODIES),
                        help="JSON from src.data.arxiv_tables ('' to index captions only).")
    parser.add_argument("--table-format", choices=["markdown", "rows"], default="markdown",
                        help="Table bodies as Markdown, or one linearized sentence per row "
                             "(TabFact horizontal template).")
    args = parser.parse_args()
    run_ingestion_pipeline(Path(args.indices_dir),
                           Path(args.table_bodies) if args.table_bodies else None,
                           table_format=args.table_format)
