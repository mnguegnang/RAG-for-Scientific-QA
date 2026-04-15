import pickle
import hashlib
import os
from rank_bm25 import BM25Okapi
import nltk

# Download tokenizer data if not present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# CONFIG
DENSE_META_PATH = 'data/indices/dense.index.meta'
SPARSE_INDEX_PATH = 'data/indices/sparse.pkl'

# Project root for path-traversal checks
_PROJECT_ROOT = os.path.realpath(os.path.dirname(__file__) + '/..')

def regenerate_sparse_index():
    print(f"Loading metadata from {DENSE_META_PATH}...")
    
    if not os.path.exists(DENSE_META_PATH):
        print(f"ERROR: Could not find {DENSE_META_PATH}")
        return

    # Path traversal guard
    resolved_meta = os.path.realpath(DENSE_META_PATH)
    if not resolved_meta.startswith(_PROJECT_ROOT + os.sep):
        raise ValueError(f"Path traversal blocked: {DENSE_META_PATH} -> {resolved_meta}")

    # 1. Load the text data we used for FAISS — SHA-256 verified if sidecar exists
    with open(resolved_meta, 'rb') as f:
        raw = f.read()

    sha256_path = resolved_meta + ".sha256"
    if os.path.exists(sha256_path):
        actual = hashlib.sha256(raw).hexdigest()
        with open(sha256_path) as fh:
            expected = fh.read().strip()
        if actual != expected:
            raise ValueError(
                f"Integrity check FAILED for {DENSE_META_PATH}: "
                f"expected sha256={expected}, got {actual}."
            )
        print("SHA-256 integrity check passed for metadata.")
    else:
        print("WARNING: No .sha256 sidecar — skipping integrity check.")

    meta_data = pickle.loads(raw)
        # meta_data is likely a dictionary {0: {...}, 1: {...}} or a list
        
    print(f"Found {len(meta_data)} documents in metadata.")

    # 2. Extract Corpus and IDs
    # We need to ensure the order matches exactly 0, 1, 2...
    corpus_text = []
    doc_ids = []
    
    # Check if meta_data is list or dict
    if isinstance(meta_data, dict):
        # Sort by index to ensure alignment
        sorted_indices = sorted(meta_data.keys())
        for idx in sorted_indices:
            item = meta_data[idx]
            corpus_text.append(item['text'])
            doc_ids.append(item['paper_id'])
    elif isinstance(meta_data, list):
        for item in meta_data:
            corpus_text.append(item['text'])
            doc_ids.append(item['paper_id'])

    print("Tokenizing corpus for BM25 (this might take a moment)...")
    tokenized_corpus = [doc.lower().split() for doc in corpus_text]

    # 3. Train BM25
    print("Training BM25 Model...")
    bm25 = BM25Okapi(tokenized_corpus)

    # 4. Save in the correct format for Phase 3
    package = {
        'model': bm25,
        'corpus': corpus_text,   # <--- This is the key you were missing
        'doc_ids': doc_ids
    }

    print(f"Saving new Sparse Index to {SPARSE_INDEX_PATH}...")
    with open(SPARSE_INDEX_PATH, 'wb') as f:
        pickle.dump(package, f)
    # Write SHA-256 sidecar for integrity verification at load time
    with open(SPARSE_INDEX_PATH, 'rb') as f:
        sha = hashlib.sha256(f.read()).hexdigest()
    with open(SPARSE_INDEX_PATH + ".sha256", 'w') as f:
        f.write(sha)
        
    print(f"SUCCESS: Sparse Index regenerated with 'corpus' key (sha256={sha[:16]}...).")

if __name__ == "__main__":
    regenerate_sparse_index()
