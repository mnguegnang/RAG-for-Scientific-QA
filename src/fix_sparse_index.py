import pickle
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

def regenerate_sparse_index():
    print(f"Loading metadata from {DENSE_META_PATH}...")
    
    if not os.path.exists(DENSE_META_PATH):
        print(f"ERROR: Could not find {DENSE_META_PATH}")
        return

    # 1. Load the text data we used for FAISS
    with open(DENSE_META_PATH, 'rb') as f:
        meta_data = pickle.load(f)
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
        
    print("SUCCESS: Sparse Index regenerated with 'corpus' key.")

if __name__ == "__main__":
    regenerate_sparse_index()
