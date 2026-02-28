# src/retrieval/sparse_store.py

from rank_bm25 import BM25Okapi
import pickle
import nltk
from nltk.tokenize import word_tokenize
from typing import List, Dict
import os

# Ensure NLTK data is present
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')

class SparseIndexer:
    def __init__(self, index_path="../data/indices/sparse.pkl"):
        self.index_path = index_path
        self.bm25 = None
        self.corpus_chunks = []

    def build_index(self, chunks: List[Dict]):
        print("Tokenizing corpus for BM25...")
        # 1. Tokenize: Break text into list of words ["the", "cat", ...]
        # We use .lower() to make it case-insensitive
        tokenized_corpus = [word_tokenize(doc['text'].lower()) for doc in chunks]
        
        # 2. Build Index: BM25Okapi calculates IDF for every word immediately
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_chunks = chunks
        print("BM25 Index built.")

    def save(self):
        # Ensure the directory exists
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        # BM25 is a Python object, so we pickle it directly
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "model": self.bm25,
                "metadata": self.corpus_chunks
            }, f)
        print(f"Saved sparse index to {self.index_path}")