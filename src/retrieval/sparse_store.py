import re
import os
import hashlib
import pickle
import nltk
from pathlib import Path
from rank_bm25 import BM25Okapi
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from typing import List, Dict

# Default index path is resolved relative to this file's location so the
# path is correct regardless of the working directory the script is run from.
_DEFAULT_SPARSE_INDEX = str(Path(__file__).resolve().parents[2] / "data" / "indices" / "sparse.pkl")

# ── NLTK data bootstrapping ────────────────────────────────────────────────────
# Each resource is checked individually so we never re-download unnecessarily.
for _resource, _path in [
    ('punkt',     'tokenizers/punkt'),
    ('punkt_tab', 'tokenizers/punkt_tab'),
    ('stopwords', 'corpora/stopwords'),
]:
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_resource, quiet=True)

# ── Module-level BM25 tokenisation utilities ──────────────────────────────────
_stemmer = PorterStemmer()
_stop_words: set = set()   # populated lazily on first call (avoids import-time cost)

# LaTeX artefacts that appear in QASPER but never in user queries.
# They inflate IDF and add noise without contributing matching signal.
_LATEX_RE = re.compile(
    r'\b(INLINEFORM|BIBREF|TABREF|FIGREF|SECREF|EQREF|URLREF)\d*\b',
    re.IGNORECASE,
)


def tokenize_for_bm25(text: str) -> List[str]:
    """
    Normalises text for BM25 indexing and querying.

    This function MUST be called identically at index-time (SparseIndexer)
    and at query-time (HybridRetriever._search_sparse) so that stem-forms align.

    Pipeline:
    1. Strip QASPER LaTeX artefacts (INLINEFORM0, BIBREF3, TABREF23, …)
       — these inflate IDF and never occur in user queries.
    2. Lowercase for case-insensitive matching.
    3. NLTK word_tokenize — handles hyphenated terms and punctuation better
       than whitespace split.
    4. Remove English stop-words (Robertson & Zaragoza, 2009 — stopwords raise
       IDF noise and dilute meaningful term weights in BM25).
    5. Porter-stem each remaining alphabetic token (Porter, 1980 — reduces
       vocabulary fragmentation: "propagate/propagation/propagated" → "propag").

    Reference:
        Robertson & Zaragoza (2009). The Probabilistic Relevance Framework:
        BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4).
    """
    global _stop_words
    if not _stop_words:
        _stop_words = set(stopwords.words('english'))

    # Step 1: strip LaTeX artefacts
    text = _LATEX_RE.sub(' ', text)
    # Steps 2–3: lowercase + tokenise
    tokens = word_tokenize(text.lower())
    # Steps 4–5: keep only alphabetic tokens, remove stopwords, then stem
    return [
        _stemmer.stem(t) for t in tokens
        if t.isalpha() and t not in _stop_words
    ]


class SparseIndexer:
    def __init__(self, index_path: str = _DEFAULT_SPARSE_INDEX):
        self.index_path = index_path
        self.bm25 = None
        self.corpus_chunks: List[Dict] = []

    def build_index(self, chunks: List[Dict]) -> None:
        """
        Builds a BM25Okapi index over the chunk corpus.

        Tokenisation uses :func:`tokenize_for_bm25` — the same function that
        HybridRetriever calls at query time — ensuring stem-form alignment.
        """
        print("Tokenizing corpus for BM25 (stop-word removal + Porter stemming)...")
        tokenized_corpus = [tokenize_for_bm25(doc['text']) for doc in chunks]
        self.bm25 = BM25Okapi(tokenized_corpus)
        self.corpus_chunks = chunks
        print("BM25 Index built.")

    def save(self) -> None:
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({
                "model": self.bm25,
                "metadata": self.corpus_chunks,
            }, f)
        # Write SHA-256 sidecar for integrity verification at load time
        with open(self.index_path, "rb") as f:
            sha = hashlib.sha256(f.read()).hexdigest()
        with open(self.index_path + ".sha256", "w") as f:
            f.write(sha)
        print(f"Saved sparse index to {self.index_path} (sha256={sha[:16]}...)")
