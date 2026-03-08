

import json
import os
import re
import time
import pickle
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import normalize

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
EMBED_DIR = BASE_DIR / "embeddings"
EMBED_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# Step 1: Load corpus
# ---------------------------------------------------------------------------
def load_corpus(path: str = None) -> dict:
    path = path or str(DATA_DIR / "corpus.json")
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Step 2: Preprocessing
# ---------------------------------------------------------------------------
_QUOTE_LINE = re.compile(r"^>.*$", re.MULTILINE)          # quoted reply lines
_HEADER_LINE = re.compile(r"^(From|Subject|Message-ID|Date|Organization|Lines|NNTP|Path|Xref|References|In-Reply-To):.*$", re.MULTILINE | re.IGNORECASE)
_EMAIL = re.compile(r"\S+@\S+\.\S+")                       # email addresses
_URL = re.compile(r"https?://\S+|www\.\S+")                # URLs
_PUNCT_DIGITS = re.compile(r"[^a-zA-Z\s]")                 # keep only alpha
_MULTI_SPACE = re.compile(r"\s+")


def clean_document(text: str) -> str:
    """
    Aggressively clean a single newsgroup post.

    Pipeline:
      1. Strip header fields (From:, Subject:, etc.)
      2. Remove quoted lines ("> …")
      3. Remove email addresses and URLs
      4. Strip punctuation and digits
      5. Lowercase and collapse whitespace
    """
    text = _HEADER_LINE.sub("", text)
    text = _QUOTE_LINE.sub("", text)
    text = _EMAIL.sub(" ", text)
    text = _URL.sub(" ", text)
    text = _PUNCT_DIGITS.sub(" ", text)
    text = text.lower()
    text = _MULTI_SPACE.sub(" ", text).strip()
    return text


def preprocess_corpus(documents: list) -> list:
    print("Preprocessing corpus…")
    t0 = time.time()
    cleaned = [clean_document(d) for d in documents]
    # Drop empty documents (degenerate posts after stripping)
    cleaned = [c if len(c) > 30 else "unknown topic" for c in cleaned]
    print(f"  Preprocessing done in {time.time()-t0:.1f}s")
    return cleaned


# ---------------------------------------------------------------------------
# Step 3: TF-IDF vectorisation
# ---------------------------------------------------------------------------
def build_tfidf(cleaned_docs: list, n_components: int = 300):
    """
    Build TF-IDF matrix and reduce to n_components via Truncated SVD (LSA).

    Returns:
      tfidf_vectorizer  — fitted TfidfVectorizer (for query encoding)
      svd               — fitted TruncatedSVD (for query encoding)
      embeddings        — shape (n_docs, n_components), L2-normalised
    """
    print(f"Building TF-IDF (min_df=5, max_df=0.9, sublinear_tf=True)…")
    t0 = time.time()

    # sublinear_tf = True: replace tf with 1 + log(tf).
    # This dampens the effect of very frequent in-document terms, which is
    # especially important for newsgroup posts where the same word can appear
    # dozens of times without adding proportional semantic weight.
    tfidf = TfidfVectorizer(
        min_df=5,           # ignore terms in fewer than 5 docs
        max_df=0.90,        # ignore terms in >90% of docs
        ngram_range=(1, 2), # unigrams + bigrams: "hard drive" ≠ "hard" + "drive"
        sublinear_tf=True,
        max_features=50000, # cap vocabulary to avoid memory blowup
        stop_words="english",
        token_pattern=r"(?u)\b[a-zA-Z]{3,}\b",  # min 3 chars, alpha only
    )
    X = tfidf.fit_transform(cleaned_docs)
    print(f"  TF-IDF shape: {X.shape}  ({time.time()-t0:.1f}s)")

    # --- Truncated SVD (LSA) ---
    # 300 components is standard for LSA on newsgroups; empirical diminishing
    # returns observed past 250 (explained variance ratio plateaus).
    print(f"Running Truncated SVD to {n_components} components…")
    t1 = time.time()
    svd = TruncatedSVD(n_components=n_components, n_iter=10, random_state=42)
    X_svd = svd.fit_transform(X)

    explained = svd.explained_variance_ratio_.sum()
    print(f"  SVD done in {time.time()-t1:.1f}s  (explained variance: {explained:.3f})")

    # L2-normalise so cosine similarity = dot product (required for fast cache lookup)
    X_norm = normalize(X_svd, norm="l2")
    return tfidf, svd, X_norm


# ---------------------------------------------------------------------------
# Step 4: Minimal Vector Store
# ---------------------------------------------------------------------------
class VectorStore:
    
   

    def __init__(self):
        self._ids: list = []
        self._vectors: np.ndarray = None   # shape (n, d)
        self._metadata: list = []          # list of dicts

    def add_documents(
        self,
        ids: list,
        vectors: np.ndarray,
        metadata: list = None,
    ):
        """Append documents to the store."""
        assert len(ids) == vectors.shape[0], "ids/vectors length mismatch"
        if metadata is None:
            metadata = [{} for _ in ids]

        self._ids.extend(ids)
        self._metadata.extend(metadata)

        if self._vectors is None:
            self._vectors = vectors
        else:
            self._vectors = np.vstack([self._vectors, vectors])

    def query(
        self,
        vector: np.ndarray,
        n_results: int = 10,
        filter_fn=None,
    ) -> list:
        """
        Return top-n_results documents by cosine similarity.

        Args:
          vector   — L2-normalised query vector shape (d,)
          filter_fn — optional callable(metadata_dict) -> bool

        Returns list of dicts: {id, similarity, metadata}
        """
        if self._vectors is None:
            return []

        # Cosine similarity: dot product on L2-normalised vectors
        sims = self._vectors @ vector  # shape (n,)

        if filter_fn is not None:
            mask = np.array([filter_fn(m) for m in self._metadata], dtype=bool)
            sims = np.where(mask, sims, -1.0)

        top_idx = np.argpartition(sims, -n_results)[-n_results:]
        top_idx = top_idx[np.argsort(-sims[top_idx])]

        return [
            {
                "id": self._ids[i],
                "similarity": float(sims[i]),
                "metadata": self._metadata[i],
            }
            for i in top_idx
        ]

    def get_by_ids(self, ids: list) -> list:
        id_to_idx = {doc_id: i for i, doc_id in enumerate(self._ids)}
        results = []
        for doc_id in ids:
            if doc_id in id_to_idx:
                i = id_to_idx[doc_id]
                results.append({
                    "id": doc_id,
                    "vector": self._vectors[i],
                    "metadata": self._metadata[i],
                })
        return results

    def __len__(self):
        return len(self._ids)

    def persist(self, directory: str):
        os.makedirs(directory, exist_ok=True)
        np.save(os.path.join(directory, "vectors.npy"), self._vectors)
        with open(os.path.join(directory, "meta.json"), "w") as f:
            json.dump({"ids": self._ids, "metadata": self._metadata}, f)
        print(f"  Vector store persisted to {directory}  ({len(self)} docs)")

    @classmethod
    def load(cls, directory: str) -> "VectorStore":
        store = cls()
        store._vectors = np.load(os.path.join(directory, "vectors.npy"))
        with open(os.path.join(directory, "meta.json")) as f:
            data = json.load(f)
        store._ids = data["ids"]
        store._metadata = data["metadata"]
        print(f"  Vector store loaded from {directory}  ({len(store)} docs)")
        return store


# ---------------------------------------------------------------------------
# Step 5: Encode a query at inference time
# ---------------------------------------------------------------------------
def encode_query(
    query: str,
    tfidf: TfidfVectorizer,
    svd: TruncatedSVD,
) -> np.ndarray:
    """
    Transform a raw query string into a normalised LSA vector.
    Same pipeline as the corpus: clean → TF-IDF → SVD → L2-normalise.
    """
    cleaned = clean_document(query)
    X_tfidf = tfidf.transform([cleaned])
    X_svd = svd.transform(X_tfidf)
    return normalize(X_svd, norm="l2")[0]


# ---------------------------------------------------------------------------
# Main: run Part 1
# ---------------------------------------------------------------------------
def run_part1():
    print("=" * 60)
    print("PART 1: Embedding & Vector Database Setup")
    print("=" * 60)

    # Load corpus
    corpus = load_corpus()
    documents = corpus["documents"]
    labels = corpus["labels"]
    category_names = corpus["category_names"]
    print(f"Loaded {len(documents)} documents across {len(category_names)} categories")

    # Preprocess
    cleaned = preprocess_corpus(documents)

    # Build TF-IDF + SVD embeddings
    tfidf, svd, embeddings = build_tfidf(cleaned, n_components=300)
    print(f"Embeddings shape: {embeddings.shape}")

    # Persist embeddings + models
    print("Persisting embeddings and models…")
    np.save(str(EMBED_DIR / "embeddings.npy"), embeddings)
    with open(str(EMBED_DIR / "tfidf.pkl"), "wb") as f:
        pickle.dump(tfidf, f)
    with open(str(EMBED_DIR / "svd.pkl"), "wb") as f:
        pickle.dump(svd, f)
    with open(str(EMBED_DIR / "labels.json"), "w") as f:
        json.dump({"labels": labels, "category_names": category_names}, f)

    # Build vector store
    print("Building vector store…")
    store = VectorStore()
    ids = [f"doc_{i}" for i in range(len(documents))]
    metadata = [
        {
            "label": labels[i],
            "doc_idx": i,
            "text_preview": documents[i][:200],
        }
        for i in range(len(documents))
    ]
    store.add_documents(ids, embeddings, metadata)
    store.persist(str(EMBED_DIR / "vector_store"))

    # Quick smoke test
    print("\nSmoke test — querying 'rocket launch orbit satellite':")
    query_vec = encode_query("rocket launch orbit satellite", tfidf, svd)
    results = store.query(query_vec, n_results=5)
    for r in results:
        print(f"  [{r['similarity']:.3f}] {r['metadata']['label']}: {r['metadata']['text_preview'][:80]}")

    print("\nPart 1 complete.")
    return tfidf, svd, embeddings, store, labels, category_names, cleaned


if __name__ == "__main__":
    run_part1()
