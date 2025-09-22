import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from collections import defaultdict
import time
import json

# -------------------------------
# Setup
# -------------------------------
# nltk.download("punkt")
# nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

def preprocess(text):
    """Remove HTML, tokenize, lowercase, remove stopwords and short tokens."""
    if not isinstance(text, str):
        return []
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# -------------------------------
# Load dataset
# -------------------------------
answers = pd.read_csv("answers.csv", parse_dates=["CreateDate"], low_memory=False)

# -------------------------------
# Build TF Inverted Index
# -------------------------------
def build_tf_index(df):
    tf_index = defaultdict(dict)  # term -> {doc_id: tf}
    start = time.perf_counter()

    for _, row in df.iterrows():
        doc_id = int(row["Id"])
        tokens = preprocess(row["Body"])
        for token in tokens:
            if doc_id not in tf_index[token]:
                tf_index[token][doc_id] = 0
            tf_index[token][doc_id] += 1

    elapsed = time.perf_counter() - start
    print(f"✅ TF Inverted index built in {elapsed:.2f} seconds")
    print(f"📦 Index size: {len(tf_index)} unique terms")
    return tf_index

tf_index = build_tf_index(answers)

# Save for reuse
with open("inverted_index_tf.json", "w", encoding="utf-8") as f:
    json.dump(tf_index, f)

# -------------------------------
# TAAT Retrieval
# -------------------------------
def taat_retrieve(query, index, top_k=50):
    """Term-at-a-time retrieval using term frequencies. Returns top-k doc IDs ranked by score."""
    query_terms = preprocess(query)
    if not query_terms:
        return []

    scores = defaultdict(int)

    # For each term, accumulate scores across documents
    for term in query_terms:
        if term in index:
            for doc_id, tf in index[term].items():
                scores[doc_id] += tf  # simple scoring by term frequency

    # Sort by score (desc), then doc ID (asc)
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    top_docs = [doc for doc, _ in ranked[:top_k]]
    return top_docs

# -------------------------------
# Example Queries
# -------------------------------
example_queries = [
    "visa passport",
    "hotel hostel"
]

for q in example_queries:
    results = taat_retrieve(q, tf_index, top_k=50)
    print(f"\nQuery: {q}")
    print(f"Top {len(results)} result IDs: {results[:10]}...")  # show first 10
