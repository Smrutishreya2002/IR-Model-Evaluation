import pandas as pd
import re
import nltk
from bs4 import BeautifulSoup
from collections import defaultdict
import time
import json

# -------------------------------
# Download tokenizer (only first time)
# -------------------------------
nltk.download("punkt")
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# -------------------------------
# Preprocessing function
# -------------------------------
def preprocess(text):
    """Remove HTML, tokenize, lowercase, remove stopwords and short tokens."""
    if not isinstance(text, str):
        return []
    text = BeautifulSoup(text, "html.parser").get_text(" ")  # strip HTML
    text = re.sub(r"[^a-z\s]", " ", text.lower())           # keep only letters
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return tokens

# -------------------------------
# Load dataset
# -------------------------------
answers = pd.read_csv("answers.csv", parse_dates=["CreateDate"], low_memory=False)

# -------------------------------
# Build Inverted Index
# -------------------------------
def build_inverted_index(df):
    inverted_index = defaultdict(set)
    start = time.perf_counter()

    for _, row in df.iterrows():
        doc_id = row["Id"]
        tokens = preprocess(row["Body"])
        for token in set(tokens):  # use set() to avoid duplicates
            inverted_index[token].add(int(doc_id))

    elapsed = time.perf_counter() - start
    print(f"✅ Inverted index built in {elapsed:.2f} seconds")
    print(f"📦 Index size: {len(inverted_index)} unique terms")
    return inverted_index

inverted_index = build_inverted_index(answers)

# (Optional) Save to disk for reuse
with open("inverted_index.json", "w", encoding="utf-8") as f:
    json.dump({k: list(v) for k, v in inverted_index.items()}, f)

# -------------------------------
# Boolean Search Functions
# -------------------------------
def boolean_retrieve(query, index, operator="AND", top_k=50):
    """Boolean search with AND / OR operators. Returns top-k doc IDs."""
    query_terms = preprocess(query)
    if not query_terms:
        return []

    postings = [set(index[t]) for t in query_terms if t in index]

    if not postings:
        return []

    if operator == "AND":
        result = set.intersection(*postings)
    elif operator == "OR":
        result = set.union(*postings)
    else:
        raise ValueError("Operator must be AND or OR")

    result = sorted(result)[:top_k]  # top_k by ID order
    return result

# -------------------------------
# Example Queries
# -------------------------------
example_queries = [
    ("visa AND passport", "AND"),
    ("hotel OR hostel", "OR")
]

for q, op in example_queries:
    results = boolean_retrieve(q, inverted_index, operator=op, top_k=50)
    print(f"\nQuery: {q} ({op})")
    print(f"Top {len(results)} result IDs: {results[:10]}...")  # show first 10
