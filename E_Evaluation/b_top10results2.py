import pandas as pd
import json
import nltk
import re
import time
from collections import defaultdict

# -------------------------------
# Load datasets
# -------------------------------
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv", parse_dates=["CreateDate"], low_memory=False)

# Load indexes
with open("inverted_index.json", "r", encoding="utf-8") as f:
    bool_index = json.load(f)

with open("inverted_index_tf.json", "r", encoding="utf-8") as f:
    tf_index = json.load(f)

# -------------------------------
# Preprocessing
# -------------------------------
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from bs4 import BeautifulSoup

STOPWORDS = set(stopwords.words("english"))
STEMMER = PorterStemmer()

def preprocess(text):
    if not isinstance(text, str):
        return []
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    tokens = [STEMMER.stem(t) for t in tokens]
    return tokens

# -------------------------------
# Boolean Retrieval (AND only)
# -------------------------------
def boolean_retrieve(query_terms, index):
    postings = []
    for term in query_terms:
        if term in index:
            postings.append(set(index[term]))
    if not postings:
        return []
    return sorted(set.intersection(*postings))  # simple AND

# -------------------------------
# TAAT Scoring
# -------------------------------
def taat_score(query_terms, index):
    scores = defaultdict(int)
    for term in query_terms:
        if term not in index:
            continue
        for doc, tf in index[term].items():
            scores[doc] += tf
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [doc for doc, _ in ranked]

# -------------------------------
# Evaluation File Generator
# -------------------------------
def generate_eval(query_file="query_set.csv", out_file="evaluation.csv"):
    queries = pd.read_csv(query_file)
    eval_rows = []

    for _, row in queries.iterrows():
        qid = row["qid"]
        qtext = str(row["query_text"])
        qbody = questions.loc[questions["Id"] == qid, "Body"].values[0] if qid in questions["Id"].values else ""

        query_terms = preprocess(qtext)

        # -----------------------
        # Boolean retrieval timing
        # -----------------------
        start = time.perf_counter()
        bool_results = boolean_retrieve(query_terms, bool_index)[:10]
        elapsed_bool = time.perf_counter() - start

        for rank, aid in enumerate(bool_results, start=1):
            eval_rows.append({
                "system": "boolean",
                "qid": qid,
                "query": qtext,
                "query_body": qbody,
                "answer_id": aid,
                "rank": rank,
                "relevance": -1,   # manual later
                "time": elapsed_bool  # ⬅ added
            })

        # -----------------------
        # TAAT retrieval timing
        # -----------------------
        start = time.perf_counter()
        taat_results = taat_score(query_terms, tf_index)[:10]
        elapsed_taat = time.perf_counter() - start

        for rank, aid in enumerate(taat_results, start=1):
            eval_rows.append({
                "system": "taat",
                "qid": qid,
                "query": qtext,
                "query_body": qbody,
                "answer_id": aid,
                "rank": rank,
                "relevance": -1,
                "time": elapsed_taat  # ⬅ added
            })

    eval_df = pd.DataFrame(eval_rows)
    eval_df.to_csv(out_file, index=False)
    print(f"✅ Evaluation file created with timing: {out_file}")
    return eval_df

# -------------------------------
# Example Run
# -------------------------------
if __name__ == "__main__":
    eval_df = generate_eval("query_set.csv", "evaluation.csv")
    print(eval_df.head(20))
