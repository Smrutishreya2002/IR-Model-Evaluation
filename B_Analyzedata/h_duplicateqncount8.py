import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from bs4 import BeautifulSoup
import re

# -------------------------------
# Load questions
# -------------------------------
questions = pd.read_csv("questions.csv")

# -------------------------------
# Clean text (remove HTML + normalize)
# -------------------------------
def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    text = re.sub(r"[^a-z\s]", " ", text.lower())
    return " ".join(text.split())

questions["clean_text"] = (
    questions["Title"].fillna("").astype(str) + " " +
    questions["Body"].fillna("").astype(str)
).apply(clean_text)

# -------------------------------
# 1. Metadata-based duplicate detection
# -------------------------------
if "DuplicateOfQuestionId" in questions.columns:
    dup_meta = questions[~questions["DuplicateOfQuestionId"].isna()]
    print(f"✅ Found {len(dup_meta)} metadata-marked duplicate relations.")
    if not dup_meta.empty:
        print("Example duplicate pair (from metadata):")
        print(dup_meta[["Id", "DuplicateOfQuestionId", "Title"]].head())
else:
    print("⚠️ No DuplicateOfQuestionId field in dataset → falling back to text similarity.")

    # -------------------------------
    # 2. Text-based duplicate detection (TF-IDF + Cosine)
    # -------------------------------
    tfidf = TfidfVectorizer(max_features=5000, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(questions["clean_text"])

    # Compute cosine similarity (pairwise)
    sim_matrix = cosine_similarity(tfidf_matrix)

    # Find pairs with similarity > 0.8 (tune threshold)
    duplicates = []
    threshold = 0.8

    for i in range(len(sim_matrix)):
        for j in range(i+1, len(sim_matrix)):
            if sim_matrix[i, j] > threshold:
                duplicates.append((questions.iloc[i]["Id"], questions.iloc[j]["Id"], sim_matrix[i, j]))

    print(f"✅ Found {len(duplicates)} potential duplicate pairs (text-based).")

    if duplicates:
        print("Example duplicate pairs:")
        for pair in duplicates[:5]:  # show first 5
            print(f"Q{pair[0]} ↔ Q{pair[1]} (similarity={pair[2]:.2f})")
