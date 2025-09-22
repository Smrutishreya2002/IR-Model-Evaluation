import pandas as pd
import nltk
import re
from bs4 import BeautifulSoup
from collections import Counter
import matplotlib.pyplot as plt

# -------------------------------
# Load data
# -------------------------------
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

# -------------------------------
# Combine text from questions + answers
# -------------------------------
text_data = (
    questions["Title"].fillna("").astype(str) + " " +
    questions["Body"].fillna("").astype(str) + " " +
    answers["Body"].fillna("").astype(str)
)

# -------------------------------
# Preprocess text
# -------------------------------
from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words("english"))

def clean_and_tokenize(text):
    if not isinstance(text, str):   # skip NaN or non-strings
        return []
    text = BeautifulSoup(str(text), "html.parser").get_text(" ")  # remove HTML safely
    text = re.sub(r"[^a-z\s]", " ", text.lower())  # keep only letters
    tokens = nltk.word_tokenize(text)
    return [t for t in tokens if t not in STOPWORDS and len(t) > 1]

all_tokens = []
for doc in text_data:
    all_tokens.extend(clean_and_tokenize(doc))

print(f"✅ Total tokens: {len(all_tokens)}")
print(f"✅ Unique tokens: {len(set(all_tokens))}")

# -------------------------------
# Frequency distribution
# -------------------------------
freq_dist = Counter(all_tokens)
sorted_freqs = sorted(freq_dist.values(), reverse=True)

# -------------------------------
# Plot Zipf's Law
# -------------------------------
plt.figure(figsize=(8,6))
plt.loglog(range(1, len(sorted_freqs)+1), sorted_freqs, marker=".")
plt.title("Zipf's Law - Travel StackExchange")
plt.xlabel("Rank (log scale)")
plt.ylabel("Frequency (log scale)")
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()
