import pandas as pd
import re
from collections import Counter
import matplotlib.pyplot as plt
import nltk

# Make sure nltk stopwords are available
nltk.download("stopwords")
from nltk.corpus import stopwords

STOPWORDS = set(stopwords.words("english"))

# --- Helper function ---
def tokenize(text):
    if not isinstance(text, str):
        return []
    # Lowercase and keep only words (a-z)
    words = re.findall(r"[a-z]+", text.lower())
    # Remove stopwords
    return [w for w in words if w not in STOPWORDS]

# --- Load dataset ---
questions = pd.read_csv("questions.csv")

# Combine Title + Body
questions["FullText"] = questions["Title"].fillna("") + " " + questions["Body"].fillna("")

# Tokenize all text
all_tokens = []
for txt in questions["FullText"]:
    all_tokens.extend(tokenize(txt))

# Count frequencies
word_freq = Counter(all_tokens)
top20 = word_freq.most_common(20)

# Separate for plotting
words, freqs = zip(*top20)

# --- Plot ---
plt.figure(figsize=(12,6))
plt.bar(words, freqs)
plt.xticks(rotation=45, ha="right")
plt.title("Top 20 Frequent Words in Questions")
plt.xlabel("Words")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()
