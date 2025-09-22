import pandas as pd
import re
from collections import Counter
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from nltk.corpus import stopwords
import nltk

# Download stopwords if not already
#nltk.download("stopwords")

# Load your CSV (make sure questions.csv exists in same folder)
df = pd.read_csv("questions.csv")

# Combine Title + Body
df["Text"] = df["Title"].fillna("") + " " + df["Body"].fillna("")
texts = " ".join(df["Text"].astype(str).tolist())

# Tokenize: words >=3 letters
tokens = re.findall(r"\b[a-zA-Z]{3,}\b", texts.lower())

# Build frequency dictionary (all tokens)
freq_all = Counter(tokens)

# Remove stopwords
STOPWORDS = set(stopwords.words("english"))
tokens_nostop = [t for t in tokens if t not in STOPWORDS]
freq_nostop = Counter(tokens_nostop)

# WordCloud with all words
wc_all = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq_all)

plt.figure(figsize=(10,5))
plt.imshow(wc_all, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - All Words")
plt.show()

# WordCloud without stopwords
wc_nostop = WordCloud(width=800, height=400, background_color="white").generate_from_frequencies(freq_nostop)

plt.figure(figsize=(10,5))
plt.imshow(wc_nostop, interpolation="bilinear")
plt.axis("off")
plt.title("WordCloud - Without Stopwords")
plt.show()
