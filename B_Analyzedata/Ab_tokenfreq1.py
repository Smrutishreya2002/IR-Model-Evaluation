'''common terms and its frequency'''
import pandas as pd
import re
from collections import Counter

# 1. Load dataset
df = pd.read_csv("questions.csv")   # make sure questions.csv is in same folder

# 2. Combine Title + Body
df["Text"] = df["Title"].fillna("") + " " + df["Body"].fillna("")

# 3. Merge into one big string
texts = " ".join(df["Text"].astype(str).tolist())

# 4. Tokenize (words >=3 letters)
tokens = re.findall(r"\b[a-zA-Z]{3,}\b", texts.lower())

# 5. Frequency count
freq = Counter(tokens)

print("Top 20 words:")
print(freq.most_common(40))
