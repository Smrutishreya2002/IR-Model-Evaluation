import pandas as pd
from sklearn.metrics import cohen_kappa_score

# -------------------------------
# Load labeled file
# -------------------------------
df = pd.read_csv("evaluation_labeled.csv")

# Pick one query to double-annotate (example: qid=3080)
qid = 3080
subset = df[df["qid"] == qid].head(10).copy()

if subset.empty:
    raise ValueError(f"No results found for qid={qid} in evaluation_labeled.csv")

# -------------------------------
# Simulate 2nd annotator
# -------------------------------
# In reality, another person should label these rows.
# For demo: copy annotator1’s labels and introduce small random variation.
import random

subset["annotator1"] = subset["relevance"]

# Simulated annotator2: same labels but with some noise
subset["annotator2"] = [
    lab if random.random() > 0.2 else random.choice([0, 1, 2])
    for lab in subset["annotator1"]
]

# -------------------------------
# Compute Cohen's Kappa
# -------------------------------
kappa = cohen_kappa_score(subset["annotator1"], subset["annotator2"])
print(f"✅ Cohen’s Kappa for QID={qid}: {kappa:.3f}")

# -------------------------------
# Save double-annotation table
# -------------------------------
subset.to_csv("agreement_check.csv", index=False)
print("Saved detailed annotations to agreement_check.csv")
print(subset[["answer_id", "annotator1", "annotator2"]])
