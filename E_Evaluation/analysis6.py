import pandas as pd
import numpy as np

# Load your labeled evaluation file
df = pd.read_csv("evaluation_labeled.csv")

# Ensure relevance is numeric
df["relevance"] = df["relevance"].fillna(0).astype(int)

K = 10

def precision_at_k(rels, k=K):
    rels = rels[:k]
    return np.sum([1 if r == 2 else (0.5 if r == 1 else 0) for r in rels]) / k

# Compute P@10 per query, per system
results = []
for system in df["system"].unique():
    sys_df = df[df["system"] == system]
    for qid, group in sys_df.groupby("qid"):
        rels = group.sort_values("rank")["relevance"].tolist()
        results.append({
            "system": system,
            "qid": qid,
            "P@10": precision_at_k(rels, K)
        })

res_df = pd.DataFrame(results)

# Pivot to compare Boolean vs TAAT side by side
pivot = res_df.pivot(index="qid", columns="system", values="P@10").fillna(0)

# Find example queries
boolean_better = pivot[pivot["boolean"] > pivot["taat"]].head(5)
taat_better = pivot[pivot["taat"] > pivot["boolean"]].head(5)

print("📌 Boolean wins examples (higher P@10):")
print(boolean_better)

print("\n📌 TAAT wins examples (higher P@10):")
print(taat_better)
