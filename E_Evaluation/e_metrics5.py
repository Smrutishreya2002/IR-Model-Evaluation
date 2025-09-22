import pandas as pd
import numpy as np

# -------------------------------
# Load labeled evaluation file
# -------------------------------
df = pd.read_csv("evaluation_labeled.csv")

# Ensure relevance is numeric
df["relevance"] = df["relevance"].fillna(-1).astype(int)

# -------------------------------
# Parameters
# -------------------------------
K = 10  # cutoff for metrics

# Mapping: how to interpret labels
# 2 = Relevant, 1 = Partially relevant, 0 = Not relevant
def relevance_to_binary(rel):
    if rel == 2:
        return 1
    elif rel == 1:
        return 0.5  # optional: count partial as 0.5
    else:
        return 0

def relevance_to_gain(rel):
    if rel == 2:
        return 2
    elif rel == 1:
        return 1
    else:
        return 0

# -------------------------------
# Metrics functions
# -------------------------------
def precision_at_k(rels, k=K):
    binary_rels = [relevance_to_binary(r) for r in rels[:k]]
    return np.sum(binary_rels) / k

def dcg_at_k(rels, k=K):
    gains = [relevance_to_gain(r) for r in rels[:k]]
    return np.sum([g / np.log2(i + 2) for i, g in enumerate(gains)])

def ndcg_at_k(rels, k=K):
    actual = dcg_at_k(rels, k)
    ideal_rels = sorted(rels, reverse=True)  # best possible ordering
    ideal = dcg_at_k(ideal_rels, k)
    return actual / ideal if ideal > 0 else 0

# -------------------------------
# Compute metrics per system
# -------------------------------
results = []
for system in df["system"].unique():
    sys_df = df[df["system"] == system]

    precs, ndcgs = [], []
    for qid, group in sys_df.groupby("qid"):
        rels = group.sort_values("rank")["relevance"].tolist()
        if all(r == -1 for r in rels):  # skip unlabeled queries
            continue
        rels = [r if r >= 0 else 0 for r in rels]  # replace -1 with 0

        precs.append(precision_at_k(rels, K))
        ndcgs.append(ndcg_at_k(rels, K))

    results.append({
        "system": system,
        f"P@{K}": np.mean(precs) if precs else 0,
        f"nDCG@{K}": np.mean(ndcgs) if ndcgs else 0,
    })

metrics_df = pd.DataFrame(results)

print("✅ Evaluation Metrics (averaged across queries):")
print(metrics_df)

# -------------------------------
# Average Retrieval Time
# -------------------------------
if "time" in df.columns:
    time_stats = df.groupby("system")["time"].mean().reset_index()
    print("\n⏱️ Average Retrieval Time per Query (seconds):")
    print(time_stats)
else:
    print("\n⚠️ No retrieval time recorded in evaluation file")
