import pandas as pd

# -------------------------------
# Load questions
# -------------------------------
questions = pd.read_csv("questions.csv")

# Ensure tags column exists
if "Tags" not in questions.columns:
    raise ValueError("Tags column missing in questions.csv. Please regenerate with Tags included.")

# -------------------------------
# Select 20 diverse queries
# -------------------------------
# Sort by Score to get popular ones first
popular = questions.sort_values("Score", ascending=False).head(50)

# Sample some short titles (< 5 words) and long ones (> 10 words)
short_qs = questions[questions["Title"].str.split().str.len() <= 5].sample(5, random_state=42)
long_qs = questions[questions["Title"].str.split().str.len() > 10].sample(5, random_state=42)

# Random sample for diversity
random_qs = questions.sample(10, random_state=42)

# Combine all, drop duplicates, keep only 20
query_set = pd.concat([popular, short_qs, long_qs, random_qs]).drop_duplicates("Id").head(20)

# -------------------------------
# Build table: qid, query_text, tags
# -------------------------------
query_table = query_set[["Id", "Title", "Tags"]].rename(
    columns={"Id": "qid", "Title": "query_text", "Tags": "tags"}
).reset_index(drop=True)

# Save to CSV
query_table.to_csv("query_set.csv", index=False)

print("✅ Query set saved to query_set.csv")
print(query_table)
