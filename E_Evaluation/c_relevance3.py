import pandas as pd

# -------------------------------
# Load data
# -------------------------------
eval_df = pd.read_csv("evaluation.csv")
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

# Ensure relevance column exists
if "relevance" not in eval_df.columns:
    eval_df["relevance"] = -1

# -------------------------------
# Helper: fetch text
# -------------------------------
def get_question(qid):
    row = questions[questions["Id"] == qid]
    if row.empty:
        return "N/A", "N/A"
    return row.iloc[0]["Title"], row.iloc[0]["Body"]

def get_answer(aid):
    row = answers[answers["Id"] == aid]
    if row.empty:
        return "N/A"
    return row.iloc[0]["Body"]

# -------------------------------
# Interactive labeling loop
# -------------------------------
for idx, row in eval_df.iterrows():
    if row["relevance"] != -1:
        continue  # already labeled

    qid = row["qid"]
    aid = row["answer_id"]

    qtitle, qbody = get_question(qid)
    atext = get_answer(aid)

    print("=" * 80)
    print(f"System: {row['system']} | QID: {qid} | AnswerID: {aid} | Rank: {row['rank']}")
    print(f"\nQUERY TITLE: {qtitle}")
    print(f"\nQUERY BODY: {qbody[:500]}...\n")  # show first 500 chars
    print(f"ANSWER (excerpt): {atext[:500]}...\n")

    label = input("Relevance? (2=Relevant, 1=Partially, 0=Not): ").strip()
    if label in ["0", "1", "2"]:
        eval_df.at[idx, "relevance"] = int(label)
    else:
        print("⚠️ Skipped, leaving as -1")

    # Save progress after each labeling step
    eval_df.to_csv("evaluation_labeled.csv", index=False)

print("\n✅ Labeling finished. Results saved to evaluation_labeled.csv")
