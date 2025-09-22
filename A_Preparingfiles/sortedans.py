import pandas as pd

# Load cleaned data
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

# Keep only questions with an AcceptedAnswerId
q_with_accepted = questions.dropna(subset=["AcceptedAnswerId"]).copy()
q_with_accepted["AcceptedAnswerId"] = q_with_accepted["AcceptedAnswerId"].astype(int)

first_is_accepted = 0
total = len(q_with_accepted)

for _, q in q_with_accepted.iterrows():
    qid = q["Id"]
    accepted = q["AcceptedAnswerId"]

    # Get all answers for this question
    ans = answers[answers["ParentId"] == qid].copy()
    if ans.empty:
        continue

    # Sort answers by CreateDate
    ans["CreateDate"] = pd.to_datetime(ans["CreateDate"])
    ans = ans.sort_values("CreateDate")

    # Check if first posted answer is the accepted one
    if ans.iloc[0]["Id"] == accepted:
        first_is_accepted += 1

print(f"Accepted answers that are first posted: {first_is_accepted} / {total} "
      f"({100*first_is_accepted/total:.2f}%)")
