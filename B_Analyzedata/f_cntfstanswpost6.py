from lxml import etree
import pandas as pd

# --- Load data ---
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

file_path = r"C:/Users/mohan/AppData/Local/Temp/81a040c1-ed11-4724-b9d6-a5c154172f92_travel.stackexchange.com.7z.f92/Users.xml"

users = []

for _, elem in etree.iterparse(file_path, events=("end",), tag="row"):
    users.append({
        "Id": elem.get("Id"),
        "DisplayName": elem.get("DisplayName"),
        "Reputation": elem.get("Reputation"),
        "CreationDate": elem.get("CreationDate"),
    })
    elem.clear()

# Save to CSV
df = pd.DataFrame(users)
df.to_csv("users.csv", index=False)

print("Saved", len(df), "users to users.csv")

users = pd.read_csv("users.csv")   # must include: Id, Reputation

# --- Preprocess ---
questions = questions.dropna(subset=["AcceptedAnswerId"]).copy()
questions["AcceptedAnswerId"] = questions["AcceptedAnswerId"].astype(int)

answers["CreateDate"] = pd.to_datetime(answers["CreateDate"])
answers["Score"] = pd.to_numeric(answers["Score"], errors="coerce").fillna(0).astype(int)

users = users.rename(columns={"Id": "OwnerUserId"})
users["Reputation"] = pd.to_numeric(users["Reputation"], errors="coerce").fillna(0).astype(int)

# --- Join answers with reputation ---
answers = answers.merge(users[["OwnerUserId", "Reputation"]], on="OwnerUserId", how="left").fillna({"Reputation": 0})

# =========================================================
# 1. How many accepted answers are the first answers posted?
# =========================================================
first_is_accepted = 0
total = len(questions)

for _, q in questions.iterrows():
    qid = q["Id"]
    accepted_id = q["AcceptedAnswerId"]

    # Get answers for this question sorted by time
    ans = answers[answers["ParentId"] == qid].sort_values("CreateDate")
    if ans.empty:
        continue

    if ans.iloc[0]["Id"] == accepted_id:
        first_is_accepted += 1

print(f"Accepted answers that are the first posted: {first_is_accepted}/{total} "
      f"({100*first_is_accepted/total:.2f}%)\n")

# =========================================================
# 2. Is there a correlation with reputation score?
# =========================================================
# Mark accepted answers
accepted_set = set(questions["AcceptedAnswerId"])
answers["IsAccepted"] = answers["Id"].isin(accepted_set).astype(int)

# Correlation between reputation and accepted answer
corr_accept = answers["IsAccepted"].corr(answers["Reputation"])
corr_score  = answers["Score"].corr(answers["Reputation"])

print(f"Correlation (Reputation vs Accepted Answer): {corr_accept:.3f}")
print(f"Correlation (Reputation vs Answer Score): {corr_score:.3f}\n")

# =========================================================
# 3. Are accepted answers always highest scoring?
# =========================================================
examples = []

for _, q in questions.iterrows():
    qid = q["Id"]
    accepted_id = q["AcceptedAnswerId"]

    ans = answers[answers["ParentId"] == qid]
    if ans.empty:
        continue

    max_score = ans["Score"].max()
    accepted_score = ans.loc[ans["Id"] == accepted_id, "Score"]

    if not accepted_score.empty and int(accepted_score) < max_score:
        examples.append((qid, accepted_id, int(accepted_score), max_score))

print(f"Total accepted answers NOT highest scoring: {len(examples)}\n")

# Show 3 examples with links
site_url = "https://travel.stackexchange.com/questions/"  # <-- change for your site
for qid, aid, ascore, mscore in examples[:3]:
    print(f"QID={qid} | Accepted Answer Score={ascore}, Max Score={mscore}")
    print(f"  Link: {site_url}{qid}\n")
