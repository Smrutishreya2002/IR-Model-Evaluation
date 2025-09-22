import pandas as pd
from lxml import etree

# --- Load questions ---
questions = pd.read_csv("questions.csv")

# --- Parse Comments.xml ---
comments_file = r"C:/Users/mohan/AppData/Local/Temp/d043ac73-633d-4af7-af01-711d70b8ed14_travel.stackexchange.com.7z.d14/Comments.xml"

comments = []
for _, elem in etree.iterparse(comments_file, events=("end",), tag="row"):
    comments.append({
        "Id": elem.get("Id"),
        "PostId": int(elem.get("PostId")),
        "Text": elem.get("Text"),
        "UserId": elem.get("UserId"),
        "CreationDate": elem.get("CreationDate"),
    })
    elem.clear()

comments_df = pd.DataFrame(comments)

# --- Join comments with questions ---
comments_on_questions = comments_df[comments_df["PostId"].isin(questions["Id"])]

# Pick 5 questions that have comments
sample_questions = comments_on_questions.groupby("PostId").head(3)  # take up to 3 comments each
sample_qids = sample_questions["PostId"].unique()[:5]

for qid in sample_qids:
    qrow = questions.loc[questions["Id"] == qid].iloc[0]
    qcomments = comments_on_questions[comments_on_questions["PostId"] == qid]

    print("\n" + "="*80)
    print(f"Question ID: {qid}")
    print("Title:", qrow["Title"])
    print("---- Comments ----")
    for _, crow in qcomments.head(5).iterrows():
        print(f"- {crow['Text']}")
