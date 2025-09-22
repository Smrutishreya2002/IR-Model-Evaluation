import pandas as pd
from lxml import etree
from bs4 import BeautifulSoup

# -------------------------------
# Helper: clean HTML
# -------------------------------
def clean_html(raw_html):
    return BeautifulSoup(raw_html, "html.parser").get_text(" ") if raw_html else ""

# -------------------------------
# File path to Posts.xml
# -------------------------------
file_path = r"C:/Users/mohan/AppData/Local/Temp/d362f019-c0a3-4134-8b99-81dc8d3b1b0d_travel.stackexchange.com.7z.b0d/Posts.xml"
# -------------------------------
# Containers for parsed data
# -------------------------------
questions = []
answers = []

# -------------------------------
# Parse XML (streaming)
# -------------------------------
for _, elem in etree.iterparse(file_path, events=("end",), tag="row"):
    post_type = elem.get("PostTypeId")

    if post_type == "1":  # Question
        questions.append({
            "Id": elem.get("Id"),
            "Title": elem.get("Title") or "",
            "Body": clean_html(elem.get("Body")),
            "Tags": elem.get("Tags") or "",
            "AcceptedAnswerId": elem.get("AcceptedAnswerId") or "",
            "CreateDate": elem.get("CreationDate") or "",
            "Score": elem.get("Score") or "0",
            "OwnerUserId": elem.get("OwnerUserId") or ""
        })

    elif post_type == "2":  # Answer
        answers.append({
            "Id": elem.get("Id"),
            "ParentId": elem.get("ParentId") or "",
            "Body": clean_html(elem.get("Body")),
            "Score": elem.get("Score") or "0",
            "CreateDate": elem.get("CreationDate") or "",
            "OwnerUserId": elem.get("OwnerUserId") or ""
        })

    elem.clear()  # free memory

# -------------------------------
# Save to CSV with Pandas
# -------------------------------
pd.DataFrame(questions).to_csv("questions.csv", index=False)
pd.DataFrame(answers).to_csv("answers.csv", index=False)

print(f"✅ Saved {len(questions)} questions and {len(answers)} answers.")
