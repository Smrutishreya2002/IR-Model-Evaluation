import pandas as pd
from bs4 import BeautifulSoup
import nltk
import textstat

# --- Helpers ---
def clean_html(text):
    if not isinstance(text, str):
        return ""
    return " ".join(BeautifulSoup(text, "html.parser").get_text(" ").split())

# --- Load data ---
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

# --- Count answers per question ---
answer_counts = answers.groupby("ParentId").size()
questions["AnswerCount"] = questions["Id"].map(answer_counts).fillna(0).astype(int)

# --- Prepare clean text ---
questions["FullText"] = (questions["Title"].fillna("") + " " + questions["Body"].fillna("")).apply(clean_html)

# --- Compute readability (Flesch Reading Ease) ---
questions["Readability"] = questions["FullText"].apply(lambda x: textstat.flesch_reading_ease(x) if x else 0)

# --- Mark answered/unanswered ---
questions["HasAnswer"] = (questions["AnswerCount"] > 0).astype(int)

# --- Compare averages ---
avg_read_answered = questions[questions["HasAnswer"] == 1]["Readability"].mean()
avg_read_unanswered = questions[questions["HasAnswer"] == 0]["Readability"].mean()

print("Average readability (answered questions):", round(avg_read_answered, 2))
print("Average readability (unanswered questions):", round(avg_read_unanswered, 2))

# --- Correlation ---
corr = questions["HasAnswer"].corr(questions["Readability"])
print("Correlation between readability and receiving an answer:", round(corr, 3))
