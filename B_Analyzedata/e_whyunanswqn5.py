'''Manually or automatically check these questions with no answer and discuss what are the possible reasons that these questions are not answered. For example,can it be the case that questions with no answers are longer than those with answers?'''
import pandas as pd
from bs4 import BeautifulSoup
import nltk

nltk.download("punkt")

# --- Helpers ---
def clean_html(text):
    if not isinstance(text, str):
        return ""
    return " ".join(BeautifulSoup(text, "html.parser").get_text(" ").split())

def count_words(text):
    return len(text.split())

def count_sentences(text):
    return len(nltk.sent_tokenize(text))

# --- Load Data ---
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

# --- Count answers per question ---
answer_counts = answers.groupby("ParentId").size()
questions["AnswerCount"] = questions["Id"].map(answer_counts).fillna(0).astype(int)

# --- Separate answered vs unanswered ---
answered = questions[questions["AnswerCount"] > 0].copy()
unanswered = questions[questions["AnswerCount"] == 0].copy()

# --- Clean text ---
questions["FullText"] = (questions["Title"].fillna("") + " " + questions["Body"].fillna("")).apply(clean_html)

# --- Add word/sentence counts ---
questions["WordCount"] = questions["FullText"].apply(count_words)
questions["SentenceCount"] = questions["FullText"].apply(count_sentences)

answered["WordCount"] = questions.loc[answered.index, "WordCount"]
unanswered["WordCount"] = questions.loc[unanswered.index, "WordCount"]

answered["SentenceCount"] = questions.loc[answered.index, "SentenceCount"]
unanswered["SentenceCount"] = questions.loc[unanswered.index, "SentenceCount"]

# --- Compute averages ---
avg_words_answered = answered["WordCount"].mean()
avg_words_unanswered = unanswered["WordCount"].mean()

avg_sent_answered = answered["SentenceCount"].mean()
avg_sent_unanswered = unanswered["SentenceCount"].mean()

# --- Print results ---
print("Total Questions:", len(questions))
print("Answered Questions:", len(answered))
print("Unanswered Questions:", len(unanswered), "\n")

print("Average Word Count (Answered):", round(avg_words_answered, 2))
print("Average Word Count (Unanswered):", round(avg_words_unanswered, 2))

print("Average Sentence Count (Answered):", round(avg_sent_answered, 2))
print("Average Sentence Count (Unanswered):", round(avg_sent_unanswered, 2))
