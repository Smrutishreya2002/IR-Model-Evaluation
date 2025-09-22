'''this code states the avg no of words  and sentences in question and answer.'''
import pandas as pd
from bs4 import BeautifulSoup
import nltk

# Download sentence tokenizer (only first time)
nltk.download("punkt")

# --- Helper functions ---
def clean_html(text):
    """Remove HTML tags and normalize whitespace."""
    if not isinstance(text, str):
        return ""
    text = BeautifulSoup(text, "html.parser").get_text(" ")
    return " ".join(text.split())

def count_words(text):
    return len(text.split())

def count_sentences(text):
    return len(nltk.sent_tokenize(text))

# --- Load datasets ---
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

# --- Prepare Question Text (Title + Body) ---
questions["FullText"] = (questions["Title"].fillna("") + " " +questions["Body"].fillna("")).apply(clean_html)

# --- Prepare Answer Text ---
answers["CleanBody"] = answers["Body"].fillna("").apply(clean_html)

# --- Word counts ---
questions["WordCount"] = questions["FullText"].apply(count_words)
answers["WordCount"] = answers["CleanBody"].apply(count_words)

# --- Sentence counts ---
questions["SentenceCount"] = questions["FullText"].apply(count_sentences)
answers["SentenceCount"] = answers["CleanBody"].apply(count_sentences)

# --- Compute averages ---
avg_q_words = questions["WordCount"].mean()
avg_q_sentences = questions["SentenceCount"].mean()
avg_a_words = answers["WordCount"].mean()
avg_a_sentences = answers["SentenceCount"].mean()

# --- Print results ---
print("Average Question Length:")
print(f"  Words: {avg_q_words:.2f}")
print(f"  Sentences: {avg_q_sentences:.2f}\n")

print("Average Answer Length:")
print(f"  Words: {avg_a_words:.2f}")
print(f"  Sentences: {avg_a_sentences:.2f}")
