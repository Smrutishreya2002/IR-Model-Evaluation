'''From your questions.csv and answers.csv:
Average #answers per question
For each question, count how many answers it has (by matching answers.ParentId to questions.Id).ake the average.
Count questions with no answer
Questions where no answer exists in answers.csv.
Count questions with an accepted answer
Questions where the column AcceptedAnswerId is not empty/NaN.
'''
import pandas as pd

# Load cleaned data
questions = pd.read_csv("questions.csv")
answers = pd.read_csv("answers.csv")

# --- Count answers per question ---
# Group answers by ParentId (which links to Question Id)
answer_counts = answers.groupby("ParentId").size()

# Add this info back into questions DataFrame
questions["AnswerCount"] = questions["Id"].map(answer_counts).fillna(0).astype(int)

# --- 1. Average answers per question ---
avg_answers = questions["AnswerCount"].mean()

# --- 2. Questions with no answers ---
no_answer_count = (questions["AnswerCount"] == 0).sum()

# --- 3. Questions with an accepted answer ---
accepted_count = questions["AcceptedAnswerId"].notna().sum()

# --- Print results ---
print(f"Average number of answers per question: {avg_answers:.2f}")
print(f"Number of questions with no answers: {no_answer_count}")
print(f"Number of questions with an accepted answer: {accepted_count}")
