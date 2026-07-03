import streamlit as st
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

data = {
    "Question": [
        "What is AI?",
        "What is Machine Learning?",
        "What is Deep Learning?",
        "What is NLP?",
        "What is Python?",
        "What is Data Science?",
        "What is Supervised Learning?",
        "What is Unsupervised Learning?",
        "What is TensorFlow?",
        "What is Scikit-learn?"
    ],

    "Answer": [
        "Artificial Intelligence is the simulation of human intelligence by machines.",
        "Machine Learning is a subset of AI that enables systems to learn from data.",
        "Deep Learning is a subset of Machine Learning based on neural networks.",
        "Natural Language Processing enables computers to understand human language.",
        "Python is a popular programming language used in AI and Data Science.",
        "Data Science is the process of extracting insights from data.",
        "Supervised Learning uses labeled data for training.",
        "Unsupervised Learning works with unlabeled data.",
        "TensorFlow is an open-source deep learning framework.",
        "Scikit-learn is a machine learning library for Python."
    ]
}

df = pd.DataFrame(data)

vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(df["Question"])

def chatbot(user_question):
    user_vector = vectorizer.transform([user_question])
    similarity_scores = cosine_similarity(user_vector, tfidf_matrix)
    index = similarity_scores.argmax()
    return df.iloc[index]["Answer"]

st.title("🤖 FAQ Chatbot")

question = st.text_input("Ask a Question")

if st.button("Get Answer"):
    if question:
        answer = chatbot(question)
        st.success(answer)
    else:
        st.warning("Please enter a question.")
