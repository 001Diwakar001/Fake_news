from flask import Flask, render_template, request
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.feature_extraction.text import TfidfVectorizer
import torch
import joblib

app = Flask(__name__)

# Load ML models
rf_model = joblib.load("random_forest.pkl")
vectorizer = joblib.load("tfidf_vectorizer.pkl")

# Load BERT model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bert_model = BertForSequenceClassification.from_pretrained(
    "backend/bert_model", local_files_only=True
).to(device)

tokenizer = BertTokenizer.from_pretrained("backend/bert_model", local_files_only=True)

@app.route("/")
def home():
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)
