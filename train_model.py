import pandas as pd
import re
import pickle
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load datasets
true_news = pd.read_csv("dataset/True.csv")
fake_news = pd.read_csv("dataset/Fake.csv")

# Add labels (1 = Fake, 0 = Real)
true_news['label'] = 0
fake_news['label'] = 1

# Merge both datasets
df = pd.concat([true_news, fake_news], ignore_index=True)

# Select necessary columns
df = df[['text', 'label']]

# Text cleaning function
def clean_text(text):
    text = text.lower()
    text = re.sub(r'\W', ' ', text)  # Remove special characters
    text = re.sub(r'\s+', ' ', text)  # Remove extra spaces
    words = text.split()
    words = [word for word in words if word not in stop_words]  # Remove stopwords
    return ' '.join(words)

df['text'] = df['text'].apply(clean_text)

# Convert text into numerical representation
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df['text'])

# Labels
y = df['label']

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the model and vectorizer
pickle.dump(vectorizer, open("tfidf_vectorizer.pkl", "wb"))
pickle.dump(model, open("random_forest.pkl", "wb"))

print("Model training complete and saved successfully!")
