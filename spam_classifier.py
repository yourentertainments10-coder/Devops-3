# spam_classifier.py

# ------------------------------
# Step 1: Import Libraries
# ------------------------------
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------------
# Step 2: Load Dataset
# (You can use "spam.csv" dataset from Kaggle/UCI)
# ------------------------------
data = pd.read_csv('data/spam.csv', encoding='latin-1')
data = data[['v1', 'v2']]  # Keep only label & text
data.columns = ['label', 'message']

# ------------------------------
# Step 3: Data Preprocessing
# ------------------------------
# Convert labels (ham=0, spam=1)
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    data['message'], data['label'], test_size=0.2, random_state=42
)

# Convert text to numerical vectors
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# ------------------------------
# Step 4: Model Training
# ------------------------------
model = MultinomialNB()
model.fit(X_train_vec, y_train)

# ------------------------------
# Step 5: Model Evaluation
# ------------------------------
y_pred = model.predict(X_test_vec)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# ------------------------------
# Step 6: User Input Prediction
# ------------------------------
while True:
    user_input = input("Enter a message (or type 'exit' to quit): ")
    if user_input.lower() == "exit":
        break
    user_vec = vectorizer.transform([user_input])
    prediction = model.predict(user_vec)
    print("Prediction:", "Spam 🚨" if prediction[0] == 1 else "Not Spam ✅")
