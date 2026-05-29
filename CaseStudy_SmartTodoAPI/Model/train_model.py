"""
Phase 1 - Step 2: Train ML Model
=================================
This script reads the CSV dataset, trains a text classification model
using scikit-learn, and saves it to disk for use in the Flask API.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib
import os

# ─────────────────────────────────────────
# 1. Load Dataset
# ─────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "../dataset/tasks.csv")
df = pd.read_csv(DATA_PATH)

print(f"✅ Loaded {len(df)} tasks")
print(df["Priority"].value_counts())
print()

X = df["Task"]
y = df["Priority"]

# ─────────────────────────────────────────
# 2. Train / Test Split
# ─────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ─────────────────────────────────────────
# 3. Build Pipeline: TF-IDF + Logistic Regression
# ─────────────────────────────────────────
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(
        ngram_range=(1, 2),   # unigrams + bigrams
        max_features=5000,
        stop_words="english"
    )),
    ("clf", LogisticRegression(
        max_iter=1000,
        C=1.0,
        solver="lbfgs",
        multi_class="multinomial"
    ))
])

pipeline.fit(X_train, y_train)

# ─────────────────────────────────────────
# 4. Evaluate
# ─────────────────────────────────────────
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"✅ Model Accuracy: {accuracy:.2%}")
print()
print("Classification Report:")
print(classification_report(y_test, y_pred))

if accuracy < 0.70:
    print("⚠️  WARNING: Accuracy below 70%. Consider adding more data or tuning.")
else:
    print("✅ Accuracy meets the ≥70% requirement.")

# ─────────────────────────────────────────
# 5. Save Model
# ─────────────────────────────────────────
MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pkl")
joblib.dump(pipeline, MODEL_PATH)
print(f"\n✅ Model saved to: {MODEL_PATH}")
