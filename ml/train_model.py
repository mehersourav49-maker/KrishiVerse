# train_model.py

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib
import os

# Path to dataset
CSV_PATH = os.path.join('data', 'crop_recommendation.csv')

print("Loading dataset from:", CSV_PATH)
df = pd.read_csv(CSV_PATH)

# Check if dataset has the required columns
expected = ['N','P','K','temperature','humidity','ph','rainfall','label']
for col in expected:
    if col not in df.columns:
        raise SystemExit(f"Missing column: {col}")

# Split into inputs (X) and target (y)
X = df[['N','P','K','temperature','humidity','ph','rainfall']]
y = df['label']

# Encode crop names (text → numbers)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split into training & testing data
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Create Random Forest model
clf = RandomForestClassifier(n_estimators=150, random_state=42)
clf.fit(X_train, y_train)

# Test accuracy
accuracy = clf.score(X_test, y_test)
print(f"Test accuracy: {accuracy:.3f}")

# Save model + label encoder
joblib.dump({'model': clf, 'label_encoder': le}, 'model.joblib')
print("✅ Model saved as model.joblib")
