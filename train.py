import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import resample

# ✅ Load Dataset
file_path = "Election.csv"
df = pd.read_csv(file_path)

# ✅ Clean column names (Remove newlines & extra spaces)
df.columns = df.columns.str.replace(r'[\n\s]+', '_', regex=True)
print("Updated Column Names:", df.columns)  # Debugging

# ✅ Fill missing values
df[['SYMBOL', 'GENDER', 'CATEGORY', 'EDUCATION']] = df[['SYMBOL', 'GENDER', 'CATEGORY', 'EDUCATION']].fillna('Unknown')
numerical_cols = ['AGE', 'TOTAL_VOTES', 'GENERAL_VOTES', 'POSTAL_VOTES', 'TOTAL_ELECTORS']
df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())

# ✅ Convert 'CRIMINAL_CASES' to numeric
df['CRIMINAL_CASES'] = pd.to_numeric(df['CRIMINAL_CASES'], errors='coerce').fillna(0).astype(int)

# ✅ Convert ASSETS & LIABILITIES to numeric values
def convert_currency(value):
    if isinstance(value, str):
        value = value.replace('Rs', '').replace(',', '').strip()
        if 'Crore' in value:
            return float(value.split()[0]) * 10**7  # Convert Crore to absolute value
        elif 'Lacs' in value:
            return float(value.split()[0]) * 10**5  # Convert Lacs to absolute value
    return 0

df['ASSETS'] = df['ASSETS'].apply(convert_currency)
df['LIABILITIES'] = df['LIABILITIES'].apply(convert_currency)

# ✅ Encode categorical variables
label_encoders = {}
for col in ['CATEGORY', 'EDUCATION', 'GENDER', 'PARTY']:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Store encoders for later use

# ✅ Balance Data (Fix always predicting "Won")
df_winners = df[df['WINNER'] == 1]  # Candidates who won
df_losers = df[df['WINNER'] == 0]   # Candidates who lost

# Make sure both classes have equal data
if len(df_winners) > len(df_losers):
    df_winners = resample(df_winners, replace=False, n_samples=len(df_losers), random_state=42)
else:
    df_losers = resample(df_losers, replace=False, n_samples=len(df_winners), random_state=42)

# Combine the balanced dataset
df_balanced = pd.concat([df_winners, df_losers])

# ✅ Select relevant features
features = ['AGE', 'TOTAL_VOTES', 'GENERAL_VOTES', 'POSTAL_VOTES', 'TOTAL_ELECTORS',
            'CRIMINAL_CASES', 'ASSETS', 'LIABILITIES', 'EDUCATION', 'CATEGORY', 'GENDER', 'PARTY']
X = df_balanced[features]
y = df_balanced['WINNER']

# ✅ Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# ✅ Train Model (Limit complexity to prevent overfitting)
rf_model = RandomForestClassifier(n_estimators=50, max_depth=5, random_state=42)
rf_model.fit(X_train, y_train)

# ✅ Evaluate Model
y_pred = rf_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

# ✅ Check if model predicts both "Won" & "Lost" (Confusion Matrix)
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# ✅ Feature Importance Check
importances = rf_model.feature_importances_
feature_names = X.columns

plt.figure(figsize=(10, 6))
plt.barh(feature_names, importances)
plt.xlabel("Feature Importance Score")
plt.ylabel("Features")
plt.title("Feature Importance in Election Prediction")
plt.show()

# ✅ Save Model
joblib.dump(rf_model, "model.pkl")
print("Model saved as model.pkl")
