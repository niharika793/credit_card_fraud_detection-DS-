import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
data = pd.read_csv('creditcard.csv')

# Checking class distribution
print("Original Class Distribution:\n", data['Class'].value_counts())

# Splitting features and target variable
X = data.drop(columns=['Class'])
y = data['Class']

# Splitting dataset before scaling (Prevents Data Leakage)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Handling Class Imbalance using SMOTE (After train-test split)
smote = SMOTE(sampling_strategy='auto', random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# Feature Scaling (Standardization)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_resampled)
X_test_scaled = scaler.transform(X_test)

# RandomForest Classifier with Regularization to prevent overfitting
model = RandomForestClassifier(
    n_estimators=100,
    max_depth=10,  # Limits depth to avoid overfitting
    min_samples_split=10,  # Prevents overly complex trees
    min_samples_leaf=5,  # Ensures nodes have at least 5 samples
    random_state=42
)
model.fit(X_train_scaled, y_train_resampled)

# Predictions
y_pred = model.predict(X_test_scaled)

# Evaluation Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)

# Print Results
print("\n Model Performance Metrics:")
print(f" Accuracy: {accuracy:.4f}")
print(f" Precision: {precision:.4f}")
print(f" Recall: {recall:.4f}")
print(f" F1 Score: {f1:.4f}")
print(f" ROC-AUC Score: {roc_auc:.4f}")

# Classification Report
print("\n Classification Report:")
print(classification_report(y_test, y_pred))

# Check for Overfitting
y_train_pred = model.predict(X_train_scaled)
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
print(f"\n Overfitting Check - Training Accuracy: {train_accuracy:.4f}")

# Plot Class Distribution Before and After SMOTE
plt.figure(figsize=(8,4))
sns.barplot(x=['Legit', 'Fraud'], 
            y=[sum(y == 0), sum(y == 1)], 
            hue=['Legit', 'Fraud'],  
            palette=['green', 'red'])
plt.legend([], [], frameon=False)
plt.title("Class Distribution Before Resampling")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()

plt.figure(figsize=(8,4))
sns.barplot(x=['Legit (Resampled)', 'Fraud (Resampled)'], 
            y=[sum(y_train_resampled == 0), sum(y_train_resampled == 1)], 
            hue=['Legit (Resampled)', 'Fraud (Resampled)'],  
            palette=['green', 'red'])
plt.legend([], [], frameon=False)
plt.title("Class Distribution After SMOTE")
plt.xlabel("Class")
plt.ylabel("Count")
plt.show()
