# Mini-Project 1: Credit Card Fraud Detection

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# --- IMPORTANT: Data Acquisition ---
# This script assumes you have downloaded the Credit Card Fraud Detection dataset from Kaggle.
# Download it from: https://www.kaggle.com/mlg-gza/creditcardfraud
# Place the 'creditcard.csv' file in the same directory as this script.
# -----------------------------------

def run_fraud_detection():
    try:
        df = pd.read_csv('creditcard.csv')
    except FileNotFoundError:
        print("Error: 'creditcard.csv' not found.")
        print("Please download the dataset from https://www.kaggle.com/mlg-gza/creditcardfraud")
        print("and place it in the same directory as 'fraud_detection.py'.")
        return

    print("Dataset loaded successfully. First 5 rows:")
    print(df.head())
    print("\nClass distribution:")
    print(df['Class'].value_counts())

    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']

    # Handle imbalanced data (Undersampling for demonstration)
    # This is a simple approach; more advanced techniques like SMOTE or ADASYN can be used.
    print("\nBalancing dataset (Undersampling)...")
    # Separate majority and minority classes
    df_majority = df[df.Class == 0]
    df_minority = df[df.Class == 1]

    # Undersample majority class
    df_majority_undersampled = df_majority.sample(n=len(df_minority), random_state=42)

    # Concatenate minority and undersampled majority
    df_balanced = pd.concat([df_majority_undersampled, df_minority])

    print("Balanced class distribution:")
    print(df_balanced['Class'].value_counts())

    X_balanced = df_balanced.drop('Class', axis=1)
    y_balanced = df_balanced['Class']

    # Split the balanced data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42, stratify=y_balanced)

    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining Logistic Regression model...")
    # Train a Logistic Regression model
    model = LogisticRegression(solver='liblinear', random_state=42)
    model.fit(X_train_scaled, y_train)

    # Make predictions
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    print("\n--- Model Evaluation ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Normal (0)', 'Fraud (1)'],
                yticklabels=['Normal (0)', 'Fraud (1)'])
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix for Fraud Detection')
    plt.tight_layout()
    plt.savefig('fraud_detection_confusion_matrix.png')
    plt.show()
    plt.close()

if __name__ == "__main__":
    run_fraud_detection()
