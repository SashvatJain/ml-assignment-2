import os
import joblib
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

def main():
    # 1. Load Dataset (UCI Breast Cancer - 30 features, 569 instances)
    data = load_breast_cancer(as_frame=True)
    df = data.frame
    X = df.drop(columns=['target'])
    y = df['target']

    # 2. Train-Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Save test dataset as CSV for Streamlit web app evaluation
    test_df = X_test.copy()
    test_df['target'] = y_test
    test_df.to_csv('test_data.csv', index=False)
    print("test_data.csv generated successfully.")

    # 3. Scaling features for distance and gradient-based models
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Directory for model storage
    os.makedirs('models', exist_ok=True)
    joblib.dump(scaler, 'models/scaler.pkl')

    # 4. Define 6 Classification Models
    models = {
        "Logistic Regression": (LogisticRegression(random_state=42, max_iter=1000), True),
        "Decision Tree": (DecisionTreeClassifier(random_state=42), False),
        "kNN": (KNeighborsClassifier(n_neighbors=5), True),
        "Naive Bayes": (GaussianNB(), True),
        "Random Forest": (RandomForestClassifier(n_estimators=100, random_state=42), False),
        "Gradient Boosting": (GradientBoostingClassifier(random_state=42), False)
    }

    results = []

    # 5. Model Training & Evaluation
    for name, (model, needs_scaling) in models.items():
        X_tr = X_train_scaled if needs_scaling else X_train
        X_te = X_test_scaled if needs_scaling else X_test

        # Train model
        model.fit(X_tr, y_train)

        # Predictions
        y_pred = model.predict(X_te)
        y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred

        # Metrics Calculation
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        results.append({
            "ML Model Name": name,
            "Accuracy": round(acc, 4),
            "AUC": round(auc, 4),
            "Precision": round(prec, 4),
            "Recall": round(rec, 4),
            "F1": round(f1, 4),
            "MCC": round(mcc, 4)
        })

        # Save model file
        filename = f"models/{name.lower().replace(' ', '_')}.pkl"
        joblib.dump(model, filename)

    # 6. Output Summary Table
    results_df = pd.DataFrame(results)
    print("\n================ EVALUATION METRICS COMPARISON ================\n")
    print(results_df.to_markdown(index=False))

if __name__ == "__main__":
    main()