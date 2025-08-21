import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_decision_tree(df, target_col="Cluster", save_path="models/decision_tree.joblib", report_folder="reports"):
    """
    Train a Decision Tree on the dataset, save model, metrics, and plots.
    """
    os.makedirs(report_folder, exist_ok=True)

    # Prepare features and target
    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Pipeline with scaling + decision tree
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", DecisionTreeClassifier(criterion="entropy", random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    logging.info("Decision Tree trained on %d samples", len(X_train))

    # Predictions and evaluation
    y_pred = pipeline.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Save metrics
    metrics = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1"],
        "Score": [acc, prec, rec, f1]
    })
    metrics.to_csv(os.path.join(report_folder, "classification_metrics.csv"), index=False)

    # Save classification report
    with open(os.path.join(report_folder, "classification_report.txt"), "w") as f:
        f.write(classification_report(y_test, y_pred))

    # Confusion matrix plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(report_folder, "confusion_matrix.png"))
    plt.close()

    # Save model
    dump(pipeline, save_path)
    logging.info("Model saved at %s", save_path)

    return pipeline
