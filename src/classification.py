import logging
import pandas as pd
from typing import Union
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    accuracy_score, precision_score, recall_score, f1_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import os

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_decision_tree(
    df: pd.DataFrame,
    target_col: str = "Cluster",
    save_path: Union[str, bytes] = "models/decision_tree.joblib",
    report_folder: str = "reports"
) -> Pipeline:
    os.makedirs(report_folder, exist_ok=True)

    X = df.drop(columns=[target_col])
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("decision_tree", DecisionTreeClassifier(criterion="entropy", random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    logging.info("Decision Tree trained on %d samples", len(X_train))

    # Evaluate
    y_pred = pipeline.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")

    # Save metrics CSV
    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "Precision", "Recall", "F1-Score"],
        "Score": [acc, prec, rec, f1]
    })
    metrics_df.to_csv(os.path.join(report_folder, "classification_metrics.csv"), index=False)

    # Save classification report
    with open(os.path.join(report_folder, "classification_report.txt"), "w") as f:
        f.write(cr)

    # Save confusion matrix plot
    plt.figure(figsize=(8,6))
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
