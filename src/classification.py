import logging
import pandas as pd
from typing import Union
from joblib import dump
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix

# Configure logging once for the module
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_decision_tree(
    df: pd.DataFrame,
    target_col: str = "Cluster",
    save_path: Union[str, bytes] = "models/decision_tree.joblib"
) -> Pipeline:
    """
    Train a Decision Tree classifier with scaling and save the pipeline.

    Parameters
    ----------
    df : pd.DataFrame
        Dataset containing features and target.
    target_col : str, optional
        Name of the target column. Default is "Cluster".
    save_path : str, optional
        Where to save the trained pipeline. Default is "models/decision_tree.joblib".

    Returns
    -------
    Pipeline
        Trained pipeline with StandardScaler and DecisionTreeClassifier.
    """
    # Split features and labels
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
    logging.info("Confusion Matrix:\n%s", confusion_matrix(y_test, y_pred))
    logging.info("Classification Report:\n%s", classification_report(y_test, y_pred))

    dump(pipeline, save_path)
    logging.info("Model saved at %s", save_path)

    return pipeline
