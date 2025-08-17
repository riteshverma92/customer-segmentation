from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import dump
import logging
import pandas as pd
from typing import Union

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    save_path: Union[str, bytes] = "models/kmeans_pipeline.joblib"
) -> Pipeline:
    """
    Train a KMeans clustering pipeline with scaling and PCA, then save it.

    The pipeline consists of:
    1. StandardScaler - standardizes features
    2. PCA - reduces dimensionality to 2 components
    3. KMeans - clustering algorithm

    Parameters
    ----------
    df : pd.DataFrame
        Input dataset with numerical features.
    n_clusters : int, optional
        Number of clusters for KMeans, by default 4.
    save_path : str or bytes, optional
        File path to save the trained pipeline, by default "models/kmeans_pipeline.joblib".

    Returns
    -------
    Pipeline
        Trained scikit-learn Pipeline containing scaler, PCA, and KMeans.
    """
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
    ])

    pipeline.fit(df)
    dump(pipeline, save_path)
    logging.info("KMeans pipeline trained and saved to %s", save_path)

    return pipeline
