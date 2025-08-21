from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from joblib import dump
import logging
import pandas as pd
from typing import Union
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score
import matplotlib.pyplot as plt
import seaborn as sns
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

def train_kmeans(
    df: pd.DataFrame,
    n_clusters: int = 4,
    save_path: Union[str, bytes] = "models/kmeans_pipeline.joblib",
    report_folder: str = "reports"
) -> Pipeline:
    os.makedirs(report_folder, exist_ok=True)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
    ])

    pipeline.fit(df)
    clusters = pipeline.predict(df)
    logging.info("KMeans pipeline trained")

    # Clustering metrics
    inertia = pipeline.named_steps['kmeans'].inertia_
    silhouette = silhouette_score(df, clusters)
    calinski = calinski_harabasz_score(df, clusters)
    davies = davies_bouldin_score(df, clusters)

    metrics_df = pd.DataFrame({
        "Metric": ["Inertia", "Silhouette Score", "Calinski-Harabasz Index", "Davies-Bouldin Index"],
        "Score": [inertia, silhouette, calinski, davies]
    })
    metrics_df.to_csv(os.path.join(report_folder, "kmeans_metrics.csv"), index=False)

    # PCA Scatter plot of clusters
    pca_df = pd.DataFrame(pipeline.named_steps['pca'].transform(df), columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="Set2", data=pca_df, s=60)
    plt.title("KMeans Clustering (PCA Reduced)")
    plt.savefig(os.path.join(report_folder, "kmeans_clusters.png"))
    plt.close()

    # Optional: Elbow method plot
    distortions = []
    K = range(1, 10)
    for k in K:
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df)
        distortions.append(km.inertia_)
    plt.figure(figsize=(8,6))
    plt.plot(K, distortions, 'bo-')
    plt.xlabel("Number of clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method for KMeans")
    plt.savefig(os.path.join(report_folder, "kmeans_elbow.png"))
    plt.close()

    dump(pipeline, save_path)
    logging.info("KMeans model saved at %s", save_path)

    return pipeline
