import logging
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from joblib import dump
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_kmeans(df, n_clusters=4, save_path="models/kmeans_pipeline.joblib", report_folder="reports"):
    """
    Train KMeans with scaling and PCA for visualization.
    """
    os.makedirs(report_folder, exist_ok=True)

    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=2)),
        ("kmeans", KMeans(n_clusters=n_clusters, random_state=42))
    ])

    pipeline.fit(df)
    clusters = pipeline.predict(df)
    logging.info("KMeans trained with %d clusters", n_clusters)

    # Metrics
    inertia = pipeline.named_steps['kmeans'].inertia_
    silhouette = silhouette_score(df, clusters)
    calinski = calinski_harabasz_score(df, clusters)
    davies = davies_bouldin_score(df, clusters)

    metrics = pd.DataFrame({
        "Metric": ["Inertia", "Silhouette", "Calinski-Harabasz", "Davies-Bouldin"],
        "Score": [inertia, silhouette, calinski, davies]
    })
    metrics.to_csv(os.path.join(report_folder, "kmeans_metrics.csv"), index=False)

    # PCA scatter plot
    pca_df = pd.DataFrame(pipeline.named_steps['pca'].transform(df), columns=["PC1", "PC2"])
    pca_df["Cluster"] = clusters
    plt.figure(figsize=(8,6))
    sns.scatterplot(x="PC1", y="PC2", hue="Cluster", palette="Set2", data=pca_df, s=60)
    plt.title("KMeans Clusters (PCA Reduced)")
    plt.savefig(os.path.join(report_folder, "kmeans_clusters.png"))
    plt.close()

    # Elbow plot
    distortions = []
    for k in range(1, 10):
        km = KMeans(n_clusters=k, random_state=42)
        km.fit(df)
        distortions.append(km.inertia_)
    plt.figure(figsize=(8,6))
    plt.plot(range(1,10), distortions, 'bo-')
    plt.xlabel("Clusters")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(os.path.join(report_folder, "kmeans_elbow.png"))
    plt.close()

    # Save model
    dump(pipeline, save_path)
    logging.info("KMeans model saved at %s", save_path)

    return pipeline
