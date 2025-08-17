import matplotlib.pyplot as plt
import seaborn as sns

def plot_heatmap(df):
    """Plot a heatmap of feature correlations."""
    plt.figure(figsize=(12, 12))
    sns.heatmap(df.corr(), annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Feature Correlation Heatmap")
    plt.show()


def plot_clusters(pca_df, clusters):
    """Plot PCA-reduced data with cluster labels."""
    plt.figure(figsize=(8, 8))
    sns.scatterplot(
        x="PCA1",
        y="PCA2",
        hue=clusters,
        palette="viridis",
        data=pca_df,
        s=100,
        edgecolor="k"
    )
    plt.title("KMeans Clusters (PCA-Reduced Data)")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Cluster")
    plt.show()
