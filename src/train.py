from src.preprocessing import load_and_preprocess
from src.clustering import train_kmeans
from src.classification import train_decision_tree

# 1. Load and preprocess
df = load_and_preprocess("data/raw/Customer Data.csv")

# 2. Train KMeans
kmeans_pipeline = train_kmeans(df)

# 3. Add cluster labels to dataframe
clusters = kmeans_pipeline.predict(df)
df["Cluster"] = clusters

# 4. Train Decision Tree
train_decision_tree(df)
