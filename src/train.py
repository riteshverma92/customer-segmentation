from src.preprocessing import load_and_preprocess
from src.clustering import train_kmeans
from src.classification import train_decision_tree
import pandas as pd
import os

# Ensure necessary folders exist
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# 1. Load and preprocess
df = load_and_preprocess("data/raw/Customer Data.csv")

# 2. Train KMeans
kmeans_pipeline = train_kmeans(df)

# 3. Add cluster labels to dataframe
clusters = kmeans_pipeline.predict(df)
df["Cluster"] = clusters

# Save processed data
processed_path = "data/processed/customer_processed.csv"
df.to_csv(processed_path, index=False)
print(f"✅ Processed data saved at {processed_path}")

# 4. Train Decision Tree on processed data
train_decision_tree(df)
