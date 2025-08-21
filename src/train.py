import os
from src.preprocessing import load_and_preprocess
from src.clustering import train_kmeans
from src.classification import train_decision_tree

# Make folders if missing
os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("reports", exist_ok=True)

# Load and preprocess data
df = load_and_preprocess("data/raw/Customer Data.csv")

# Train KMeans and add clusters
kmeans = train_kmeans(df)
df["Cluster"] = kmeans.predict(df)

# Save processed CSV
processed_file = "data/processed/customer_processed.csv"
df.to_csv(processed_file, index=False)
print(f"Processed data saved at {processed_file}")

# Train decision tree
train_decision_tree(df)
