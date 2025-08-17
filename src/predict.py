from src.preprocessing import load_and_preprocess
from joblib import load

# Load and preprocess
df_new = load_and_preprocess("data/raw/Customer Data.csv")

# Load KMeans model
kmeans_pipeline = load("models/kmeans_pipeline.joblib")
df_new["Cluster"] = kmeans_pipeline.predict(df_new)

# Load Decision Tree model
dt_pipeline = load("models/decision_tree.joblib")
predictions = dt_pipeline.predict(df_new.drop(columns=["Cluster"]))

df_new["Predicted_Class"] = predictions

print(df_new.head())
