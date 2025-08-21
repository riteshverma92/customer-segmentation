# 🧑‍🤝‍🧑 Customer Segmentation & Classification

This project applies **KMeans Clustering** to segment customers based on their purchasing behavior and then builds
a **Decision Tree Classifier** to predict which cluster a new customer belongs to.  

It is useful for:
- 📈 Marketing strategies  
- 🎯 Targeted campaigns  
- 🛒 Customer behavior insights  

## 📂 Project Structure
customer-segmentation/
│
├── data/
│ ├── raw/ # Original dataset
│ ├── processed/ # Preprocessed dataset with clusters
│
├── models/ # Saved KMeans + Decision Tree models
├── reports/ # Evaluation reports & confusion matrix
│
├── src/ # Source code
│ ├── preprocessing.py # Data cleaning & preprocessing
│ ├── clustering.py # KMeans clustering pipeline
│ ├── classification.py # Decision Tree training
│ ├── evaluation.py # Evaluation of clustering & classification
│ ├── predict.py # Make predictions on new data
│ ├── train.py # Main pipeline (runs preprocessing + training)
│
├── notebooks/ # Jupyter notebooks (optional exploration)
│
├── README.md # Project documentation
└── requirements.txt # Python dependencies



---

## 📊 Dataset

The dataset contains customer transactions and behavior patterns.  

**Sample Columns:**
- `BALANCE` → Balance amount left in the account  
- `PURCHASES` → Total purchase amount  
- `ONEOFF_PURCHASES` → One-time purchases  
- `CASH_ADVANCE` → Cash advance amount  
- `PURCHASES_FREQUENCY` → Frequency of purchases  
- `CREDIT_LIMIT` → Credit limit for the customer  
- `PAYMENTS` → Payment amount  
- `MINIMUM_PAYMENTS` → Minimum payments made  
- `TENURE` → Customer tenure with the bank  
- `Cluster` → Assigned cluster label  

---

## 🚀 Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation

2. Create Virtual Environment
python -m venv venv

3. Activate Environment
Windows
venv\Scripts\activate

Mac/Linux
source venv/bin/activate

4. Install Dependencies
pip install -r requirements.txt

🏃 Running the Project
1. Train Models
python -m src.train

2. Evaluate Models
python -m src.evaluation

📌 Outputs:

reports/evaluation_report.txt → Accuracy, precision, recall, F1
reports/confusion_matrix.png → Confusion matrix
reports/kmeans_clusters.png → PCA cluster visualization

3. Make Predictions
Predict new customer clusters:
python -m src.predict
