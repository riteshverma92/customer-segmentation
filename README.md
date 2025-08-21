## Credit Card Customer Segmentation & Classification 💳

# Project Overview
This project focuses on a comprehensive analysis of credit card usage data to perform customer segmentation and classification. Using unsupervised learning techniques, specifically K-Means Clustering, customers are grouped into distinct segments based on their spending behaviors. Subsequently, a Decision Tree Classifier is trained to predict the cluster a new customer would belong to, providing a powerful tool for targeted marketing and strategic financial planning.
# 🎯 Objectives
Unsupervised Segmentation: Group credit card customers into meaningful clusters based on their transaction and behavioral data.

Behavioral Analysis: Analyze and characterize each customer segment to understand their unique spending habits and financial profiles.

Supervised Classification: Build a predictive model to classify new customers into one of the identified segments.

Actionable Insights: Generate insights that can be utilized by marketing and finance teams to optimize strategies, enhance customer service, and reduce churn.
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
