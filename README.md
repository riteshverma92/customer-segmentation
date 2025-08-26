# Credit Card Customer Segmentation & Classification
<img width="650" height="320" alt="7c2b3102-b703-4e90-ac02-2ca6fe5461c1" src="https://github.com/user-attachments/assets/4c320060-b936-4026-8103-9b0925cc0e30" />

 ## 📑 Table of Contents  

- [Project Overview](#-project-overview)   
- [Tech Stack](#-tech-stack)  
- [Dataset](#-dataset)  
- [Setup Instructions](#-setup-instructions)  
  - [1. Clone the Repository](#1-clone-the-repository)  
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)  
  - [3. Activate the Virtual Environment](#3-activate-the-virtual-environment)  
  - [4. Install Dependencies](#4-install-dependencies)  
- [Running the Project](#-running-the-project)  
  - [1. Train Models](#1-train-models)  
  - [2. Predict Models](#2-predict-models)
- [Project Flow](#-project-flow)
- [Results & Insights](#-results--insights)
- [Use Cases](#-Use-Cases)
<p align="right">
  <a href="#credit-card-customer-segmentation--classification">
    🔝 Back to Top
  </a>
</p>


---

## 📌 Project Overview
Project applies a **Machine Learning pipeline** on credit card usage data to perform customer segmentation and classification. Customers are first grouped into segments using **K-Means Clustering**, and then a **Decision Tree Classifier** predicts the segment of new customers, supporting targeted marketing and strategic financial planning.

## 🛠 Tech Stack  
This project leverages a modern data science and machine learning stack to ensure efficient data processing, robust modeling, and professional visualization.  

### Languages & Core Libraries  
- ![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python) Primary programming language for analysis and model development.  
- ![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-lightgrey?logo=numpy) High-performance numerical computations.  
- ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?logo=pandas) Data manipulation and preprocessing.  

### Machine Learning & Modeling  
- ![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-green?logo=scikitlearn) Algorithms and evaluation (K-Means, DBSCAN, Decision Tree Classifier).  

### Visualization  
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?logo=plotly) Core data visualization.  
- ![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-9cf) Statistical and exploratory data analysis.  

### Development & Collaboration  
- ![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter) Interactive development and experimentation.  
- ![venv](https://img.shields.io/badge/venv-Virtual%20Environment-yellow) Isolated project environment for dependencies.  
- ![GitHub](https://img.shields.io/badge/GitHub-Version%20Control-black?logo=github) Version control and collaboration platform.
  <p align="right">[🔝 Back to Top](#top)</p>
---
## 📂 Repository Structure
```
CUSTOMER-SEGMENTATION/
├── data/
│   ├── raw/
│   │   └── Customer Data.csv
│   └── processed/
│       └── customer_processed.csv
│
├── models/
│   ├── decision_tree.joblib
│   └── kmeans_pipeline.joblib
│
├── reports/
│   ├── classification_metrics.csv
│   ├── classification_report.txt
│   ├── confusion_matrix.png
│   ├── distribution_of_data_for_missing_Value.png
│   ├── kmeans_clusters.png
│   └── kmeans_metrics.csv
│
├── src/
│   ├── __pycache__/
│   ├── __init__.py
│   ├── clustering.py
│   ├── detect.py
│   ├── preprocessing.py
│   └── train.py
│
├── Project_Report.txt
├── README.md
└── requirements.txt

```
<p align="right">[🔝 Back to Top](#top)</p>
---
## 📊 Dataset  

The dataset provides comprehensive information on **credit card customers**, capturing their **transaction patterns, spending frequency, cash advances, credit utilization, payment behavior, and tenure with the bank**. It includes features such as purchase amounts, installment purchases, cash advance transactions, credit limits, and repayment history, making it well-suited for **customer segmentation** and **predictive modeling**.  

📂 **Dataset Access:** [Credit Card Customer Segmentation – Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)  
# Preview:
<img width="1767" height="341" alt="Screenshot 2025-08-26 171500" src="https://github.com/user-attachments/assets/28a5ecbe-d6ed-4ca0-96b1-dafa86d09714" />

### 🔑 Key Categories & Features  

- **👤 Customer Identity**  
  - `CUST_ID`: Unique customer identifier  

- **💳 Transaction Behavior**  
  - `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `PURCHASES_TRX`  
  - `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`  

- **🏦 Credit & Cash Advances**  
  - `BALANCE`, `BALANCE_FREQUENCY`, `CREDIT_LIMIT`  
  - `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`  

- **📈 Repayment & Financial Discipline**  
  - `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`  

- **⏳ Customer Lifecycle**  
  - `TENURE`: Length of customer’s relationship with the bank  
<p align="right">[🔝 Back to Top](#top)</p>
---

## 🚀 Setup Instructions  

Follow the steps below to set up and run the project locally:  

### 1. Clone the Repository  
```bash
git clone https://github.com/your-username/customer-segmentation.git
cd customer-segmentation
```  

### 2. Create a Virtual Environment  
```bash
python -m venv venv
```  

### 3. Activate the Virtual Environment  
- **Windows**  
  ```bash
  venv\Scripts\activate
  ```  
- **Mac/Linux**  
  ```bash
  source venv/bin/activate
  ```  

### 4. Install Dependencies  
```bash
pip install -r requirements.txt
```  

---
<p align="right">[🔝 Back to Top](#top)</p>
## 🏃 Running the Project  

### 1. Train Models  
```bash
python -m src.train
```  

### 2. Predict Models  
```bash
python -m src.predict

```
<p align="right">[🔝 Back to Top](#top)</p>
---

## 🔄 Pipeline Flow 


The project follows a **structured pipeline** to ensure reproducibility, clarity, and scalability:  


### 1️⃣ Data Collection & Understanding   
- Load and explore the dataset (customer transactions & credit card behavior).  
- Perform initial **exploratory data analysis (EDA)**.
  
### 2️⃣ Data Preprocessing  
- Handle missing values (e.g., `MINIMUM_PAYMENTS`).  
- Normalize/scale numerical features for clustering.  
- Feature engineering and selection.  

### 3️⃣ Unsupervised Learning (Segmentation)  
 
- Apply **K-Means Clustering** to group customers based on behavior.  
- Experiment with **DBSCAN** to capture non-linear clusters.  
- Evaluate clusters using metrics (e.g., silhouette score).  

### 4️⃣ Cluster Profiling & Insights  
- Analyze behavioral patterns in each cluster.  
- Identify high-value customers, risky profiles, and unique segments.  


### 5️⃣ Supervised Learning (Classification)  
- Train a **Decision Tree Classifier** to predict customer clusters.  
- Validate model performance on unseen data.  


### 6️⃣ Visualization & Reporting   
- Summarize findings with plots and statistical insights.  
- Document results for **business impact**.
<p align="right">[🔝 Back to Top](#top)</p>
---
## 📊 Results & Insights 

The clustering and classification models provided **valuable insights** into customer behavior, enabling data-driven decision-making for financial services.  

<table>
  <tr>
    <th align="center">📈 Classification Report</th>
    <th align="center">🔍 KMeans Clusters</th>
  </tr>
  <tr>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/9a0732b0-7a2f-4d6a-8dc0-a57a2da890fe" alt="Classification Report" width="400"/>
    </td>
    <td align="center">
      <img src="https://github.com/user-attachments/assets/eb69395a-c1cf-4cb8-b548-b73918f75dde" alt="KMeans Clusters" width="400"/>
    </td>
  </tr>
</table>

-  The Decision Tree Classifier achieved **91% accuracy**, with precision, recall, and F1-scores consistently high (0.85–0.94), ensuring reliable classification across all customer groups.  
-  Macro-average scores of **0.89** indicate fair treatment of smaller clusters, while weighted averages of **0.91** confirm strong performance on larger segments.  
-  High spenders, installment-focused users, and cash-advance-reliant customers were clearly identified.  
-  Each cluster showed unique patterns in spending frequency, repayment discipline, and credit utilization.  
<p align="right">[🔝 Back to Top](#top)</p>
---

## 💼 Use Cases  

- **Targeted Marketing:** Design personalized campaigns for each customer group to increase engagement and conversions.  
- **Credit Risk Assessment:** Identify risky customers based on repayment behavior and credit utilization trends.  
- **Customer Retention:** Offer tailored incentives and loyalty programs to reduce churn in vulnerable segments.  
- **Financial Planning:** Help institutions optimize lending strategies and improve overall portfolio management.  
- **Strategic Growth:** Use insights to expand services, cross-sell products, and maximize customer lifetime value (CLV).  

<p align="right">[🔝 Back to Top](#top)</p>

