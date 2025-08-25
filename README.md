# Credit Card Customer Segmentation & Classification 💳  

![Credit Card Segmentation](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExMXR0NzJxdTFwNmQ3a3RnZWRudG8wYmhzZjgzM2pjaTJpb2M4anEwaiZlcD12MV9naWZzX3NlYXJjaCZjdD1n/3o7btPCcdNniyf0ArS/giphy.gif)  

---

## Project Overview  
This project focuses on a comprehensive analysis of credit card usage data to perform customer segmentation and classification. Using unsupervised learning techniques, specifically K-Means Clustering, customers are grouped into distinct segments based on their spending behaviors. Subsequently, a Decision Tree Classifier is trained to predict the cluster a new customer would belong to, providing a powerful tool for targeted marketing and strategic financial planning.  

---
## 📑 Table of Contents  

- [📌 Project Overview](#-project-overview)  
- [🎯 Objectives](#-objectives)  
- [🛠 Tech Stack](#-tech-stack)  
- [📂 Dataset](#-dataset)  
- [🚀 Setup Instructions](#-setup-instructions)  
  - [1. Clone the Repository](#1-clone-the-repository)  
  - [2. Create a Virtual Environment](#2-create-a-virtual-environment)  
  - [3. Activate the Virtual Environment](#3-activate-the-virtual-environment)  
  - [4. Install Dependencies](#4-install-dependencies)  
- [🏃 Running the Project](#-running-the-project)  
  - [1. Train Models](#1-train-models)  
  - [2. Predict Models](#2-predict-models)  
- [📊 Results & Insights](#-results--insights)  
- [📈 Future Enhancements](#-future-enhancements)  
- [🤝 Contributing](#-contributing)  
- [📜 License](#-license)  

## 🎯 Objectives  

- **Customer Segmentation:** Apply unsupervised learning (K-Means Clustering) to categorize credit card customers into distinct, data-driven groups based on spending patterns and behavioral attributes.  
- **Behavioral Profiling:** Analyze and interpret each cluster to uncover unique financial habits, transaction trends, and customer characteristics.  
- **Predictive Classification:** Develop a Decision Tree Classifier to accurately assign new customers to the most relevant segment.  
- **Business Impact:** Deliver actionable insights to support targeted marketing, personalized financial services, customer retention strategies, and improved decision-making.  

---

## 🛠 Tech Stack  

## 🛠 Tech Stack  

This project leverages a modern data science and machine learning stack to ensure efficient data processing, robust modeling, and professional visualization.  

### Languages & Core Libraries  
- ![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python) — Primary programming language for analysis and model development.  
- ![NumPy](https://img.shields.io/badge/NumPy-Scientific%20Computing-lightgrey?logo=numpy) — High-performance numerical computations.  
- ![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-orange?logo=pandas) — Data manipulation and preprocessing.  

### Machine Learning & Modeling  
- ![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-ML%20Models-green?logo=scikitlearn) — Algorithms and evaluation (K-Means, DBSCAN, Decision Tree Classifier).  

### Visualization  
- ![Matplotlib](https://img.shields.io/badge/Matplotlib-Visualization-blueviolet?logo=plotly) — Core data visualization.  
- ![Seaborn](https://img.shields.io/badge/Seaborn-Statistical%20Plots-9cf) — Statistical and exploratory data analysis.  

### Development & Collaboration  
- ![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-orange?logo=jupyter) — Interactive development and experimentation.  
- ![venv](https://img.shields.io/badge/venv-Virtual%20Environment-yellow) — Isolated project environment for dependencies.  
- ![GitHub](https://img.shields.io/badge/GitHub-Version%20Control-black?logo=github) — Version control and collaboration platform.  


## 📊 Dataset  

The dataset provides comprehensive information on credit card customers, capturing their transaction patterns, spending frequency, cash advances, credit utilization, payment behavior, and tenure with the bank. It includes features such as purchase amounts, installment purchases, cash advance transactions, credit limits, and repayment history, making it well-suited for customer segmentation and predictive modeling. This dataset is widely used in clustering and classification projects to derive insights into customer behavior and financial profiles.  

📂 You can access a public version of the dataset here: [Credit Card Customer Segmentation Dataset – Kaggle](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata)  

---

The dataset captures multiple dimensions of customer credit card usage, organized into four key categories:  

- **Customer Identity**  
  - `CUST_ID`: Unique customer identifier  

- **Transaction Behavior**  
  - `PURCHASES`, `ONEOFF_PURCHASES`, `INSTALLMENTS_PURCHASES`, `PURCHASES_TRX`  
  - `PURCHASES_FREQUENCY`, `ONEOFF_PURCHASES_FREQUENCY`, `PURCHASES_INSTALLMENTS_FREQUENCY`  

- **Credit & Cash Advances**  
  - `BALANCE`, `BALANCE_FREQUENCY`, `CREDIT_LIMIT`  
  - `CASH_ADVANCE`, `CASH_ADVANCE_FREQUENCY`, `CASH_ADVANCE_TRX`  

- **Repayment & Financial Discipline**  
  - `PAYMENTS`, `MINIMUM_PAYMENTS`, `PRC_FULL_PAYMENT`  

- **Customer Lifecycle**  
  - `TENURE`: Length of customer’s relationship with the bank  

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

## 🏃 Running the Project  

### 1. Train Models  
```bash
python -m src.train
```  

### 2. Predict Models  
```bash
python -m src.predict
```  

---

## 📽️ Project Flow  

![Project Flow](https://media.giphy.com/media/v1.Y2lkPTc5MGI3NjExb3A5Nnd6bWh3M2U5M3p3Y2lnOWtmNmNqNnI1c2k0OXI2aGplazQxOSZlcD12MV9naWZzX3NlYXJjaCZjdD1n/26n6WywJyh39n1pBu/giphy.gif)  

---
