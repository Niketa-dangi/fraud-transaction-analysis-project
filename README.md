# Fraud Transaction Analytics Project

## Overview

This project analyzes financial transaction data to identify fraud patterns using Python, SQL, Machine Learning, and Power BI.

The project demonstrates the complete end-to-end data analytics workflow including:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Feature Engineering
* Fraud Risk Analysis
* Dashboard Development
* Customer Insights
* Machine Learning

---

## Project Objectives

* Detect fraudulent transactions
* Analyze fraud behavior patterns
* Identify high-risk customers and merchants
* Build interactive Power BI dashboards
* Generate business insights from transaction data

---

## Dataset Information

* Total Transactions: 555,719
* Fraud Transactions: 2,145
* Fraud Rate: 0.39%

Dataset Source:
Kaggle Fraud Transaction Dataset

---

## Tools & Technologies Used

### Programming & Analysis

* Python
* SQL
* Pandas
* NumPy
* Scikit-Learn

### Visualization

* Power BI
* Matplotlib
* Seaborn
* Plotly

---

## Project Workflow

### 1. Data Cleaning

* Removed unnecessary columns
* Converted data types
* Checked duplicates and missing values
* Created analytical features

### 2. Exploratory Data Analysis

* Fraud vs Non-Fraud analysis
* Category-wise fraud analysis
* State-wise fraud analysis
* Transaction trend analysis

### 3. Feature Engineering

Created additional features including:

* Transaction Hour
* Customer Risk Score
* Merchant Fraud Rate
* Customer Fraud Rate
* Age
* Amount Categories

### 4. Machine Learning

Implemented:

* Logistic Regression
* Random Forest
* XGBoost

### 5. Dashboard Development

Designed interactive Power BI dashboards for:

* Executive Overview
* Fraud Risk Analysis
* Customer Insights

---

## Dashboard Preview

### Executive Overview

![Executive Dashboard](dashboard_screenshots/executive_overview.png)
<img width="879" height="493" alt="executive_overview png" src="https://github.com/user-attachments/assets/9aff932f-d64c-40cd-88c8-74bd586e7eef" />

![Fraud Risk Analysis](dashboard_screenshots/fraud_risk_analysis.png)
<img width="884" height="499" alt="fraud_risk_analysis png" src="https://github.com/user-attachments/assets/c25d960e-842c-461f-a30a-866b02244379" />

![Customer Insights](dashboard_screenshots/customer_insights.png)
<img width="882" height="496" alt="customer_insights png" src="https://github.com/user-attachments/assets/158aeb0b-3c60-45d9-a148-84f51530e793" />

## Key Insights

* Fraud rate in the dataset was approximately 0.39%
* Shopping and grocery categories showed higher fraud activity
* Certain transaction hours showed increased fraud frequency
* High-risk customer and merchant patterns were identified
* Fraud distribution varied across states and customer demographics

---

## Power BI Dashboard Features

### Executive Dashboard

* KPI Cards
* Fraud Trend Analysis
* Fraud by Category
* Fraud by State
* Fraud vs Non-Fraud Comparison

### Fraud Risk Analysis Dashboard

* Fraud by Hour
* Fraud Amount Distribution
* Top Fraud Merchants
* Category-wise Fraud Analysis

### Customer Insights Dashboard

* Customer Age Distribution
* High-Risk Customers
* Occupation Analysis
* Customer Location Analysis

---

## Project Structure

```bash
Fraud_Analytics_Project/
│
├── dashboard_screenshots/
│   ├── executive_overview.png
│   ├── fraud_risk_analysis.png
│   └── customer_insights.png
│
├── data_cleaning.py
├── eda.py
├── visualization.py
├── model.py
├── requirements.txt
├── README.md
├── FINAL_PROJECT_REPORT.md
└── powerbi_dashboard.pbix
```


## Author

Niketa Dangi

## Connect With Me

LinkedIn: www.linkedin.com/in/niketa-dangi-bbb3533bb
GitHub: https://github.com/Niketa-dangi
