# Final Project Report

## Project Title
Credit Card Fraud Transaction Analytics and Detection

## Problem Statement
Financial fraud creates revenue leakage, customer dissatisfaction, and operational risk. The goal of this project is to analyze fraud transaction patterns and build predictive models to support fraud detection.

## Objective
Inspect the uploaded dataset, clean it, perform EDA, generate business insights, and build machine learning models for fraud classification.

## Dataset Description
The uploaded dataset contains 555,719 rows and 23 original columns. The target variable is is_fraud. The actual fraud rate is approximately 0.386 percent, which indicates severe class imbalance.

## Actual Dataset Columns
Unnamed: 0, trans_date_trans_time, cc_num, merchant, category, amt, first, last, gender, street, city, state, zip, lat, long, city_pop, job, dob, trans_num, unix_time, merch_lat, merch_long, is_fraud

## Data Cleaning Summary
Column names were standardized, the technical index column was removed, date fields were parsed, numeric fields were corrected, duplicates were checked, missing values were handled, and new analytical features such as age, transaction hour, amount category, fraud rates, and risk score were created.

## EDA Findings
Fraud is rare but concentrated in specific merchants, categories, time windows, and customer groups. Transaction amount and merchant-level patterns are especially useful for identifying elevated risk.

## Key Insights
Fraud transactions are highly imbalanced compared with non-fraud transactions. Some merchant categories exhibit higher fraud rates. Certain hours of the day show elevated fraud intensity. Merchant and customer fraud rates are strong analytical indicators.

## Machine Learning Results
The project trains Logistic Regression, Random Forest, and XGBoost if available. Evaluation includes Accuracy, Precision, Recall, F1 Score, and ROC-AUC. Results are saved to the output folder for comparison.

## Business Recommendations
Focus monitoring on high-risk merchants, add real-time rules for high-risk hours, prioritize large outlier transactions for review, and use model scores with merchant and customer fraud rates in fraud operations workflows.

## Conclusion
This project demonstrates a complete analytics workflow suitable for a GitHub portfolio, resume project, internship application, and analyst portfolio.
