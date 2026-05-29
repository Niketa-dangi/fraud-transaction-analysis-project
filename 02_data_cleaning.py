import os
import numpy as np
import pandas as pd

DATA_PATH = 'fraudTest.csv'
OUTPUT_DIR = 'output'
os.makedirs(OUTPUT_DIR, exist_ok=True)


def standardize_columns(df_name):
    df_name.columns = [
        col.strip().lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace(':', '')
        for col in df_name.columns
    ]
    return df_name


def add_features(df_name):
    df_name['full_name'] = df_name['first'].astype(str) + ' ' + df_name['last'].astype(str)
    df_name['transaction_hour'] = df_name['trans_date_trans_time'].dt.hour
    df_name['transaction_day'] = df_name['trans_date_trans_time'].dt.day_name()
    df_name['transaction_month'] = df_name['trans_date_trans_time'].dt.month_name()
    df_name['transaction_year_month'] = df_name['trans_date_trans_time'].dt.to_period('M').astype(str)
    df_name['weekend_indicator'] = df_name['trans_date_trans_time'].dt.dayofweek.isin([5, 6]).astype(int)
    df_name['age'] = ((df_name['trans_date_trans_time'] - df_name['dob']).dt.days / 365.25).fillna(0).astype(int)
    df_name['merchant_name'] = df_name['merchant'].astype(str).str.replace('fraud_', '', regex=False)
    df_name['amount_category'] = pd.cut(
        df_name['amt'],
        bins=[-np.inf, 10, 50, 100, 500, np.inf],
        labels=['very_low', 'low', 'medium', 'high', 'very_high']
    )
    return df_name


def detect_outliers_iqr(df_name, column_name):
    q1_val = df_name[column_name].quantile(0.25)
    q3_val = df_name[column_name].quantile(0.75)
    iqr_val = q3_val - q1_val
    lower_val = q1_val - 1.5 * iqr_val
    upper_val = q3_val + 1.5 * iqr_val
    outlier_flag = ((df_name[column_name] < lower_val) | (df_name[column_name] > upper_val)).astype(int)
    return outlier_flag, lower_val, upper_val


def clean_data():
    df_name = pd.read_csv(DATA_PATH)
    before_summary = {
        'rows': len(df_name),
        'columns': df_name.shape[1],
        'missing_values': int(df_name.isna().sum().sum()),
        'duplicates': int(df_name.duplicated().sum())
    }

    df_name = standardize_columns(df_name)
    if 'unnamed_0' in df_name.columns:
        df_name = df_name.drop(columns=['unnamed_0'])

    df_name['trans_date_trans_time'] = pd.to_datetime(df_name['trans_date_trans_time'], errors='coerce')
    df_name['dob'] = pd.to_datetime(df_name['dob'], errors='coerce')
    df_name['cc_num'] = df_name['cc_num'].astype(str)
    df_name['zip'] = df_name['zip'].astype(str).str.zfill(5)
    df_name['is_fraud'] = pd.to_numeric(df_name['is_fraud'], errors='coerce').fillna(0).astype(int)
    df_name['amt'] = pd.to_numeric(df_name['amt'], errors='coerce')
    df_name['city_pop'] = pd.to_numeric(df_name['city_pop'], errors='coerce')

    df_name = df_name.drop_duplicates().copy()

    numeric_fill_cols = ['amt', 'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long']
    for col_name in numeric_fill_cols:
        if col_name in df_name.columns:
            df_name[col_name] = df_name[col_name].fillna(df_name[col_name].median())

    categorical_fill_cols = ['merchant', 'category', 'first', 'last', 'gender', 'street', 'city', 'state', 'job', 'trans_num']
    for col_name in categorical_fill_cols:
        if col_name in df_name.columns:
            df_name[col_name] = df_name[col_name].fillna('unknown')

    df_name = add_features(df_name)
    df_name['amt_outlier_iqr'], amt_lower, amt_upper = detect_outliers_iqr(df_name, 'amt')
    df_name['city_pop_outlier_iqr'], pop_lower, pop_upper = detect_outliers_iqr(df_name, 'city_pop')

    df_name['customer_fraud_rate'] = df_name.groupby('cc_num')['is_fraud'].transform('mean')
    df_name['merchant_fraud_rate'] = df_name.groupby('merchant_name')['is_fraud'].transform('mean')
    df_name['category_fraud_rate'] = df_name.groupby('category')['is_fraud'].transform('mean')
    df_name['risk_score'] = (
        df_name['customer_fraud_rate'] * 0.4 +
        df_name['merchant_fraud_rate'] * 0.3 +
        df_name['category_fraud_rate'] * 0.2 +
        df_name['amt_outlier_iqr'] * 0.1
    ).round(4)

    after_summary = {
        'rows': len(df_name),
        'columns': df_name.shape[1],
        'missing_values': int(df_name.isna().sum().sum()),
        'duplicates': int(df_name.duplicated().sum()),
        'fraud_count': int(df_name['is_fraud'].sum()),
        'non_fraud_count': int((df_name['is_fraud'] == 0).sum()),
        'fraud_percentage': round(df_name['is_fraud'].mean() * 100, 4)
    }

    cleaning_log = pd.DataFrame([
        ['Dropped technical index column', 'unnamed_0 removed'],
        ['Standardized column names', 'lowercase and underscore format'],
        ['Fixed data types', 'datetime, numeric, and text identifiers corrected'],
        ['Handled missing values', 'median for numeric and unknown for categorical'],
        ['Removed duplicates', 'exact duplicate records dropped'],
        ['Created analytical features', 'time, age, amount, merchant, and risk features'],
        ['Flagged outliers', 'IQR based flags for amount and city population']
    ], columns=['step', 'details'])

    pd.DataFrame([before_summary]).to_csv(os.path.join(OUTPUT_DIR, 'before_cleaning_summary.csv'), index=False)
    pd.DataFrame([after_summary]).to_csv(os.path.join(OUTPUT_DIR, 'after_cleaning_summary.csv'), index=False)
    cleaning_log.to_csv(os.path.join(OUTPUT_DIR, 'cleaning_log.csv'), index=False)
    df_name.to_csv(os.path.join(OUTPUT_DIR, 'fraud_cleaned.csv'), index=False)

    print('Data cleaning completed successfully.')
    print(pd.DataFrame([before_summary]))
    print(pd.DataFrame([after_summary]))
    print(cleaning_log)


if __name__ == '__main__':
    clean_data()
