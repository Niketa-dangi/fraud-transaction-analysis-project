import os
import pandas as pd
import numpy as np

OUTPUT_DIR = 'output'
CLEAN_DATA_PATH = os.path.join(OUTPUT_DIR, 'fraud_cleaned.csv')


def generate_eda_report():
    df_name = pd.read_csv(CLEAN_DATA_PATH, parse_dates=['trans_date_trans_time', 'dob'])

    dataset_overview = pd.DataFrame({
        'metric': ['rows', 'columns', 'fraud_count', 'non_fraud_count', 'fraud_percentage'],
        'value': [
            len(df_name),
            df_name.shape[1],
            int(df_name['is_fraud'].sum()),
            int((df_name['is_fraud'] == 0).sum()),
            round(df_name['is_fraud'].mean() * 100, 4)
        ]
    })

    statistical_summary = df_name.describe(include='all').transpose()

    fraud_vs_nonfraud = df_name['is_fraud'].value_counts().rename_axis('is_fraud').reset_index(name='count')
    amount_by_fraud = df_name.groupby('is_fraud')['amt'].agg(['count', 'mean', 'median', 'sum', 'max']).reset_index()
    fraud_by_category = df_name.groupby('category')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index().sort_values('mean', ascending=False)
    fraud_by_hour = df_name.groupby('transaction_hour')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    fraud_by_month = df_name.groupby('transaction_year_month')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index()
    top_risky_customers = df_name.groupby(['cc_num', 'full_name'])['is_fraud'].agg(['count', 'sum', 'mean']).reset_index().sort_values(['sum', 'mean'], ascending=False).head(20)
    top_risky_merchants = df_name.groupby('merchant_name')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index().sort_values(['sum', 'mean'], ascending=False).head(20)
    fraud_by_state = df_name.groupby('state')['is_fraud'].agg(['count', 'sum', 'mean']).reset_index().sort_values('sum', ascending=False)

    corr_cols = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'age', 'transaction_hour', 'weekend_indicator', 'risk_score', 'is_fraud']
    correlation_matrix = df_name[corr_cols].corr(numeric_only=True)

    dataset_overview.to_csv(os.path.join(OUTPUT_DIR, 'eda_dataset_overview.csv'), index=False)
    statistical_summary.to_csv(os.path.join(OUTPUT_DIR, 'eda_statistical_summary.csv'))
    fraud_vs_nonfraud.to_csv(os.path.join(OUTPUT_DIR, 'eda_fraud_vs_nonfraud.csv'), index=False)
    amount_by_fraud.to_csv(os.path.join(OUTPUT_DIR, 'eda_amount_by_fraud.csv'), index=False)
    fraud_by_category.to_csv(os.path.join(OUTPUT_DIR, 'eda_fraud_by_category.csv'), index=False)
    fraud_by_hour.to_csv(os.path.join(OUTPUT_DIR, 'eda_fraud_by_hour.csv'), index=False)
    fraud_by_month.to_csv(os.path.join(OUTPUT_DIR, 'eda_fraud_by_month.csv'), index=False)
    top_risky_customers.to_csv(os.path.join(OUTPUT_DIR, 'eda_top_risky_customers.csv'), index=False)
    top_risky_merchants.to_csv(os.path.join(OUTPUT_DIR, 'eda_top_risky_merchants.csv'), index=False)
    fraud_by_state.to_csv(os.path.join(OUTPUT_DIR, 'eda_fraud_by_state.csv'), index=False)
    correlation_matrix.to_csv(os.path.join(OUTPUT_DIR, 'eda_correlation_matrix.csv'))

    print('EDA report generated successfully.')
    print(dataset_overview)
    print(fraud_by_category.head(10))
    print(top_risky_customers.head(10))
    print(top_risky_merchants.head(10))


if __name__ == '__main__':
    generate_eda_report()
