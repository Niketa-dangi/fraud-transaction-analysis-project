import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

OUTPUT_DIR = 'output'
CLEAN_DATA_PATH = os.path.join(OUTPUT_DIR, 'fraud_cleaned.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set_theme(style='whitegrid')


def save_matplotlib_plot(file_name):
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, file_name), dpi=300, bbox_inches='tight')
    plt.close()


def save_plotly_plot(fig_name, fig_obj):
    fig_obj.write_image(os.path.join(OUTPUT_DIR, fig_name), scale=2)


def create_visuals():
    df_name = pd.read_csv(CLEAN_DATA_PATH, parse_dates=['trans_date_trans_time', 'dob'])

    fraud_counts = df_name['is_fraud'].map({0: 'Non-Fraud', 1: 'Fraud'}).value_counts()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=fraud_counts.index, y=fraud_counts.values, palette=['#4CAF50', '#E53935'])
    plt.title('Fraud vs Non-Fraud Transactions')
    plt.xlabel('Transaction Class')
    plt.ylabel('Transaction Count')
    save_matplotlib_plot('01_fraud_vs_nonfraud_bar.png')

    plt.figure(figsize=(7, 7))
    plt.pie(fraud_counts.values, labels=fraud_counts.index, autopct='%1.2f%%', colors=['#66BB6A', '#EF5350'], startangle=90)
    plt.title('Fraud vs Non-Fraud Share')
    save_matplotlib_plot('02_fraud_vs_nonfraud_pie.png')

    plt.figure(figsize=(10, 6))
    sns.histplot(df_name['amt'], bins=50, kde=True, color='#1E88E5')
    plt.title('Transaction Amount Distribution')
    plt.xlabel('Transaction Amount')
    plt.ylabel('Frequency')
    save_matplotlib_plot('03_transaction_amount_histogram.png')

    plt.figure(figsize=(10, 6))
    sns.boxplot(data=df_name, x='is_fraud', y='amt', palette=['#4CAF50', '#E53935'])
    plt.title('Transaction Amount by Fraud Status')
    plt.xlabel('Is Fraud')
    plt.ylabel('Amount')
    save_matplotlib_plot('04_amount_by_fraud_boxplot.png')

    corr_cols = ['amt', 'city_pop', 'lat', 'long', 'merch_lat', 'merch_long', 'age', 'transaction_hour', 'weekend_indicator', 'risk_score', 'is_fraud']
    plt.figure(figsize=(11, 8))
    sns.heatmap(df_name[corr_cols].corr(numeric_only=True), annot=True, cmap='coolwarm', fmt='.2f')
    plt.title('Correlation Heatmap')
    save_matplotlib_plot('05_correlation_heatmap.png')

    monthly_fraud = df_name.groupby('transaction_year_month')['is_fraud'].sum().reset_index()
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=monthly_fraud, x='transaction_year_month', y='is_fraud', marker='o', color='#8E24AA')
    plt.title('Monthly Fraud Trend')
    plt.xlabel('Year-Month')
    plt.ylabel('Fraud Transactions')
    plt.xticks(rotation=45)
    save_matplotlib_plot('06_monthly_fraud_trend_line.png')

    top_categories = df_name.groupby('category')['is_fraud'].mean().reset_index().sort_values('is_fraud', ascending=False).head(10)
    plt.figure(figsize=(12, 6))
    sns.barplot(data=top_categories, x='is_fraud', y='category', palette='Reds_r')
    plt.title('Top 10 Categories by Fraud Rate')
    plt.xlabel('Fraud Rate')
    plt.ylabel('Category')
    save_matplotlib_plot('07_top_category_fraud_rate_bar.png')

    top_states = df_name.groupby('state')['is_fraud'].sum().reset_index().sort_values('is_fraud', ascending=False).head(15)
    plt.figure(figsize=(12, 7))
    sns.barplot(data=top_states, x='is_fraud', y='state', palette='rocket')
    plt.title('Top 15 States by Fraud Count')
    plt.xlabel('Fraud Transactions')
    plt.ylabel('State')
    save_matplotlib_plot('08_top_states_by_fraud_bar.png')

    risky_customers = df_name.groupby('full_name').agg({'is_fraud': 'sum', 'amt': 'mean'}).reset_index().sort_values('is_fraud', ascending=False).head(20)
    plt.figure(figsize=(12, 7))
    sns.scatterplot(data=risky_customers, x='amt', y='is_fraud', s=120, color='#FB8C00')
    plt.title('Risky Customers Scatter Plot')
    plt.xlabel('Average Amount')
    plt.ylabel('Fraud Count')
    save_matplotlib_plot('09_risky_customers_scatter.png')

    hour_analysis = df_name.groupby('transaction_hour')['is_fraud'].mean().reset_index()
    fig_obj = px.line(hour_analysis, x='transaction_hour', y='is_fraud', markers=True, title='Fraud Rate by Transaction Hour')
    save_plotly_plot('10_fraud_rate_by_hour_plotly.png', fig_obj)

    merchant_analysis = df_name.groupby('merchant_name')['is_fraud'].sum().reset_index().sort_values('is_fraud', ascending=False).head(15)
    fig_obj = px.bar(merchant_analysis, x='merchant_name', y='is_fraud', title='Top Merchants by Fraud Count')
    fig_obj.update_layout(xaxis_tickangle=-45)
    save_plotly_plot('11_top_merchants_by_fraud_plotly.png', fig_obj)

    print('All visualizations saved successfully in the output folder.')


if __name__ == '__main__':
    create_visuals()
