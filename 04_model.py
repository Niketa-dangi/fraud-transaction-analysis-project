import os
import warnings
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report

warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    xgb_available = True
except Exception:
    xgb_available = False

OUTPUT_DIR = 'output'
CLEAN_DATA_PATH = os.path.join(OUTPUT_DIR, 'fraud_cleaned.csv')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def evaluate_model(model_name, fitted_pipeline, x_test_df, y_test_sr):
    pred_vals = fitted_pipeline.predict(x_test_df)
    prob_vals = fitted_pipeline.predict_proba(x_test_df)[:, 1]
    result_dict = {
        'model': model_name,
        'accuracy': round(accuracy_score(y_test_sr, pred_vals), 4),
        'precision': round(precision_score(y_test_sr, pred_vals, zero_division=0), 4),
        'recall': round(recall_score(y_test_sr, pred_vals, zero_division=0), 4),
        'f1_score': round(f1_score(y_test_sr, pred_vals, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test_sr, prob_vals), 4)
    }
    cm_df = pd.DataFrame(confusion_matrix(y_test_sr, pred_vals))
    cm_df.to_csv(os.path.join(OUTPUT_DIR, model_name.lower().replace(' ', '_') + '_confusion_matrix.csv'), index=False)
    report_df = pd.DataFrame(classification_report(y_test_sr, pred_vals, output_dict=True)).transpose()
    report_df.to_csv(os.path.join(OUTPUT_DIR, model_name.lower().replace(' ', '_') + '_classification_report.csv'))
    return result_dict


def train_models():
    df_name = pd.read_csv(CLEAN_DATA_PATH, parse_dates=['trans_date_trans_time', 'dob'])

    drop_cols = ['is_fraud', 'trans_num', 'first', 'last', 'street']
    feature_df = df_name.drop(columns=drop_cols, errors='ignore').copy()
    feature_df['trans_date_trans_time'] = feature_df['trans_date_trans_time'].astype('int64') // 10**9
    feature_df['dob'] = feature_df['dob'].astype('int64') // 10**9

    target_sr = df_name['is_fraud']
    categorical_cols = feature_df.select_dtypes(include=['object', 'category']).columns.tolist()
    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', Pipeline(steps=[('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())]), numeric_cols),
            ('cat', Pipeline(steps=[('imputer', SimpleImputer(strategy='most_frequent')), ('encoder', OneHotEncoder(handle_unknown='ignore'))]), categorical_cols)
        ]
    )

    x_train_df, x_test_df, y_train_sr, y_test_sr = train_test_split(
        feature_df,
        target_sr,
        test_size=0.2,
        random_state=42,
        stratify=target_sr
    )

    model_specs = {
        'Logistic Regression': LogisticRegression(max_iter=1000, class_weight='balanced'),
        'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced_subsample', n_jobs=-1)
    }

    if xgb_available:
        model_specs['XGBoost'] = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            eval_metric='logloss',
            random_state=42
        )

    results_list = []
    for model_name, model_obj in model_specs.items():
        pipeline_obj = Pipeline(steps=[('preprocessor', preprocessor), ('model', model_obj)])
        pipeline_obj.fit(x_train_df, y_train_sr)
        results_list.append(evaluate_model(model_name, pipeline_obj, x_test_df, y_test_sr))

    results_df = pd.DataFrame(results_list).sort_values('roc_auc', ascending=False)
    results_df.to_csv(os.path.join(OUTPUT_DIR, 'model_comparison.csv'), index=False)
    print(results_df)
    print('Model training and evaluation completed successfully.')


if __name__ == '__main__':
    train_models()
