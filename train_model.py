"""
PCOS Detection - Model Training Script
This script trains the XGBoost model on clinical parameters data
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score
import pickle
from collections import Counter

def preprocess_data(df):
    """Preprocess PCOS data"""
    # helper to find a column by keyword substrings (case-insensitive)
    def find_column(df, keywords):
        for col in df.columns:
            lc = col.lower()
            for kw in keywords:
                if kw in lc:
                    return col
        return None

    # Drop common junk column if present
    if 'Unnamed: 44' in df.columns:
        df = df.drop(['Unnamed: 44'], axis=1)

    # Locate weight and height columns (support variants)
    weight_col = find_column(df, ['weight'])
    height_col = find_column(df, ['height'])
    if weight_col is None or height_col is None:
        raise ValueError(f"Required columns for BMI not found. Available columns: {list(df.columns)}")

    # Ensure numeric and compute BMI
    df[weight_col] = pd.to_numeric(df[weight_col], errors='coerce')
    df[height_col] = pd.to_numeric(df[height_col], errors='coerce')
    df['BMI'] = df[weight_col] / (df[height_col] / 100) ** 2

    # Optionally drop 'FSH/LH' if present
    if 'FSH/LH' in df.columns:
        df = df.drop(['FSH/LH'], axis=1)

    # Waist:Hip ratio
    waist_col = find_column(df, ['waist'])
    hip_col = find_column(df, ['hip'])
    if waist_col is not None and hip_col is not None:
        df[waist_col] = pd.to_numeric(df[waist_col], errors='coerce')
        df[hip_col] = pd.to_numeric(df[hip_col], errors='coerce')
        df['Waist:Hip Ratio'] = df[waist_col] / df[hip_col]

    # Convert commonly named hormonal columns if present
    beta_col = find_column(df, ['beta-hcg', 'beta hcg', 'beta', 'hcg'])
    if beta_col is not None:
        df[beta_col] = pd.to_numeric(df[beta_col], errors='coerce')

    amh_col = find_column(df, ['amh'])
    if amh_col is not None:
        df[amh_col] = pd.to_numeric(df[amh_col], errors='coerce')

    # Drop rows with NaN in critical columns
    # Ensure target column exists
    target_col = find_column(df, ['pcos'])
    if target_col is None:
        raise ValueError(f"Target column containing 'pcos' not found. Available columns: {list(df.columns)}")

    df.dropna(inplace=True)

    return df

def train_model(data_path='PCOS_data_without_infertility.xlsx', output_path='best_xgboost_model.pkl'):
    """Train XGBoost model with nested cross-validation"""
    
    print("Loading data...")
    # Auto-detect the sheet that contains clinical data (looks for keywords)
    xls = pd.ExcelFile(data_path)
    preferred_keywords = ['weight', 'pcos', 'age']
    selected_sheet = None
    for name in xls.sheet_names:
        try:
            tmp = pd.read_excel(xls, sheet_name=name, nrows=5)
        except Exception:
            continue
        cols = ' '.join([str(c).lower() for c in tmp.columns])
        if any(k in cols for k in preferred_keywords):
            selected_sheet = name
            break

    if selected_sheet is None:
        selected_sheet = xls.sheet_names[0]
        print(f"No sheet matched keywords; falling back to first sheet: {selected_sheet}")
    else:
        print(f"Using sheet: {selected_sheet}")

    df = pd.read_excel(xls, sheet_name=selected_sheet)
    
    print("Preprocessing data...")
    df = preprocess_data(df)
    
    # Prepare features and target
    y = df['PCOS (Y/N)']
    X = df.drop(columns=['PCOS (Y/N)'])

    # Coerce non-numeric columns to numeric where possible (force numeric dtype for XGBoost)
    for col in X.columns:
        if X[col].dtype == object:
            X[col] = pd.to_numeric(X[col], errors='coerce')

    # Fill remaining NaNs with column median
    X = X.fillna(X.median())

    # Ensure target is numeric
    y = pd.to_numeric(y, errors='coerce')
    # Drop rows with missing target or features
    valid_idx = y.dropna().index.intersection(X.dropna().index)
    X = X.loc[valid_idx].reset_index(drop=True)
    y = y.loc[valid_idx].astype(int).reset_index(drop=True)

    print(f"Dataset size: {len(X)} samples, {len(X.columns)} features")
    print(f"PCOS cases: {(y == 1).sum()}, Non-PCOS cases: {(y == 0).sum()}")
    
    # Hyperparameter grid
    param_grid = {
        'n_estimators': [50, 100, 200],
        'max_depth': [3, 5, 7],
        'learning_rate': [0.01, 0.1, 0.2],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Nested cross-validation
    outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    best_params_list = []
    outer_scores = []
    all_metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': [], 'roc_auc': []}
    
    fold_num = 1
    print("\nRunning nested cross-validation...")
    
    for train_idx, test_idx in outer_cv.split(X, y):
        print(f"\nFold {fold_num}/5")
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        # Inner cross-validation for hyperparameter tuning
        xgb_model = xgb.XGBClassifier(eval_metric='logloss', random_state=42)
        grid_search = GridSearchCV(xgb_model, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=0)
        grid_search.fit(X_train, y_train)
        
        best_params_list.append(frozenset(grid_search.best_params_.items()))
        
        # Evaluate on outer test set
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred_proba)
        
        all_metrics['accuracy'].append(accuracy)
        all_metrics['precision'].append(precision)
        all_metrics['recall'].append(recall)
        all_metrics['f1'].append(f1)
        all_metrics['roc_auc'].append(roc_auc)
        
        print(f"  Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}, ROC-AUC: {roc_auc:.4f}")
        outer_scores.append(accuracy)
        fold_num += 1
    
    print("\n" + "="*60)
    print("CROSS-VALIDATION RESULTS")
    print("="*60)
    print(f"Average Accuracy: {np.mean(all_metrics['accuracy']):.4f} (+/- {np.std(all_metrics['accuracy']):.4f})")
    print(f"Average Precision: {np.mean(all_metrics['precision']):.4f} (+/- {np.std(all_metrics['precision']):.4f})")
    print(f"Average Recall: {np.mean(all_metrics['recall']):.4f} (+/- {np.std(all_metrics['recall']):.4f})")
    print(f"Average F1-Score: {np.mean(all_metrics['f1']):.4f} (+/- {np.std(all_metrics['f1']):.4f})")
    print(f"Average ROC-AUC: {np.mean(all_metrics['roc_auc']):.4f} (+/- {np.std(all_metrics['roc_auc']):.4f})")
    print("="*60)
    
    # Train final model on full dataset
    print("\nTraining final model on full dataset...")
    # Determine most common hyperparameters from CV; fall back to sensible defaults if empty
    if len(best_params_list) == 0:
        most_common_params = {
            'n_estimators': 100,
            'max_depth': 5,
            'learning_rate': 0.1,
            'subsample': 0.8,
            'colsample_bytree': 0.8
        }
        print("Warning: no best params collected from CV; using defaults:", most_common_params)
    else:
        most_common = Counter(best_params_list).most_common(1)[0][0]
        most_common_params = dict(most_common)
        print(f"Best hyperparameters: {most_common_params}")
    
    final_model = xgb.XGBClassifier(**most_common_params, eval_metric='logloss', random_state=42)
    final_model.fit(X, y)
    
    # Get feature importance
    print("\nTop 20 Most Important Features:")
    feature_importance = final_model.get_booster().get_score(importance_type='gain')
    importance_df = pd.DataFrame(feature_importance.items(), columns=['Feature', 'Importance']).sort_values(by='Importance', ascending=False)
    print(importance_df.head(20).to_string(index=False))
    
    # Save model
    print(f"\nSaving model to {output_path}...")
    with open(output_path, 'wb') as f:
        pickle.dump(final_model, f)
    
    print("✅ Model training complete!")
    
    return final_model, importance_df

if __name__ == "__main__":
    train_model()
