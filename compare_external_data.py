#!/usr/bin/env python3
"""
External Data Comparison Experiments
Compare SynthERA-835 against Kaggle datasets for IEEE paper validation.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False
    print("XGBoost not available")

def load_kaggle_healthcare_claims():
    """Load Kaggle Synthetic Healthcare Claims Dataset."""
    path = Path("KaggleDatasets/Synthetic Healthcare Claims Dataset/claim_data.csv")
    df = pd.read_csv(path)
    
    print(f"\n=== Kaggle Healthcare Claims Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    print(f"\nOutcome distribution:")
    print(df['Outcome'].value_counts())
    
    return df

def load_kaggle_ar_medical():
    """Load Kaggle Synthetic AR Medical Dataset."""
    path = Path("KaggleDatasets/Synthetic AR Medical Dataset with Realistic Denial/Synthetic AR Medical Dataset with Realistic Denial.csv")
    df = pd.read_csv(path)
    
    print(f"\n=== Kaggle AR Medical Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()}")
    
    return df

def load_synthera835():
    """Load SynthERA-835 generated dataset."""
    claims_path = Path("synthera835_output_10k/claims.csv")
    labels_path = Path("synthera835_output_10k/labels.csv")
    
    claims_df = pd.read_csv(claims_path)
    labels_df = pd.read_csv(labels_path)
    
    # Aggregate line-level to claim-level
    agg_df = claims_df.groupby('claim_id').agg({
        'charge_amount': 'sum',
        'procedure_code': 'first',
        'payer_id': 'first'
    }).reset_index()
    
    # Add line counts
    line_counts = claims_df.groupby('claim_id').size().reset_index(name='num_lines')
    agg_df = agg_df.merge(line_counts, on='claim_id')
    
    # Merge labels
    df = agg_df.merge(labels_df, on='claim_id')
    
    print(f"\n=== SynthERA-835 Dataset ===")
    print(f"Shape: {df.shape}")
    print(f"Denial rate: {df['is_denied'].mean():.1%}")
    
    return df

def prepare_kaggle_features(df):
    """Prepare features from Kaggle Healthcare Claims dataset."""
    # Binary target: Denied vs others
    df['is_denied'] = (df['Outcome'] == 'Denied').astype(int)
    
    # Feature engineering
    le = LabelEncoder()
    
    # Available numeric features
    feature_cols = []
    
    if 'Claim Amount' in df.columns:
        df['claim_amount_clean'] = pd.to_numeric(df['Claim Amount'].replace('[$,]', '', regex=True), errors='coerce')
        feature_cols.append('claim_amount_clean')
    
    if 'Provider ID' in df.columns:
        df['provider_encoded'] = le.fit_transform(df['Provider ID'].astype(str))
        feature_cols.append('provider_encoded')
    
    if 'Diagnosis Code' in df.columns:
        df['diagnosis_encoded'] = le.fit_transform(df['Diagnosis Code'].astype(str))
        feature_cols.append('diagnosis_encoded')
    
    if 'Procedure Code' in df.columns:
        df['procedure_encoded'] = le.fit_transform(df['Procedure Code'].astype(str))
        feature_cols.append('procedure_encoded')
    
    if 'Prior Authorization Required' in df.columns:
        df['prior_auth'] = (df['Prior Authorization Required'] == 'Yes').astype(int)
        feature_cols.append('prior_auth')
    
    # Fill NaN
    df[feature_cols] = df[feature_cols].fillna(0)
    
    X = df[feature_cols].values
    y = df['is_denied'].values
    
    print(f"  Features: {feature_cols}")
    print(f"  Denial rate: {y.mean():.1%}")
    
    return X, y, feature_cols

def prepare_synthera_features(df):
    """Prepare features from SynthERA-835 dataset."""
    le = LabelEncoder()
    
    # Feature engineering
    df['charge_log'] = np.log1p(df['charge_amount'])
    df['avg_charge'] = df['charge_amount'] / df['num_lines']
    df['payer_encoded'] = le.fit_transform(df['payer_id'].astype(str))
    df['procedure_encoded'] = le.fit_transform(df['procedure_code'].astype(str))
    
    feature_cols = ['charge_amount', 'num_lines', 'charge_log', 'avg_charge', 
                    'payer_encoded', 'procedure_encoded']
    
    X = df[feature_cols].values
    y = df['is_denied'].values
    
    print(f"  Features: {feature_cols}")
    print(f"  Denial rate: {y.mean():.1%}")
    
    return X, y, feature_cols

def run_experiment(X, y, dataset_name):
    """Train RF and XGBoost, return metrics."""
    results = {}
    
    # Check for sufficient samples
    if len(y) < 50 or y.sum() < 10:
        print(f"  WARNING: Insufficient samples for {dataset_name}")
        return None
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Random Forest
    print(f"\n  Training Random Forest on {dataset_name}...")
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                 class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_pred_rf = rf.predict(X_test_scaled)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    
    results['RandomForest'] = {
        'accuracy': accuracy_score(y_test, y_pred_rf),
        'precision': precision_score(y_test, y_pred_rf, zero_division=0),
        'recall': recall_score(y_test, y_pred_rf, zero_division=0),
        'f1': f1_score(y_test, y_pred_rf, zero_division=0),
        'auc': roc_auc_score(y_test, y_prob_rf)
    }
    
    print(f"    RF AUROC: {results['RandomForest']['auc']:.3f}")
    
    # XGBoost
    if XGB_AVAILABLE:
        print(f"  Training XGBoost on {dataset_name}...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, 
                                       learning_rate=0.1, random_state=42,
                                       use_label_encoder=False, eval_metric='logloss')
        xgb_model.fit(X_train_scaled, y_train)
        y_pred_xgb = xgb_model.predict(X_test_scaled)
        y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
        
        results['XGBoost'] = {
            'accuracy': accuracy_score(y_test, y_pred_xgb),
            'precision': precision_score(y_test, y_pred_xgb, zero_division=0),
            'recall': recall_score(y_test, y_pred_xgb, zero_division=0),
            'f1': f1_score(y_test, y_pred_xgb, zero_division=0),
            'auc': roc_auc_score(y_test, y_prob_xgb)
        }
        
        print(f"    XGB AUROC: {results['XGBoost']['auc']:.3f}")
    
    return results

def main():
    print("=" * 60)
    print("External Data Comparison Experiments")
    print("=" * 60)
    
    all_results = {}
    
    # 1. Kaggle Healthcare Claims
    try:
        df_kaggle = load_kaggle_healthcare_claims()
        X_kaggle, y_kaggle, _ = prepare_kaggle_features(df_kaggle)
        results_kaggle = run_experiment(X_kaggle, y_kaggle, "Kaggle Healthcare Claims")
        if results_kaggle:
            all_results['Kaggle_Healthcare'] = results_kaggle
    except Exception as e:
        print(f"Kaggle Healthcare failed: {e}")
    
    # 2. SynthERA-835
    try:
        df_synthera = load_synthera835()
        X_synthera, y_synthera, _ = prepare_synthera_features(df_synthera)
        results_synthera = run_experiment(X_synthera, y_synthera, "SynthERA-835")
        if results_synthera:
            all_results['SynthERA835'] = results_synthera
    except Exception as e:
        print(f"SynthERA-835 failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    print("\n| Dataset | Model | AUROC | Accuracy | F1 |")
    print("|---------|-------|-------|----------|-----|")
    
    for dataset, results in all_results.items():
        for model, metrics in results.items():
            print(f"| {dataset} | {model} | {metrics['auc']:.3f} | {metrics['accuracy']:.1%} | {metrics['f1']:.3f} |")
    
    # Save results
    output_path = Path("synthera835/comparison_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to {output_path}")
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)

if __name__ == "__main__":
    main()
