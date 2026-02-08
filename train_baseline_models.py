#!/usr/bin/env python3
"""
Baseline ML Models for Denial Prediction

Train and evaluate Random Forest and XGBoost models for:
1. Binary denial prediction (denied vs. paid)
2. Denial category classification
3. Recoverability prediction

Metrics: AUC-ROC, Precision, Recall, F1-Score
For IEEE Access paper experiments.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    print("Warning: XGBoost not available. Install with: pip install xgboost")
    XGB_AVAILABLE = False


def load_dataset(data_dir: str) -> tuple:
    """Load claims dataset and labels."""
    data_path = Path(data_dir)
    
    # Load claims CSV (line-level data)
    claims_lines_df = pd.read_csv(data_path / "claims.csv")
    labels_df = pd.read_csv(data_path / "labels.csv")
    
    print(f"Loaded {len(claims_lines_df)} claim lines, {len(labels_df)} claims")
    
    # Aggregate lines to claim level (only feature columns, not labels)
    claims_df = claims_lines_df.groupby('claim_id').agg({
        'patient_id': 'first',
        'provider_npi': 'first',
        'payer_id': 'first',
        'date_of_service': 'first',
        'charge_amount': 'sum',  # Total charge
        'paid_amount': 'sum',    # Total paid
    }).reset_index()
    
    # Calculate num_lines
    line_counts = claims_lines_df.groupby('claim_id').size().reset_index(name='num_lines')
    claims_df = claims_df.merge(line_counts, on='claim_id')
    
    # ================================================================
    # PROCEDURE CODE FEATURES (pre-adjudication clinical data)
    # ================================================================
    # Get the primary (first) procedure code for each claim
    primary_proc = claims_lines_df.groupby('claim_id').agg({
        'procedure_code': 'first'  # Primary procedure
    }).reset_index()
    primary_proc = primary_proc.rename(columns={'procedure_code': 'primary_procedure'})
    claims_df = claims_df.merge(primary_proc, on='claim_id')
    
    # Count unique procedures per claim
    unique_procs = claims_lines_df.groupby('claim_id')['procedure_code'].nunique().reset_index()
    unique_procs = unique_procs.rename(columns={'procedure_code': 'unique_procedures'})
    claims_df = claims_df.merge(unique_procs, on='claim_id')
    
    # Get all procedure codes for the claim (for one-hot encoding)
    proc_lists = claims_lines_df.groupby('claim_id')['procedure_code'].apply(list).reset_index()
    proc_lists = proc_lists.rename(columns={'procedure_code': 'all_procedures'})
    claims_df = claims_df.merge(proc_lists, on='claim_id')
    
    # Add patient_responsibility (charge - paid) - kept for reference but NOT for ML
    claims_df['patient_responsibility'] = claims_df['charge_amount'] - claims_df['paid_amount']
    claims_df['patient_responsibility'] = claims_df['patient_responsibility'].clip(lower=0)
    
    # Rename for consistency
    claims_df = claims_df.rename(columns={
        'charge_amount': 'total_charge',
        'paid_amount': 'total_paid'
    })
    
    # Merge with labels (labels have is_denied, denial_category, is_recoverable, recovery_action)
    claims_df = claims_df.merge(labels_df, on='claim_id', how='inner')
    
    print(f"Aggregated to {len(claims_df)} claims with labels")
    print(f"Procedure features added: primary_procedure, unique_procedures, all_procedures")
    
    return claims_df


def engineer_features(claims_df: pd.DataFrame) -> tuple:
    """Engineer features for ML models.
    
    IMPORTANT: We only use PRE-ADJUDICATION features.
    Features like total_paid, payment_ratio are POST-ADJUDICATION
    and would cause data leakage (they encode the denial outcome).
    """
    df = claims_df.copy()
    
    # ================================================================
    # PRE-ADJUDICATION FEATURES ONLY (to avoid data leakage)
    # ================================================================
    # These are known BEFORE the payer decides to pay or deny:
    
    # Derived from charge amount (pre-decision)
    df['avg_charge_per_line'] = df['total_charge'] / (df['num_lines'] + 1)
    df['charge_log'] = np.log1p(df['total_charge'])  # Log-scaled charge
    
    # ================================================================
    # REMOVED (DATA LEAKAGE - these encode the denial outcome!):
    # - payment_ratio = total_paid / total_charge  <- LEAKY!
    # - patient_ratio = patient_responsibility / total_charge <- LEAKY!
    # - total_paid <- directly zero for denials
    # - patient_responsibility <- set after adjudication
    # ================================================================
    
    # Date features (if available) - these are pre-decision
    if 'date_of_service' in df.columns:
        df['date_of_service'] = pd.to_datetime(df['date_of_service'], errors='coerce')
        df['day_of_week'] = df['date_of_service'].dt.dayofweek.fillna(0).astype(int)
        df['month'] = df['date_of_service'].dt.month.fillna(1).astype(int)
        df['quarter'] = df['date_of_service'].dt.quarter.fillna(1).astype(int)
    
    # Encode categorical features (pre-decision)
    cat_cols = ['payer_id']
    label_encoders = {}
    
    for col in cat_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[f'{col}_encoded'] = le.fit_transform(df[col].astype(str))
            label_encoders[col] = le
    
    # ================================================================
    # PROCEDURE CODE FEATURES (clinical/pre-adjudication)
    # ================================================================
    if 'primary_procedure' in df.columns:
        # Encode primary procedure
        le_proc = LabelEncoder()
        df['primary_procedure_encoded'] = le_proc.fit_transform(df['primary_procedure'].astype(str))
        label_encoders['primary_procedure'] = le_proc
    
    # One-hot encode top procedures (if all_procedures column exists)
    if 'all_procedures' in df.columns:
        # Get top N most common procedures
        all_procs = []
        for proc_list in df['all_procedures']:
            if isinstance(proc_list, list):
                all_procs.extend(proc_list)
        from collections import Counter
        proc_counts = Counter(all_procs)
        top_procs = [p for p, _ in proc_counts.most_common(10)]  # Top 10
        
        # Create one-hot columns
        for proc in top_procs:
            col_name = f'has_proc_{proc}'
            df[col_name] = df['all_procedures'].apply(
                lambda x: 1 if isinstance(x, list) and proc in x else 0
            )
        print(f"Added {len(top_procs)} procedure one-hot features: {top_procs}")
    
    return df, label_encoders


def prepare_features(df: pd.DataFrame) -> np.ndarray:
    """Select and prepare feature matrix.
    
    Only uses PRE-ADJUDICATION features to avoid data leakage.
    """
    # ================================================================
    # SAFE FEATURES (known before payer decision)
    # ================================================================
    feature_cols = [
        'total_charge',           # Billed amount (pre-decision)
        'num_lines',              # Number of service lines (pre-decision)
        'avg_charge_per_line',    # Derived from charge (pre-decision)
        'charge_log',             # Log-scaled charge (pre-decision)
        'unique_procedures',      # Count of unique procedures (pre-decision)
    ]
    
    # ================================================================
    # REMOVED FEATURES (cause data leakage!):
    # - 'total_paid'             <- ZERO for denials = perfect predictor
    # - 'payment_ratio'          <- ZERO for denials = perfect predictor  
    # - 'patient_responsibility' <- Set after adjudication
    # - 'patient_ratio'          <- Derived from leaked feature
    # ================================================================
    
    # Add encoded categoricals if present (pre-decision)
    for col in df.columns:
        if col.endswith('_encoded'):
            feature_cols.append(col)
    
    # Add procedure one-hot features (pre-decision clinical data)
    for col in df.columns:
        if col.startswith('has_proc_'):
            feature_cols.append(col)
    
    # Add date features if present (pre-decision)
    for col in ['day_of_week', 'month', 'quarter']:
        if col in df.columns:
            feature_cols.append(col)
    
    # Keep only available columns
    available_cols = [c for c in feature_cols if c in df.columns]
    
    print(f"\n⚠️  Using ONLY pre-adjudication features (no leakage):")
    for col in available_cols:
        print(f"    ✓ {col}")
    
    X = df[available_cols].values
    return X, available_cols


def train_random_forest(X_train, y_train, X_test, y_test, task_name: str):
    """Train and evaluate Random Forest model."""
    print(f"\n{'='*60}")
    print(f"Random Forest - {task_name}")
    print('='*60)
    
    # Train model
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Predictions
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)
    
    # Metrics
    results = evaluate_model(y_test, y_pred, y_prob, task_name)
    results['model'] = 'RandomForest'
    results['feature_importance'] = dict(zip(
        [f'feature_{i}' for i in range(X_train.shape[1])],
        rf.feature_importances_.tolist()
    ))
    
    return rf, results


def train_xgboost(X_train, y_train, X_test, y_test, task_name: str):
    """Train and evaluate XGBoost model."""
    print(f"\n{'='*60}")
    print(f"XGBoost - {task_name}")
    print('='*60)
    
    if not XGB_AVAILABLE:
        print("Skipping XGBoost (not installed)")
        return None, None
    
    # Calculate scale_pos_weight for imbalanced data
    unique, counts = np.unique(y_train, return_counts=True)
    if len(counts) == 2:
        scale_pos_weight = counts[0] / counts[1] if counts[1] > 0 else 1
    else:
        scale_pos_weight = 1
    
    # Train model
    xgb_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )
    
    xgb_model.fit(X_train, y_train)
    
    # Predictions
    y_pred = xgb_model.predict(X_test)
    y_prob = xgb_model.predict_proba(X_test)
    
    # Metrics
    results = evaluate_model(y_test, y_pred, y_prob, task_name)
    results['model'] = 'XGBoost'
    results['feature_importance'] = dict(zip(
        [f'feature_{i}' for i in range(X_train.shape[1])],
        xgb_model.feature_importances_.tolist()
    ))
    
    return xgb_model, results


def evaluate_model(y_true, y_pred, y_prob, task_name: str) -> dict:
    """Evaluate model performance."""
    results = {
        'task': task_name,
        'samples': len(y_true),
        'classes': len(np.unique(y_true))
    }
    
    # Binary or multi-class
    if results['classes'] == 2:
        # Binary classification
        results['accuracy'] = (y_pred == y_true).mean()
        results['precision'] = precision_score(y_true, y_pred, zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, zero_division=0)
        
        # AUC (use positive class probability)
        if y_prob is not None and len(y_prob.shape) > 1:
            results['auc'] = roc_auc_score(y_true, y_prob[:, 1])
        else:
            results['auc'] = 0.5
            
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall:    {results['recall']:.4f}")
        print(f"F1-Score:  {results['f1']:.4f}")
        print(f"AUC-ROC:   {results['auc']:.4f}")
        
    else:
        # Multi-class classification
        results['accuracy'] = (y_pred == y_true).mean()
        results['precision'] = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        results['recall'] = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        results['f1'] = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Multi-class AUC (One-vs-Rest)
        try:
            results['auc'] = roc_auc_score(y_true, y_prob, multi_class='ovr', average='weighted')
        except:
            results['auc'] = 0.5
            
        print(f"Accuracy:  {results['accuracy']:.4f}")
        print(f"Precision: {results['precision']:.4f} (weighted)")
        print(f"Recall:    {results['recall']:.4f} (weighted)")
        print(f"F1-Score:  {results['f1']:.4f} (weighted)")
        print(f"AUC-ROC:   {results['auc']:.4f} (OvR weighted)")
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    results['confusion_matrix'] = cm.tolist()
    
    return results


def main():
    print("=" * 60)
    print("SynthERA-835: Baseline ML Model Training")
    print("=" * 60)
    
    # Load data (already merged with labels)
    data_dir = Path(__file__).parent.parent / "synthera835_output_10k"
    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        print("Run generate_10k_dataset.py first!")
        return
    
    df = load_dataset(data_dir)
    print(f"Dataset: {len(df)} rows")
    
    # Engineer features
    df, encoders = engineer_features(df)
    X, feature_names = prepare_features(df)
    
    print(f"\nFeatures ({len(feature_names)}):")
    for i, name in enumerate(feature_names):
        print(f"  {i}: {name}")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # All results
    all_results = []
    
    # =========================================
    # Task 1: Binary Denial Prediction
    # =========================================
    print("\n" + "=" * 60)
    print("TASK 1: Binary Denial Prediction")
    print("=" * 60)
    
    y_denial = df['is_denied'].astype(int).values
    
    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y_denial, test_size=0.2, random_state=42, stratify=y_denial
    )
    
    print(f"\nTrain: {len(y_train)} samples ({y_train.sum()} denials)")
    print(f"Test:  {len(y_test)} samples ({y_test.sum()} denials)")
    
    # Random Forest
    rf_model_1, rf_results_1 = train_random_forest(
        X_train, y_train, X_test, y_test, "Binary Denial Prediction"
    )
    all_results.append(rf_results_1)
    
    # XGBoost
    xgb_model_1, xgb_results_1 = train_xgboost(
        X_train, y_train, X_test, y_test, "Binary Denial Prediction"
    )
    if xgb_results_1:
        all_results.append(xgb_results_1)
    
    # =========================================
    # Task 2: Denial Category Classification
    # =========================================
    print("\n" + "=" * 60)
    print("TASK 2: Denial Category Classification")
    print("=" * 60)
    
    # Only on denied claims
    denied_mask = df['is_denied'] == True
    denied_df = df[denied_mask].copy()
    
    if 'denial_category' in denied_df.columns and denied_df['denial_category'].notna().sum() > 0:
        # Encode categories
        le_cat = LabelEncoder()
        denied_df['category_encoded'] = le_cat.fit_transform(
            denied_df['denial_category'].fillna('unknown')
        )
        
        X_denied = X_scaled[denied_mask]
        y_category = denied_df['category_encoded'].values
        
        # Split
        X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
            X_denied, y_category, test_size=0.2, random_state=42, stratify=y_category
        )
        
        print(f"\nCategories: {list(le_cat.classes_)}")
        print(f"Train: {len(y_train_c)} samples")
        print(f"Test:  {len(y_test_c)} samples")
        
        # Random Forest
        rf_model_2, rf_results_2 = train_random_forest(
            X_train_c, y_train_c, X_test_c, y_test_c, "Denial Category Classification"
        )
        rf_results_2['categories'] = list(le_cat.classes_)
        all_results.append(rf_results_2)
        
        # XGBoost
        xgb_model_2, xgb_results_2 = train_xgboost(
            X_train_c, y_train_c, X_test_c, y_test_c, "Denial Category Classification"
        )
        if xgb_results_2:
            xgb_results_2['categories'] = list(le_cat.classes_)
            all_results.append(xgb_results_2)
    else:
        print("No denial category labels available for Task 2")
    
    # =========================================
    # Task 3: Recoverability Prediction
    # =========================================
    print("\n" + "=" * 60)
    print("TASK 3: Recoverability Prediction (on denied claims)")
    print("=" * 60)
    
    if 'is_recoverable' in denied_df.columns:
        y_recoverable = denied_df['is_recoverable'].astype(int).values
        
        # Split
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
            X_denied, y_recoverable, test_size=0.2, random_state=42, stratify=y_recoverable
        )
        
        print(f"\nTrain: {len(y_train_r)} samples ({y_train_r.sum()} recoverable)")
        print(f"Test:  {len(y_test_r)} samples ({y_test_r.sum()} recoverable)")
        
        # Random Forest
        rf_model_3, rf_results_3 = train_random_forest(
            X_train_r, y_train_r, X_test_r, y_test_r, "Recoverability Prediction"
        )
        all_results.append(rf_results_3)
        
        # XGBoost
        xgb_model_3, xgb_results_3 = train_xgboost(
            X_train_r, y_train_r, X_test_r, y_test_r, "Recoverability Prediction"
        )
        if xgb_results_3:
            all_results.append(xgb_results_3)
    else:
        print("No recoverability labels available for Task 3")
    
    # =========================================
    # Save Results
    # =========================================
    output_dir = data_dir / "ml_results"
    output_dir.mkdir(exist_ok=True)
    
    # Save experiment results
    results_file = output_dir / "baseline_results.json"
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'dataset_size': len(df),
            'feature_names': feature_names,
            'experiments': all_results
        }, f, indent=2)
    
    print(f"\n{'='*60}")
    print("Results Summary")
    print('='*60)
    
    # Summary table
    print("\n{:<35} {:<15} {:>8} {:>8} {:>8}".format(
        "Task", "Model", "AUC", "F1", "Recall"
    ))
    print("-" * 75)
    
    for r in all_results:
        if r:
            print("{:<35} {:<15} {:>8.4f} {:>8.4f} {:>8.4f}".format(
                r['task'][:35], r['model'], r['auc'], r['f1'], r['recall']
            ))
    
    print(f"\n✓ Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
