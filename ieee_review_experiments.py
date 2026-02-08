#!/usr/bin/env python3
"""
IEEE Review Experiments - Cross-Domain Validation

Implements three key experiments:
1. 5-Fold CV with confidence intervals for Table III
2. Cross-domain training: Train on Kaggle → Test on SynthERA-835
3. Cross-domain training: Train on SynthERA-835 → Test on Kaggle
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False

# ============================================================
# Data Loading Functions
# ============================================================

def load_synthera835():
    """Load SynthERA-835 generated dataset."""
    base_paths = [
        Path("../synthera835_output_10k"),
        Path("synthera835_output_10k"),
        Path(__file__).parent.parent / "synthera835_output_10k"
    ]
    
    data_dir = None
    for p in base_paths:
        if p.exists():
            data_dir = p
            break
    
    if data_dir is None:
        raise FileNotFoundError("SynthERA-835 dataset not found")
    
    claims_df = pd.read_csv(data_dir / "claims.csv")
    labels_df = pd.read_csv(data_dir / "labels.csv")
    
    # Aggregate line-level to claim-level
    agg_df = claims_df.groupby('claim_id').agg({
        'charge_amount': 'sum',
        'procedure_code': 'first',
        'payer_id': 'first',
        'date_of_service': 'first'
    }).reset_index()
    
    # Add line counts
    line_counts = claims_df.groupby('claim_id').size().reset_index(name='num_lines')
    agg_df = agg_df.merge(line_counts, on='claim_id')
    
    # Merge labels
    df = agg_df.merge(labels_df, on='claim_id')
    
    print(f"SynthERA-835: {len(df)} claims, {df['is_denied'].mean():.1%} denied")
    return df


def load_kaggle_claims():
    """Load Kaggle Healthcare Claims Dataset."""
    base_paths = [
        Path("../KaggleDatasets/Synthetic Healthcare Claims Dataset/claim_data.csv"),
        Path("KaggleDatasets/Synthetic Healthcare Claims Dataset/claim_data.csv"),
        Path(__file__).parent.parent / "KaggleDatasets/Synthetic Healthcare Claims Dataset/claim_data.csv"
    ]
    
    kaggle_path = None
    for p in base_paths:
        if p.exists():
            kaggle_path = p
            break
    
    if kaggle_path is None:
        raise FileNotFoundError("Kaggle dataset not found")
    
    df = pd.read_csv(kaggle_path)
    df['is_denied'] = (df['Outcome'] == 'Denied').astype(int)
    
    print(f"Kaggle: {len(df)} claims, {df['is_denied'].mean():.1%} denied")
    return df


def prepare_common_features_synthera(df):
    """Extract common features from SynthERA-835."""
    le = LabelEncoder()
    
    df = df.copy()
    df['charge_log'] = np.log1p(df['charge_amount'])
    df['avg_charge'] = df['charge_amount'] / (df['num_lines'] + 1)
    df['payer_encoded'] = le.fit_transform(df['payer_id'].astype(str))
    df['procedure_encoded'] = le.fit_transform(df['procedure_code'].astype(str))
    
    # Date features
    df['date_of_service'] = pd.to_datetime(df['date_of_service'], errors='coerce')
    df['month'] = df['date_of_service'].dt.month.fillna(1).astype(int)
    df['day_of_week'] = df['date_of_service'].dt.dayofweek.fillna(0).astype(int)
    
    feature_cols = ['charge_amount', 'num_lines', 'charge_log', 'avg_charge', 
                    'payer_encoded', 'month', 'day_of_week']
    
    X = df[feature_cols].fillna(0).values
    y = df['is_denied'].astype(int).values
    
    return X, y, feature_cols


def prepare_common_features_kaggle(df):
    """Extract common features from Kaggle (aligned with SynthERA)."""
    le = LabelEncoder()
    
    df = df.copy()
    
    # Parse claim amount
    if 'Claim Amount' in df.columns:
        df['charge_amount'] = pd.to_numeric(
            df['Claim Amount'].replace('[$,]', '', regex=True), errors='coerce'
        ).fillna(0)
    else:
        df['charge_amount'] = 100.0
    
    df['charge_log'] = np.log1p(df['charge_amount'])
    df['num_lines'] = 1  # Kaggle doesn't have line-level data
    df['avg_charge'] = df['charge_amount']
    
    if 'Provider ID' in df.columns:
        df['payer_encoded'] = le.fit_transform(df['Provider ID'].astype(str))
    else:
        df['payer_encoded'] = 0
    
    # Date features
    if 'Claim Date' in df.columns:
        df['Claim Date'] = pd.to_datetime(df['Claim Date'], errors='coerce')
        df['month'] = df['Claim Date'].dt.month.fillna(1).astype(int)
        df['day_of_week'] = df['Claim Date'].dt.dayofweek.fillna(0).astype(int)
    else:
        df['month'] = 6
        df['day_of_week'] = 2
    
    feature_cols = ['charge_amount', 'num_lines', 'charge_log', 'avg_charge', 
                    'payer_encoded', 'month', 'day_of_week']
    
    X = df[feature_cols].fillna(0).values
    y = df['is_denied'].astype(int).values
    
    return X, y, feature_cols


# ============================================================
# Experiment 1: 5-Fold CV with Confidence Intervals
# ============================================================

def run_cv_with_ci(X, y, n_folds=5):
    """Run 5-fold stratified CV and return metrics with CI."""
    print(f"\n{'='*60}")
    print(f"5-Fold Cross-Validation with Confidence Intervals")
    print(f"{'='*60}")
    
    results = {}
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    scaler = StandardScaler()
    
    # Calculate scale_pos_weight for XGBoost
    unique, counts = np.unique(y, return_counts=True)
    scale_pos_weight = counts[0] / counts[1] if len(counts) == 2 and counts[1] > 0 else 1
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    }
    
    if XGB_AVAILABLE:
        models['XGBoost'] = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
    
    for model_name, model_template in models.items():
        print(f"\n  {model_name}:")
        
        fold_metrics = {'auc': [], 'f1': [], 'accuracy': [], 'precision': [], 'recall': []}
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Scale
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # Clone and train
            import copy
            model = copy.deepcopy(model_template)
            model.fit(X_train_scaled, y_train)
            
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            fold_metrics['auc'].append(roc_auc_score(y_test, y_prob))
            fold_metrics['f1'].append(f1_score(y_test, y_pred, zero_division=0))
            fold_metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            fold_metrics['precision'].append(precision_score(y_test, y_pred, zero_division=0))
            fold_metrics['recall'].append(recall_score(y_test, y_pred, zero_division=0))
        
        results[model_name] = {
            'auc_mean': np.mean(fold_metrics['auc']),
            'auc_std': np.std(fold_metrics['auc']),
            'f1_mean': np.mean(fold_metrics['f1']),
            'f1_std': np.std(fold_metrics['f1']),
            'accuracy_mean': np.mean(fold_metrics['accuracy']),
            'accuracy_std': np.std(fold_metrics['accuracy']),
            'precision_mean': np.mean(fold_metrics['precision']),
            'precision_std': np.std(fold_metrics['precision']),
            'recall_mean': np.mean(fold_metrics['recall']),
            'recall_std': np.std(fold_metrics['recall']),
            'fold_aucs': fold_metrics['auc'],
            'fold_f1s': fold_metrics['f1']
        }
        
        print(f"    AUROC: {results[model_name]['auc_mean']:.3f} ± {results[model_name]['auc_std']:.3f}")
        print(f"    F1:    {results[model_name]['f1_mean']:.3f} ± {results[model_name]['f1_std']:.3f}")
    
    return results


# ============================================================
# Experiment 2 & 3: Cross-Domain Training
# ============================================================

def run_cross_domain_experiments(X_synth, y_synth, X_kaggle, y_kaggle):
    """Train on one dataset, test on another."""
    print(f"\n{'='*60}")
    print(f"Cross-Domain Training Experiments")
    print(f"{'='*60}")
    
    results = {}
    scaler = StandardScaler()
    
    # Calculate scale_pos_weight for each dataset
    unique_synth, counts_synth = np.unique(y_synth, return_counts=True)
    scale_synth = counts_synth[0] / counts_synth[1] if len(counts_synth) == 2 and counts_synth[1] > 0 else 1
    
    unique_kaggle, counts_kaggle = np.unique(y_kaggle, return_counts=True)
    scale_kaggle = counts_kaggle[0] / counts_kaggle[1] if len(counts_kaggle) == 2 and counts_kaggle[1] > 0 else 1
    
    # Experiment A: Train on Kaggle → Test on SynthERA-835
    print("\n  [A] Train on Kaggle → Test on SynthERA-835")
    
    X_train_scaled = scaler.fit_transform(X_kaggle)
    X_test_scaled = scaler.transform(X_synth)
    
    rf_a = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    rf_a.fit(X_train_scaled, y_kaggle)
    y_pred_a = rf_a.predict(X_test_scaled)
    y_prob_a = rf_a.predict_proba(X_test_scaled)[:, 1]
    
    results['Kaggle_to_SynthERA'] = {
        'RandomForest': {
            'auc': roc_auc_score(y_synth, y_prob_a),
            'f1': f1_score(y_synth, y_pred_a, zero_division=0),
            'accuracy': accuracy_score(y_synth, y_pred_a)
        }
    }
    print(f"      RF AUROC: {results['Kaggle_to_SynthERA']['RandomForest']['auc']:.3f}")
    print(f"      RF F1:    {results['Kaggle_to_SynthERA']['RandomForest']['f1']:.3f}")
    
    if XGB_AVAILABLE:
        xgb_a = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   scale_pos_weight=scale_kaggle, random_state=42,
                                   use_label_encoder=False, eval_metric='logloss', verbosity=0)
        xgb_a.fit(X_train_scaled, y_kaggle)
        y_pred_xgb_a = xgb_a.predict(X_test_scaled)
        y_prob_xgb_a = xgb_a.predict_proba(X_test_scaled)[:, 1]
        
        results['Kaggle_to_SynthERA']['XGBoost'] = {
            'auc': roc_auc_score(y_synth, y_prob_xgb_a),
            'f1': f1_score(y_synth, y_pred_xgb_a, zero_division=0),
            'accuracy': accuracy_score(y_synth, y_pred_xgb_a)
        }
        print(f"      XGB AUROC: {results['Kaggle_to_SynthERA']['XGBoost']['auc']:.3f}")
    
    # Experiment B: Train on SynthERA-835 → Test on Kaggle
    print("\n  [B] Train on SynthERA-835 → Test on Kaggle")
    
    X_train_scaled = scaler.fit_transform(X_synth)
    X_test_scaled = scaler.transform(X_kaggle)
    
    rf_b = RandomForestClassifier(n_estimators=100, max_depth=10, 
                                   class_weight='balanced', random_state=42, n_jobs=-1)
    rf_b.fit(X_train_scaled, y_synth)
    y_pred_b = rf_b.predict(X_test_scaled)
    y_prob_b = rf_b.predict_proba(X_test_scaled)[:, 1]
    
    results['SynthERA_to_Kaggle'] = {
        'RandomForest': {
            'auc': roc_auc_score(y_kaggle, y_prob_b),
            'f1': f1_score(y_kaggle, y_pred_b, zero_division=0),
            'accuracy': accuracy_score(y_kaggle, y_pred_b)
        }
    }
    print(f"      RF AUROC: {results['SynthERA_to_Kaggle']['RandomForest']['auc']:.3f}")
    print(f"      RF F1:    {results['SynthERA_to_Kaggle']['RandomForest']['f1']:.3f}")
    
    if XGB_AVAILABLE:
        xgb_b = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1,
                                   scale_pos_weight=scale_synth, random_state=42,
                                   use_label_encoder=False, eval_metric='logloss', verbosity=0)
        xgb_b.fit(X_train_scaled, y_synth)
        y_pred_xgb_b = xgb_b.predict(X_test_scaled)
        y_prob_xgb_b = xgb_b.predict_proba(X_test_scaled)[:, 1]
        
        results['SynthERA_to_Kaggle']['XGBoost'] = {
            'auc': roc_auc_score(y_kaggle, y_prob_xgb_b),
            'f1': f1_score(y_kaggle, y_pred_xgb_b, zero_division=0),
            'accuracy': accuracy_score(y_kaggle, y_pred_xgb_b)
        }
        print(f"      XGB AUROC: {results['SynthERA_to_Kaggle']['XGBoost']['auc']:.3f}")
    
    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("IEEE REVIEW EXPERIMENTS")
    print("Cross-Domain Validation & Confidence Intervals")
    print("=" * 60)
    
    all_results = {}
    
    # Load datasets
    try:
        df_synth = load_synthera835()
        X_synth, y_synth, _ = prepare_common_features_synthera(df_synth)
    except Exception as e:
        print(f"ERROR loading SynthERA-835: {e}")
        return
    
    try:
        df_kaggle = load_kaggle_claims()
        X_kaggle, y_kaggle, _ = prepare_common_features_kaggle(df_kaggle)
    except Exception as e:
        print(f"ERROR loading Kaggle: {e}")
        X_kaggle, y_kaggle = None, None
    
    # Experiment 1: 5-Fold CV with CI
    print("\n" + "=" * 60)
    print("EXPERIMENT 1: SynthERA-835 5-Fold CV")
    print("=" * 60)
    cv_results = run_cv_with_ci(X_synth, y_synth)
    all_results['synthera_cv'] = cv_results
    
    # Experiments 2 & 3: Cross-Domain
    if X_kaggle is not None:
        cross_results = run_cross_domain_experiments(X_synth, y_synth, X_kaggle, y_kaggle)
        all_results['cross_domain'] = cross_results
    
    # Save results
    output_path = Path(__file__).parent / "ieee_review_results.json"
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=float)
    
    # Summary for paper
    print("\n" + "=" * 60)
    print("SUMMARY FOR IEEE PAPER")
    print("=" * 60)
    
    print("\n### Table III: 5-Fold CV with Confidence Intervals")
    print("| Model | AUROC | Accuracy | Precision | Recall | F1 |")
    print("|-------|-------|----------|-----------|--------|-----|")
    for model, m in cv_results.items():
        print(f"| {model} | {m['auc_mean']:.3f}±{m['auc_std']:.3f} | "
              f"{m['accuracy_mean']:.1%}±{m['accuracy_std']:.1%} | "
              f"{m['precision_mean']:.1%}±{m['precision_std']:.1%} | "
              f"{m['recall_mean']:.1%}±{m['recall_std']:.1%} | "
              f"{m['f1_mean']:.3f}±{m['f1_std']:.3f} |")
    
    if 'cross_domain' in all_results:
        print("\n### Table VI: Cross-Domain Transfer Learning")
        print("| Training Set | Test Set | Model | AUROC | F1 |")
        print("|--------------|----------|-------|-------|-----|")
        for exp_name, models in all_results['cross_domain'].items():
            train_set = exp_name.split('_to_')[0].replace('_', ' ')
            test_set = exp_name.split('_to_')[1].replace('_', '-')
            for model_name, metrics in models.items():
                print(f"| {train_set} | {test_set} | {model_name} | "
                      f"{metrics['auc']:.3f} | {metrics['f1']:.3f} |")
    
    print(f"\n✓ Results saved to: {output_path}")


if __name__ == "__main__":
    main()
