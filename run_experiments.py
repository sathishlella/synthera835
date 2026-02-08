#!/usr/bin/env python3
"""
Cross-Validation and Ablation Experiments

For IEEE Access paper with rigorous experimental validation:
1. 5-Fold Stratified Cross-Validation
2. Feature Ablation Study
3. Different Dataset Sizes (learning curves)

Provides confidence intervals and statistical significance.
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
from datetime import datetime

from sklearn.model_selection import StratifiedKFold, cross_val_score, learning_curve
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, f1_score, make_scorer

try:
    import xgboost as xgb
    XGB_AVAILABLE = True
except ImportError:
    XGB_AVAILABLE = False


def load_and_prepare_data(data_dir: Path):
    """Load and prepare data for experiments."""
    # Load claims CSV (line-level data)
    claims_lines_df = pd.read_csv(data_dir / "claims.csv")
    labels_df = pd.read_csv(data_dir / "labels.csv")
    
    # Aggregate lines to claim level
    claims_df = claims_lines_df.groupby('claim_id').agg({
        'patient_id': 'first',
        'provider_npi': 'first',
        'payer_id': 'first',
        'date_of_service': 'first',
        'charge_amount': 'sum',
        'paid_amount': 'sum',
    }).reset_index()
    
    # Calculate num_lines
    line_counts = claims_lines_df.groupby('claim_id').size().reset_index(name='num_lines')
    claims_df = claims_df.merge(line_counts, on='claim_id')
    
    # Add patient_responsibility
    claims_df['patient_responsibility'] = (
        claims_df['charge_amount'] - claims_df['paid_amount']
    ).clip(lower=0)
    
    # Rename columns
    claims_df = claims_df.rename(columns={
        'charge_amount': 'total_charge',
        'paid_amount': 'total_paid'
    })
    
    # Merge with labels
    df = claims_df.merge(labels_df, on='claim_id', how='inner')
    
    # ================================================================
    # PRE-ADJUDICATION FEATURES ONLY (no data leakage)
    # REMOVED: payment_ratio, patient_ratio - these encode denial outcome!
    # ================================================================
    df['charge_log'] = np.log1p(df['total_charge'])
    df['avg_charge_per_line'] = df['total_charge'] / (df['num_lines'] + 1)
    
    # Date features
    df['date_of_service'] = pd.to_datetime(df['date_of_service'], errors='coerce')
    df['day_of_week'] = df['date_of_service'].dt.dayofweek.fillna(0).astype(int)
    df['month'] = df['date_of_service'].dt.month.fillna(1).astype(int)
    
    # Encode payer
    le = LabelEncoder()
    df['payer_id_encoded'] = le.fit_transform(df['payer_id'].astype(str))
    
    return df


def get_feature_matrix(df: pd.DataFrame, feature_subset: list = None):
    """Extract feature matrix.
    
    Uses only PRE-ADJUDICATION features to avoid data leakage.
    """
    # SAFE features only (known before payer decision)
    all_features = [
        'total_charge', 'num_lines', 'avg_charge_per_line', 'charge_log',
        'payer_id_encoded', 'day_of_week', 'month'
    ]
    
    features = feature_subset if feature_subset else all_features
    X = df[features].values
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, features


def run_cross_validation(X, y, model_name, n_folds=5):
    """Run stratified k-fold cross-validation."""
    print(f"\n  {model_name} - {n_folds}-Fold CV...")
    
    if model_name == "RandomForest":
        model = RandomForestClassifier(
            n_estimators=100, max_depth=10, min_samples_split=5,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
    elif model_name == "XGBoost" and XGB_AVAILABLE:
        unique, counts = np.unique(y, return_counts=True)
        scale_pos_weight = counts[0] / counts[1] if len(counts) == 2 and counts[1] > 0 else 1
        model = xgb.XGBClassifier(
            n_estimators=100, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_pos_weight, random_state=42,
            use_label_encoder=False, eval_metric='logloss', verbosity=0
        )
    else:
        return None
    
    cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)
    
    # Custom AUC scorer
    auc_scorer = make_scorer(roc_auc_score, needs_proba=True)
    f1_scorer = make_scorer(f1_score, average='binary' if len(np.unique(y)) == 2 else 'weighted')
    
    # Run CV
    auc_scores = cross_val_score(model, X, y, cv=cv, scoring=auc_scorer)
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring=f1_scorer)
    
    results = {
        'model': model_name,
        'n_folds': n_folds,
        'auc_mean': float(np.mean(auc_scores)),
        'auc_std': float(np.std(auc_scores)),
        'f1_mean': float(np.mean(f1_scores)),
        'f1_std': float(np.std(f1_scores)),
        'auc_scores': auc_scores.tolist(),
        'f1_scores': f1_scores.tolist()
    }
    
    print(f"    AUC: {results['auc_mean']:.4f} ± {results['auc_std']:.4f}")
    print(f"    F1:  {results['f1_mean']:.4f} ± {results['f1_std']:.4f}")
    
    return results


def run_ablation_study(df, y, feature_names):
    """Feature ablation study - remove one feature at a time."""
    print("\n" + "="*60)
    print("Feature Ablation Study")
    print("="*60)
    
    results = []
    
    # Baseline with all features
    X_all, _ = get_feature_matrix(df, feature_names)
    baseline = run_cross_validation(X_all, y, "RandomForest", n_folds=5)
    baseline['ablated_feature'] = 'none (baseline)'
    results.append(baseline)
    
    # Remove each feature
    for feature in feature_names:
        remaining = [f for f in feature_names if f != feature]
        X_subset, _ = get_feature_matrix(df, remaining)
        
        ablation = run_cross_validation(X_subset, y, "RandomForest", n_folds=5)
        ablation['ablated_feature'] = feature
        ablation['auc_drop'] = baseline['auc_mean'] - ablation['auc_mean']
        results.append(ablation)
        
        print(f"    → Removing {feature}: AUC drop = {ablation['auc_drop']:.4f}")
    
    return results


def run_learning_curve_experiment(X, y, sizes=[0.1, 0.25, 0.5, 0.75, 1.0]):
    """Learning curve experiment - how does performance scale with data?"""
    print("\n" + "="*60)
    print("Learning Curve Experiment")
    print("="*60)
    
    model = RandomForestClassifier(
        n_estimators=100, max_depth=10, min_samples_split=5,
        class_weight='balanced', random_state=42, n_jobs=-1
    )
    
    # Use scikit-learn's learning_curve
    train_sizes, train_scores, test_scores = learning_curve(
        model, X, y,
        train_sizes=sizes,
        cv=5,
        scoring='roc_auc',
        n_jobs=-1,
        random_state=42
    )
    
    results = []
    for i, size in enumerate(train_sizes):
        result = {
            'train_size': int(size),
            'train_auc_mean': float(np.mean(train_scores[i])),
            'train_auc_std': float(np.std(train_scores[i])),
            'test_auc_mean': float(np.mean(test_scores[i])),
            'test_auc_std': float(np.std(test_scores[i]))
        }
        results.append(result)
        print(f"  n={size}: Train AUC={result['train_auc_mean']:.4f}, Test AUC={result['test_auc_mean']:.4f}")
    
    return results


def main():
    print("=" * 60)
    print("SynthERA-835: Cross-Validation & Ablation Experiments")
    print("=" * 60)
    
    # Load data
    data_dir = Path(__file__).parent.parent / "synthera835_output_10k"
    if not data_dir.exists():
        print(f"Error: Dataset not found at {data_dir}")
        return
    
    df = load_and_prepare_data(data_dir)
    print(f"Dataset: {len(df)} claims")
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'dataset_size': len(df)
    }
    
    # Feature names - PRE-ADJUDICATION ONLY (no leakage)
    feature_names = [
        'total_charge', 'num_lines', 'avg_charge_per_line', 'charge_log',
        'payer_id_encoded', 'day_of_week', 'month'
    ]
    
    # =========================================
    # Experiment 1: 5-Fold Cross-Validation
    # =========================================
    print("\n" + "="*60)
    print("Experiment 1: 5-Fold Stratified Cross-Validation")
    print("="*60)
    
    y_denial = df['is_denied'].astype(int).values
    X, _ = get_feature_matrix(df, feature_names)
    
    cv_results = []
    for model_name in ["RandomForest", "XGBoost"]:
        result = run_cross_validation(X, y_denial, model_name, n_folds=5)
        if result:
            cv_results.append(result)
    
    all_results['cross_validation'] = cv_results
    
    # =========================================
    # Experiment 2: Feature Ablation
    # =========================================
    ablation_results = run_ablation_study(df, y_denial, feature_names)
    all_results['ablation_study'] = ablation_results
    
    # =========================================
    # Experiment 3: Learning Curve
    # =========================================
    learning_results = run_learning_curve_experiment(X, y_denial)
    all_results['learning_curve'] = learning_results
    
    # =========================================
    # Save Results
    # =========================================
    output_dir = data_dir / "ml_results"
    output_dir.mkdir(exist_ok=True)
    
    results_file = output_dir / "experiments_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "="*60)
    print("Experiment Summary")
    print("="*60)
    
    print("\n5-Fold CV Results:")
    print("{:<15} {:>12} {:>12}".format("Model", "AUC", "F1"))
    print("-" * 40)
    for r in cv_results:
        print("{:<15} {:>8.4f}±{:<3.4f} {:>8.4f}±{:<3.4f}".format(
            r['model'], r['auc_mean'], r['auc_std'], r['f1_mean'], r['f1_std']
        ))
    
    print("\nTop Feature Importance (by ablation AUC drop):")
    sorted_ablation = sorted(
        [r for r in ablation_results if 'auc_drop' in r],
        key=lambda x: x['auc_drop'],
        reverse=True
    )[:5]
    for r in sorted_ablation:
        print(f"  {r['ablated_feature']}: {r['auc_drop']:.4f}")
    
    print(f"\n✓ Results saved to: {results_file}")
    print("=" * 60)


if __name__ == "__main__":
    main()
