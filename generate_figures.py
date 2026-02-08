#!/usr/bin/env python3
"""
Generate Figures for SynthERA-835 IEEE Paper
1. ROC Curve (RF vs XGBoost)
2. System Architecture Diagram
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_curve, auc
import xgboost as xgb

# Set style for publication quality
try:
    plt.style.use('seaborn-v0_8-whitegrid')
except:
    try:
        plt.style.use('seaborn-whitegrid')
    except:
        plt.style.use('ggplot')

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'legend.fontsize': 12,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'figure.figsize': (10, 8),
    'lines.linewidth': 2.5
})

def load_and_prep_data():
    """Load data using logic from train_baseline_models.py"""
    data_dir = Path("synthera835_output_10k")
    if not data_dir.exists():
        # Fallback for running from parent dir
        data_dir = Path("..") / "synthera835_output_10k"
    
    print(f"Loading data from {data_dir}...")
    claims_lines_df = pd.read_csv(data_dir / "claims.csv")
    labels_df = pd.read_csv(data_dir / "labels.csv")
    
    # Aggregate
    claims_df = claims_lines_df.groupby('claim_id').agg({
        'charge_amount': 'sum',
        'paid_amount': 'sum', 
        'date_of_service': 'first',
        'payer_id': 'first'
    }).reset_index()
    
    # Line counts
    line_counts = claims_lines_df.groupby('claim_id').size().reset_index(name='num_lines')
    
    # Procedures
    unique_procs = claims_lines_df.groupby('claim_id')['procedure_code'].nunique().reset_index(name='unique_procedures')
    
    # Merge
    df = claims_df.merge(labels_df, on='claim_id').merge(line_counts, on='claim_id').merge(unique_procs, on='claim_id')
    
    # Rename
    df = df.rename(columns={'charge_amount': 'total_charge', 'paid_amount': 'total_paid'})
    
    # Feature Engineering
    df['avg_charge_per_line'] = df['total_charge'] / df['num_lines']
    df['charge_log'] = np.log1p(df['total_charge'])
    
    # Date features
    df['date_of_service'] = pd.to_datetime(df['date_of_service'])
    df['month'] = df['date_of_service'].dt.month
    
    # Encode payer
    le = LabelEncoder()
    df['payer_id_encoded'] = le.fit_transform(df['payer_id'].astype(str))
    
    # Select Features
    features = ['total_charge', 'num_lines', 'avg_charge_per_line', 'charge_log', 
                'unique_procedures', 'payer_id_encoded', 'month']
    
    X = df[features].values
    y = df['is_denied'].astype(int).values
    
    return X, y, features

def plot_roc_curve(X, y):
    """Generate and save ROC Curve"""
    print("Generating ROC Curve...")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train RF
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, class_weight='balanced', random_state=42)
    rf.fit(X_train_scaled, y_train)
    y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]
    fpr_rf, tpr_rf, _ = roc_curve(y_test, y_prob_rf)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    # Train XGBoost
    xgb_model = xgb.XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='logloss')
    xgb_model.fit(X_train_scaled, y_train)
    y_prob_xgb = xgb_model.predict_proba(X_test_scaled)[:, 1]
    fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_prob_xgb)
    roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.plot(fpr_rf, tpr_rf, color='#2c3e50', label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot(fpr_xgb, tpr_xgb, color='#e74c3c', linestyle='--', label=f'XGBoost (AUC = {roc_auc_xgb:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle=':', label='Random Chance')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1 - Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity)')
    plt.title('ROC Curve: Binary Denial Prediction')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Save
    output_path = Path('papers/figures/roc_curve.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to {output_path}")

def plot_architecture_diagram():
    """Generate System Architecture Diagram using basic geometric approach"""
    print("Generating Architecture Diagram...")
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')
    
    # Helper to draw box
    def draw_box(x, y, width, height, color, text, subtext=""):
        rect = plt.Rectangle((x, y), width, height, facecolor=color, edgecolor='black', alpha=0.8)
        ax.add_patch(rect)
        ax.text(x + width/2, y + height/2 + 0.3, text, ha='center', va='center', fontsize=12, fontweight='bold', color='white')
        if subtext:
            ax.text(x + width/2, y + height/2 - 0.3, subtext, ha='center', va='center', fontsize=9, color='white')
    
    # Helper to draw arrow
    def draw_arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#34495e"))

    # Components
    # 1. Config
    draw_box(0.5, 2.5, 2, 1.5, '#3498db', 'Configuration', 'Provider Profile\nPayer Mix\nCommon CPTs')
    
    # Arrow
    draw_arrow(2.5, 3.25, 3.5, 3.25)
    
    # 2. Generator Core
    draw_box(3.5, 2.0, 3, 2.5, '#2c3e50', 'SynthERA-835 Core', 'Patient Generator\nClaim Assembler\nDenial Probability\nModel')
    
    # Arrow
    draw_arrow(6.5, 3.25, 7.5, 3.25)
    
    # 3. Output
    draw_box(7.5, 3.5, 2.5, 1.5, '#e67e22', 'Output Files', 'X12 835 EDI\nCSV Labels')
    
    # Arrow 2 (Denial Logic)
    draw_arrow(5.0, 1.0, 5.0, 2.0)
    
    # 4. Denial Logic
    draw_box(3.5, 0.2, 3, 0.8, '#27ae60', 'Denial Logic', 'Rules + Probabilities')
    
    # Title
    plt.title('SynthERA-835 System Architecture', fontsize=16, pad=20)
    
    # Save
    output_path = Path('papers/figures/architecture.png')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Saved architecture diagram to {output_path}")

if __name__ == "__main__":
    try:
        X, y, features = load_and_prep_data()
        plot_roc_curve(X, y)
        plot_architecture_diagram()
        print("Done!")
    except Exception as e:
        print(f"Error: {e}")
