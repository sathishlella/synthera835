# 🔴 Data Leakage Audit Report

## Summary: CRITICAL LEAKAGE FOUND

**AUC = 1.0 is explained by direct label leakage in features.**

---

## The Problem

### How Data is Generated:

```python
# For DENIED claims (_apply_denial, line 356-358):
line.paid_amount = 0.0  # ← ZERO for denied!
line.adjustment_amount = line.charge_amount

# For PAID claims (_apply_payment, line 390):
line.paid_amount = charge_amount - adjustment  # ← NON-ZERO for paid!
```

### How Features are Computed (train_baseline_models.py):

```python
# Line 54-55: Aggregation
'paid_amount': 'sum',  # Sum of line paid amounts

# Line 85-86: Derived features
df['payment_ratio'] = df['total_paid'] / (df['total_charge'] + 0.01)
df['patient_ratio'] = df['patient_responsibility'] / (df['total_charge'] + 0.01)
```

### The Leakage:

| Claim Status | total_paid | payment_ratio |
|--------------|------------|---------------|
| **DENIED** | 0.0 | 0.0 |
| **PAID** | >0 | 0.7-0.9 |

**The model doesn't need to learn anything - it just sees:**
- `payment_ratio ≈ 0` → DENIED
- `payment_ratio > 0.5` → PAID

This is **100% deterministic** - hence AUC = 1.0

---

## Visual Proof

```
Feature: payment_ratio

Denied Claims: ████████████████████████████████ 0.0
Paid Claims:   ░░░░░░░░░░░░░░░░████████████████ 0.7+

                0.0                           1.0
                ↑                             ↑
           Perfect separator              No overlap
```

**There is ZERO overlap between classes on this feature.**

---

## Affected Features (Must Remove)

| Feature | Leakage Type | Action |
|---------|--------------|--------|
| `total_paid` | **DIRECT** - zero for denied | ❌ REMOVE |
| `payment_ratio` | **DIRECT** - derived from total_paid | ❌ REMOVE |
| `patient_responsibility` | **INDIRECT** - set differently | ❌ REMOVE |
| `patient_ratio` | **INDIRECT** - derived | ❌ REMOVE |

## Safe Features (Can Keep)

| Feature | Why Safe |
|---------|----------|
| `total_charge` | Set BEFORE denial decision |
| `num_lines` | Set BEFORE denial decision |
| `avg_charge_per_line` | Derived from safe features |
| `payer_id_encoded` | Set BEFORE denial decision |
| `day_of_week` | Date feature |
| `month` | Date feature |

---

## Expected Results After Fix

| Metric | Before (Leaky) | After (Fixed) |
|--------|----------------|---------------|
| AUC | 1.0000 | ~0.55-0.70 |
| F1 | 1.0000 | ~0.30-0.50 |

**This is REALISTIC.** Denial prediction from pre-adjudication features is HARD. 
If it were easy, insurance companies would already do it.

---

## Code Fix Required

### Option 1: Remove Leaky Features
```python
# In train_baseline_models.py, line 110-113:
feature_cols = [
    'total_charge', 'num_lines',  # Keep
    'avg_charge_per_line',         # Keep 
    # REMOVED: 'total_paid', 'patient_responsibility',
    # REMOVED: 'payment_ratio', 'patient_ratio'
]
```

### Option 2: Use Pre-Adjudication Features Only
Add features that exist BEFORE the payer decides:
- Procedure codes
- Diagnosis codes
- Provider history
- Payer history
- Claim history

---

## Implications for the Paper

| Aspect | Impact |
|--------|--------|
| Results Section | Must re-run all experiments with fixed features |
| Contribution | Now showing "realistic baseline" not "perfect prediction" |
| Novelty | Actually MORE interesting - showing the problem is HARD |
| Related Work | Can compare to actual claims denial literature |

---

## Next Steps

1. [ ] Fix `train_baseline_models.py` to remove leaky features
2. [ ] Re-run experiments with fixed features
3. [ ] Document realistic AUC (expect ~0.55-0.70)
4. [ ] Add more pre-adjudication features (procedure codes etc.)
5. [ ] Update paper claims accordingly

---

*Audit completed: 2026-02-06*
