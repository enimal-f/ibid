#!/usr/bin/env python3
"""
================================================================================
I-BID — Multi-Variable Linear Regression for Compressive Strength Prediction
================================================================================
Predicts 2-day, 7-day, and 28-day compressive strengths from:
  • Blaine (fineness, cm²/g)
  • SO₃ (sulfate content, %)
  • PAF (loss on ignition, %)
  • Insoluble residue (proxy for granite/calcaire ratio, %)
  • Refus 45µm (particle size, %)
  • Humidity (%)
  • K/C proxy (clinker ratio, inferred from product type)
  • Product type dummy variables

Data source: PRODUCTION MOYENNE 2024 & 2025 SITE 1 (lab quality data)
Author: I-BID Analytics Module
================================================================================
"""

import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# ──────────────────────────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────────────────────────
QUALITY_DATA_PATH = "quality_data.json"
OUTPUT_SUMMARY_PATH = "strength_regression_summary.json"
OUTPUT_PREDICTIONS_PATH = "strength_predictions.json"
OUTPUT_MODEL_PATH = "strength_regression_models.pkl"

# K/C proxy values by product (industry-standard approximations for CEM II/B-M)
KC_PROXY = {
    'CPA 52.5': 0.95,   # CEM I — nearly pure clinker
    'CPJ 42.5': 0.70,   # CEM II/B-M — ~70% clinker
    'CPJ 32.5': 0.60,   # CEM II/B-M — ~60% clinker (more additions)
}

FEATURE_COLS = ['blaine', 'so3', 'paf', 'res_insoluble', 'refus_45um', 'humidite']
TARGET_COLS = ['resistance_2j', 'resistance_7j', 'resistance_28j']

# ──────────────────────────────────────────────────────────────────────────────
# LOAD & PREPARE DATA
# ──────────────────────────────────────────────────────────────────────────────
print("=" * 70)
print(" I-BID STRENGTH PREDICTION — MULTI-VARIABLE LINEAR REGRESSION")
print("=" * 70)

with open(QUALITY_DATA_PATH, 'r') as f:
    quality_data = json.load(f)

df = pd.DataFrame(quality_data)
df['date'] = pd.to_datetime(df['date'])

# Filter to valid date range (remove erroneous OCR dates)
df = df[(df['date'] >= '2024-01-01') & (df['date'] <= '2025-12-31')]
print(f"\n📊 Loaded {len(df)} quality records from {df['date'].min().date()} to {df['date'].max().date()}")

# Add K/C proxy
df['kc_proxy'] = df['product'].map(KC_PROXY)

# Add product dummies
product_dummies = pd.get_dummies(df['product'], prefix='prod')
df = pd.concat([df, product_dummies], axis=1)

# Build complete feature list
all_features = FEATURE_COLS + ['kc_proxy'] + list(product_dummies.columns)
print(f"🔧 Features: {', '.join(all_features)}")

# Drop incomplete cases
model_df = df[all_features + TARGET_COLS].dropna()
print(f"✅ Complete cases for regression: {len(model_df)}")
print(f"   By product:\n{model_df.groupby(df.loc[model_df.index, 'product']).size()}")

# ──────────────────────────────────────────────────────────────────────────────
# RUN REGRESSIONS
# ──────────────────────────────────────────────────────────────────────────────
X = model_df[all_features].values
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

results = {}
summary = {}

for target in TARGET_COLS:
    y = model_df[target].values
    
    # Fit OLS
    reg = LinearRegression()
    reg.fit(X, y)
    y_pred = reg.predict(X)
    
    # Metrics
    r2 = r2_score(y, y_pred)
    rmse = np.sqrt(mean_squared_error(y, y_pred))
    mae = np.mean(np.abs(y - y_pred))
    
    # Store
    results[target] = {
        'model': reg,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'y_pred': y_pred,
        'y_actual': y,
        'residuals': y - y_pred,
        'coefficients': dict(zip(all_features, reg.coef_)),
        'intercept': reg.intercept_
    }
    
    # Build equation string
    eq_parts = [f"{reg.coef_[i]:+.4f}×{all_features[i]}" for i in range(len(all_features))]
    equation = f"{target} = {reg.intercept_:.4f} " + " ".join(eq_parts)
    
    summary[target] = {
        'r2': round(r2, 4),
        'rmse': round(rmse, 2),
        'mae': round(mae, 2),
        'intercept': round(reg.intercept_, 4),
        'coefficients': {k: round(v, 4) for k, v in zip(all_features, reg.coef_)},
        'equation': equation
    }
    
    print(f"\n{'─' * 70}")
    print(f"  {target.upper().replace('_', ' ')}")
    print(f"{'─' * 70}")
    print(f"  R² = {r2:.4f}  |  RMSE = {rmse:.2f} MPa  |  MAE = {mae:.2f} MPa")
    print(f"  Intercept = {reg.intercept_:.4f}")
    print(f"  Equation: {equation}")

# ──────────────────────────────────────────────────────────────────────────────
# SAVE PREDICTION DATASET
# ──────────────────────────────────────────────────────────────────────────────
model_df['date'] = df.loc[model_df.index, 'date'].values
model_df['product'] = df.loc[model_df.index, 'product'].values

for target in TARGET_COLS:
    model_df[f'{target}_pred'] = results[target]['y_pred']
    model_df[f'{target}_resid'] = results[target]['residuals']

pred_data = model_df[['date', 'product'] + all_features + TARGET_COLS +
                     [f'{t}_pred' for t in TARGET_COLS] +
                     [f'{t}_resid' for t in TARGET_COLS]].copy()
pred_data['date'] = pred_data['date'].dt.strftime('%Y-%m-%d')
pred_records = pred_data.to_dict('records')

with open(OUTPUT_PREDICTIONS_PATH, 'w') as f:
    json.dump(pred_records, f, indent=2, default=str)
print(f"\n💾 Predictions saved to: {OUTPUT_PREDICTIONS_PATH}")

with open(OUTPUT_SUMMARY_PATH, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"💾 Summary saved to: {OUTPUT_SUMMARY_PATH}")

import pickle
with open(OUTPUT_MODEL_PATH, 'wb') as f:
    pickle.dump({'results': results, 'scaler': scaler, 'features': all_features, 'model_df': model_df}, f)
print(f"💾 Models saved to: {OUTPUT_MODEL_PATH}")

# ──────────────────────────────────────────────────────────────────────────────
# INTERPRETATION SUMMARY
# ──────────────────────────────────────────────────────────────────────────────
print(f"\n{'=' * 70}")
print(" INTERPRETATION")
print(f"{'=' * 70}")

print("""
• Blaine (+) : Higher fineness → higher strength (more surface area for hydration)
• SO₃ (−)    : Excess sulfate can retard strength development
• PAF (−)    : Higher loss-on-ignition (unburned material) reduces strength
• Res. Insol. (−) : Higher insoluble residue (granite proxy) dilutes clinker strength
• Refus 45µm (−)  : Coarser particles hydrate slower → lower early strength
• Humidité (+)    : Moderate moisture aids early hydration
• K/C proxy (+)   : Higher clinker ratio → higher strength (expected)
""")

print("=" * 70)
print(" DONE")
print("=" * 70)
