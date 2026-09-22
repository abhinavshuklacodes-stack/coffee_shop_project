"""
Coffee Shop Revenue Predictor - Model Training Pipeline
=======================================================
Business Problem:
  Predict the DAILY total revenue for a specific store location,
  given: store, day of week, month, and week of year.

This turns 149,116 raw transactions into 543 daily aggregated records,
then trains a Linear Regression model on the resulting time-aware features.

Run:    python train_model.py
Output: model/model.pkl, model/scaler.pkl, model/le_store.pkl,
        model/metrics.json, model/diagnostics.png
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import json
import os

# ── Directory setup ──────────────────────────────────────────────
os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

# ── Load & Aggregate to Daily Level ─────────────────────────────
print("Loading dataset...")
df = pd.read_csv('data/Coffee_Shop_Sales.csv')
print(f"  Raw transactions loaded: {len(df):,}")

df['transaction_date'] = pd.to_datetime(df['transaction_date'], format='%m/%d/%y')
df['revenue']          = df['unit_price'] * df['transaction_qty']

# Aggregate to one row per (store, day)
daily = df.groupby(['store_location', 'transaction_date']).agg(
    daily_revenue   =('revenue',          'sum'),
    num_transactions=('transaction_id',   'count'),
    avg_qty         =('transaction_qty',  'mean'),
    avg_unit_price  =('unit_price',       'mean'),
).reset_index()

daily['day_of_week']  = daily['transaction_date'].dt.dayofweek   # 0=Mon
daily['month']        = daily['transaction_date'].dt.month
daily['week_of_year'] = daily['transaction_date'].dt.isocalendar().week.astype(int)

print(f"  Daily aggregated records: {len(daily)}")
print(f"  Daily revenue  — min: ${daily['daily_revenue'].min():.2f}  "
      f"max: ${daily['daily_revenue'].max():.2f}  "
      f"mean: ${daily['daily_revenue'].mean():.2f}")

# ── Encode Store Location ────────────────────────────────────────
le_store            = LabelEncoder()
daily['store_enc']  = le_store.fit_transform(daily['store_location'])
print(f"  Stores: {list(le_store.classes_)}")

# ── Features & Target ────────────────────────────────────────────
FEATURES = ['store_enc', 'day_of_week', 'month', 'week_of_year']
X = daily[FEATURES]
y = daily['daily_revenue']

# ── Train / Test Split ───────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print(f"\n  Training samples: {len(X_train)}")
print(f"  Test samples    : {len(X_test)}")

# ── Feature Scaling ──────────────────────────────────────────────
scaler         = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

# ── Model Training ───────────────────────────────────────────────
print("\nTraining Linear Regression model...")
model = LinearRegression()
model.fit(X_train_scaled, y_train)

# ── Evaluation ───────────────────────────────────────────────────
y_pred = model.predict(X_test_scaled)

r2   = r2_score(y_test, y_pred)
mae  = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print(f"\n  R² Score : {r2:.4f}")
print(f"  MAE      : ${mae:.2f}")
print(f"  RMSE     : ${rmse:.2f}")

# Model equation
intercept = model.intercept_
coefs     = model.coef_
eq_parts  = " + ".join(
    f"{coefs[i]:.4f} * {FEATURES[i]}" for i in range(len(FEATURES))
)
equation = f"Daily Revenue = {intercept:.2f} + {eq_parts}"
print(f"\n  Model Equation:\n  {equation}")

# ── Save Artefacts ───────────────────────────────────────────────
joblib.dump(model,   'model/model.pkl')
joblib.dump(scaler,  'model/scaler.pkl')
joblib.dump(le_store,'model/le_store.pkl')

# Revenue summary by store for the API
revenue_by_store    = daily.groupby('store_location')['daily_revenue'].sum().round(2).to_dict()
revenue_by_month    = daily.groupby('month')['daily_revenue'].sum().round(2).to_dict()
revenue_by_dow      = daily.groupby('day_of_week')['daily_revenue'].mean().round(2).to_dict()

metrics = {
    'r2':                round(r2,   4),
    'mae':               round(mae,  2),
    'rmse':              round(rmse, 2),
    'intercept':         round(float(intercept), 4),
    'coefficients':      {FEATURES[i]: round(float(coefs[i]), 4) for i in range(len(FEATURES))},
    'equation':          equation,
    'training_samples':  int(len(X_train)),
    'test_samples':      int(len(X_test)),
    'total_records':     int(len(df)),
    'daily_records':     int(len(daily)),
    'store_locations':   list(le_store.classes_),
    'features':          FEATURES,
    'target':            'daily_revenue (sum of unit_price * transaction_qty per store per day)',
    'revenue_by_store':  {k: float(v) for k, v in revenue_by_store.items()},
    'revenue_by_month':  {int(k): float(v) for k, v in revenue_by_month.items()},
    'revenue_by_dow':    {int(k): float(v) for k, v in revenue_by_dow.items()},
    'mean_daily_revenue': round(float(daily['daily_revenue'].mean()), 2),
    'max_daily_revenue':  round(float(daily['daily_revenue'].max()), 2),
    'min_daily_revenue':  round(float(daily['daily_revenue'].min()), 2),
}
with open('model/metrics.json', 'w') as f:
    json.dump(metrics, f, indent=2)

# ── Diagnostic Plots ─────────────────────────────────────────────
sns.set_style('whitegrid')
fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle('Coffee Shop Daily Revenue Model – Diagnostics', fontsize=13, fontweight='bold')

y_test_arr  = np.array(y_test)
residuals   = y_test_arr - y_pred

# 1. Actual vs Predicted
axes[0].scatter(y_test_arr, y_pred, alpha=0.55, color='steelblue', s=28, edgecolors='white', linewidths=0.3)
lo = min(y_test_arr.min(), y_pred.min())
hi = max(y_test_arr.max(), y_pred.max())
axes[0].plot([lo, hi], [lo, hi], 'r--', linewidth=1.8, label='Perfect fit')
axes[0].set_xlabel('Actual Daily Revenue ($)')
axes[0].set_ylabel('Predicted Daily Revenue ($)')
axes[0].set_title(f'Actual vs Predicted  (R²={r2:.3f})')
axes[0].legend()

# 2. Residuals
axes[1].scatter(y_pred, residuals, alpha=0.55, color='darkorange', s=28, edgecolors='white', linewidths=0.3)
axes[1].axhline(0, color='red', linestyle='--', linewidth=1.8)
axes[1].set_xlabel('Predicted Daily Revenue ($)')
axes[1].set_ylabel('Residuals ($)')
axes[1].set_title('Residual Plot')

plt.tight_layout()
plt.savefig('model/diagnostics.png', dpi=120, bbox_inches='tight')
plt.close()
print("\nSaved all model artefacts to model/")
print("Training complete! Run 'python backend/app.py' to start the API.")
