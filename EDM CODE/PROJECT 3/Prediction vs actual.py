import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Read sheets
su_df = pd.read_excel("EDM3.xlsx", sheet_name="Surface Undulation Su")
svr_df = pd.read_excel("EDM3.xlsx", sheet_name="Substance Vaporization Rate SVR")
kc_df  = pd.read_excel("EDM3.xlsx", sheet_name="Kerf Clearence KC")  # or Kerf Width sheet name

datasets = {
    "Su": {
        "Actual": su_df["Exp.Value"].values,
        "Predicted": su_df["Ensemble"].values
    },

    "svr": {
        "Actual": svr_df["Exp.Value"].values,
        "Predicted": svr_df["Ensemble"].values
    },

    "KC": {
        "Actual": kc_df["Exp.Value"].values,
        "Predicted": kc_df["Ensemble"].values
    }
}

# ===== PERFORMANCE METRICES =====
# Extract actual and predicted values

su_actual = datasets["Su"]["Actual"]
su_pred   = datasets["Su"]["Predicted"]

svr_actual = datasets["svr"]["Actual"]
svr_pred   = datasets["svr"]["Predicted"]

kc_actual = datasets["KC"]["Actual"]
kc_pred   = datasets["KC"]["Predicted"]

# ===== PERFORMANCE METRICS =====
targets = {
    "SU": (su_actual, su_pred),
    "SVR": (svr_actual, svr_pred),
    "KC": (kc_actual, kc_pred)
}

performance_metrics = {}
for name, (y_true, y_pred) in targets.items():
    rmse = mean_squared_error(y_true, y_pred)**0.5
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    performance_metrics[name] = {"RMSE": rmse, "MAE": mae, "R²": r2}

# ===== PLOT FUNCTION =====
def plot_actual_vs_pred(y_true, y_pred, title, metrics):
    plt.figure(figsize=(6, 5))
    plt.scatter(y_true, y_pred, color="blue", edgecolor="k", alpha=0.7, s=60)

    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], "r--", lw=1.5)

    plt.title(f"{title}\nEnsemble Prediction", fontsize=16, fontweight="bold")
    plt.xlabel("Actual Values", fontsize=14, fontweight="bold")
    plt.ylabel("Predicted Values", fontsize=14, fontweight="bold")
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    # Add metrics box
    metrics_text = f"RMSE: {metrics['RMSE']:.4f}\nMAE: {metrics['MAE']:.4f}\nR²: {metrics['R²']:.4f}"
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes,
             fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.grid(True, linestyle="--", alpha=0.3)
    plt.tight_layout()
    plt.show()

# ===== Plot 1:  =====
plot_actual_vs_pred(su_actual, su_pred, "SU",
                    performance_metrics["SU"])

# ===== Plot 2: SVR =====
plot_actual_vs_pred(svr_actual, svr_pred, "SVR",
                    performance_metrics["SVR"])

# ===== Plot 3: KC =====
plot_actual_vs_pred(kc_actual, kc_pred, "KC",
                    performance_metrics["KC"])

# ===== Print Performance summary =====
print("\nEnsemble Model Performance Summary")
print("=" * 55)
print(f"{'Target':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 55)

for name, m in performance_metrics.items():
    print(f"{name:<25} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['R²']:<10.4f}")

print("=" * 55)