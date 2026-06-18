import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd

# Read sheets
esr_df = pd.read_excel("EDM2.xlsx", sheet_name="External Surface Roughness")
srr_df = pd.read_excel("EDM2.xlsx", sheet_name="Substance Removal Rate SRR")
kg_df  = pd.read_excel("EDM2.xlsx", sheet_name="Kerf Gap KG")  # or Kerf Width sheet name

datasets = {
    "ESR": {
        "Actual": esr_df["Exp.Value"].values,
        "Predicted": esr_df["Ensemble"].values
    },

    "srr": {
        "Actual": srr_df["Exp.Value"].values,
        "Predicted": srr_df["Ensemble"].values
    },

    "KG": {
        "Actual": kg_df["Exp.Value"].values,
        "Predicted": kg_df["Ensemble"].values
    }
}

# ===== PERFORMANCE METRICES =====
# Extract actual and predicted values

esr_actual = datasets["ESR"]["Actual"]
esr_pred   = datasets["ESR"]["Predicted"]

srr_actual = datasets["srr"]["Actual"]
srr_pred   = datasets["srr"]["Predicted"]

kg_actual = datasets["KG"]["Actual"]
kg_pred   = datasets["KG"]["Predicted"]

# ===== PERFORMANCE METRICS =====
targets = {
    "ESR": (esr_actual, esr_pred),
    "SRR": (srr_actual, srr_pred),
    "KG": (kg_actual, kg_pred)
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

# ===== Plot 1: ESR =====
plot_actual_vs_pred(esr_actual, esr_pred, "ESR",
                    performance_metrics["ESR"])

# ===== Plot 2: SRR =====
plot_actual_vs_pred(srr_actual, srr_pred, "SRR",
                    performance_metrics["SRR"])

# ===== Plot 3: KG =====
plot_actual_vs_pred(kg_actual, kg_pred, "KG",
                    performance_metrics["KG"])

# ===== Print Performance summary =====
print("\nEnsemble Model Performance Summary")
print("=" * 55)
print(f"{'Target':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 55)

for name, m in performance_metrics.items():
    print(f"{name:<25} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['R²']:<10.4f}")

print("=" * 55)