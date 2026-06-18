import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== DATA =====
datasets = {
    "Surface Roughness": {
        'Actual': [4.137, 3.752, 2.458, 3.795, 4.225, 3.782, 2.692, 3.320, 3.561, 3.885,
                   3.431, 3.354, 2.777, 3.645, 3.284, 3.279, 3.859, 3.774, 4.577, 4.487,
                   2.209, 2.285, 4.026, 3.937, 3.331, 3.415, 4.829, 3.638, 3.925, 4.910,
                   3.725, 4.017, 5.060, 4.944, 2.963, 3.042, 4.722, 4.812, 3.895, 3.805,
                   3.531, 3.612],
        'Predicted': [4.155, 3.775, 2.527, 3.670, 4.155, 3.775, 2.742, 3.099, 3.524, 3.768,
                      3.426, 3.369, 3.099, 3.513, 3.513, 3.340, 3.873, 3.873, 4.489, 4.489,
                      2.433, 2.322, 4.002, 4.002, 3.421, 3.421, 3.998, 3.998, 4.495, 4.495,
                      3.998, 3.938, 4.797, 4.874, 3.022, 3.022, 4.717, 4.717, 3.914, 3.863,
                      3.615, 3.615]
    },

    "MRR": {
        'Actual': [5.866, 5.556, 4.389, 5.695, 5.957, 5.653, 4.919, 5.382, 5.442, 5.737,
                   5.544, 5.561, 5.024, 5.535, 5.643, 5.172, 5.736, 5.644, 5.901, 5.797,
                   4.290, 4.393, 5.728, 5.683, 5.338, 5.447, 6.589, 5.517, 5.752, 6.668,
                   5.562, 5.850, 6.577, 6.467, 4.848, 4.950, 6.258, 6.331, 5.773, 5.684,
                   5.519, 5.614],
        'Predicted': [5.878, 5.610, 4.482, 5.618, 5.878, 5.610, 4.965, 5.212, 5.425, 5.622,
                      5.519, 5.526, 5.212, 5.568, 5.568, 5.269, 5.744, 5.744, 5.824, 5.824,
                      4.497, 4.423, 5.725, 5.725, 5.447, 5.447, 5.826, 5.826, 6.215, 6.215,
                      5.826, 5.775, 6.413, 6.427, 4.915, 4.915, 6.255, 6.255, 5.794, 5.757,
                      5.591, 5.591]
    },

    "Kerf Width": {
        'Actual': [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
                   1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
                   0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
                   1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
                   1.222, 1.241],
        'Predicted': [1.364, 1.275, 0.960, 1.265, 1.364, 1.275, 0.996, 1.002, 1.215, 1.280,
                      1.192, 1.155, 1.002, 1.226, 1.226, 1.189, 1.295, 1.295, 1.479, 1.479,
                      0.934, 0.921, 1.322, 1.322, 1.193, 1.193, 1.344, 1.344, 1.466, 1.466,
                      1.344, 1.318, 1.539, 1.569, 1.093, 1.093, 1.529, 1.529, 1.323, 1.322,
                      1.240, 1.240]
    }
}

surface_actual = np.array(datasets["Surface Roughness"]["Actual"])
surface_pred   = np.array(datasets["Surface Roughness"]["Predicted"])

mrr_actual = np.array(datasets["MRR"]["Actual"])
mrr_pred   = np.array(datasets["MRR"]["Predicted"])

kerf_actual = np.array(datasets["Kerf Width"]["Actual"])
kerf_pred   = np.array(datasets["Kerf Width"]["Predicted"])

# ===== PERFORMANCE METRICES =====
targets = {
    "Surface Roughness": (surface_actual, surface_pred),
    "MRR": (mrr_actual, mrr_pred),
    "Kerf Width": (kerf_actual, kerf_pred)
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

# ===== Plot 1: Surface Roughness =====
plot_actual_vs_pred(surface_actual, surface_pred, "Surface Roughness",
                    performance_metrics["Surface Roughness"])

# ===== Plot 2: MRR =====
plot_actual_vs_pred(mrr_actual, mrr_pred, "MRR",
                    performance_metrics["MRR"])

# ===== Plot 3: Kerf Width =====
plot_actual_vs_pred(kerf_actual, kerf_pred, "Kerf Width",
                    performance_metrics["Kerf Width"])

# ===== Print Performance summary =====
print("\nEnsemble Model Performance Summary")
print("=" * 55)
print(f"{'Target':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
print("-" * 55)

for name, m in performance_metrics.items():
    print(f"{name:<25} {m['RMSE']:<10.4f} {m['MAE']:<10.4f} {m['R²']:<10.4f}")

print("=" * 55)