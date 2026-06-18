import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ===== Data =====
datasets = {
    "Surface Roughness": {
        'Exp.value': [4.137, 3.752, 2.458, 3.795, 4.225, 3.782, 2.692, 3.320, 3.561, 3.885,
                      3.431, 3.354, 2.777, 3.645, 3.284, 3.279, 3.859, 3.774, 4.577, 4.487,
                      2.209, 2.285, 4.026, 3.937, 3.331, 3.415, 4.829, 3.638, 3.925, 4.910,
                      3.725, 4.017, 5.060, 4.944, 2.963, 3.042, 4.722, 4.812, 3.895, 3.805,
                      3.531, 3.612],
        'Ensemble': [4.155, 3.775, 2.527, 3.670, 4.155, 3.775, 2.742, 3.099, 3.524, 3.768,
                     3.426, 3.369, 3.099, 3.513, 3.513, 3.340, 3.873, 3.873, 4.489, 4.489,
                     2.433, 2.322, 4.002, 4.002, 3.421, 3.421, 3.998, 3.998, 4.495, 4.495,
                     3.998, 3.938, 4.797, 4.874, 3.022, 3.022, 4.717, 4.717, 3.914, 3.863,
                     3.615, 3.615]
    },
    "MRR": {
        'Exp.value': [5.866, 5.556, 4.389, 5.695, 5.957, 5.653, 4.919, 5.382, 5.442, 5.737,
                      5.544, 5.561, 5.024, 5.535, 5.643, 5.172, 5.736, 5.644, 5.901, 5.797,
                      4.290, 4.393, 5.728, 5.683, 5.338, 5.447, 6.589, 5.517, 5.752, 6.668,
                      5.562, 5.850, 6.577, 6.467, 4.848, 4.950, 6.258, 6.331, 5.773, 5.684,
                      5.519, 5.614],
        'Ensemble': [5.878, 5.610, 4.482, 5.618, 5.878, 5.610, 4.965, 5.212, 5.425, 5.622,
                     5.519, 5.526, 5.212, 5.568, 5.568, 5.269, 5.744, 5.744, 5.824, 5.824,
                     4.497, 4.423, 5.725, 5.725, 5.447, 5.447, 5.826, 5.826, 6.215, 6.215,
                     5.826, 5.775, 6.413, 6.427, 4.915, 4.915, 6.255, 6.255, 5.794, 5.757,
                     5.591, 5.591]
    },
    "Kerf Width": {
        'Exp.value': [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
                      1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
                      0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
                      1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
                      1.222, 1.241],
        'Ensemble': [1.364, 1.275, 0.960, 1.265, 1.364, 1.275, 0.996, 1.002, 1.215, 1.280,
                     1.192, 1.155, 1.002, 1.226, 1.226, 1.189, 1.295, 1.295, 1.479, 1.479,
                     0.934, 0.921, 1.322, 1.322, 1.193, 1.193, 1.344, 1.344, 1.466, 1.466,
                     1.344, 1.318, 1.539, 1.569, 1.093, 1.093, 1.529, 1.529, 1.323, 1.322,
                     1.240, 1.240]
    }
}

# ===== PLOTTING =====
for name, data in datasets.items():
    df = pd.DataFrame(data)
    y_true = df['Exp.value'].values
    y_pred = df['Ensemble'].values

    residuals = y_true - y_pred
    std = np.std(residuals)
    pi_upper = y_pred + 1.96 * std
    pi_lower = y_pred - 1.96 * std

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # ===== TITLE AND AXIS LABEL =====
    plt.figure(figsize=(12, 6))
    x = np.arange(len(y_true))
    plt.plot(x, y_true, "o-", color='blue', lw=2, markersize=5, label="Experimental")
    plt.plot(x, y_pred, "s--", color='red', lw=2, markersize=5, label="Ensemble Prediction")
    plt.fill_between(x, pi_lower, pi_upper, color='gray', alpha=0.3, label="95% Prediction Interval")

    plt.xlabel("Sample Index", fontsize=14, fontweight='bold')
    plt.ylabel(name, fontsize=14, fontweight='bold')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.title(f"{name}: Experimental vs Ensemble Prediction", fontsize=16, fontweight='bold')

    textstr = f"R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}"
    plt.text(0.02, 0.95, textstr, transform=plt.gca().transAxes,
             fontsize=12, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.legend()
    plt.tight_layout()
    plt.show()
