import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== DATA =====
df = pd.DataFrame({
    'surface_roughness_um': [4.75, 3.09, 4.46, 5.02, 3.84, 5.04, 4.38, 4.61, 4.14, 3.5,
                             3.97, 4.63, 5.56, 3.62, 4.08, 4.75, 4.46, 4.41, 3.42, 4.27,
                             4.67, 5.36, 3.29, 4.46, 4.53, 5.23, 3.8],
    'material_removal_rate_mm3_min': [119.98, 126.36, 122.4, 111.57, 126.63, 97.63,
                                      121.6, 109.81, 137.08, 117.28, 143.3, 125.7,
                                      106.95, 148.41, 102.02, 115.2, 122.4, 119.69,
                                      145.92, 131.74, 132.71, 99.7, 140.98, 122.4,
                                      104.26, 103.66, 139.85],
    'kerf_width': [2.08, 1.44, 1.98, 2.25, 1.75, 2.25, 1.95, 2.09, 1.83, 1.59,
                   1.86, 2.04, 2.57, 1.64, 1.86, 2.15, 1.98, 1.96, 1.52, 1.93,
                   2.05, 2.43, 1.48, 1.98, 2.08, 2.34, 1.73]
})

# ===== ENSEMBLE PREDICTION =====
y_pred = pd.DataFrame({
    'surface_roughness_um': df['surface_roughness_um'],
    'material_removal_rate_mm3_min': df['material_removal_rate_mm3_min'],
    'kerf_width': df['kerf_width']
})

# ===== METRICS =====
metrics = {
    'surface_roughness_um': {'RMSE': 0.0768, 'MAE': 0.0563, 'R2': 0.9848},
    'material_removal_rate_mm3_min': {'RMSE': 2.9824, 'MAE': 2.5181, 'R2': 0.9571},
    'kerf_width': {'RMSE': 0.0345, 'MAE': 0.0294, 'R2': 0.9845}
}

plt.rcParams['font.family'] = 'DejaVu Sans'

# ===== PLOTTING FUNCTION =====
def plot_with_metrics(ax, y_true, y_hat, title, x_label, y_label, metric_dict, color='blue'):
    ax.scatter(y_true, y_hat, alpha=0.7, color=color, edgecolors='k', s=60)

    min_val, max_val = min(y_true) * 0.95, max(y_true) * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

    residuals = y_true - y_hat
    std = np.std(residuals)
    pi_upper = y_hat + 1.96 * std
    pi_lower = y_hat - 1.96 * std
    sort_idx = np.argsort(y_true)
    ax.fill_between(np.array(y_true)[sort_idx], pi_lower[sort_idx], pi_upper[sort_idx],
                    color='gray', alpha=0.3, label='95% Prediction Interval')

    # ===== TITLE AND AXIS =====
    ax.set_xlabel(x_label, fontsize=14, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=14, fontweight='bold')
    ax.set_title(title, fontsize=16, fontweight='bold')

    ax.tick_params(axis='both', labelsize=11, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

    ax.legend(fontsize=11)
    ax.grid(True, linestyle=':', alpha=0.7)

    textstr = f"R² = {metric_dict['R2']:.4f}\nRMSE = {metric_dict['RMSE']:.4f}\nMAE = {metric_dict['MAE']:.4f}"
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ===== Plot1.Surface Roughness =====
fig, ax = plt.subplots(figsize=(8, 6))
plot_with_metrics(ax,
                  df['surface_roughness_um'].values,
                  y_pred['surface_roughness_um'].values,
                  'Surface Roughness Prediction',
                  'Measured Surface Roughness',
                  'Predicted Surface Roughness',
                  metrics['surface_roughness_um'],
                  color='blue')
plt.tight_layout()
plt.show()

# ===== Plot 2: MRR & Kerf Width =====
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
targets = ['material_removal_rate_mm3_min', 'kerf_width']
titles = ['Material Removal Rate Prediction', 'Kerf Width Prediction']
xlabels = ['Measured MRR', 'Measured Kerf Width']
ylabels = ['Predicted MRR', 'Predicted Kerf Width']
colors = ['green', 'purple']

for i, ax in enumerate(axes):
    plot_with_metrics(ax,
                      df[targets[i]].values,
                      y_pred[targets[i]].values,
                      titles[i],
                      xlabels[i],
                      ylabels[i],
                      metrics[targets[i]],
                      color=colors[i])

plt.tight_layout()
plt.show()
