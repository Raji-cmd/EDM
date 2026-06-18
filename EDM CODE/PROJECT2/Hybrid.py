import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== DATA =====
df = pd.DataFrame({
    'External Surface Roughness': [4.174,3.775,2.457,3.778,4.174,3.775,2.716,3.080,3.530,3.868,
3.426,3.316,3.080,3.501,3.501,3.299,3.856,3.856,4.513,4.513,
2.229,2.272,4.050,4.050,3.422,3.422,3.999,3.999,4.474,4.474,
3.999,3.974,4.883,4.961,3.004,3.004,4.747,4.747,3.877,3.842,
3.574,3.574],
    'Substance Removal Rate': [5.918,5.631,4.388,5.686,5.918,5.631,4.953,5.214,5.434,5.735,
5.540,5.526,5.214,5.573,5.573,5.202,5.729,5.729,5.829,5.829,
4.309,4.382,5.753,5.753,5.440,5.440,5.821,5.821,6.220,6.220,
5.821,5.813,6.467,6.485,4.899,4.899,6.277,6.277,5.759,5.725,
5.563,5.563],
    'Kerf Gap': [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,
1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,
0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,
1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,
1.222,1.241]
})

# ===== ENSEMBLE PREDICTION =====
y_pred = pd.DataFrame({
    'External Surface Roughness': df['External Surface Roughness'],
    'Substance Removal Rate': df['Substance Removal Rate'],
    'Kerf Gap': df['Kerf Gap']
})

# ===== METRICS =====
metrics = {
    'External Surface Roughness': {'RMSE': 0.2007, 'MAE': 0.1098, 'R2': 0.9165},
    'Substance Removal Rate': {'RMSE': 0.1768, 'MAE': 0.0939, 'R2': 0.8917},
    'Kerf Gap': {'RMSE': 0.0486, 'MAE': 0.0244, 'R2': 0.9306}
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

# ===== Plot1.ESR =====
fig, ax = plt.subplots(figsize=(8, 6))
plot_with_metrics(ax,
                  df['External Surface Roughness'].values,
                  y_pred['External Surface Roughness'].values,
                  'External Surface Roughness Prediction',
                  'Measured ESR',
                  'Predicted ESR',
                  metrics['External Surface Roughness'],
                  color='blue')
plt.tight_layout()
plt.show()

# ===== Plot 2: SRR & Kerf Gap =====
fig, axes = plt.subplots(1, 2, figsize=(16, 6))
targets = ['Substance Removal Rate', 'Kerf Gap']
titles = ['Substance Removal Rate Prediction', 'Kerf Gap Prediction']
xlabels = ['Measured SRR', 'Measured Kerf Gap']
ylabels = ['Predicted SRR', 'Predicted Kerf Gap']
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
