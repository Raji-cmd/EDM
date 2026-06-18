import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== DATA =====
datasets = {
    "SW": pd.read_excel('EDM4.xlsx', sheet_name='Surface Waviness SW'),
    "MVR": pd.read_excel('EDM4.xlsx', sheet_name='Material Vaporization Rate MVR'),
    "KD": pd.read_excel('EDM4.xlsx', sheet_name='Kerf Distance KD')
}

# ===== METRICS =====
metrics = {
    'SW': {'RMSE': 0.2078, 'MAE': 0.1262, 'R2': 0.9105},
    'MVR': {'RMSE': 0.1828, 'MAE': 0.1089, 'R2': 0.8841},
    'KD': {'RMSE': 0.0522, 'MAE': 0.0310, 'R2': 0.9200}
}

plt.rcParams['font.family'] = 'DejaVu Sans'

# ===== PLOTTING FUNCTION =====
def plot_with_metrics(ax, y_true, y_hat, title, x_label, y_label, metric_dict, color='blue'):
    y_true = np.array(y_true).flatten()
    y_hat = np.array(y_hat).flatten()

    ax.scatter(y_true, y_hat, alpha=0.7, color=color, edgecolors='k', s=60)

    min_val, max_val = y_true.min() * 0.95, y_true.max() * 1.05
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Ideal Prediction')

    residuals = y_true - y_hat
    std = np.std(residuals)

    pi_upper = y_hat + 1.96 * std
    pi_lower = y_hat - 1.96 * std

    sort_idx = np.argsort(y_true)

    ax.fill_between(
        y_true[sort_idx],
        pi_lower[sort_idx],
        pi_upper[sort_idx],
        color='gray',
        alpha=0.3,
        label='95% Prediction Interval'
    )

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


# ===== SW PLOT =====
fig, ax = plt.subplots(figsize=(8, 6))

plot_with_metrics(
    ax,
    datasets['SW']['Exp.Value'],
    datasets['SW']['Ensemble'],
    'SW Prediction',
    'Measured SW',
    'Predicted SW',
    metrics['SW'],
    color='blue'
)

plt.tight_layout()
plt.show()


# ===== MVR & KD PLOT =====
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

targets = ['MVR', 'KD']
titles = ['MVR Prediction', 'KD Prediction']
colors = ['green', 'purple']

for i, ax in enumerate(axes):
    plot_with_metrics(
        ax,
        datasets[targets[i]]['Exp.Value'],
        datasets[targets[i]]['Ensemble'],
        titles[i],
        f'Measured {targets[i]}',
        f'Predicted {targets[i]}',
        metrics[targets[i]],
        color=colors[i]
    )

plt.tight_layout()
plt.show()