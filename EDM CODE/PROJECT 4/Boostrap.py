import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Read all sheets
excel_file = pd.ExcelFile("EDM4.xlsx")

for sheet in excel_file.sheet_names:

    # Read current sheet
    df = pd.read_excel("EDM4.xlsx", sheet_name=sheet)

    # Skip sheet if required columns are missing
    if 'Exp.Value' not in df.columns or 'Ensemble' not in df.columns:
        print(f"Skipping {sheet}: Required columns not found")
        continue

    y_true = df['Exp.Value'].values
    y_pred = df['Ensemble'].values

    # Residuals and prediction interval
    residuals = y_true - y_pred
    std = np.std(residuals)

    pi_upper = y_pred + 1.96 * std
    pi_lower = y_pred - 1.96 * std

    # Metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Plot
    plt.figure(figsize=(12, 6))

    x = np.arange(len(y_true))

    plt.plot(x, y_true, 'o-', color='blue',
             linewidth=2, markersize=5,
             label='Experimental')

    plt.plot(x, y_pred, 's--', color='red',
             linewidth=2, markersize=5,
             label='Ensemble Prediction')

    plt.fill_between(x, pi_lower, pi_upper,
                     color='gray', alpha=0.3,
                     label='95% Prediction Interval')

    plt.xlabel("Sample Index", fontsize=14, fontweight='bold')
    plt.ylabel(sheet, fontsize=14, fontweight='bold')

    plt.title(f"{sheet}: Experimental vs Ensemble Prediction",
              fontsize=16, fontweight='bold')

    textstr = (
        f"R² = {r2:.4f}\n"
        f"RMSE = {rmse:.4f}\n"
        f"MAE = {mae:.4f}"
    )

    plt.text(0.02, 0.95, textstr,
             transform=plt.gca().transAxes,
             fontsize=12,
             verticalalignment='top',
             bbox=dict(boxstyle='round',
                       facecolor='wheat',
                       alpha=0.8))

    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')

    plt.grid(True, linestyle=':')
    plt.legend()
    plt.tight_layout()
    plt.show()