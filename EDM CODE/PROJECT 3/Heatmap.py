import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ===== DATA =====

datasets = {
    "SU": pd.read_excel('EDM3.xlsx', sheet_name='Surface Undulation Su'),
    "SVR": pd.read_excel('EDM3.xlsx', sheet_name='Substance Vaporization Rate SVR'),
    "KC": pd.read_excel('EDM3.xlsx', sheet_name='Kerf Clearence KC')
}

# ===== DATAFRAMES =====
datasets = {name: pd.DataFrame(values) for name, values in datasets.items()}


plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14

# ===== TITLE AND AXIS =====
for name, df in datasets.items():
    plt.figure(figsize=(12, 4))

    heat_matrix = np.vstack([df['Exp.Value'].values, df['Ensemble'].values])

    im = plt.imshow(heat_matrix, aspect='auto', cmap='Blues')
    plt.xticks(fontsize=12, fontweight='bold')
    plt.yticks(fontsize=12, fontweight='bold')
    plt.yticks([0, 1], ['Actual', 'Predicted'], fontsize=14, fontweight='bold')
    plt.xlabel('Sample Index', fontsize=14, fontweight='bold')
    plt.title(f'{name}: Actual vs Predicted Heatmap',
              fontsize=16, fontweight='bold')
    cbar = plt.colorbar(im)
    cbar.set_label('Value', fontsize=18, fontweight='bold')
    cbar.ax.tick_params(labelsize=13)
    plt.tight_layout()
    plt.show()
