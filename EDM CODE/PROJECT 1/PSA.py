import pandas as pd
import matplotlib.pyplot as plt

# ===== DATASET =====
data = {
    "Pulse On": [12,12,8,8,12,12,8,8,12,12,16,16,8,12,12,16,12,12,16,16,8,8,14,14,12,12,8,8,12,12,8,12,12,16,16,16,16,16,12,12,10,10],
    "Pulse Off": [12,12,14,14,12,12,10,10,14,14,10,10,10,13,13,10,12,12,14,14,14,14,12,12,12,12,10,10,11,11,10,12,11,10,14,14,14,14,12,12,12,12],
    "Servo Voltage": [55,60,70,50,55,60,70,70,60,70,70,70,70,60,60,70,60,60,50,50,50,70,60,60,65,65,50,50,60,60,50,60,60,50,70,70,50,50,50,45,60,60],
    "Applied Current": [4,3.5,5,5,4,3.5,5,5,4,5,5,5,5,4,4,3.5,4.5,4.5,3,3,3,3,4,4,4,4,3,3,4,4,3,4,4,5,3,3,5,5,4,4.5,4,4],
    "Angle of Wire": [60,60,30,90,60,60,90,30,60,90,30,90,30,60,60,90,60,60,90,90,90,90,60,60,60,60,30,30,75,75,30,60,60,90,30,30,30,30,30,45,60,60],
    "Surface Roughness": [4.137,3.752,2.458,3.795,4.225,3.782,2.692,3.320,3.561,3.885,3.431,3.354,2.777,3.645,3.284,3.279,3.859,3.774,4.577,4.487,2.209,2.285,4.026,3.937,3.331,3.415,4.829,3.638,3.925,4.910,3.725,4.017,5.060,4.944,2.963,3.042,4.722,4.812,3.895,3.805,3.531,3.612],
    "MRR": [5.866,5.556,4.389,5.695,5.957,5.653,4.919,5.382,5.442,5.737,5.544,5.561,5.024,5.535,5.643,5.172,5.736,5.644,5.901,5.797,4.290,4.393,5.728,5.683,5.338,5.447,6.589,5.517,5.752,6.668,5.562,5.850,6.577,6.467,4.848,4.950,6.258,6.331,5.773,5.684,5.519,5.614],
    "Kerf": [1.378,1.277,0.943,1.289,1.400,1.286,1.004,0.965,1.227,1.311,1.194,1.135,1.026,1.249,1.219,1.155,1.305,1.283,1.489,1.467,0.879,0.898,1.347,1.325,1.169,1.191,1.556,1.249,1.322,1.576,1.272,1.347,1.615,1.584,1.074,1.093,1.528,1.551,1.314,1.291,1.222,1.241]
}

df = pd.DataFrame(data)
plt.rcParams['font.family'] = 'DejaVu Sans'

def bold_axes(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
    ax.tick_params(axis='both', labelsize=11, width=1.2)
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight('bold')

# ===== Plot 1: Surface Roughness =====
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(df['Pulse On'], df['Pulse Off'],
                     c=df['Surface Roughness'], cmap='viridis',
                     s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
ax.set_xlabel('Pulse On', fontsize=14, fontweight='bold')
ax.set_ylabel('Pulse Off', fontsize=14, fontweight='bold')
ax.set_title('Surface Roughness', fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)
cbar = plt.colorbar(scatter)
cbar.set_label('Surface Roughness', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.show()

# ===== Plot 2: MRR =====
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(df['Servo Voltage'], df['Applied Current'],
                     c=df['MRR'], cmap='plasma',
                     s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
ax.set_xlabel('Servo Voltage', fontsize=14, fontweight='bold')
ax.set_ylabel('Applied Current', fontsize=14, fontweight='bold')
ax.set_title('MRR', fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)
cbar = plt.colorbar(scatter)
cbar.set_label('MRR', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()

# ===== Plot 3: KERF WIDTH =====
fig, ax = plt.subplots(figsize=(8,6))
scatter = ax.scatter(df['Applied Current'], df['Angle of Wire'],
                     c=df['Kerf'], cmap='cool',
                     s=100, alpha=0.8, edgecolors='black', linewidth=1.5)
ax.set_xlabel('Applied Current', fontsize=14, fontweight='bold')
ax.set_ylabel('Angle of Wire', fontsize=14, fontweight='bold')
ax.set_title('Kerf Width', fontsize=16, fontweight='bold')
ax.grid(True, linestyle='--', alpha=0.3)
bold_axes(ax)
cbar = plt.colorbar(scatter)
cbar.set_label('Kerf Width', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.show()
