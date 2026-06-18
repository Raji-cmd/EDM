import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ===== DATA =====
data = {
    "Pulse_On": [12, 12, 8, 8, 12, 12, 8, 8, 12, 12, 16, 16, 8, 12, 12, 16, 12, 12, 16, 16, 8, 8, 14, 14, 12, 12, 8, 8,
                 12, 12, 8, 12, 12, 16, 16, 16, 16, 16, 12, 12, 10, 10],
    "Pulse_Off": [12, 12, 14, 14, 12, 12, 10, 10, 14, 14, 10, 10, 10, 13, 13, 10, 12, 12, 14, 14, 14, 14, 12, 12, 12,
                  12, 10, 10, 11, 11, 10, 12, 11, 10, 14, 14, 14, 14, 12, 12, 12, 12],
    "Servo_V": [55, 60, 70, 50, 55, 60, 70, 70, 60, 70, 70, 70, 70, 60, 60, 70, 60, 60, 50, 50, 50, 70, 60, 60, 65, 65,
                50, 50, 60, 60, 50, 60, 60, 50, 70, 70, 50, 50, 50, 45, 60, 60],
    "Current": [4, 3.5, 5, 5, 4, 3.5, 5, 5, 4, 5, 5, 5, 5, 4, 4, 3.5, 4.5, 4.5, 3, 3, 3, 3, 4, 4, 4, 4, 3, 3, 4, 4, 3,
                4, 4, 5, 3, 3, 5, 5, 4, 4.5, 4, 4],
    "Angle": [60, 60, 30, 90, 60, 60, 90, 30, 60, 90, 30, 90, 30, 60, 60, 90, 60, 60, 90, 90, 90, 90, 60, 60, 60, 60,
              30, 30, 75, 75, 30, 60, 60, 90, 30, 30, 30, 30, 30, 45, 60, 60],
    "Surface Roughness": [4.137, 3.752, 2.458, 3.795, 4.225, 3.782, 2.692, 3.320, 3.561, 3.885, 3.431, 3.354, 2.777,
                          3.645, 3.284, 3.279, 3.859, 3.774, 4.577, 4.487, 2.209, 2.285, 4.026, 3.937, 3.331, 3.415,
                          4.829, 3.638, 3.925, 4.910, 3.725, 4.017, 5.060, 4.944, 2.963, 3.042, 4.722, 4.812, 3.895,
                          3.805, 3.531, 3.612],
    "MRR": [5.866, 5.556, 4.389, 5.695, 5.957, 5.653, 4.919, 5.382, 5.442, 5.737, 5.544, 5.561, 5.024, 5.535, 5.643,
            5.172, 5.736, 5.644, 5.901, 5.797, 4.290, 4.393, 5.728, 5.683, 5.338, 5.447, 6.589, 5.517, 5.752, 6.668,
            5.562, 5.850, 6.577, 6.467, 4.848, 4.950, 6.258, 6.331, 5.773, 5.684, 5.519, 5.614],
    "Kerf Width": [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311, 1.194, 1.135, 1.026, 1.249,
                   1.219,
                   1.155, 1.305, 1.283, 1.489, 1.467, 0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322,
                   1.576,
                   1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291, 1.222, 1.241]
}

df = pd.DataFrame(data)
kerf = df['Kerf Width'].values  # ← FIXED
mrr = df['MRR'].values

# ===== PARETO OPTIMAL IDENTIFICATION =====
def identify_pareto(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] < c, axis=1) | \
                                         np.any(scores[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

scores = np.column_stack([kerf, -mrr])
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([kerf[pareto_mask], mrr[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ===== PLOT =====
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.scatter(kerf, mrr, color='gray', alpha=0.6, s=50, label='Experimental Points', zorder=1)
plt.scatter(pareto_points[:, 0], pareto_points[:, 1], color='blue', s=80,
            edgecolors='black', linewidth=1, label='Pareto-optimal Solutions', zorder=3)
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8, zorder=2)

best_quality_idx = np.argmin(kerf)
best_productivity_idx = np.argmax(mrr)
balanced_idx = np.argmin(np.sqrt((kerf - kerf.min())**2 + (mrr - mrr.max())**2))

key_points = [
    (kerf[best_quality_idx], mrr[best_quality_idx], "Min Kerf", 's', 'red'),
    (kerf[best_productivity_idx], mrr[best_productivity_idx], "Max MRR", '^', 'green'),
    (kerf[balanced_idx], mrr[balanced_idx], "Balanced", 'D', 'purple')
]

for x, y, label, marker, color in key_points:
    plt.scatter(x, y, s=150, marker=marker, color=color, edgecolors='black', linewidth=2, zorder=4)
    plt.annotate(label, xy=(x, y), xytext=(30, -30),
                 textcoords='offset points', fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', lw=1.2))

if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points)
    plt.fill(pareto_points[hull.vertices, 0], pareto_points[hull.vertices, 1],
             'blue', alpha=0.1, label='Feasible Region')

plt.xlabel('Kerf Width', fontsize=14, fontweight='bold')
plt.ylabel('MRR ', fontsize=14, fontweight='bold')
plt.title('Pareto Frontier for EDM Process\nTrade-off between MRR and Kerf Width',
          fontsize=16, fontweight='bold', pad=20)

for axis in ['top', 'bottom', 'left', 'right']:
    plt.gca().spines[axis].set_linewidth(1.5)

plt.xticks(fontsize=12, fontweight='bold')
plt.yticks(fontsize=12, fontweight='bold')

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper left', frameon=True)
plt.tight_layout()
plt.show()


# ===== SUMMARY TABLE =====
best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]
best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary_table = pd.DataFrame({
    'Parameter': [
        'Pulse On (µs)', 'Pulse Off (µs)', 'Servo Voltage (V)',
        'Current (A)', 'Angle (°)', 'Kerf Width (mm)', 'MRR (mm³/min)'
    ],
    'Min Kerf': [
        best_quality['Pulse_On'], best_quality['Pulse_Off'], best_quality['Servo_V'],
        best_quality['Current'], best_quality['Angle'],
        best_quality['Kerf Width'], best_quality['MRR']
    ],
    'Max MRR': [
        best_productivity['Pulse_On'], best_productivity['Pulse_Off'], best_productivity['Servo_V'],
        best_productivity['Current'], best_productivity['Angle'],
        best_productivity['Kerf Width'], best_productivity['MRR']
    ],
    'Balanced': [
        balanced['Pulse_On'], balanced['Pulse_Off'], balanced['Servo_V'],
        balanced['Current'], balanced['Angle'],
        balanced['Kerf Width'], balanced['MRR']
    ]
})

print("\n" + "="*70)
print("Optimal EDM Process Parameter Settings")
print("="*70)
print(summary_table.to_string(index=False))