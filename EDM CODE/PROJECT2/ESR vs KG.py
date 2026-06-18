import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ===== DATASET =====
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
"ESR": [4.174, 3.775, 2.457, 3.778, 4.174, 3.775, 2.716, 3.080, 3.530, 3.868,
        3.426, 3.316, 3.080, 3.501, 3.501, 3.299, 3.856, 3.856, 4.513, 4.513,
        2.229, 2.272, 4.050, 4.050, 3.422, 3.422, 3.999, 3.999, 4.474, 4.474,
        3.999, 3.974, 4.883, 4.961, 3.004, 3.004, 4.747, 4.747, 3.877, 3.842,
        3.574, 3.574],
"SRR": [5.918, 5.631, 4.388, 5.686, 5.918, 5.631, 4.953, 5.214, 5.434, 5.735,
        5.540, 5.526, 5.214, 5.573, 5.573, 5.202, 5.729, 5.729, 5.829, 5.829,
        4.309, 4.382, 5.753, 5.753, 5.440, 5.440, 5.821, 5.821, 6.220, 6.220,
        5.821, 5.813, 6.467, 6.485, 4.899, 4.899, 6.277, 6.277, 5.759, 5.725,
        5.563, 5.563],
"KG": [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
       1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
       0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
       1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
       1.222, 1.241]
}

# ===== DATAFRAME =====
df = pd.DataFrame(data)
ESR = df['ESR'].values
KG = df['KG'].values


# ===== PARETO FUNCTION =====
def pareto_frontier(X, Y, maximizeX=False, maximizeY=False):
    """Return the Pareto frontier points (for minimization problems by default)."""
    # Sort by X
    sorted_points = sorted(zip(X, Y), reverse=maximizeX)
    pareto_front = [sorted_points[0]]
    for x, y in sorted_points[1:]:
        if maximizeY:
            if y >= pareto_front[-1][1]:
                pareto_front.append((x, y))
        else:
            if y <= pareto_front[-1][1]:
                pareto_front.append((x, y))
    return np.array(pareto_front)

# ===== COMBINE PARETO FRONTIER =====
pareto_points = pareto_frontier(ESR, KG, maximizeX=False, maximizeY=False)
plt.figure(figsize=(10, 8))
plt.scatter(ESR, KG, color='gray', alpha=0.6, s=50, label='Experimental Points', zorder=1)
plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=80, edgecolors='black', linewidth=1.2,
            label='Pareto-optimal Solutions', zorder=3)
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8, zorder=2)
best_quality_idx = np.argmin(ESR)
min_kerf_idx = np.argmin(KG)
balanced_idx = np.argmin(np.sqrt((ESR - ESR.min())**2 + (KG - KG.min())**2))

key_points = [
    (ESR[best_quality_idx], KG[best_quality_idx],
     "Best ESR\n({:.2f} μm)".format(ESR[best_quality_idx]),
     's', 'red', (40, 20)),

    (ESR[min_kerf_idx], KG[min_kerf_idx],
     "Min KG\n({:.3f} mm)".format(KG[min_kerf_idx]),
     '^', 'green', (-60, -30)),

    (ESR[balanced_idx], KG[balanced_idx],
     "Balanced\n(ESR={:.2f}, KG={:.3f} mm)".format(ESR[balanced_idx],
                                                    KG[balanced_idx]),
     'D', 'purple', (50, -40))
]


for x, y, label, marker, color, offset in key_points:
    plt.scatter(x, y, s=150, marker=marker, color=color, edgecolors='black', linewidth=2, zorder=4)
    plt.annotate(label,
                 xy=(x, y), xytext=offset, textcoords='offset points',
                 fontsize=10, fontweight='bold',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8),
                 arrowprops=dict(arrowstyle='->', lw=1.2, color='black'))

if len(pareto_points) >= 3:
    hull = ConvexHull(pareto_points)
    plt.fill(pareto_points[hull.vertices, 0], pareto_points[hull.vertices, 1],
             'blue', alpha=0.1, label='Feasible Region')

# ===== STYLE =====
plt.xlabel('ESR',
           fontsize=14, fontweight='bold', fontname='DejaVu Sans')
plt.ylabel('KG ',
           fontsize=14, fontweight='bold', fontname='DejaVu Sans')
plt.title('Pareto Frontier for EDM Process\nTrade-off between ESR and KG',
          fontsize=16, fontweight='bold', fontname='DejaVu Sans', pad=20)

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right', frameon=True, prop={'family': 'DejaVu Sans', 'size': 10})
plt.tick_params(axis='both', labelsize=12)
plt.xticks(fontname='DejaVu Sans', fontweight='bold')
plt.yticks(fontname='DejaVu Sans', fontweight='bold')

plt.xlim(ESR.min() * 0.95, ESR.max() * 1.05)
plt.ylim(KG.min() * 0.95, KG.max() * 1.05)
plt.tight_layout()
plt.show()

# ====================== Summary Table ======================
best_quality = df.iloc[best_quality_idx]   # Minimum Ra
best_kerf = df.iloc[min_kerf_idx]          # Minimum Kerf
balanced = df.iloc[balanced_idx]           # Balanced (Ra + Kerf)

summary_table = pd.DataFrame({
    'Parameter': [
        'Pulse On (µs)',
        'Pulse Off (µs)',
        'Servo Voltage (V)',
        'Current (A)',
        'Angle (°)',
        'ESR',
        'KG '
    ],
    'Best Quality (Min ESR)': [
        best_quality['Pulse_On'],
        best_quality['Pulse_Off'],
        best_quality['Servo_V'],
        best_quality['Current'],
        best_quality['Angle'],
        best_quality['ESR'],
        best_quality['KG']
    ],
    'Min KG (mm)': [
        best_kerf['Pulse_On'],
        best_kerf['Pulse_Off'],
        best_kerf['Servo_V'],
        best_kerf['Current'],
        best_kerf['Angle'],
        best_kerf['ESR'],
        best_kerf['KG']
    ],
    'Balanced': [
        balanced['Pulse_On'],
        balanced['Pulse_Off'],
        balanced['Servo_V'],
        balanced['Current'],
        balanced['Angle'],
        balanced['ESR'],
        balanced['KG']
    ]
})


print("\n" + "="*70)
print("Table 3: Optimal Process Parameter Settings for Different Objectives")
print("="*70)
print(summary_table.to_string(index=False))
