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

# ===== DATAFRAME =====
df = pd.DataFrame(data)
SR = df['Surface Roughness'].values
kerf_angle = df['Kerf Width'].values


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
pareto_points = pareto_frontier(SR, kerf_angle, maximizeX=False, maximizeY=False)
plt.figure(figsize=(10, 8))
plt.scatter(SR, kerf_angle, color='gray', alpha=0.6, s=50, label='Experimental Points', zorder=1)
plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=80, edgecolors='black', linewidth=1.2,
            label='Pareto-optimal Solutions', zorder=3)
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8, zorder=2)
best_quality_idx = np.argmin(SR)
min_kerf_idx = np.argmin(kerf_angle)
balanced_idx = np.argmin(np.sqrt((SR - SR.min())**2 + (kerf_angle - kerf_angle.min())**2))

key_points = [
    (SR[best_quality_idx], kerf_angle[best_quality_idx],
     "Best Ra\n({:.2f} μm)".format(SR[best_quality_idx]),
     's', 'red', (40, 20)),

    (SR[min_kerf_idx], kerf_angle[min_kerf_idx],
     "Min Kerf Width\n({:.3f} mm)".format(kerf_angle[min_kerf_idx]),
     '^', 'green', (-60, -30)),

    (SR[balanced_idx], kerf_angle[balanced_idx],
     "Balanced\n(SR={:.2f}, Kerf={:.3f} mm)".format(SR[balanced_idx],
                                                    kerf_angle[balanced_idx]),
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
plt.xlabel('Surface Roughness',
           fontsize=14, fontweight='bold', fontname='DejaVu Sans')
plt.ylabel('Kerf Width ',
           fontsize=14, fontweight='bold', fontname='DejaVu Sans')
plt.title('Pareto Frontier for EDM Process\nTrade-off between Surface Roughness and Kerf Width',
          fontsize=16, fontweight='bold', fontname='DejaVu Sans', pad=20)

plt.grid(True, linestyle='--', alpha=0.4)
plt.legend(loc='upper right', frameon=True, prop={'family': 'DejaVu Sans', 'size': 10})
plt.tick_params(axis='both', labelsize=12)
plt.xticks(fontname='DejaVu Sans', fontweight='bold')
plt.yticks(fontname='DejaVu Sans', fontweight='bold')

plt.xlim(SR.min() * 0.95, SR.max() * 1.05)
plt.ylim(kerf_angle.min() * 0.95, kerf_angle.max() * 1.05)
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
        'Surface Roughness ',
        'Kerf Width '
    ],
    'Best Quality (Min SR)': [
        best_quality['Pulse_On'],
        best_quality['Pulse_Off'],
        best_quality['Servo_V'],
        best_quality['Current'],
        best_quality['Angle'],
        best_quality['Surface Roughness'],
        best_quality['Kerf Width']
    ],
    'Min Kerf (mm)': [
        best_kerf['Pulse_On'],
        best_kerf['Pulse_Off'],
        best_kerf['Servo_V'],
        best_kerf['Current'],
        best_kerf['Angle'],
        best_kerf['Surface Roughness'],
        best_kerf['Kerf Width']
    ],
    'Balanced': [
        balanced['Pulse_On'],
        balanced['Pulse_Off'],
        balanced['Servo_V'],
        balanced['Current'],
        balanced['Angle'],
        balanced['Surface Roughness'],
        balanced['Kerf Width']
    ]
})


print("\n" + "="*70)
print("Table 3: Optimal Process Parameter Settings for Different Objectives")
print("="*70)
print(summary_table.to_string(index=False))
