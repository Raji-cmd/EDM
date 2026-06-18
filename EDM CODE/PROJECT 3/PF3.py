import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import ConvexHull

# ===== DATASET =====
data = {
    "Pulse_On": [12,12,8,8,12,12,8,8,12,12,16,16,8,12,12,16,12,12,16,16,8,8,14,14,12,12,8,8,12,12,8,12,12,16,16,16,16,16,12,12,10,10],
    "Pulse_Off": [12,12,14,14,12,12,10,10,14,14,10,10,10,13,13,10,12,12,14,14,14,14,12,12,12,12,10,10,11,11,10,12,11,10,14,14,14,14,12,12,12,12],
    "Servo_V": [55,60,70,50,55,60,70,70,60,70,70,70,70,60,60,70,60,60,50,50,50,70,60,60,65,65,50,50,60,60,50,60,60,50,70,70,50,50,50,45,60,60],
    "Current": [4,3.5,5,5,4,3.5,5,5,4,5,5,5,5,4,4,3.5,4.5,4.5,3,3,3,3,4,4,4,4,3,3,4,4,3,4,4,5,3,3,5,5,4,4.5,4,4],
    "Angle": [60,60,30,90,60,60,90,30,60,90,30,90,30,60,60,90,60,60,90,90,90,90,60,60,60,60,30,30,75,75,30,60,60,90,30,30,30,30,30,45,60,60]
}

df = pd.DataFrame(data)

# =====================================================
# ENSEMBLE PREDICTIONS
# =====================================================

SU = np.array([
4.129,3.796,2.507,3.664,4.129,3.796,2.767,3.053,3.490,3.748,
3.425,3.314,3.053,3.553,3.553,3.349,3.898,3.898,4.500,4.500,
2.428,2.317,4.031,4.031,3.460,3.460,4.044,4.044,4.355,4.355,
4.044,3.940,4.595,4.895,3.010,3.010,4.719,4.719,3.865,3.917,
3.611,3.611
])

SVR = np.array([
5.879,5.630,4.483,5.593,5.879,5.630,5.023,5.209,5.423,5.611,
5.515,5.490,5.209,5.584,5.584,5.307,5.784,5.784,5.828,5.828,
4.525,4.415,5.777,5.777,5.483,5.483,5.895,5.895,6.118,6.118,
5.895,5.788,6.263,6.442,4.900,4.900,6.247,6.247,5.767,5.827,
5.561,5.561
])

KC = np.array([
1.367,1.282,0.955,1.259,1.367,1.282,1.000,0.994,1.203,1.275,
1.188,1.152,0.994,1.234,1.234,1.180,1.304,1.304,1.476,1.476,
0.932,0.915,1.336,1.336,1.207,1.207,1.356,1.356,1.442,1.442,
1.356,1.325,1.494,1.566,1.089,1.089,1.528,1.528,1.308,1.330,
1.242,1.242
])

df["SU"] = SU
df["SVR"] = SVR
df["KC"] = KC

# ===== PARETO OPTIMAL IDENTIFICATION =====
def identify_pareto(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = np.any(scores[is_efficient] < c, axis=1) | \
                                         np.any(scores[is_efficient] > c, axis=1)
            is_efficient[i] = True
    return is_efficient

scores = np.column_stack([KC, -SVR])
pareto_mask = identify_pareto(scores)
pareto_points = np.column_stack([KC[pareto_mask], SVR[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 1].argsort()]

# ===== PLOT =====
plt.figure(figsize=(10, 8))
plt.rcParams['font.family'] = 'DejaVu Sans'

plt.scatter(KC, SVR, color='gray', alpha=0.6, s=50, label='Experimental Points', zorder=1)
plt.scatter(pareto_points[:, 0], pareto_points[:, 1], color='blue', s=80,
            edgecolors='black', linewidth=1, label='Pareto-optimal Solutions', zorder=3)
plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.8, zorder=2)

best_quality_idx = np.argmin(KC)
best_productivity_idx = np.argmax(SVR)
balanced_idx = np.argmin(np.sqrt((KC - KC.min())**2 + (SVR - SVR.max())**2))

key_points = [
    (KC[best_quality_idx], SVR[best_quality_idx], "Min KC", 's', 'red'),
    (KC[best_productivity_idx], SVR[best_productivity_idx], "Max SVR", '^', 'green'),
    (KC[balanced_idx], SVR[balanced_idx], "Balanced", 'D', 'purple')
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

plt.xlabel('KC', fontsize=14, fontweight='bold')
plt.ylabel('SVR ', fontsize=14, fontweight='bold')
plt.title('Pareto Frontier for EDM Process\nTrade-off between SVR and KC',
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
        'Current (A)', 'Angle (°)', 'KC (mm)', 'SVR (mm³/min)'
    ],
    'Min KC': [
        best_quality['Pulse_On'], best_quality['Pulse_Off'], best_quality['Servo_V'],
        best_quality['Current'], best_quality['Angle'],
        best_quality['KC'], best_quality['SVR']
    ],
    'Max SVR': [
        best_productivity['Pulse_On'], best_productivity['Pulse_Off'], best_productivity['Servo_V'],
        best_productivity['Current'], best_productivity['Angle'],
        best_productivity['KC'], best_productivity['SVR']
    ],
    'Balanced': [
        balanced['Pulse_On'], balanced['Pulse_Off'], balanced['Servo_V'],
        balanced['Current'], balanced['Angle'],
        balanced['KC'], balanced['SVR']
    ]
})

print("\n" + "="*70)
print("Optimal EDM Process Parameter Settings")
print("="*70)
print(summary_table.to_string(index=False))