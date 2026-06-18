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
    "Surface Waviness": [4.174, 3.775, 2.457, 3.778, 4.174, 3.775, 2.716, 3.080, 3.530, 3.868,
           3.426, 3.316, 3.080, 3.501, 3.501, 3.299, 3.856, 3.856, 4.513, 4.513,
           2.229, 2.272, 4.050, 4.050, 3.422, 3.422, 3.999, 3.999, 4.474, 4.474,
           3.999, 3.974, 4.883, 4.961, 3.004, 3.004, 4.747, 4.747, 3.877, 3.842,
           3.574, 3.574],
    "Material Vaporization Rate": [5.918, 5.631, 4.388, 5.686, 5.918, 5.631, 4.953, 5.214, 5.434, 5.735,
            5.540, 5.526, 5.214, 5.573, 5.573, 5.202, 5.729, 5.729, 5.829, 5.829,
            4.309, 4.382, 5.753, 5.753, 5.440, 5.440, 5.821, 5.821, 6.220, 6.220,
            5.821, 5.813, 6.467, 6.485, 4.899, 4.899, 6.277, 6.277, 5.759, 5.725,
            5.563, 5.563],
    "KD": [1.378, 1.277, 0.943, 1.289, 1.400, 1.286, 1.004, 0.965, 1.227, 1.311,
           1.194, 1.135, 1.026, 1.249, 1.219, 1.155, 1.305, 1.283, 1.489, 1.467,
           0.879, 0.898, 1.347, 1.325, 1.169, 1.191, 1.556, 1.249, 1.322, 1.576,
           1.272, 1.347, 1.615, 1.584, 1.074, 1.093, 1.528, 1.551, 1.314, 1.291,
           1.222, 1.241]

}

df = pd.DataFrame(data)
SW = df["Surface Waviness"].values
MVR = df["Material Vaporization Rate"].values

# ===== PARETO FRONTIER IDENTIFICATION =====
def identify_pareto(scores):
    is_efficient = np.ones(scores.shape[0], dtype=bool)
    for i, c in enumerate(scores):
        if is_efficient[i]:
            is_efficient[is_efficient] = (
                np.any(scores[is_efficient] < c, axis=1) |
                np.any(scores[is_efficient] > c, axis=1)
            )
            is_efficient[i] = True
    return is_efficient

scores = np.column_stack([SW, -MVR])
pareto_mask = identify_pareto(scores)

pareto_points = np.column_stack([SW[pareto_mask], MVR[pareto_mask]])
pareto_points = pareto_points[pareto_points[:, 0].argsort()]
best_quality_idx = np.argmin(SW)
best_productivity_idx = np.argmax(MVR)
balanced_idx = np.argmin(np.sqrt((SW - SW.min())**2 + (MVR - MVR.max())**2))


# =============== Plotting Pareto Front ===================
plt.figure(figsize=(10, 8))
plt.scatter(SW, MVR, color='gray', alpha=0.6, s=50, label="Experimental Points")

plt.scatter(pareto_points[:, 0], pareto_points[:, 1],
            color='blue', s=80, edgecolors='black', linewidth=1,
            label="Pareto-optimal Solutions")

plt.plot(pareto_points[:, 0], pareto_points[:, 1], 'b--', lw=1.6)
special = [
    (best_quality_idx, "Best Quality", 's', 'red'),
    (best_productivity_idx, "Best Productivity", '^', 'green'),
    (balanced_idx, "Balanced", 'D', 'purple')
]

for idx, name, marker, color in special:
    x, y = SW[idx], MVR[idx]
    plt.scatter(x, y, s=150, marker=marker, color=color,
                edgecolors='black', linewidth=2)
    plt.annotate(f"{name}\nSW={x:.2f}, MVR={y:.2f}",
                 xy=(x, y), xytext=(20, -40),
                 textcoords='offset points', fontsize=10,
                 bbox=dict(facecolor='white', alpha=0.8),
                 arrowprops=dict(arrowstyle='->'))

plt.xlabel("Surface Waviness(µm)", fontsize=14, fontweight="bold")
plt.ylabel("Material Vaporization Rate (mm³/min)", fontsize=14, fontweight="bold")
plt.xticks(fontsize=12, fontweight="bold")
plt.yticks(fontsize=12, fontweight="bold")
plt.title("Pareto Frontier for EDM Process\nTrade-off between SW and MVR"
          , fontsize=16, fontweight="bold")
plt.grid(True, linestyle='--', alpha=0.4)
plt.legend()
plt.tight_layout()
plt.show()

# ===== SUMMARY =====
print("\n=========== Summary ===========")
print(f"Total Pareto-optimal solutions = {pareto_mask.sum()}")

best_quality = df.iloc[best_quality_idx]
best_productivity = df.iloc[best_productivity_idx]
balanced = df.iloc[balanced_idx]

summary = pd.DataFrame({
    "Parameter": df.columns,
    "Best Quality (Min SW)": best_quality.values,
    "Best Productivity (Max MVR)": best_productivity.values,
    "Balanced": balanced.values
})

print(summary.to_string(index=False))
